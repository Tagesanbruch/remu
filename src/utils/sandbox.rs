use crate::common::PAddr;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

const SERIAL_WINDOW_LIMIT: usize = 8192;
const HOT_REGION_SHIFT: u32 = 12;

#[derive(Clone)]
struct PhaseEvent {
    name: String,
    marker: String,
    inst: u64,
    host_us: u128,
}

#[derive(Default, Clone)]
struct IoStats {
    read_count: u64,
    write_count: u64,
    read_bytes: u64,
    write_bytes: u64,
}

#[derive(Default, Clone)]
struct TlbKindStats {
    hits: u64,
    misses: u64,
    walks: u64,
    success: u64,
    fail: u64,
}

#[derive(Clone)]
pub struct LazyPageEvent {
    pub page_idx: usize,
    pub paddr: PAddr,
    pub first_touch_inst: u64,
    pub first_touch_us: u128,
    pub access_kind: &'static str,
    pub stall_us: u128,
    pub touch_count: u64,
    pub dirty: bool,
    pub prefetched: bool,
}

struct SandboxMetrics {
    enabled: bool,
    report_dir: Option<PathBuf>,
    stop_pattern: Option<String>,
    serial_window: String,
    phase_seen: HashSet<String>,
    phases: Vec<PhaseEvent>,
    start: Instant,
    stop_triggered: bool,
    mmio: HashMap<String, IoStats>,
    paddr: IoStats,
    paddr_pages: HashMap<u64, IoStats>,
    interrupts: HashMap<String, u64>,
    tlb: HashMap<i32, TlbKindStats>,
    va_regions: HashMap<u64, u64>,
    pa_regions: HashMap<u64, u64>,
    lazy_events: Vec<LazyPageEvent>,
    lazy_prefetch_pages: u64,
    lazy_resident_pages: u64,
}

impl SandboxMetrics {
    fn new(report_dir: Option<PathBuf>, stop_pattern: Option<String>) -> Self {
        Self {
            enabled: report_dir.is_some() || stop_pattern.is_some(),
            report_dir,
            stop_pattern,
            serial_window: String::new(),
            phase_seen: HashSet::new(),
            phases: Vec::new(),
            start: Instant::now(),
            stop_triggered: false,
            mmio: HashMap::new(),
            paddr: IoStats::default(),
            paddr_pages: HashMap::new(),
            interrupts: HashMap::new(),
            tlb: HashMap::new(),
            va_regions: HashMap::new(),
            pa_regions: HashMap::new(),
            lazy_events: Vec::new(),
            lazy_prefetch_pages: 0,
            lazy_resident_pages: 0,
        }
    }

    fn elapsed_us(&self) -> u128 {
        self.start.elapsed().as_micros()
    }

    fn record_phase(&mut self, name: &str, marker: &str) {
        if self.phase_seen.insert(name.to_string()) {
            self.phases.push(PhaseEvent {
                name: name.to_string(),
                marker: marker.to_string(),
                inst: crate::cpu::execute::guest_inst_count(),
                host_us: self.elapsed_us(),
            });
        }
    }
}

lazy_static::lazy_static! {
    static ref METRICS: Mutex<SandboxMetrics> = Mutex::new(SandboxMetrics::new(None, None));
}

pub fn init(report_dir: Option<PathBuf>, stop_pattern: Option<String>) {
    let mut metrics = METRICS.lock().unwrap();
    *metrics = SandboxMetrics::new(report_dir, stop_pattern);
}

pub fn observe_serial_byte(ch: u8) {
    let mut metrics = METRICS.lock().unwrap();
    if !metrics.enabled {
        return;
    }

    metrics.serial_window.push(ch as char);
    if metrics.serial_window.len() > SERIAL_WINDOW_LIMIT {
        let drop_len = metrics.serial_window.len() - SERIAL_WINDOW_LIMIT;
        metrics.serial_window.drain(..drop_len);
    }

    for (name, marker) in [
        ("opensbi", "OpenSBI v"),
        ("kernel", "Linux version"),
        ("init", "Run /init as init process"),
        ("welcome", "Welcome to REMU Linux"),
        ("shell", "~ #"),
    ] {
        if metrics.serial_window.contains(marker) {
            metrics.record_phase(name, marker);
        }
    }

    let stop_hit = metrics
        .stop_pattern
        .as_deref()
        .map(|pattern| pattern_matches(pattern, &metrics.serial_window))
        .unwrap_or(false);
    if stop_hit && !metrics.stop_triggered {
        metrics.stop_triggered = true;
        crate::Log!(
            "REMU-Sandbox stop-on-serial matched at inst={} host_us={}",
            crate::cpu::execute::guest_inst_count(),
            metrics.elapsed_us()
        );
        crate::utils::set_state(crate::common::RemuState::Stop);
    }
}

fn pattern_matches(pattern: &str, text: &str) -> bool {
    for alt in pattern.split('|') {
        let alt = alt.trim();
        if alt.is_empty() {
            continue;
        }
        if ordered_fragments_match(alt, text) {
            return true;
        }
    }
    false
}

fn ordered_fragments_match(pattern: &str, text: &str) -> bool {
    if !pattern.contains(".*") {
        return text.contains(pattern);
    }

    let mut pos = 0;
    for frag in pattern.split(".*") {
        if frag.is_empty() {
            continue;
        }
        let Some(idx) = text[pos..].find(frag) else {
            return false;
        };
        pos += idx + frag.len();
    }
    true
}

pub fn record_mmio(device: &str, is_write: bool, len: usize) {
    let mut metrics = METRICS.lock().unwrap();
    if !metrics.enabled {
        return;
    }

    let stats = metrics.mmio.entry(device.to_string()).or_default();
    if is_write {
        stats.write_count += 1;
        stats.write_bytes += len as u64;
    } else {
        stats.read_count += 1;
        stats.read_bytes += len as u64;
    }
}

pub fn record_paddr(addr: PAddr, len: usize, is_write: bool) {
    let mut metrics = METRICS.lock().unwrap();
    if !metrics.enabled {
        return;
    }

    if is_write {
        metrics.paddr.write_count += 1;
        metrics.paddr.write_bytes += len as u64;
    } else {
        metrics.paddr.read_count += 1;
        metrics.paddr.read_bytes += len as u64;
    }

    let page = addr >> HOT_REGION_SHIFT;
    let page_stats = metrics.paddr_pages.entry(page).or_default();
    if is_write {
        page_stats.write_count += 1;
        page_stats.write_bytes += len as u64;
    } else {
        page_stats.read_count += 1;
        page_stats.read_bytes += len as u64;
    }
}

pub fn record_interrupt(cause: u64, is_intr: bool) {
    let mut metrics = METRICS.lock().unwrap();
    if !metrics.enabled {
        return;
    }

    let key = format!("{}:{}", if is_intr { "intr" } else { "exception" }, cause);
    *metrics.interrupts.entry(key).or_default() += 1;
}

pub fn record_tlb(kind: i32, hit: bool) {
    let mut metrics = METRICS.lock().unwrap();
    if !metrics.enabled {
        return;
    }

    let stats = metrics.tlb.entry(kind).or_default();
    if hit {
        stats.hits += 1;
    } else {
        stats.misses += 1;
    }
}

pub fn record_page_walk(kind: i32) {
    let mut metrics = METRICS.lock().unwrap();
    if !metrics.enabled {
        return;
    }
    metrics.tlb.entry(kind).or_default().walks += 1;
}

pub fn record_mmu_translate(vaddr: u64, paddr: u64, kind: i32, success: bool) {
    let mut metrics = METRICS.lock().unwrap();
    if !metrics.enabled {
        return;
    }

    let stats = metrics.tlb.entry(kind).or_default();
    if success {
        stats.success += 1;
        *metrics
            .va_regions
            .entry(vaddr >> HOT_REGION_SHIFT)
            .or_default() += 1;
        *metrics
            .pa_regions
            .entry(paddr >> HOT_REGION_SHIFT)
            .or_default() += 1;
    } else {
        stats.fail += 1;
    }
}

pub fn record_lazy_page(
    page_idx: usize,
    paddr: PAddr,
    access_kind: &'static str,
    stall_us: u128,
    dirty: bool,
    prefetched: bool,
) {
    let mut metrics = METRICS.lock().unwrap();
    if !metrics.enabled {
        return;
    }

    metrics.lazy_resident_pages += 1;
    if prefetched {
        metrics.lazy_prefetch_pages += 1;
    }
    let first_touch_us = metrics.elapsed_us();
    metrics.lazy_events.push(LazyPageEvent {
        page_idx,
        paddr,
        first_touch_inst: crate::cpu::execute::guest_inst_count(),
        first_touch_us,
        access_kind,
        stall_us,
        touch_count: 1,
        dirty,
        prefetched,
    });
}

pub fn export_reports() {
    let metrics = METRICS.lock().unwrap();
    let Some(dir) = metrics.report_dir.as_ref() else {
        return;
    };
    if let Err(err) = fs::create_dir_all(dir) {
        eprintln!(
            "Failed to create sandbox report dir {}: {}",
            dir.display(),
            err
        );
        return;
    }

    let _ = write_boot_phase(dir, &metrics);
    let _ = write_tlb_summary(dir, &metrics);
    let _ = write_page_walk_summary(dir, &metrics);
    let _ = write_mmio_hotspot(dir, &metrics);
    let _ = write_interrupt_summary(dir, &metrics);
    let _ = write_paddr_summary(dir, &metrics);
    let _ = write_hot_regions(dir, "va_hot_region.csv", &metrics.va_regions);
    let _ = write_hot_regions(dir, "pa_hot_region.csv", &metrics.pa_regions);
    let _ = write_lazy_events(dir, &metrics);
    let _ = write_run_summary(dir, &metrics);
}

fn create_report_file(dir: &std::path::Path, name: &str) -> std::io::Result<File> {
    File::create(dir.join(name))
}

fn write_boot_phase(dir: &std::path::Path, metrics: &SandboxMetrics) -> std::io::Result<()> {
    let mut file = create_report_file(dir, "boot_phase_timeline.csv")?;
    writeln!(file, "phase,marker,guest_inst,host_us")?;
    for phase in &metrics.phases {
        writeln!(
            file,
            "{},{},{},{}",
            phase.name,
            csv_escape(&phase.marker),
            phase.inst,
            phase.host_us
        )?;
    }
    Ok(())
}

fn write_tlb_summary(dir: &std::path::Path, metrics: &SandboxMetrics) -> std::io::Result<()> {
    let mut file = create_report_file(dir, "tlb_summary.csv")?;
    writeln!(file, "kind,hits,misses,hit_rate,success,fail")?;
    for (kind, stats) in sorted_tlb(&metrics.tlb) {
        let total = stats.hits + stats.misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            stats.hits as f64 / total as f64
        };
        writeln!(
            file,
            "{},{},{},{:.6},{},{}",
            kind, stats.hits, stats.misses, hit_rate, stats.success, stats.fail
        )?;
    }
    Ok(())
}

fn write_page_walk_summary(dir: &std::path::Path, metrics: &SandboxMetrics) -> std::io::Result<()> {
    let mut file = create_report_file(dir, "page_walk_summary.csv")?;
    writeln!(file, "kind,page_walks")?;
    for (kind, stats) in sorted_tlb(&metrics.tlb) {
        writeln!(file, "{},{}", kind, stats.walks)?;
    }
    Ok(())
}

fn write_mmio_hotspot(dir: &std::path::Path, metrics: &SandboxMetrics) -> std::io::Result<()> {
    let mut file = create_report_file(dir, "mmio_hotspot.csv")?;
    writeln!(
        file,
        "device,read_count,write_count,read_bytes,write_bytes,total_count"
    )?;
    let mut rows: Vec<_> = metrics.mmio.iter().collect();
    rows.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.read_count + stats.write_count));
    for (device, stats) in rows {
        writeln!(
            file,
            "{},{},{},{},{},{}",
            csv_escape(device),
            stats.read_count,
            stats.write_count,
            stats.read_bytes,
            stats.write_bytes,
            stats.read_count + stats.write_count
        )?;
    }
    Ok(())
}

fn write_interrupt_summary(dir: &std::path::Path, metrics: &SandboxMetrics) -> std::io::Result<()> {
    let mut file = create_report_file(dir, "interrupt_summary.csv")?;
    writeln!(file, "kind,cause,count")?;
    let mut rows: Vec<_> = metrics.interrupts.iter().collect();
    rows.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
    for (key, count) in rows {
        let mut parts = key.split(':');
        let kind = parts.next().unwrap_or("unknown");
        let cause = parts.next().unwrap_or("0");
        writeln!(file, "{},{},{}", kind, cause, count)?;
    }
    Ok(())
}

fn write_paddr_summary(dir: &std::path::Path, metrics: &SandboxMetrics) -> std::io::Result<()> {
    let mut file = create_report_file(dir, "paddr_summary.csv")?;
    writeln!(
        file,
        "scope,page,paddr,read_count,write_count,read_bytes,write_bytes,total_count"
    )?;
    writeln!(
        file,
        "total,,,{},{},{},{},{}",
        metrics.paddr.read_count,
        metrics.paddr.write_count,
        metrics.paddr.read_bytes,
        metrics.paddr.write_bytes,
        metrics.paddr.read_count + metrics.paddr.write_count
    )?;
    let mut rows: Vec<_> = metrics.paddr_pages.iter().collect();
    rows.sort_by_key(|(_, stats)| std::cmp::Reverse(stats.read_count + stats.write_count));
    for (page, stats) in rows.into_iter().take(256) {
        writeln!(
            file,
            "page,{},0x{:016x},{},{},{},{},{}",
            page,
            page << HOT_REGION_SHIFT,
            stats.read_count,
            stats.write_count,
            stats.read_bytes,
            stats.write_bytes,
            stats.read_count + stats.write_count
        )?;
    }
    Ok(())
}

fn write_hot_regions(
    dir: &std::path::Path,
    name: &str,
    regions: &HashMap<u64, u64>,
) -> std::io::Result<()> {
    let mut file = create_report_file(dir, name)?;
    writeln!(file, "page,addr,count")?;
    let mut rows: Vec<_> = regions.iter().collect();
    rows.sort_by_key(|(_, count)| std::cmp::Reverse(**count));
    for (page, count) in rows.into_iter().take(512) {
        writeln!(
            file,
            "{},0x{:016x},{}",
            page,
            page << HOT_REGION_SHIFT,
            count
        )?;
    }
    Ok(())
}

fn write_lazy_events(dir: &std::path::Path, metrics: &SandboxMetrics) -> std::io::Result<()> {
    let mut file = create_report_file(dir, "lazy_page_events.csv")?;
    writeln!(
        file,
        "page_idx,paddr,first_touch_inst,first_touch_us,access_kind,stall_us,touch_count,dirty,prefetched"
    )?;
    for event in &metrics.lazy_events {
        writeln!(
            file,
            "{},0x{:016x},{},{},{},{},{},{},{}",
            event.page_idx,
            event.paddr,
            event.first_touch_inst,
            event.first_touch_us,
            event.access_kind,
            event.stall_us,
            event.touch_count,
            event.dirty,
            event.prefetched
        )?;
    }
    Ok(())
}

fn write_run_summary(dir: &std::path::Path, metrics: &SandboxMetrics) -> std::io::Result<()> {
    let mut file = create_report_file(dir, "run_summary.json")?;
    let (tlb_hits, tlb_misses, page_walks) = crate::isa::riscv32::system::mmu::tlb_stats();
    writeln!(file, "{{")?;
    writeln!(file, "  \"xlen\": {},", crate::common::xlen())?;
    writeln!(
        file,
        "  \"guest_inst\": {},",
        crate::cpu::execute::guest_inst_count()
    )?;
    writeln!(file, "  \"host_us\": {},", metrics.elapsed_us())?;
    writeln!(file, "  \"stop_triggered\": {},", metrics.stop_triggered)?;
    writeln!(file, "  \"boot_phase_count\": {},", metrics.phases.len())?;
    writeln!(file, "  \"tlb_hits\": {},", tlb_hits)?;
    writeln!(file, "  \"tlb_misses\": {},", tlb_misses)?;
    writeln!(file, "  \"page_walks\": {},", page_walks)?;
    writeln!(
        file,
        "  \"lazy_resident_pages\": {},",
        metrics.lazy_resident_pages
    )?;
    writeln!(
        file,
        "  \"lazy_prefetch_pages\": {}",
        metrics.lazy_prefetch_pages
    )?;
    writeln!(file, "}}")?;
    Ok(())
}

fn sorted_tlb(tlb: &HashMap<i32, TlbKindStats>) -> Vec<(i32, TlbKindStats)> {
    let mut rows: Vec<_> = tlb
        .iter()
        .map(|(kind, stats)| (*kind, stats.clone()))
        .collect();
    rows.sort_by_key(|(kind, _)| *kind);
    rows
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

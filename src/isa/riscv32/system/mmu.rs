use super::csr::{CSR_MSTATUS, CSR_SATP};
use crate::common::{PAddr, PrivMode, VAddr, Word};
use crate::memory::paddr::paddr_read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

pub const MMU_DIRECT: i32 = 0;
pub const MMU_TRANSLATE: i32 = 1;
pub const MMU_FAIL: i32 = 2;

pub const MEM_TYPE_IFETCH: i32 = 0;
pub const MEM_TYPE_READ: i32 = 1;
pub const MEM_TYPE_WRITE: i32 = 2;

const PTE_V: Word = 1 << 0;
const PTE_R: Word = 1 << 1;
const PTE_W: Word = 1 << 2;
const PTE_X: Word = 1 << 3;

#[derive(Clone, Copy, Default)]
struct TlbEntry {
    valid: bool,
    satp_key: Word,
    vbase: VAddr,
    pbase: PAddr,
    page_mask: Word,
    perms: Word,
}

impl TlbEntry {
    fn allows(self, type_: i32) -> bool {
        match type_ {
            MEM_TYPE_IFETCH => (self.perms & PTE_X) != 0,
            MEM_TYPE_READ => (self.perms & PTE_R) != 0,
            MEM_TYPE_WRITE => (self.perms & PTE_W) != 0,
            _ => false,
        }
    }
}

struct SplitTlb {
    itlb: Vec<TlbEntry>,
    dtlb: Vec<TlbEntry>,
}

impl SplitTlb {
    fn new() -> Self {
        let entries = crate::generated::config::TLB_ENTRIES.max(1) as usize;
        Self {
            itlb: vec![TlbEntry::default(); entries],
            dtlb: vec![TlbEntry::default(); entries],
        }
    }

    fn flush(&mut self) {
        for entry in self.itlb.iter_mut().chain(self.dtlb.iter_mut()) {
            entry.valid = false;
        }
    }

    fn lookup(&mut self, satp_key: Word, vaddr: VAddr, type_: i32) -> Option<PAddr> {
        if !crate::generated::config::TLB {
            return None;
        }

        let bank = if type_ == MEM_TYPE_IFETCH {
            &mut self.itlb
        } else {
            &mut self.dtlb
        };
        let idx = tlb_index(vaddr, satp_key, bank.len());
        let entry = bank[idx];
        if !entry.valid || entry.satp_key != satp_key || !entry.allows(type_) {
            crate::utils::tlb_trace::trace_tlb(vaddr, 0, type_, false);
            crate::utils::sandbox::record_tlb(type_, false);
            TLB_MISS.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        if (vaddr & !entry.page_mask) != entry.vbase {
            crate::utils::tlb_trace::trace_tlb(vaddr, 0, type_, false);
            crate::utils::sandbox::record_tlb(type_, false);
            TLB_MISS.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let paddr = entry.pbase | (vaddr & entry.page_mask);
        crate::utils::tlb_trace::trace_tlb(vaddr, paddr, type_, true);
        crate::utils::sandbox::record_tlb(type_, true);
        TLB_HIT.fetch_add(1, Ordering::Relaxed);
        Some(paddr)
    }

    fn insert(
        &mut self,
        satp_key: Word,
        vaddr: VAddr,
        paddr: PAddr,
        page_mask: Word,
        perms: Word,
        type_: i32,
    ) {
        if !crate::generated::config::TLB {
            return;
        }

        let bank = if type_ == MEM_TYPE_IFETCH {
            &mut self.itlb
        } else {
            &mut self.dtlb
        };
        let idx = tlb_index(vaddr, satp_key, bank.len());
        bank[idx] = TlbEntry {
            valid: true,
            satp_key,
            vbase: vaddr & !page_mask,
            pbase: paddr & !page_mask,
            page_mask,
            perms,
        };
    }
}

lazy_static::lazy_static! {
    static ref TLB: Mutex<SplitTlb> = Mutex::new(SplitTlb::new());
}

static TLB_HIT: AtomicU64 = AtomicU64::new(0);
static TLB_MISS: AtomicU64 = AtomicU64::new(0);
static PAGE_WALK: AtomicU64 = AtomicU64::new(0);

pub fn isa_mmu_check(
    cpu: &crate::cpu::state::CpuState,
    _vaddr: VAddr,
    _len: usize,
    _type: i32,
) -> i32 {
    let satp = cpu.csr[CSR_SATP as usize];
    let mode = cpu.mode;
    let _mstatus = cpu.csr[CSR_MSTATUS as usize];

    if satp_mode(satp) != 0 && mode != PrivMode::Machine {
        MMU_TRANSLATE
    } else {
        MMU_DIRECT
    }
}

pub fn isa_mmu_translate(
    cpu: &crate::cpu::state::CpuState,
    vaddr: VAddr,
    _len: usize,
    type_: i32,
) -> Result<PAddr, Word> {
    let satp = cpu.csr[CSR_SATP as usize];
    let key = satp_key(satp);

    if let Some(paddr) = TLB.lock().unwrap().lookup(key, vaddr, type_) {
        crate::utils::mmu_trace::trace_mmu(vaddr, paddr, type_, true);
        crate::utils::sandbox::record_mmu_translate(vaddr, paddr, type_, true);
        return Ok(paddr);
    }

    PAGE_WALK.fetch_add(1, Ordering::Relaxed);
    crate::utils::sandbox::record_page_walk(type_);
    let walk = if crate::generated::config::RV64 {
        translate_sv39(satp, vaddr, type_)
    } else {
        translate_sv32(satp, vaddr, type_)
    };

    match walk {
        Ok(result) => {
            TLB.lock().unwrap().insert(
                key,
                vaddr,
                result.paddr,
                result.page_mask,
                result.perms,
                type_,
            );
            crate::utils::mmu_trace::trace_mmu(vaddr, result.paddr, type_, true);
            crate::utils::sandbox::record_mmu_translate(vaddr, result.paddr, type_, true);
            Ok(result.paddr)
        }
        Err(cause) => {
            crate::utils::mmu_trace::trace_mmu(vaddr, 0, type_, false);
            crate::utils::sandbox::record_mmu_translate(vaddr, 0, type_, false);
            Err(cause)
        }
    }
}

pub fn flush_tlb() {
    TLB.lock().unwrap().flush();
    crate::utils::tlb_trace::trace_tlb_flush();
}

pub fn tlb_stats() -> (u64, u64, u64) {
    (
        TLB_HIT.load(Ordering::Relaxed),
        TLB_MISS.load(Ordering::Relaxed),
        PAGE_WALK.load(Ordering::Relaxed),
    )
}

struct WalkResult {
    paddr: PAddr,
    page_mask: Word,
    perms: Word,
}

fn translate_sv32(satp: Word, vaddr: VAddr, type_: i32) -> Result<WalkResult, Word> {
    let ppn_base = satp & 0x003f_ffff;
    let vpn = [(vaddr >> 12) & 0x3ff, (vaddr >> 22) & 0x3ff];
    let mut pte_addr = (ppn_base << 12) + vpn[1] * 4;

    for level in (0..=1).rev() {
        let pte = paddr_read(pte_addr, 4);
        if invalid_pte(pte) {
            return Err(report_pf(vaddr, type_));
        }

        if leaf_pte(pte) {
            check_perms(pte, vaddr, type_)?;
            let ppn = (pte >> 10) & 0x003f_ffff;
            let page_mask = if level == 1 { 0x003f_ffff } else { 0x0000_0fff };
            if level == 1 && (ppn & 0x3ff) != 0 {
                return Err(report_pf(vaddr, type_));
            }
            let paddr = (ppn << 12) | (vaddr & page_mask);
            return Ok(WalkResult {
                paddr,
                page_mask,
                perms: pte & 0xe,
            });
        }

        if level == 0 {
            return Err(report_pf(vaddr, type_));
        }

        let next_ppn = (pte >> 10) & 0x003f_ffff;
        pte_addr = (next_ppn << 12) + vpn[0] * 4;
    }

    Err(report_pf(vaddr, type_))
}

fn translate_sv39(satp: Word, vaddr: VAddr, type_: i32) -> Result<WalkResult, Word> {
    if !canonical_sv39(vaddr) {
        return Err(report_pf(vaddr, type_));
    }

    let ppn_base = satp & 0x0000_0fff_ffff_ffff;
    let vpn = [
        (vaddr >> 12) & 0x1ff,
        (vaddr >> 21) & 0x1ff,
        (vaddr >> 30) & 0x1ff,
    ];
    let mut pte_addr = (ppn_base << 12) + vpn[2] * 8;

    for level in (0..=2).rev() {
        let pte = paddr_read(pte_addr, 8);
        if invalid_pte(pte) {
            return Err(report_pf(vaddr, type_));
        }

        if leaf_pte(pte) {
            check_perms(pte, vaddr, type_)?;
            let ppn = (pte >> 10) & 0x0000_0fff_ffff_ffff;
            let page_mask = match level {
                2 => 0x3fff_ffff,
                1 => 0x001f_ffff,
                _ => 0x0000_0fff,
            };
            if (level == 2 && (ppn & 0x3ffff) != 0) || (level == 1 && (ppn & 0x1ff) != 0) {
                return Err(report_pf(vaddr, type_));
            }
            let paddr = (ppn << 12) | (vaddr & page_mask);
            return Ok(WalkResult {
                paddr,
                page_mask,
                perms: pte & 0xe,
            });
        }

        if level == 0 {
            return Err(report_pf(vaddr, type_));
        }

        let next_ppn = (pte >> 10) & 0x0000_0fff_ffff_ffff;
        pte_addr = (next_ppn << 12) + vpn[level - 1] * 8;
    }

    Err(report_pf(vaddr, type_))
}

fn satp_mode(satp: Word) -> Word {
    if crate::generated::config::RV64 {
        satp >> 60
    } else {
        (satp >> 31) & 1
    }
}

fn satp_key(satp: Word) -> Word {
    if crate::generated::config::RV64 {
        satp & 0xffff_ffff_ffff_ffff
    } else {
        satp & 0xffff_ffff
    }
}

fn tlb_index(vaddr: VAddr, satp_key: Word, entries: usize) -> usize {
    let vpn = vaddr >> 12;
    ((vpn ^ satp_key) as usize) % entries
}

fn invalid_pte(pte: Word) -> bool {
    (pte & PTE_V) == 0 || (pte & (PTE_W | PTE_R)) == PTE_W
}

fn leaf_pte(pte: Word) -> bool {
    (pte & (PTE_R | PTE_W | PTE_X)) != 0
}

fn check_perms(pte: Word, vaddr: VAddr, type_: i32) -> Result<(), Word> {
    let ok = match type_ {
        MEM_TYPE_IFETCH => (pte & PTE_X) != 0,
        MEM_TYPE_READ => (pte & PTE_R) != 0,
        MEM_TYPE_WRITE => (pte & PTE_W) != 0,
        _ => false,
    };
    if ok {
        Ok(())
    } else {
        Err(report_pf(vaddr, type_))
    }
}

fn canonical_sv39(vaddr: VAddr) -> bool {
    let bit38 = (vaddr >> 38) & 1;
    let upper = vaddr >> 39;
    if bit38 == 0 {
        upper == 0
    } else {
        upper == 0x1ff_ffff
    }
}

fn report_pf(vaddr: VAddr, type_: i32) -> Word {
    let code = match type_ {
        MEM_TYPE_IFETCH => 12,
        MEM_TYPE_READ => 13,
        MEM_TYPE_WRITE => 15,
        _ => 13,
    };
    log::error!(
        "Page Fault: type={}, code={} at vaddr=0x{:016x}",
        type_,
        code,
        vaddr
    );
    code
}

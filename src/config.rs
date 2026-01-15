// Configuration parsing and management

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "remu")]
#[command(about = "RISC-V Emulator (REMU) - Rust implementation of NEMU")]
#[command(version = "0.1.0")]
pub struct Config {
    /// Run in batch mode (no interactive debugger)
    #[arg(short = 'b', long = "batch")]
    pub batch: bool,

    /// Log file path (default: build/remu-log.txt)
    #[arg(short = 'l', long = "log", value_name = "FILE")]
    pub log_file: Option<String>,

    /// Difftest SO file path for differential testing
    #[arg(short = 'd', long = "diff", value_name = "REF_SO")]
    pub diff_so: Option<String>,

    /// Difftest port number (default: 1234)
    #[arg(short = 'p', long = "port", value_name = "PORT", default_value = "1234")]
    pub difftest_port: u16,

    /// ELF file for symbol loading (function tracing)
    #[arg(short = 'e', long = "elf", value_name = "ELF_FILE")]
    pub elf_file: Option<String>,

    /// ELF symbol offset (e.g. 0x400000)
    #[arg(long = "elf-offset", value_name = "OFFSET", default_value = "0")]
    pub elf_offset: String,

    /// Image file to load (positional argument)
    #[arg(value_name = "IMAGE")]
    pub image: Option<std::path::PathBuf>,
}

// Runtime configuration from generated/config.rs
// We use the crate root to access the generated module
use crate::generated::config::*;

#[derive(Debug)]
pub struct RuntimeConfig {
    pub mbase: u32,
    pub msize: u32,
    pub pc_reset_offset: u32,
    pub mem_random: bool,
    
    pub trace_start: u64,
    pub trace_end: u64,
    
    pub has_serial: bool,
    pub serial_mmio: u32,
    
    pub has_timer: bool,
    pub rtc_mmio: u32,
    
    pub has_keyboard: bool,
    pub keyboard_mmio: u32,
    
    pub has_vga: bool,
    pub fb_addr: u32,
    pub vgactl_mmio: u32,
    
    pub has_audio: bool,
    pub audio_addr: u32,
    
    pub has_disk: bool,
    pub disk_mmio: u32,
    
    pub has_clint: bool,
    pub has_plic: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            mbase: MBASE,
            msize: MSIZE,
            pc_reset_offset: PC_RESET_OFFSET,
            mem_random: MEM_RANDOM,
            
            trace_start: TRACE_START,
            trace_end: TRACE_END,
            
            has_serial: HAS_SERIAL,
            serial_mmio: SERIAL_MMIO,
            
            has_timer: HAS_TIMER,
            rtc_mmio: RTC_MMIO,
            
            has_keyboard: HAS_KEYBOARD,
            keyboard_mmio: I8042_DATA_MMIO,
            
            has_vga: HAS_VGA,
            fb_addr: FB_ADDR,
            vgactl_mmio: VGA_CTL_MMIO,
            
            has_audio: HAS_AUDIO,
            audio_addr: SB_ADDR,
            
            has_disk: HAS_DISK,
            disk_mmio: DISK_CTL_MMIO,
            
            has_clint: HAS_CLINT,
            has_plic: HAS_PLIC,
        }
    }
}

pub fn parse_args() -> Result<Config, Box<dyn std::error::Error>> {
    let config = Config::parse();
    Ok(config)
}

// Reset vector = MBASE + PC_RESET_OFFSET
pub fn reset_vector(cfg: &RuntimeConfig) -> u32 {
    cfg.mbase + cfg.pc_reset_offset
}

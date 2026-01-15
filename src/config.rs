// Configuration parsing and management

use clap::Parser;
use std::path::PathBuf;

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

    /// Image file to load (positional argument)
    #[arg(value_name = "IMAGE")]
    pub image: Option<std::path::PathBuf>,
}

// Runtime configuration constants
// These would normally be generated from Kconfig
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
    pub i8042_data_mmio: u32,
    
    pub has_vga: bool,
    pub fb_addr: u32,
    pub vga_ctl_mmio: u32,
    pub vga_show_screen: bool,
    
    pub has_audio: bool,
    pub sb_addr: u32,
    pub sb_size: u32,
    pub audio_ctl_mmio: u32,
    
    pub has_disk: bool,
    pub disk_ctl_mmio: u32,
    pub disk_img_path: String,
    
    pub has_clint: bool,
    pub has_plic: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            // Memory configuration (matching NEMU .config)
            mbase: 0x80000000,
            msize: 0x8000000,  // 128MB
            pc_reset_offset: 0,
            mem_random: true,
            
            // Trace configuration
            trace_start: 0,
            trace_end: 10000,
            
            // Device configuration
            has_serial: cfg!(feature = "serial"),
            serial_mmio: 0xa00003f8,
            
            has_timer: cfg!(feature = "timer"),
            rtc_mmio: 0xa0000048,
            
            has_keyboard: cfg!(feature = "keyboard"),
            i8042_data_mmio: 0xa0000060,
            
            has_vga: cfg!(feature = "vga"),
            fb_addr: 0xa1000000,
            vga_ctl_mmio: 0xa0000100,
            vga_show_screen: true,
            
            has_audio: cfg!(feature = "audio"),
            sb_addr: 0xa1200000,
            sb_size: 0x10000,
            audio_ctl_mmio: 0xa0000200,
            
            has_disk: cfg!(feature = "disk"),
            disk_ctl_mmio: 0xa0000300,
            disk_img_path: String::new(),
            
            has_clint: cfg!(feature = "clint"),
            has_plic: cfg!(feature = "plic"),
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

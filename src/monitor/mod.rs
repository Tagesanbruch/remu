// Monitor module - initialization and image loading

use crate::config::{Config, RuntimeConfig};
use crate::memory::load_image;
use crate::Log;
use std::fs;
use std::io::Read;

static mut EXIT_BAD: bool = false;

pub fn init_monitor(cfg: &Config) {
    Log!("Initializing monitor...");
    
    // Initialize memory
    crate::memory::init_mem();
    
    // Initialize CPU
    crate::cpu::init_cpu();
    
    // Load image
    load_img(cfg);
    
    // Initialize FTRACE
    if let Some(elf_file) = &cfg.elf_file {
        crate::utils::ftrace::init_ftrace(elf_file);
    }
    
    // Initialize devices
    if crate::generated::config::DEVICE {
        crate::device::init_device();
    }
    
    welcome();
}

fn load_img(cfg: &Config) {
    if let Some(ref img_path) = cfg.image {
        // Load from file
        match fs::File::open(img_path) {
            Ok(mut file) => {
                let mut buffer = Vec::new();
                match file.read_to_end(&mut buffer) {
                    Ok(size) => {
                        Log!("The image is {}, size = {} bytes", 
                                 img_path.display(), size);
                        
                        let rt_cfg = RuntimeConfig::default();
                        let reset_vec = crate::config::reset_vector(&rt_cfg);
                        
                        match load_image(&buffer, reset_vec) {
                            Ok(_) => Log!("Image loaded successfully at 0x{:08x}", reset_vec),
                            Err(e) => {
                                eprintln!("Failed to load image: {}", e);
                                std::process::exit(1);
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to read image file: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("Cannot open '{}': {}", img_path.display(), e);
                std::process::exit(1);
            }
        }
    } else {
        Log!("No image is given. Use the default built-in image.");
        copy_builtin_image_to_memory();
    }
}

fn copy_builtin_image_to_memory() {
    let rt_cfg = RuntimeConfig::default();
    let reset_vec = crate::config::reset_vector(&rt_cfg);
    
    // Simple infinite loop + ebreak
    let image: [u32; 4] = [
        0x00000297,  // auipc t0, 0
        0x01028823,  // sb a0, 16(t0)
        0x0102c503,  // lbu a0, 16(t0)
        0x00100073,  // ebreak
    ];
    
    // Convert to bytes
    let mut bytes = Vec::new();
    for inst in image.iter() {
        bytes.extend_from_slice(&inst.to_le_bytes());
    }
    
    match load_image(&bytes, reset_vec) {
        Ok(_) => Log!("Built-in image loaded at 0x{:08x}", reset_vec),
        Err(e) => {
            eprintln!("Failed to load built-in image: {}", e);
            std::process::exit(1);
        }
    }
}

fn welcome() {
    Log!("Trace: {}", if crate::generated::config::TRACE { crate::common::colored("ON", crate::common::ANSI_FG_GREEN) } else { crate::common::colored("OFF", crate::common::ANSI_FG_RED) });
    if crate::generated::config::TRACE {
        Log!("If trace is enabled, a log file will be generated to record the trace. This may lead to a large log file. If it is not necessary, you can disable it in menuconfig");
    }
    // ANSI color codes (matching C NEMU)
    pub const ANSI_NONE: &str = "\x1b[0m";
    pub const ANSI_FG_YELLOW: &str = "\x1b[33m";
    pub const ANSI_BG_RED: &str = "\x1b[41m";
    // Print welcome message (not logged to file, direct to stdout)
    println!("Welcome to {}{}{}{}-REMU!",
        ANSI_FG_YELLOW, ANSI_BG_RED, "riscv32", ANSI_NONE);
    println!("For help, type \"help\"");
}

pub fn set_exit_status_bad() {
    unsafe { EXIT_BAD = true; }
}

pub fn is_exit_status_bad() -> bool {
    unsafe { EXIT_BAD }
}

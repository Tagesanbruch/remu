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
    
    // Initialize devices
    #[cfg(feature = "device")]
    crate::device::init_device();
    
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
        // Load built-in image (like NEMU's init.c)
        Log!("No image is given. Use the default built-in image.");
        load_builtin_image();
    }
}

fn load_builtin_image() {
    // Built-in test program (from NEMU's riscv32/init.c)  
    // Note: Only 5 words, ebreak is at 0x8000000c
    let img: [u32; 5] = [
        0x00000297,  // auipc t0,0
        0x00028823,  // sb  zero,16(t0)
        0x0102c503,  // lbu a0,16(t0)
        0x00100073,  // ebreak (NEMU trap)
        0xdeadbeef,  // some data
    ];
    
    let rt_cfg = RuntimeConfig::default();
    let reset_vec = crate::config::reset_vector(&rt_cfg);
    
    // Convert to bytes with correct endianness
    let mut bytes = Vec::new();
    for inst in &img {
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
    use crate::utils::log::{ANSI_FG_GREEN, ANSI_FG_RED, ANSI_FG_YELLOW, ANSI_BG_RED, ANSI_NONE};
    
    #[cfg(feature = "trace")]
    let trace_status = format!("{}ON{}", ANSI_FG_GREEN, ANSI_NONE);
    
    #[cfg(not(feature = "trace"))]
    let trace_status = format!("{}OFF{}", ANSI_FG_RED, ANSI_NONE);
    
    Log!("Trace: {}", trace_status);
    
    #[cfg(feature = "trace")]
    Log!("If trace is enabled, a log file will be generated to record the trace. This may lead to a large log file. If it is not necessary, you can disable it in menuconfig");
    
    // Print welcome message (not logged to file, direct to stdout)
    println!("Welcome to {}{}{}{}-NEMU!",
        ANSI_FG_YELLOW, ANSI_BG_RED, "riscv32", ANSI_NONE);
    println!("For help, type \"help\"");
}

pub fn is_exit_status_bad() -> bool {
    unsafe { EXIT_BAD }
}

pub fn set_exit_bad() {
    unsafe {
        EXIT_BAD = true;
    }
}

use std::process;

// Common types and utilities
pub mod common;
pub mod config;
pub mod utils;
pub mod generated;

pub mod cpu;
pub mod device;
pub mod engine;
pub mod isa;
pub mod memory;
pub mod monitor;

// Import macros
#[macro_use]
extern crate lazy_static;

fn main() {
    // Parse arguments
    let config = match config::parse_args() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("{}", e);
            process::exit(1);
        }
    };

    // Initialize custom logging (NEMU-style)
    let log_file = config.log_file.as_deref().unwrap_or("build/remu-log.txt");
    std::fs::create_dir_all("build").ok();
    crate::utils::log::init_log(log_file);
    
    Log!("REMU starting...");
    
    // Initialize monitor (memory, devices, ISA)
    monitor::init_monitor(&config);

    // Start the engine (debugger or batch mode)
    engine::start(&config);

    // Check exit status
    let exit_code = if monitor::is_exit_status_bad() { 1 } else { 0 };
    process::exit(exit_code);
}

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
// lazy_static macro is used in modules

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
    std::fs::create_dir_all("build").ok();
    crate::utils::log::init_log(log_file);
    crate::utils::log::init_panic_hook();
    
    Log!("REMU starting...");
    
    // Set runtime flags
    config::set_ebreak_halt(config.ebreak_halt);
    
    // Initialize monitor (memory, devices, ISA)
    monitor::init_monitor(&config);

    // Register Ctrl+C handler
    ctrlc::set_handler(move || {
        crate::utils::set_state(crate::common::RemuState::Abort);
        println!("\nCtrl-C pressed. Aborting...");
        crate::utils::itrace::show_itrace();
        crate::utils::mtrace::show_mtrace();
        crate::utils::ftrace::show_ftrace();

        use crate::utils::state::REMU_STATE;

        use crate::common::RemuState;
    
        let state_guard = REMU_STATE.lock().unwrap();
        let state = state_guard.state;
        let halt_pc = state_guard.halt_pc;
        let halt_ret = state_guard.halt_ret;
        drop(state_guard);

        let trap_msg = if state == RemuState::Abort {
        format!("{}ABORT{}", crate::utils::log::ANSI_FG_RED, crate::utils::log::ANSI_NONE)
    } else {
        if halt_ret == 0 {
            format!("{}HIT GOOD TRAP{}", crate::utils::log::ANSI_FG_GREEN, crate::utils::log::ANSI_NONE)
        } else {
            format!("{}HIT BAD TRAP{}", crate::utils::log::ANSI_FG_RED, crate::utils::log::ANSI_NONE)
        }
    };
    
    Log!("{}Remu: {} at pc = 0x{:08x}{}",
        crate::utils::log::ANSI_FG_BLUE,
        trap_msg,
        halt_pc,
        crate::utils::log::ANSI_NONE);

        // crate::cpu::execute::statistic(); // DEADLOCK WARNING: Holds NPU/CPU lock potentially
        std::process::exit(0);
    }).expect("Error setting Ctrl-C handler");

    // Start the engine (debugger or batch mode)
    engine::start(&config);

    // Check exit status
    let exit_code = if monitor::is_exit_status_bad() { 1 } else { 0 };
    process::exit(exit_code);
}

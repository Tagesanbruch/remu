// CPU execution loop

use crate::common::RemuState;
use crate::cpu::state::{CPU, CpuState};
use crate::isa::riscv32;
use crate::utils::{get_state, set_state};
use crate::Log;
use std::time::Instant;

static mut GUEST_INST_COUNT: u64 = 0;
static mut HOST_START_TIME: Option<Instant> = None;

pub fn init_cpu() {
    Log!("Initializing CPU...");
    
    // Initialize CPU state (already done in state::new())
    let cpu = CPU.lock().unwrap();
    let pc = cpu.pc;
    drop(cpu);
    
    Log!("CPU initialized: PC = 0x{:08x}", pc);
}

pub fn cpu_exec(n: u64) {
    let state = get_state();
    match state {
        RemuState::End | RemuState::Abort => {
            log::warn!("Program execution has ended. To restart the program, exit REMU and run again.");
            return;
        }
        _ => {
            set_state(RemuState::Running);
        }
    }

    unsafe {
        GUEST_INST_COUNT = 0;
        HOST_START_TIME = Some(Instant::now());
    }
    
    execute(n);
    
    // Print statistics
    statistic();
}

fn execute(n: u64) {
    // Lock CPU state once for the entire execution loop
    let mut cpu_guard = CPU.lock().unwrap();
    let cpu = &mut *cpu_guard;

    for i in 0..n {
        // Check interrupts every 1024 instructions to throttle timer syscalls
        exec_once(cpu, (i & 0x3ff) == 0);
        
        unsafe {
            GUEST_INST_COUNT += 1;
        }
        // Check for external state change (e.g. from SDL or CTRL+C)
        if crate::utils::get_state() != RemuState::Running {
            break;
        }

        // The following lines are added as part of the instruction,
        // but `timer_start` is not defined in the current context.
        // This might indicate a missing part of the change or an incomplete instruction.
        // For now, it's included as is, assuming `timer_start` would be defined elsewhere.
        // If `timer_start` is not defined, this code will cause a compilation error.
        // if n > 0 && timer_start.elapsed().as_micros() as u64 > 500000 { // Check every 0.5s
        //     let duration = timer_start.elapsed().as_micros() as u64;
        // }
        
        // Update devices
        // Update devices (Throttle: Check every 65536 instructions)
        // ~333Hz at 20MIPS, sufficient for 60Hz VGA/Serial
        if crate::generated::config::DEVICE && (i & 0xffff) == 0 {
            crate::device::device_update();
        }
    }
}

fn exec_once(cpu: &mut CpuState, check_intr: bool) {
    let pc = cpu.pc;
    
    // Check for interrupts
    if check_intr {
        let intr = crate::isa::riscv32::system::intr::isa_query_intr(cpu);
        if intr != 0 {
             let new_pc = crate::isa::riscv32::system::intr::isa_raise_intr(cpu, intr, pc);
             cpu.pc = new_pc;
             return;
        }
    }

    // Execute one instruction
    riscv32::isa_exec_once(cpu, pc);
}

pub fn statistic() {
    use crate::utils::log::{ANSI_FG_GREEN, ANSI_FG_RED, ANSI_FG_BLUE, ANSI_NONE};
    use crate::utils::state::REMU_STATE;
    
    let state_guard = REMU_STATE.lock().unwrap();
    let state = state_guard.state;
    let halt_pc = state_guard.halt_pc;
    let halt_ret = state_guard.halt_ret;
    drop(state_guard);
    
    // Determine trap message
    let trap_msg = if state == RemuState::Abort {
        format!("{}ABORT{}", ANSI_FG_RED, ANSI_NONE)
    } else {
        if halt_ret == 0 {
            format!("{}HIT GOOD TRAP{}", ANSI_FG_GREEN, ANSI_NONE)
        } else {
            format!("{}HIT BAD TRAP{}", ANSI_FG_RED, ANSI_NONE)
        }
    };
    
    // Print trap message with Log! macro format
    Log!("{}Remu: {} at pc = 0x{:08x}{}",
        ANSI_FG_BLUE,
        trap_msg,
        halt_pc,
        ANSI_NONE);
    
    // Calculate statistics
    let guest_inst = unsafe { GUEST_INST_COUNT };
    let elapsed = unsafe {
        HOST_START_TIME.map(|start| start.elapsed()).unwrap_or_default()
    };
    let host_time_us = elapsed.as_micros();
    let freq = if host_time_us > 0 {
        (guest_inst as f64) / (host_time_us as f64) * 1_000_000.0
    } else {
        0.0
    };
    
    // Format with thousand separators
    let time_str = format!("{}", host_time_us);
    let time_formatted = time_str.as_bytes().rchunks(3)
        .rev()
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect::<Vec<_>>()
        .join(",");
    
    Log!("{}host time spent = {} us{}",
        ANSI_FG_BLUE, time_formatted, ANSI_NONE);
    Log!("{}total guest instructions = {}{}",
        ANSI_FG_BLUE, guest_inst, ANSI_NONE);
    Log!("{}simulation frequency = {:.0} inst/s{}",
        ANSI_FG_BLUE, freq, ANSI_NONE);
    
    // Show traces
    if crate::generated::config::TRACE {
        crate::utils::print_trace_summary();
    }
    
    // Set bad exit status if needed
    if halt_ret != 0 && state != RemuState::Abort {
        crate::monitor::set_exit_status_bad();
    }
}

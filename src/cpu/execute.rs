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
    statistic();
}

fn execute(n: u64) {
    let mut cpu_guard = CPU.lock().unwrap();
    let cpu = &mut *cpu_guard;

    for i in 0..n {
        // 目前每执行 1024 条指令检查一次中断
        exec_once(cpu, (i & 0x3ff) == 0);
        
        unsafe {
            GUEST_INST_COUNT += 1;
        }
        if crate::utils::get_state() != RemuState::Running {
            break;
        }

        // Update devices
        if crate::generated::config::DEVICE && (i & 0xffff) == 0 {
            crate::device::device_update();
        }
    }
}

fn exec_once(cpu: &mut CpuState, check_intr: bool) {
    let pc = cpu.pc;
    if check_intr {
        let intr = crate::isa::riscv32::system::intr::isa_query_intr(cpu);
        if intr != 0 {
             let new_pc = crate::isa::riscv32::system::intr::isa_raise_intr(cpu, intr, pc);
             cpu.pc = new_pc;
             return;
        }
    }
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
    let trap_msg = if state == RemuState::Abort {
        format!("{}ABORT{}", ANSI_FG_RED, ANSI_NONE)
    } else {
        if halt_ret == 0 {
            format!("{}HIT GOOD TRAP{}", ANSI_FG_GREEN, ANSI_NONE)
        } else {
            format!("{}HIT BAD TRAP{}", ANSI_FG_RED, ANSI_NONE)
        }
    };
    
    Log!("{}Remu: {} at pc = 0x{:08x}{}",
        ANSI_FG_BLUE,
        trap_msg,
        halt_pc,
        ANSI_NONE);
    
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
    
    if crate::generated::config::TRACE {
        crate::utils::print_trace_summary();
    }
    
    if halt_ret != 0 && state != RemuState::Abort {
        crate::monitor::set_exit_status_bad();
    }
}

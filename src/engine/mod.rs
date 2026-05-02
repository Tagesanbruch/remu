// Execution engine - interpreter mode

use crate::config::Config;
use crate::cpu::cpu_exec;
// use crate::utils::get_state;  // Unused
// use crate::common::RemuState;  // Unused

pub fn start(cfg: &Config) {
    if !run_startup_commands(cfg) {
        return;
    }

    if cfg.batch {
        // Batch mode - run until completion
        cpu_exec(u64::MAX);
    } else {
        // Interactive mode - simple debugger
        sdb_mainloop(cfg);
    }
}

fn run_startup_commands(cfg: &Config) -> bool {
    if let Some(script) = &cfg.sdb_script {
        let content = match std::fs::read_to_string(script) {
            Ok(content) => content,
            Err(err) => {
                eprintln!("Failed to read SDB script {}: {}", script.display(), err);
                return false;
            }
        };

        for (line_no, line) in content.lines().enumerate() {
            let cmd = line.trim();
            if cmd.is_empty() || cmd.starts_with('#') {
                continue;
            }
            println!("(sdb-script:{}): {}", line_no + 1, cmd);
            if !handle_command(cmd) {
                return false;
            }
        }
    }

    for cmd in &cfg.sdb_cmd {
        let cmd = cmd.trim();
        if cmd.is_empty() {
            continue;
        }
        println!("(sdb-cmd): {}", cmd);
        if !handle_command(cmd) {
            return false;
        }
    }

    true
}

fn sdb_mainloop(_cfg: &Config) {
    use std::io::{self, Write};

    loop {
        print!("(remu) ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {
                let cmd = input.trim();
                if cmd.is_empty() {
                    continue;
                }

                if !handle_command(cmd) {
                    break;
                }
            }
            Err(e) => {
                log::error!("Error reading input: {}", e);
                break;
            }
        }
    }
}

fn handle_command(cmd: &str) -> bool {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return true;
    }

    match parts[0] {
        "c" | "continue" => {
            cpu_exec(u64::MAX);
        }
        "q" | "quit" => {
            return false;
        }
        "si" => {
            let n = if parts.len() > 1 {
                parts[1].parse().unwrap_or(1)
            } else {
                1
            };
            cpu_exec(n);
        }
        "info" => {
            if parts.len() > 1 {
                match parts[1] {
                    "pc" => {
                        let cpu = crate::cpu::state::CPU.lock().unwrap();
                        println!("PC: 0x{:08x} mode={:?}", cpu.pc, cpu.mode);
                    }
                    "r" => {
                        // Print registers
                        let cpu = crate::cpu::state::CPU.lock().unwrap();
                        println!("PC: 0x{:08x}", cpu.pc);
                        for i in 0..32 {
                            print!("x{:<2} = 0x{:08x}  ", i, cpu.get_gpr(i));
                            if (i + 1) % 4 == 0 {
                                println!();
                            }
                        }
                    }
                    "csr" => {
                        print_csr_info();
                    }
                    "trace" => {
                        crate::utils::print_trace_summary(parts.get(2).copied());
                    }
                    _ => println!("Unknown info command"),
                }
            }
        }
        "trace" => {
            crate::utils::print_trace_summary(parts.get(1).copied());
        }
        "x" => {
            inspect_vaddr(&parts);
        }
        "xp" => {
            inspect_paddr(&parts);
        }
        "help" => {
            println!("Available commands:");
            println!("  c, continue      - Continue execution");
            println!("  q, quit          - Exit the emulator");
            println!("  si [N]           - Step N instructions (default 1)");
            println!("  x N ADDR         - Examine N virtual words from ADDR");
            println!("  xp N ADDR        - Examine N physical words from ADDR");
            println!("  info pc          - Print PC and privilege mode");
            println!("  info r           - Print registers");
            println!("  info csr         - Print common CSRs");
            println!("  trace [KIND]     - Dump trace ring buffers (all/itrace/dtrace/intr/mmu/ecall/ftrace)");
            println!("  help             - Show this help");
        }
        _ => {
            println!("Unknown command: {}", parts[0]);
        }
    }

    true
}

fn parse_u32(value: &str) -> Option<u32> {
    if let Some(hex) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        u32::from_str_radix(hex, 16).ok()
    } else {
        value.parse().ok()
    }
}

fn inspect_vaddr(parts: &[&str]) {
    if parts.len() != 3 {
        println!("Usage: x N ADDR");
        return;
    }

    let Some(count) = parse_u32(parts[1]) else {
        println!("Invalid count: {}", parts[1]);
        return;
    };
    let Some(addr) = parse_u32(parts[2]) else {
        println!("Invalid address: {}", parts[2]);
        return;
    };

    let cpu = crate::cpu::state::CPU.lock().unwrap();
    for i in 0..count {
        let cur = addr.wrapping_add(i.wrapping_mul(4));
        match crate::memory::vaddr::vaddr_read(&cpu, cur, 4) {
            Ok(data) => println!("0x{:08x}: 0x{:08x}", cur, data),
            Err(cause) => {
                println!("0x{:08x}: fault cause {}", cur, cause);
                break;
            }
        }
    }
}

fn inspect_paddr(parts: &[&str]) {
    if parts.len() != 3 {
        println!("Usage: xp N ADDR");
        return;
    }

    let Some(count) = parse_u32(parts[1]) else {
        println!("Invalid count: {}", parts[1]);
        return;
    };
    let Some(addr) = parse_u32(parts[2]) else {
        println!("Invalid address: {}", parts[2]);
        return;
    };

    for i in 0..count {
        let cur = addr.wrapping_add(i.wrapping_mul(4));
        let data = crate::memory::paddr::paddr_read(cur, 4);
        println!("0x{:08x}: 0x{:08x}", cur, data);
    }
}

fn print_csr_info() {
    let cpu = crate::cpu::state::CPU.lock().unwrap();
    let csrs = [
        ("mstatus", crate::isa::riscv32::system::csr::CSR_MSTATUS),
        ("mie", crate::isa::riscv32::system::csr::CSR_MIE),
        ("mip", crate::isa::riscv32::system::csr::CSR_MIP),
        ("medeleg", crate::isa::riscv32::system::csr::CSR_MEDELEG),
        ("mideleg", crate::isa::riscv32::system::csr::CSR_MIDELEG),
        ("mtvec", crate::isa::riscv32::system::csr::CSR_MTVEC),
        ("mepc", crate::isa::riscv32::system::csr::CSR_MEPC),
        ("mcause", crate::isa::riscv32::system::csr::CSR_MCAUSE),
        ("mtval", crate::isa::riscv32::system::csr::CSR_MTVAL),
        ("sstatus", crate::isa::riscv32::system::csr::CSR_SSTATUS),
        ("sie", crate::isa::riscv32::system::csr::CSR_SIE),
        ("sip", crate::isa::riscv32::system::csr::CSR_SIP),
        ("stvec", crate::isa::riscv32::system::csr::CSR_STVEC),
        ("sepc", crate::isa::riscv32::system::csr::CSR_SEPC),
        ("scause", crate::isa::riscv32::system::csr::CSR_SCAUSE),
        ("stval", crate::isa::riscv32::system::csr::CSR_STVAL),
        ("satp", crate::isa::riscv32::system::csr::CSR_SATP),
    ];

    println!("PC: 0x{:08x} mode={:?}", cpu.pc, cpu.mode);
    for (name, csr) in csrs {
        let value = crate::isa::riscv32::system::csr::isa_csr_read(&cpu, csr);
        println!("{:<8} 0x{:08x}", name, value);
    }
}

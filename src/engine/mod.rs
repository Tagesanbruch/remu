// Execution engine - interpreter mode

use crate::config::Config;
use crate::cpu::cpu_exec;
// use crate::utils::get_state;  // Unused
// use crate::common::RemuState;  // Unused

pub fn start(cfg: &Config) {
    if cfg.batch {
        // Batch mode - run until completion
        cpu_exec(u64::MAX);
    } else {
        // Interactive mode - simple debugger
        sdb_mainloop(cfg);
    }
}

fn sdb_mainloop(_cfg: &Config) {
    use std::io::{self, Write};
    
    loop {
        print!("(remu) ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break,  // EOF
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
                    _ => println!("Unknown info command"),
                }
            }
        }
        "help" => {
            println!("Available commands:");
            println!("  c, continue      - Continue execution");
            println!("  q, quit          - Exit the emulator");
            println!("  si [N]           - Step N instructions (default 1)");
            println!("  info r           - Print registers");
            println!("  help             - Show this help");
        }
        _ => {
            println!("Unknown command: {}", parts[0]);
        }
    }
    
    true
}

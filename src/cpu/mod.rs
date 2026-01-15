// CPU module

pub mod state;
pub mod execute;

pub use state::CpuState;
pub use execute::cpu_exec;

// Initialize CPU state
pub fn init_cpu() {
    log::info!("Initializing CPU...");
    let mut cpu = state::CPU.lock().unwrap();
    cpu.init();
}

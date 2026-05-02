// CPU module

pub mod execute;
pub mod state;

pub use execute::cpu_exec;
pub use state::CpuState;

// Initialize CPU state
pub fn init_cpu() {
    log::info!("Initializing CPU...");
    let mut cpu = state::CPU.lock().unwrap();
    cpu.init();
}

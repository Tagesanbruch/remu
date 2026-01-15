// Utility modules

pub mod ringbuffer;
pub mod state;
pub mod log;
pub mod itrace;
pub mod mtrace;
pub mod ftrace;
pub mod dtrace;
pub mod intr_trace;
pub mod mmu_trace;
pub mod ecall_trace;

pub use ringbuffer::RingBuffer;
pub use state::{GlobalState, get_state, set_state, set_halt};

pub fn print_trace_summary() {
    if !crate::generated::config::TRACE { return; }
    
    // ITRACE
    crate::utils::itrace::show_itrace();
    
    // MTRACE (not yet implemented in RingBuffer fully? mtrace.rs exists)
    // mtrace.rs seems to have its own show()
    crate::utils::mtrace::show_mtrace();
    
    // DTRACE
    crate::utils::dtrace::show_dtrace();

    // INTR
    crate::utils::intr_trace::show_intr_trace();
    
    // MMU
    crate::utils::mmu_trace::show_mmu_trace();

    // ECALL
    crate::utils::ecall_trace::show_ecall_trace();
    
    // FTRACE
    crate::utils::ftrace::show_ftrace();
}

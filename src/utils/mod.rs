// Utility modules

pub mod dtrace;
pub mod ecall_trace;
pub mod ftrace;
pub mod intr_trace;
pub mod itrace;
pub mod log;
pub mod mmu_trace;
pub mod mtrace;
pub mod ringbuffer;
pub mod state;

pub use ringbuffer::RingBuffer;
pub use state::{get_state, set_halt, set_state, GlobalState};

pub fn print_trace_summary(kind: Option<&str>) {
    if !crate::generated::config::TRACE {
        return;
    }

    match kind.unwrap_or("all") {
        "all" => {
            crate::utils::itrace::show_itrace();
            crate::utils::mtrace::show_mtrace();
            crate::utils::dtrace::show_dtrace();
            crate::utils::intr_trace::show_intr_trace();
            crate::utils::mmu_trace::show_mmu_trace();
            crate::utils::ecall_trace::show_ecall_trace();
            crate::utils::ftrace::show_ftrace();
        }
        "i" | "itrace" => crate::utils::itrace::show_itrace(),
        "m" | "mtrace" => crate::utils::mtrace::show_mtrace(),
        "d" | "dtrace" => crate::utils::dtrace::show_dtrace(),
        "intr" | "irq" => crate::utils::intr_trace::show_intr_trace(),
        "mmu" => crate::utils::mmu_trace::show_mmu_trace(),
        "e" | "ecall" => crate::utils::ecall_trace::show_ecall_trace(),
        "f" | "ftrace" => crate::utils::ftrace::show_ftrace(),
        other => {
            crate::Log!("unknown trace kind '{}'", other);
            crate::Log!("trace kinds: all, itrace, mtrace, dtrace, intr, mmu, ecall, ftrace");
        }
    }
}

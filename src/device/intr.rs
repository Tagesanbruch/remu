use std::sync::atomic::{AtomicU32, Ordering};

// Shared interrupt state for devices to signal CPU/PLIC
// Bit 9 = SEIP (Supervisor External Interrupt)
// Bit 11 = MEIP (Machine External Interrupt)

pub static INTR_STATE: AtomicU32 = AtomicU32::new(0);

// Helper to set/clear SEIP (Bit 9)
pub fn set_seip(val: bool) {
    if val {
        INTR_STATE.fetch_or(1 << 9, Ordering::Relaxed);
    } else {
        INTR_STATE.fetch_and(!(1 << 9), Ordering::Relaxed);
    }
}

pub fn get_intr_state() -> u32 {
    INTR_STATE.load(Ordering::Relaxed)
}

// Global emulator state

use crate::common::RemuState;
use std::sync::{Arc, Mutex};

pub struct GlobalState {
    pub state: RemuState,
    pub halt_pc: u32,
    pub halt_ret: i32,
}

impl Default for GlobalState {
    fn default() -> Self {
        Self {
            state: RemuState::Stop,
            halt_pc: 0,
            halt_ret: 0,
        }
    }
}

impl GlobalState {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(Self::default()))
    }
}

// Global state singleton
lazy_static::lazy_static! {
    pub static ref REMU_STATE: Arc<Mutex<GlobalState>> = GlobalState::new();
}

static ATOMIC_STATE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(1); // Default Stop (1)

pub fn get_state() -> RemuState {
    let val = ATOMIC_STATE.load(std::sync::atomic::Ordering::Relaxed);
    match val {
        0 => RemuState::Running,
        1 => RemuState::Stop,
        2 => RemuState::End,
        3 => RemuState::Abort,
        4 => RemuState::Quit,
        _ => RemuState::Stop,
    }
}

pub fn set_state(state: RemuState) {
    ATOMIC_STATE.store(state as i32, std::sync::atomic::Ordering::Relaxed);
    REMU_STATE.lock().unwrap().state = state;
}

pub fn set_halt(pc: u32, ret: i32) {
    let mut state = REMU_STATE.lock().unwrap();
    state.halt_pc = pc;
    state.halt_ret = ret;
}

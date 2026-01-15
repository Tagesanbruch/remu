// Global emulator state

use crate::common::NemuState;
use std::sync::{Arc, Mutex};

pub struct GlobalState {
    pub state: NemuState,
    pub halt_pc: u32,
    pub halt_ret: i32,
}

impl Default for GlobalState {
    fn default() -> Self {
        Self {
            state: NemuState::Stop,
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
    pub static ref NEMU_STATE: Arc<Mutex<GlobalState>> = GlobalState::new();
}

pub fn get_state() -> NemuState {
    NEMU_STATE.lock().unwrap().state
}

pub fn set_state(state: NemuState) {
    NEMU_STATE.lock().unwrap().state = state;
}

pub fn set_halt(pc: u32, ret: i32) {
    let mut state = NEMU_STATE.lock().unwrap();
    state.halt_pc = pc;
    state.halt_ret = ret;
}

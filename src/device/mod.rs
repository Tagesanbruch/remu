// Device module

pub mod serial;
pub mod timer;
pub mod clint;
pub mod plic;

use crate::config::RuntimeConfig;

pub fn init_device() {
    log::info!("Initializing devices...");
    
    let cfg = RuntimeConfig::default();
    
    if cfg.has_serial {
        serial::init();
    }
    
    if cfg.has_timer {
        timer::init();
    }
    
    if cfg.has_clint {
        clint::init();
    }
    
    if cfg.has_plic {
        plic::init();
    }
}

pub fn device_update() {
    // Update devices as needed
    timer::update();
}

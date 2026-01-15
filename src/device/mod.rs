// Device management

pub mod timer;
pub mod serial;
pub mod keyboard;
pub mod vga;
pub mod audio;
pub mod disk;
pub mod clint;
pub mod plic;
pub mod sdl;

pub fn init_device() {
    crate::Log!("Initializing devices...");
    
    // Timer (Core time source)
    timer::init_timer();
    
    // Interrupt Controllers
    clint::init_clint();
    plic::init_plic();
    
    // Peripherals
    serial::init_serial();
    keyboard::init_keyboard();
    vga::init_vga();
    
    // Init SDL
    // sdl::init_sdl(); // Called by init_vga now
    audio::init_audio();
    disk::init_disk();
}

pub fn device_update() {
    use std::sync::Mutex;
    use std::time::{Instant, Duration};
    
    // Throttle to ~60Hz to avoid killing performance
    // NEMU uses 1000000 / TIMER_HZ. Let's assume 60Hz.
    lazy_static::lazy_static! {
        static ref LAST_UPDATE: Mutex<Instant> = Mutex::new(Instant::now());
    }
    
    let now = Instant::now();
    let mut last = LAST_UPDATE.lock().unwrap();
    if now.duration_since(*last) < Duration::from_millis(16) { // ~60Hz
        return;
    }
    *last = now;

    // Poll SDL events
    serial::serial_update();
    sdl::poll_events();
}

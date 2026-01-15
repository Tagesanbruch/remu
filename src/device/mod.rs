// Device management

pub mod timer;
pub mod serial;
pub mod keyboard;
pub mod vga;
pub mod audio;
pub mod disk;
pub mod clint;
pub mod plic;

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
    audio::init_audio();
    disk::init_disk();
}

pub fn device_update() {
    // Poll SDL events, etc.
    serial::serial_update();
    
    // TODO: Poll SDL events and feed keyboard/VGA
}

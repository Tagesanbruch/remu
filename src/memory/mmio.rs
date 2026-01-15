// Memory-mapped I/O (MMIO) handling

use crate::common::{Word, PAddr};
use crate::config::RuntimeConfig;

pub fn mmio_read(addr: PAddr, len: usize) -> Word {
    let cfg = RuntimeConfig::default();
    
    // Serial device
    if cfg.has_serial && addr >= cfg.serial_mmio && addr < cfg.serial_mmio + 0x100 {
        return crate::device::serial::serial_read(addr - cfg.serial_mmio);
    }
    
    // Timer device
    if cfg.has_timer && addr == cfg.rtc_mmio {
        return crate::device::timer::timer_read();
    }
    
    // CLINT
    if cfg.has_clint && addr >= 0x02000000 && addr < 0x0200c000 {
        return crate::device::clint::clint_read(addr, len);
    }
    
    // PLIC
    if cfg.has_plic && addr >= 0x0c000000 && addr < 0x10000000 {
        return crate::device::plic::plic_read(addr, len);
    }
    
    log::warn!("MMIO read from unmapped address 0x{:08x} (len={})", addr, len);
    0
}

pub fn mmio_write(addr: PAddr, len: usize, data: Word) {
    let cfg = RuntimeConfig::default();
    
    // Serial device
    if cfg.has_serial && addr >= cfg.serial_mmio && addr < cfg.serial_mmio + 0x100 {
        crate::device::serial::serial_write(addr - cfg.serial_mmio, data);
        return;
    }
    
    // Timer device (read-only)
    
    // CLINT
    if cfg.has_clint && addr >= 0x02000000 && addr < 0x0200c000 {
        crate::device::clint::clint_write(addr, len, data);
        return;
    }
    
    // PLIC
    if cfg.has_plic && addr >= 0x0c000000 && addr < 0x10000000 {
        crate::device::plic::plic_write(addr, len, data);
        return;
    }
    
    log::warn!("MMIO write to unmapped address 0x{:08x} = 0x{:08x} (len={})", addr, data, len);
}

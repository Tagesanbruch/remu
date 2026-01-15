// Memory Mapped I/O

use crate::common::{PAddr, Word};
use std::sync::Mutex;

// MMIO map entry
struct MmioMap {
    name: String,
    start: PAddr,
    end: PAddr,
    callback: Box<dyn Fn(PAddr, usize, bool, Word) -> Word + Send + Sync>, // addr, len, is_write, data -> return data
}

lazy_static::lazy_static! {
    static ref MMIO_MAPS: Mutex<Vec<MmioMap>> = Mutex::new(Vec::new());
}

pub fn init_mmio() {
    let mut maps = MMIO_MAPS.lock().unwrap();
    maps.clear();
}

pub fn register_mmio(name: &str, start: PAddr, len: usize, 
                     callback: Box<dyn Fn(PAddr, usize, bool, Word) -> Word + Send + Sync>) {
    let mut maps = MMIO_MAPS.lock().unwrap();
    maps.push(MmioMap {
        name: name.to_string(),
        start,
        end: start + len as u32,
        callback,
    });
    crate::Log!("Add mmio map '{}' at [0x{:08x}, 0x{:08x}]", name, start, start + len as u32 - 1);
}

pub fn mmio_read(addr: PAddr, len: usize) -> Word {
    let maps = MMIO_MAPS.lock().unwrap();
    for map in maps.iter() {
        if addr >= map.start && addr < map.end {
            let ret = (map.callback)(addr, len, false, 0);
            crate::utils::dtrace::trace_dtrace(addr, len, ret, false, &map.name);
            return ret;
        }
    }
    
    // Using log::error to avoid panic, consistent with previous behavior but safe
    log::error!("MMIO read: unmapped address 0x{:08x}", addr);
    0
}

pub fn mmio_write(addr: PAddr, len: usize, data: Word) {
    let maps = MMIO_MAPS.lock().unwrap();
    for map in maps.iter() {
        if addr >= map.start && addr < map.end {
            (map.callback)(addr, len, true, data);
            crate::utils::dtrace::trace_dtrace(addr, len, data, true, &map.name);
            return;
        }
    }
    
    log::error!("MMIO write: unmapped address 0x{:08x}", addr);
}

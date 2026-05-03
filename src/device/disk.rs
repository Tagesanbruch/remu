// Disk Device (Simple Stub)

use crate::common::{PAddr, Word};
use crate::generated::config::*;
use crate::memory::mmio::register_mmio;

pub fn init_disk() {
    if !HAS_DISK {
        return;
    }

    register_mmio("disk", DISK_CTL_MMIO as PAddr, 8, Box::new(disk_callback));
}

fn disk_callback(addr: PAddr, _len: usize, is_write: bool, _data: Word) -> Word {
    let _offset = addr - DISK_CTL_MMIO as PAddr;
    if is_write {
        0
    } else {
        0
    }
}

// VGA Device

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};
use std::sync::Mutex;

// VGA Registers
const VGA_CTL_WIDTH: u32 = 0;
const VGA_CTL_HEIGHT: u32 = 4;
const VGA_CTL_SYNC: u32 = 8;

struct VgaState {
    width: u32,
    height: u32,
    sync: u32,
    vmem: Vec<u8>,
}

lazy_static::lazy_static! {
    static ref VGA_STATE: Mutex<VgaState> = Mutex::new(VgaState {
        width: 400,
        height: 300,
        sync: 0,
        vmem: vec![0; (400 * 300 * 4) as usize], // Default size
    });
}

pub fn init_vga() {
    if !HAS_VGA { return; }
    
    // Register VMEM (Framebuffer)
    register_mmio("vmem", FB_ADDR, 0x200000, Box::new(vmem_callback));
    
    // Register VGA Control
    register_mmio("vga_ctl", VGA_CTL_MMIO, 8, Box::new(vga_ctl_callback));
    
    // TODO: Init SDL2 window/texture if display enabled
}

fn vmem_callback(addr: PAddr, len: usize, is_write: bool, data: Word) -> Word {
    let offset = (addr - FB_ADDR) as usize;
    let mut state = VGA_STATE.lock().unwrap();
    
    // Resize vmem if needed (or just ensure capacity)
    if offset + len > state.vmem.len() {
        state.vmem.resize(offset + len, 0);
    }
    
    if is_write {
        match len {
            1 => state.vmem[offset] = data as u8,
            2 => {
                 let bytes = (data as u16).to_le_bytes();
                 state.vmem[offset] = bytes[0];
                 state.vmem[offset+1] = bytes[1];
            }
            4 => {
                 let bytes = data.to_le_bytes();
                 state.vmem[offset] = bytes[0];
                 state.vmem[offset+1] = bytes[1];
                 state.vmem[offset+2] = bytes[2];
                 state.vmem[offset+3] = bytes[3];
            }
            _ => {}
        }
        0
    } else {
        // Read from vmem
        let mut ret: Word = 0;
        for i in 0..len {
            if offset + i < state.vmem.len() {
                ret |= (state.vmem[offset + i] as Word) << (i * 8);
            }
        }
        ret
    }
}

fn vga_ctl_callback(addr: PAddr, _len: usize, is_write: bool, data: Word) -> Word {
    let offset = (addr - VGA_CTL_MMIO) as u32;
    let mut state = VGA_STATE.lock().unwrap();
    
    if is_write {
        if offset == VGA_CTL_SYNC {
            state.sync = data;
            if data != 0 {
                // TODO: Update SDL screen
                update_screen(&state);
                state.sync = 0; // Clear sync
            }
        }
        0
    } else {
        match offset {
            VGA_CTL_WIDTH => state.width << 16 | state.height, // Usually packed, check AM implementation
            VGA_CTL_HEIGHT => state.height,
            VGA_CTL_SYNC => state.sync,
            _ => 0
        }
    }
}

fn update_screen(_state: &VgaState) {
    // SDL update logic here
}

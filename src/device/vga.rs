// VGA Device

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};
use std::sync::Mutex;

// VGA Registers
// VGA Registers
const VGA_CTL_SIZE: u32 = 0; // Packed width/height (NEMU convention)
const VGA_CTL_SYNC: u32 = 4; // Sync Trigger

struct VgaState {
    width: u32,
    height: u32,
    sync: u32,
    vmem: Vec<u8>,
}

lazy_static::lazy_static! {
    static ref VGA_STATE: Mutex<VgaState> = Mutex::new(VgaState {
        width: VGA_WIDTH,
        height: VGA_HEIGHT,
        sync: 0,
        vmem: vec![0; (VGA_WIDTH * VGA_HEIGHT * 4) as usize],
    });
}

pub fn init_vga() {
    if !HAS_VGA { return; }
    
    // Register VMEM (Framebuffer)
    register_mmio("vmem", FB_ADDR, 0x200000, Box::new(vmem_callback));
    
    // Register VGA Control
    register_mmio("vga_ctl", VGA_CTL_MMIO, 8, Box::new(vga_ctl_callback));
    
    // vmem is cleared to 0 (black/transparent) by vec! default
    
    // TODO: Init SDL2 window/texture if display enabled
    crate::device::sdl::init_sdl();
    
    // Force initial update to show pattern
    if VGA_SHOW_SCREEN {
        let state = VGA_STATE.lock().unwrap();
        crate::device::sdl::update_screen(&state.vmem);
    }
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
            // Note: We do NOT update screen here immediately. 
            // We wait for vga_update_screen called by device_update (throttled).
            // This prevents performance kill if guest syncs every pixel.
        }
        0
    } else {
        match offset {
            VGA_CTL_SIZE => (state.width << 16) | state.height,
            VGA_CTL_SYNC => state.sync,
            _ => 0
        }
    }
}

pub fn vga_update_screen() {
    let mut state = VGA_STATE.lock().unwrap();
    if state.sync != 0 {
        // Debug: Sum vmem content (Keep debug for now to confirm)
        let mut sum: u64 = 0;
        let mut non_zero_count: u64 = 0;
        for &byte in &state.vmem {
            sum += byte as u64;
            if byte != 0 { non_zero_count += 1; }
        }
        println!("[VGA Debug] Sync handled. vga_ctl_sync={}, vmem_sum={}, non_zero_bytes={}", state.sync, sum, non_zero_count);
        
        crate::device::sdl::update_screen(&state.vmem);
        state.sync = 0;
    }
}

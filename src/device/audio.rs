// Audio Device (Simple Stub)

use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use crate::common::{PAddr, Word};

pub fn init_audio() {
    if !HAS_AUDIO { return; }
    
    // Audio Controller: 0xa0000200
    register_mmio("audio", AUDIO_CTL_MMIO, 24, Box::new(audio_ctl_callback));
    
    // Audio Stream Buffer: 0xa1200000 (64KB)
    register_mmio("audio-sbuf", SB_ADDR, SB_SIZE as usize, Box::new(audio_sbuf_callback));
}

fn audio_ctl_callback(addr: PAddr, _len: usize, is_write: bool, _data: Word) -> Word {
    let _offset = addr - AUDIO_CTL_MMIO;
    if is_write { 0 } else { 0 }
}

fn audio_sbuf_callback(_addr: PAddr, _len: usize, is_write: bool, _data: Word) -> Word {
    if is_write { 0 } else { 0 }
}

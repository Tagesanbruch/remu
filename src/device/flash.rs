//! Flash Device Simulation for Model Storage
//!
//! Memory Map (Base: 0x3000_0000):
//!   0x0000_0000 - 0x00FF_FFFF: Flash Storage (16MB)
//!
//! This device simulates a SPI Flash or similar non-volatile storage
//! for storing AI model weights and other large data.
//!
//! Features:
//!   - Read-only during normal operation
//!   - Can be pre-loaded from file at startup
//!   - Supports memory-mapped reads

use crate::common::{PAddr, Word};
use crate::memory::mmio::register_mmio;
use std::sync::Mutex;
use std::fs;
use std::path::Path;
use lazy_static::lazy_static;

pub const FLASH_BASE: u32 = 0x30000000;
pub const FLASH_SIZE: usize = 16 * 1024 * 1024; // 16MB

struct FlashState {
    data: Vec<u8>,
    loaded: bool,
    file_path: Option<String>,
}

impl FlashState {
    fn new() -> Self {
        Self {
            data: vec![0xFF; FLASH_SIZE], // Flash default is 0xFF
            loaded: false,
            file_path: None,
        }
    }
}

lazy_static! {
    static ref FLASH: Mutex<FlashState> = Mutex::new(FlashState::new());
}

pub fn init_flash() {
    crate::Log!("Flash: Initializing at 0x{:08x}, size {} MB", FLASH_BASE, FLASH_SIZE / 1024 / 1024);
    register_mmio("flash", FLASH_BASE, FLASH_SIZE, Box::new(flash_callback));
}

/// Load flash content from a binary file
pub fn flash_load_file(path: &str) -> Result<usize, String> {
    let mut state = FLASH.lock().unwrap();
    
    if !Path::new(path).exists() {
        return Err(format!("Flash file not found: {}", path));
    }
    
    let data = fs::read(path).map_err(|e| format!("Failed to read flash file: {}", e))?;
    
    if data.len() > FLASH_SIZE {
        return Err(format!("Flash file too large: {} bytes (max {})", data.len(), FLASH_SIZE));
    }
    
    // Copy to flash memory
    state.data[..data.len()].copy_from_slice(&data);
    state.loaded = true;
    state.file_path = Some(path.to_string());
    
    crate::Log!("Flash: Loaded {} bytes from {}", data.len(), path);
    Ok(data.len())
}

/// Get flash data pointer for direct access (used by NPU DMA etc)
pub fn flash_get_data() -> Vec<u8> {
    let state = FLASH.lock().unwrap();
    state.data.clone()
}

fn flash_callback(addr: PAddr, len: usize, is_write: bool, data: Word) -> Word {
    let offset = (addr - FLASH_BASE) as usize;
    let state = FLASH.lock().unwrap();
    
    if is_write {
        // Flash is read-only in normal operation
        // Could implement erase/program commands via special registers
        log::warn!("Flash: Write ignored at offset 0x{:x}", offset);
        return 0;
    }
    
    // Read operation
    if offset + len > FLASH_SIZE {
        log::error!("Flash: Read out of bounds at offset 0x{:x}", offset);
        return 0;
    }
    
    match len {
        1 => state.data[offset] as Word,
        2 => {
            (state.data[offset] as Word) |
            ((state.data[offset + 1] as Word) << 8)
        }
        4 => {
            (state.data[offset] as Word) |
            ((state.data[offset + 1] as Word) << 8) |
            ((state.data[offset + 2] as Word) << 16) |
            ((state.data[offset + 3] as Word) << 24)
        }
        _ => 0,
    }
}

/// Dump flash statistics
pub fn dump_flash_info() -> String {
    let state = FLASH.lock().unwrap();
    format!(
        r#"{{
  "Flash Base": "0x{:08x}",
  "Flash Size": "{} MB",
  "Loaded": {},
  "File": "{}"
}}"#,
        FLASH_BASE,
        FLASH_SIZE / 1024 / 1024,
        state.loaded,
        state.file_path.as_deref().unwrap_or("none")
    )
}

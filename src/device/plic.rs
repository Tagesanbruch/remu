// PLIC Device

use crate::common::{PAddr, Word};
use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use std::sync::Mutex;

const PLIC_BASE: PAddr = 0x0c000000;
const PLIC_SIZE: usize = 0x400000;
const MAX_SOURCE: usize = 64;
const CONTEXTS: usize = 2;
const PRIORITY_BASE: u32 = 0x000000;
const PENDING_BASE: u32 = 0x001000;
const ENABLE_BASE: u32 = 0x002000;
const ENABLE_STRIDE: u32 = 0x80;
const CONTEXT_BASE: u32 = 0x200000;
const CONTEXT_STRIDE: u32 = 0x1000;
const CLAIM_COMPLETE: u32 = 0x04;

struct PlicState {
    priority: [u32; MAX_SOURCE],
    pending: u64,
    line_level: u64,
    claimed: u64,
    enable: [u64; CONTEXTS],
    threshold: [u32; CONTEXTS],
}

impl PlicState {
    fn new() -> Self {
        Self {
            priority: [1; MAX_SOURCE],
            pending: 0,
            line_level: 0,
            claimed: 0,
            enable: [0; CONTEXTS],
            threshold: [0; CONTEXTS],
        }
    }

    fn set_line_level(&mut self, source: u32, level: bool) {
        let bit = source_bit(source);
        if bit == 0 {
            return;
        }

        if level {
            self.line_level |= bit;
            if (self.claimed & bit) == 0 {
                self.pending |= bit;
            }
        } else {
            self.line_level &= !bit;
            self.pending &= !bit;
        }
    }

    fn claim(&mut self, context: usize) -> u32 {
        let Some(source) = self.best_pending(context) else {
            return 0;
        };

        let bit = source_bit(source);
        self.pending &= !bit;
        self.claimed |= bit;
        source
    }

    fn complete(&mut self, source: u32) {
        let bit = source_bit(source);
        if bit == 0 {
            return;
        }

        self.claimed &= !bit;
        if (self.line_level & bit) != 0 {
            self.pending |= bit;
        }
    }

    fn best_pending(&self, context: usize) -> Option<u32> {
        if context >= CONTEXTS {
            return None;
        }

        let candidates = self.pending & self.enable[context];
        let threshold = self.threshold[context];
        let mut best_source = 0;
        let mut best_priority = 0;

        for source in 1..MAX_SOURCE {
            let bit = 1u64 << source;
            if (candidates & bit) == 0 {
                continue;
            }

            let priority = self.priority[source];
            if priority > threshold && priority > best_priority {
                best_source = source as u32;
                best_priority = priority;
            }
        }

        if best_source == 0 {
            None
        } else {
            Some(best_source)
        }
    }

    fn has_claimable_irq(&self) -> bool {
        (0..CONTEXTS).any(|context| self.best_pending(context).is_some())
    }
}

lazy_static::lazy_static! {
    static ref PLIC: Mutex<PlicState> = Mutex::new(PlicState::new());
}

pub fn init_plic() {
    if !HAS_PLIC {
        return;
    }

    // 0x0c000000 - 0x0c200000+ (4MB range usually)
    register_mmio("plic", PLIC_BASE, PLIC_SIZE, Box::new(plic_callback));
}

pub fn set_irq_level(source: u32, level: bool) {
    if !HAS_PLIC {
        return;
    }

    let mut state = PLIC.lock().unwrap();
    state.set_line_level(source, level);
    refresh_external_irq_locked(&state);
}

fn plic_callback(addr: PAddr, _len: usize, is_write: bool, data: Word) -> Word {
    let offset = addr - PLIC_BASE;
    let mut state = PLIC.lock().unwrap();

    if is_write {
        if write_priority(&mut state, offset, data) {
            refresh_external_irq_locked(&state);
            return 0;
        }

        if write_enable(&mut state, offset, data) {
            refresh_external_irq_locked(&state);
            return 0;
        }

        if let Some((context, reg)) = context_reg(offset) {
            if context < CONTEXTS {
                if reg == 0 {
                    state.threshold[context] = data;
                } else if reg == CLAIM_COMPLETE {
                    state.complete(data);
                }
                refresh_external_irq_locked(&state);
                return 0;
            }
        }

        refresh_external_irq_locked(&state);
        0
    } else {
        let ret = if offset >= PRIORITY_BASE && offset < PENDING_BASE && (offset & 0x3) == 0 {
            let source = (offset / 4) as usize;
            if source < MAX_SOURCE {
                state.priority[source]
            } else {
                0
            }
        } else if offset >= PENDING_BASE && offset < ENABLE_BASE {
            read_word_bits(state.pending, offset - PENDING_BASE)
        } else if let Some(value) = read_enable(&state, offset) {
            value
        } else if let Some((context, reg)) = context_reg(offset) {
            if context < CONTEXTS {
                if reg == 0 {
                    state.threshold[context]
                } else if reg == CLAIM_COMPLETE {
                    state.claim(context)
                } else {
                    0
                }
            } else {
                0
            }
        } else {
            0
        };

        refresh_external_irq_locked(&state);
        ret
    }
}

fn source_bit(source: u32) -> u64 {
    if source == 0 || source as usize >= MAX_SOURCE {
        0
    } else {
        1u64 << source
    }
}

fn write_priority(state: &mut PlicState, offset: u32, data: Word) -> bool {
    if offset >= PENDING_BASE || (offset & 0x3) != 0 {
        return false;
    }

    let source = (offset / 4) as usize;
    if source < MAX_SOURCE {
        state.priority[source] = data & 0x7;
    }
    true
}

fn write_enable(state: &mut PlicState, offset: u32, data: Word) -> bool {
    if offset < ENABLE_BASE || offset >= CONTEXT_BASE {
        return false;
    }

    let rel = offset - ENABLE_BASE;
    let context = (rel / ENABLE_STRIDE) as usize;
    let word = (rel % ENABLE_STRIDE) / 4;
    if context >= CONTEXTS || word >= 2 {
        return true;
    }

    let shift = word * 32;
    let mask = 0xffff_ffffu64 << shift;
    state.enable[context] = (state.enable[context] & !mask) | ((data as u64) << shift);
    true
}

fn read_enable(state: &PlicState, offset: u32) -> Option<Word> {
    if offset < ENABLE_BASE || offset >= CONTEXT_BASE {
        return None;
    }

    let rel = offset - ENABLE_BASE;
    let context = (rel / ENABLE_STRIDE) as usize;
    let word = (rel % ENABLE_STRIDE) / 4;
    if context >= CONTEXTS || word >= 2 {
        return Some(0);
    }

    Some(read_word_bits(state.enable[context], word * 4))
}

fn read_word_bits(bits: u64, byte_offset: u32) -> Word {
    if (byte_offset & 0x3) != 0 {
        return 0;
    }

    let word = byte_offset / 4;
    if word >= 2 {
        0
    } else {
        ((bits >> (word * 32)) & 0xffff_ffff) as Word
    }
}

fn context_reg(offset: u32) -> Option<(usize, u32)> {
    if offset < CONTEXT_BASE {
        return None;
    }

    let rel = offset - CONTEXT_BASE;
    let context = (rel / CONTEXT_STRIDE) as usize;
    let reg = rel % CONTEXT_STRIDE;
    Some((context, reg))
}

fn refresh_external_irq_locked(state: &PlicState) {
    crate::device::intr::set_seip(state.has_claimable_irq());
}

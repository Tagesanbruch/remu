// Serial Device (UART)

use crate::common::{PAddr, Word};
use crate::generated::config::*;
use crate::memory::mmio::register_mmio;
use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

const FIFO_LEN: usize = 1024;
const UART0_IRQ: u32 = 10;

const IER_RDI: u8 = 0x01;
const IER_THRI: u8 = 0x02;

const IIR_NO_INT: u8 = 0x01;
const IIR_THRI: u8 = 0x02;
const IIR_RDI: u8 = 0x04;
const IIR_FIFO_BITS: u8 = 0xc0;

const LCR_DLAB: u8 = 0x80;
const LSR_DR: u8 = 0x01;
const LSR_THRE: u8 = 0x20;
const LSR_TEMT: u8 = 0x40;

struct SerialState {
    rx_fifo: VecDeque<u8>,
    ier: u8,
    fcr: u8,
    lcr: u8,
    mcr: u8,
    dll: u8,
    dlm: u8,
    scr: u8,
    thr_empty: bool,
    thre_pending: bool,
}

pub struct SerialSnapshot {
    pub rx_fifo: Vec<u8>,
    pub ier: u8,
    pub fcr: u8,
    pub lcr: u8,
    pub mcr: u8,
    pub dll: u8,
    pub dlm: u8,
    pub scr: u8,
    pub thr_empty: bool,
    pub thre_pending: bool,
}

impl SerialState {
    fn new() -> Self {
        Self {
            rx_fifo: VecDeque::with_capacity(FIFO_LEN),
            ier: 0,
            fcr: 0,
            lcr: 0,
            mcr: 0,
            dll: 0,
            dlm: 0,
            scr: 0,
            thr_empty: true,
            thre_pending: false,
        }
    }

    fn dlab(&self) -> bool {
        (self.lcr & LCR_DLAB) != 0
    }

    fn rx_ready(&self) -> bool {
        !self.rx_fifo.is_empty()
    }

    fn rx_irq_pending(&self) -> bool {
        (self.ier & IER_RDI) != 0 && self.rx_ready()
    }

    fn thri_irq_pending(&self) -> bool {
        (self.ier & IER_THRI) != 0 && self.thre_pending
    }

    fn irq_pending(&self) -> bool {
        self.rx_irq_pending() || self.thri_irq_pending()
    }

    fn iir(&mut self) -> u8 {
        let fifo_bits = if (self.fcr & 0x01) != 0 {
            IIR_FIFO_BITS
        } else {
            0
        };
        if self.rx_irq_pending() {
            fifo_bits | IIR_RDI
        } else if self.thri_irq_pending() {
            self.thre_pending = false;
            fifo_bits | IIR_THRI
        } else {
            fifo_bits | IIR_NO_INT
        }
    }

    fn lsr(&self) -> u8 {
        let mut lsr = LSR_THRE | LSR_TEMT;
        if self.rx_ready() {
            lsr |= LSR_DR;
        }
        lsr
    }
}

lazy_static::lazy_static! {
    static ref SERIAL: Mutex<SerialState> = Mutex::new(SerialState::new());
}

static STDIN_ENABLED: AtomicBool = AtomicBool::new(false);
static STDIN_CONFIGURED: AtomicBool = AtomicBool::new(false);

pub fn set_stdin_enabled(enabled: bool) {
    STDIN_ENABLED.store(enabled, Ordering::Relaxed);
}

pub fn init_serial() {
    if !HAS_SERIAL {
        return;
    }

    register_mmio("serial", SERIAL_MMIO as PAddr, 8, Box::new(serial_callback));
}

pub fn serial_update() {
    if !HAS_SERIAL || !STDIN_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    configure_stdin_nonblocking();
    drain_stdin();
}

fn serial_callback(addr: PAddr, _len: usize, is_write: bool, data: Word) -> Word {
    let offset = addr - SERIAL_MMIO as PAddr;
    let mut state = SERIAL.lock().unwrap();
    if is_write {
        let value = data as u8;
        match offset {
            0 => {
                if state.dlab() {
                    state.dll = value;
                } else {
                    state.thr_empty = false;
                    print!("{}", value as char);
                    let _ = io::stdout().flush();
                    crate::utils::sandbox::observe_serial_byte(value);
                    state.thr_empty = true;
                    if (state.ier & IER_THRI) != 0 {
                        state.thre_pending = true;
                    }
                }
            }
            1 => {
                if state.dlab() {
                    state.dlm = value;
                } else {
                    let old_ier = state.ier;
                    state.ier = value & 0x0f;
                    if (old_ier & IER_THRI) == 0 && (state.ier & IER_THRI) != 0 && state.thr_empty {
                        state.thre_pending = true;
                    }
                    if (state.ier & IER_THRI) == 0 {
                        state.thre_pending = false;
                    }
                }
            }
            2 => {
                state.fcr = value;
                if (value & 0x02) != 0 {
                    state.rx_fifo.clear();
                }
                if (value & 0x04) != 0 {
                    state.thre_pending = false;
                }
            }
            3 => state.lcr = value,
            4 => state.mcr = value,
            7 => state.scr = value,
            _ => {}
        }
        update_irq_locked(&state);
        0
    } else {
        let ret = match offset {
            0 => {
                if state.dlab() {
                    state.dll
                } else {
                    state.rx_fifo.pop_front().unwrap_or(0)
                }
            }
            1 => {
                if state.dlab() {
                    state.dlm
                } else {
                    state.ier
                }
            }
            2 => state.iir(),
            3 => state.lcr,
            4 => state.mcr,
            5 => {
                // LSR: data-ready plus immediately-empty transmitter.
                state.lsr()
            }
            6 => 0xb0, // CTS/DSR/DCD asserted.
            7 => state.scr,
            _ => 0,
        };
        update_irq_locked(&state);
        ret as Word
    }
}

fn enqueue_rx_byte(ch: u8) {
    let mut state = SERIAL.lock().unwrap();
    if state.rx_fifo.len() < FIFO_LEN {
        state.rx_fifo.push_back(ch);
    }
    update_irq_locked(&state);
}

pub fn enqueue_rx_bytes(bytes: &[u8]) {
    let mut state = SERIAL.lock().unwrap();
    for &ch in bytes {
        if state.rx_fifo.len() < FIFO_LEN {
            state.rx_fifo.push_back(ch);
        }
    }
    update_irq_locked(&state);
}

fn update_irq_locked(state: &SerialState) {
    crate::device::plic::set_irq_level(UART0_IRQ, state.irq_pending());
}

pub fn snapshot_state() -> SerialSnapshot {
    let state = SERIAL.lock().unwrap();
    SerialSnapshot {
        rx_fifo: state.rx_fifo.iter().copied().collect(),
        ier: state.ier,
        fcr: state.fcr,
        lcr: state.lcr,
        mcr: state.mcr,
        dll: state.dll,
        dlm: state.dlm,
        scr: state.scr,
        thr_empty: state.thr_empty,
        thre_pending: state.thre_pending,
    }
}

pub fn restore_state(snapshot: SerialSnapshot) {
    let mut state = SERIAL.lock().unwrap();
    state.rx_fifo = snapshot.rx_fifo.into_iter().collect();
    state.ier = snapshot.ier;
    state.fcr = snapshot.fcr;
    state.lcr = snapshot.lcr;
    state.mcr = snapshot.mcr;
    state.dll = snapshot.dll;
    state.dlm = snapshot.dlm;
    state.scr = snapshot.scr;
    state.thr_empty = snapshot.thr_empty;
    state.thre_pending = snapshot.thre_pending;
    update_irq_locked(&state);
}

#[cfg(unix)]
fn configure_stdin_nonblocking() {
    if STDIN_CONFIGURED.swap(true, Ordering::Relaxed) {
        return;
    }

    unsafe {
        let fd = libc::STDIN_FILENO;
        let flags = libc::fcntl(fd, libc::F_GETFL);
        if flags >= 0 {
            let _ = libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }
    }
}

#[cfg(not(unix))]
fn configure_stdin_nonblocking() {}

#[cfg(unix)]
fn drain_stdin() {
    loop {
        let mut buf = [0u8; 128];
        let n = unsafe {
            libc::read(
                libc::STDIN_FILENO,
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len(),
            )
        };

        if n > 0 {
            for &ch in &buf[..n as usize] {
                enqueue_rx_byte(ch);
            }
            continue;
        }

        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() != io::ErrorKind::WouldBlock {
                crate::Log!("serial stdin read disabled after error: {}", err);
                STDIN_ENABLED.store(false, Ordering::Relaxed);
            }
        }
        break;
    }
}

#[cfg(not(unix))]
fn drain_stdin() {}

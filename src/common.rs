// Common types and macros

// Architectural word/address types.  RV32 execution masks register writes back
// to 32 bits, while RV64 uses the full value.
pub type PAddr = u64;
pub type VAddr = u64;
pub type Word = u64;
pub type SWord = i64;

pub fn xlen() -> u32 {
    if crate::generated::config::RV64 {
        64
    } else {
        32
    }
}

pub fn xlen_mask() -> Word {
    if crate::generated::config::RV64 {
        u64::MAX
    } else {
        0xffff_ffff
    }
}

pub fn mask_xlen(value: Word) -> Word {
    value & xlen_mask()
}

pub fn sign_extend(value: Word, width: u32) -> Word {
    if width == 0 || width >= 64 {
        value
    } else {
        let shift = 64 - width;
        (((value << shift) as i64) >> shift) as Word
    }
}

pub fn sign_extend_xlen(value: Word) -> Word {
    if crate::generated::config::RV64 {
        value
    } else {
        sign_extend(value & 0xffff_ffff, 32)
    }
}

pub fn as_signed_xlen(value: Word) -> SWord {
    if crate::generated::config::RV64 {
        value as i64
    } else {
        (value as u32 as i32) as i64
    }
}

pub fn sext32(value: u32) -> Word {
    (value as i32 as i64) as Word
}

// CPU state enum
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RemuState {
    Running,
    Stop,
    End,
    Abort,
    Quit,
}

// Privilege modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivMode {
    User = 0,
    Supervisor = 1,
    Machine = 3,
}

// Color codes for terminal output
pub const ANSI_FG_BLACK: &str = "\x1b[30m";
pub const ANSI_FG_RED: &str = "\x1b[31m";
pub const ANSI_FG_GREEN: &str = "\x1b[32m";
pub const ANSI_FG_YELLOW: &str = "\x1b[33m";
pub const ANSI_FG_BLUE: &str = "\x1b[34m";
pub const ANSI_FG_MAGENTA: &str = "\x1b[35m";
pub const ANSI_FG_CYAN: &str = "\x1b[36m";
pub const ANSI_FG_WHITE: &str = "\x1b[37m";
pub const ANSI_BG_RED: &str = "\x1b[41m";
pub const ANSI_RESET: &str = "\x1b[0m";

pub fn colored(text: &str, color: &str) -> String {
    format!("{}{}{}", color, text, ANSI_RESET)
}

// Custom Panic Macro that dumps state
#[macro_export]
macro_rules! panic_remu {
    ($($arg:tt)*) => {
        eprintln!("\x1b[31mPANIC: {}\x1b[0m", format_args!($($arg)*));
        crate::utils::itrace::show_itrace();
        crate::utils::mtrace::show_mtrace();
        crate::monitor::set_exit_status_bad();
        std::process::exit(1);
    }
}

// Custom Assert Macro
#[macro_export]
macro_rules! assert_remu {
    ($cond:expr) => {
        if !$cond {
            panic_remu!("Assertion failed: {}", stringify!($cond));
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if !$cond {
            panic_remu!("Assertion failed: {}", format_args!($($arg)+));
        }
    };
}

// Common types and macros used throughout RustNEMU

// use std::fmt;  // Unused

// Word type for RISC-V32
pub type Word = u32;
pub type SWord = i32;
pub type PAddr = u32;
pub type VAddr = u32;

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

// Macros for compile-time configuration
#[macro_export]
macro_rules! ifdef {
    ($feature:expr, $code:expr) => {
        #[cfg(feature = $feature)]
        {
            $code
        }
    };
}

#[macro_export]
macro_rules! ifndef {
    ($feature:expr, $code:expr) => {
        #[cfg(not(feature = $feature))]
        {
            $code
        }
    };
}

// Bit manipulation macros
#[macro_export]
macro_rules! bits {
    ($val:expr, $hi:expr, $lo:expr) => {
        (($val >> $lo) & ((1 << ($hi - $lo + 1)) - 1))
    };
}

#[macro_export]
macro_rules! sext {
    ($val:expr, $bits:expr) => {{
        let shift = 32 - $bits;
        (($val << shift) as i32 >> shift) as u32
    }};
}

// Panic with formatted message
#[macro_export]
macro_rules! panic_remu {
    ($($arg:tt)*) => {
        panic!("{}", format!($($arg)*))
    };
}

// Assert with better error messages
#[macro_export]
macro_rules! assert_remu {
    ($cond:expr, $($arg:tt)*) => {
        if !$cond {
            panic_remu!($($arg)*);
        }
    };
}

// Formatting helpers
pub fn fmt_word(w: Word) -> String {
    format!("0x{:08x}", w)
}

pub fn fmt_paddr(p: PAddr) -> String {
    format!("0x{:08x}", p)
}

pub fn colored(text: &str, color: &str) -> String {
    format!("{}{}{}", color, text, ANSI_RESET)
}

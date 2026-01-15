// Common types and macros

// Word type for RISC-V32
pub type PAddr = u32;
pub type VAddr = u32;
pub type Word = u32;
pub type SWord = i32;

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

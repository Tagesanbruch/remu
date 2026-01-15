// NEMU-style logging system
// Format: [src/module/file.rs:line function_name] message
// Log messages are in BLUE color

use std::sync::Mutex;
use std::fs::File;
use std::io::Write;
use lazy_static::lazy_static;

// ANSI color codes (matching C NEMU)
pub const ANSI_NONE: &str = "\x1b[0m";
pub const ANSI_FG_BLACK: &str = "\x1b[30m";
pub const ANSI_FG_RED: &str = "\x1b[31m";
pub const ANSI_FG_GREEN: &str = "\x1b[32m";
pub const ANSI_FG_YELLOW: &str = "\x1b[33m";
pub const ANSI_FG_BLUE: &str = "\x1b[34m";
pub const ANSI_FG_MAGENTA: &str = "\x1b[35m";
pub const ANSI_FG_CYAN: &str = "\x1b[36m";
pub const ANSI_FG_WHITE: &str = "\x1b[37m";
pub const ANSI_BG_BLACK: &str = "\x1b[40m";
pub const ANSI_BG_RED: &str = "\x1b[41m";
pub const ANSI_BG_GREEN: &str = "\x1b[42m";
pub const ANSI_BG_YELLOW: &str = "\x1b[43m";
pub const ANSI_BG_BLUE: &str = "\x1b[44m";
pub const ANSI_BG_MAGENTA: &str = "\x1b[45m";
pub const ANSI_BG_CYAN: &str = "\x1b[46m";
pub const ANSI_BG_WHITE: &str = "\x1b[47m";

// Helper function to format with ANSI colors
#[macro_export]
macro_rules! ansi_fmt {
    ($msg:expr, $color:expr) => {
        format!("{}{}{}", $color, $msg, $crate::utils::log::ANSI_NONE)
    };
}

lazy_static! {
    static ref LOG_FILE: Mutex<Option<File>> = Mutex::new(None);
}

pub fn init_log(logfile: &str) {
    let file = File::create(logfile).expect("Failed to create log file");
    *LOG_FILE.lock().unwrap() = Some(file);
    _log(file!(), line!(), "", &format!("Log is written to {}", logfile));
}

pub fn _log(file: &str, line: u32, func: &str, message: &str) {
    // Extract filename from full path
    let filename = file.split('/').last().unwrap_or(file);
    
    // Format: [src/file.rs:line function] message (in BLUE)
    let log_msg = if func.is_empty() {
        format!("{}[{}:{}] {}{}\n",
            ANSI_FG_BLUE, filename, line, message, ANSI_NONE)
    } else {
        format!("{}[{}:{} {}] {}{}\n",
            ANSI_FG_BLUE, filename, line, func, message, ANSI_NONE)
    };
    
    print!("{}", log_msg);
    
    // Also write to log file (without colors)
    if let Some(ref mut file) = *LOG_FILE.lock().unwrap() {
        let plain_msg = if func.is_empty() {
            format!("[{}:{}] {}\n", filename, line, message)
        } else {
            format!("[{}:{} {}] {}\n", filename, line, func, message)
        };
        let _ = file.write_all(plain_msg.as_bytes());
    }
}

// Log macro - simplified to not require function name (Rust limitation)
#[macro_export]
macro_rules! Log {
    ($($arg:tt)*) => {{
        $crate::utils::log::_log(file!(), line!(), "", &format!($($arg)*));
    }};
}

// Panic macro
#[macro_export]
macro_rules! panic_on_cond{
    ($cond:expr, $($arg:tt)*) => {
        if $cond {
            panic!("{}", format!($($arg)*));
        }
    };
}

// Assert macro
#[macro_export]
macro_rules! Assert {
    ($cond:expr, $($arg:tt)*) => {
        if !($cond) {
            panic!("{}", format!($($arg)*));
        }
    };
}

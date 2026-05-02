// ECALL Trace

use crate::common::Word;
use crate::generated::config::*;

#[derive(Clone)]
pub struct EcallTraceEntry {
    pub pc: Word,
    pub cause: Word,
    pub mode: u8,
    pub args: [Word; 8],
}

impl ToString for EcallTraceEntry {
    fn to_string(&self) -> String {
        let mode_str = match self.mode {
            3 => "Machine",
            1 => "Supervisor",
            0 => "User",
            _ => "Unknown",
        };
        let syscall = self.args[7];
        let syscall_name = syscall_name(syscall);
        format!(
            "ECALL: Mode={} Cause={} @ PC=0x{:08x} a7={}{} a0=0x{:08x} a1=0x{:08x} a2=0x{:08x} a3=0x{:08x} a4=0x{:08x} a5=0x{:08x} a6=0x{:08x}",
            mode_str,
            self.cause,
            self.pc,
            syscall,
            syscall_name,
            self.args[0],
            self.args[1],
            self.args[2],
            self.args[3],
            self.args[4],
            self.args[5],
            self.args[6],
        )
    }
}

lazy_static::lazy_static! {
    static ref ECALL_BUF: std::sync::Mutex<crate::utils::ringbuffer::RingBuffer<EcallTraceEntry>> = {
        let size = if crate::generated::config::TRACE_ECALL {
            crate::generated::config::TRACE_ECALL_RINGBUF as usize
        } else { 1 };
        std::sync::Mutex::new(crate::utils::ringbuffer::RingBuffer::new(size))
    };
}

pub fn trace_ecall(pc: Word, cause: Word, mode: u8, args: [Word; 8]) {
    if !TRACE_ECALL {
        return;
    }

    let entry = EcallTraceEntry {
        pc,
        cause,
        mode,
        args,
    };

    ECALL_BUF.lock().unwrap().push(entry);
}

pub fn show_ecall_trace() {
    if !TRACE_ECALL {
        return;
    }

    crate::Log!("--- RingBuffer Content ---");
    let buf = ECALL_BUF.lock().unwrap();
    if buf.is_empty() {
        crate::Log!("(empty)");
    } else {
        for entry in buf.iter() {
            crate::Log!("{}", entry.to_string());
        }
    }
    crate::Log!("--------------------------");
}

fn syscall_name(no: Word) -> &'static str {
    match no {
        17 => " (getcwd)",
        25 => " (fcntl)",
        29 => " (ioctl)",
        34 => " (mkdirat)",
        35 => " (unlinkat)",
        48 => " (faccessat)",
        49 => " (chdir)",
        56 => " (openat)",
        57 => " (close)",
        61 => " (getdents64)",
        62 => " (lseek)",
        63 => " (read)",
        64 => " (write)",
        66 => " (writev)",
        78 => " (readlinkat)",
        79 => " (fstatat)",
        80 => " (fstat)",
        93 => " (exit)",
        94 => " (exit_group)",
        98 => " (futex)",
        101 => " (nanosleep)",
        102 => " (getitimer)",
        113 => " (clock_gettime)",
        129 => " (kill)",
        134 => " (rt_sigaction)",
        135 => " (rt_sigprocmask)",
        154 => " (setpgid)",
        155 => " (getpgid)",
        160 => " (uname)",
        172 => " (getpid)",
        174 => " (getuid)",
        175 => " (geteuid)",
        176 => " (getgid)",
        177 => " (getegid)",
        214 => " (brk)",
        215 => " (munmap)",
        220 => " (clone)",
        221 => " (execve)",
        222 => " (mmap)",
        226 => " (mprotect)",
        259 => " (riscv_flush_icache)",
        260 => " (wait4)",
        261 => " (prlimit64)",
        291 => " (statx)",
        293 => " (rseq)",
        278 => " (getrandom)",
        403 => " (clock_gettime64)",
        413 => " (pselect6_time64)",
        414 => " (ppoll_time64)",
        416 => " (io_pgetevents_time64)",
        422 => " (futex_time64)",
        1039 => " (syslog)",
        _ => "",
    }
}

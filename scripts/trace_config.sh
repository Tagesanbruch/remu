#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/trace_config.sh MODE

MODE:
  off    Disable REMU trace buffers.
  light  Enable instruction/interrupt/ecall ring buffers.
  mmu    Enable interrupt/ecall/MMU/TLB ring buffers.
  full   Enable instruction/memory/device/function/interrupt/ecall/MMU/TLB buffers.

After updating .config, this regenerates src/generated/config.rs.
USAGE
}

mode="${1:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
remu_home="$(cd "$script_dir/.." && pwd)"
config="$remu_home/.config"

if [[ -z "$mode" || "$mode" == "-h" || "$mode" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -f "$config" ]]; then
  echo "Missing $config. Run make menuconfig or make defconfig first." >&2
  exit 1
fi

set_config() {
  local key="$1"
  local value="$2"
  if grep -q "^CONFIG_${key}=" "$config"; then
    sed -i.bak "s|^CONFIG_${key}=.*|CONFIG_${key}=${value}|" "$config"
  elif grep -q "^# CONFIG_${key} is not set" "$config"; then
    sed -i.bak "s|^# CONFIG_${key} is not set|CONFIG_${key}=${value}|" "$config"
  else
    printf 'CONFIG_%s=%s\n' "$key" "$value" >> "$config"
  fi
  rm -f "$config.bak"
}

unset_config() {
  local key="$1"
  if grep -q "^CONFIG_${key}=" "$config"; then
    sed -i.bak "s|^CONFIG_${key}=.*|# CONFIG_${key} is not set|" "$config"
  elif ! grep -q "^# CONFIG_${key} is not set" "$config"; then
    printf '# CONFIG_%s is not set\n' "$key" >> "$config"
  fi
  rm -f "$config.bak"
}

set_common_ranges() {
  set_config TRACE_START 0
  set_config TRACE_END 0
}

disable_trace_detail() {
  unset_config ITRACE
  unset_config MTRACE
  unset_config FTRACE
  unset_config DTRACE
  unset_config TRACE_INTR
  unset_config TRACE_MMU
  unset_config TRACE_TLB
  unset_config TRACE_PLIC
  unset_config TRACE_ECALL
  set_config ITRACE_RINGBUF 0
  set_config MTRACE_RINGBUF 0
  set_config FTRACE_BUF 0
  set_config DTRACE_RINGBUF 0
  set_config TRACE_INTR_RINGBUF 0
  set_config TRACE_MMU_RINGBUF 0
  set_config TRACE_TLB_RINGBUF 0
  set_config TRACE_ECALL_RINGBUF 0
}

case "$mode" in
  off)
    unset_config TRACE
    disable_trace_detail
    ;;
  light)
    set_config TRACE y
    set_common_ranges
    disable_trace_detail
    set_config ITRACE y
    set_config ITRACE_COND '"true"'
    set_config ITRACE_RINGBUF 128
    set_config TRACE_INTR y
    set_config TRACE_INTR_RINGBUF 256
    set_config TRACE_ECALL y
    set_config TRACE_ECALL_RINGBUF 256
    ;;
  mmu)
    set_config TRACE y
    set_common_ranges
    disable_trace_detail
    set_config TRACE_INTR y
    set_config TRACE_INTR_RINGBUF 512
    set_config TRACE_ECALL y
    set_config TRACE_ECALL_RINGBUF 256
    set_config TRACE_MMU y
    set_config TRACE_MMU_RINGBUF 1024
    set_config TRACE_TLB y
    set_config TRACE_TLB_RINGBUF 512
    ;;
  full)
    set_config TRACE y
    set_common_ranges
    disable_trace_detail
    set_config ITRACE y
    set_config ITRACE_COND '"true"'
    set_config ITRACE_RINGBUF 256
    set_config MTRACE y
    set_config MTRACE_COND '"true"'
    set_config MTRACE_RINGBUF 256
    set_config FTRACE y
    set_config FTRACE_COND '"true"'
    set_config FTRACE_BUF 1024
    set_config DTRACE y
    set_config DTRACE_COND '"true"'
    set_config DTRACE_RINGBUF 256
    set_config TRACE_INTR y
    set_config TRACE_INTR_RINGBUF 512
    set_config TRACE_ECALL y
    set_config TRACE_ECALL_RINGBUF 256
    set_config TRACE_MMU y
    set_config TRACE_MMU_RINGBUF 1024
    set_config TRACE_TLB y
    set_config TRACE_TLB_RINGBUF 512
    ;;
  *)
    echo "Unknown trace mode: $mode" >&2
    usage >&2
    exit 1
    ;;
esac

python3 "$remu_home/scripts/gen_config.py"
echo "Trace configuration set to: $mode"

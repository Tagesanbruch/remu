#include <am.h>
#include <klib-macros.h>
#include <ysyxsoc.h>

extern char _heap_start;
int main(const char *args);

extern char _pmem_start;
#define PMEM_SIZE (8 * 1024)
#define PMEM_END ((uintptr_t)&_pmem_start + PMEM_SIZE)

Area heap = RANGE(&_heap_start, PMEM_END);
#ifndef MAINARGS
#define MAINARGS ""
#endif
static const char mainargs[] = MAINARGS;

void putch(char ch) { outb(SERIAL_PORT, ch); }

void halt(int code) {
  ysyxsoc_trap(code);
  while (1)
    ;
}

void _trm_init() {
  extern char _data_lma, _data, _edata;
  char *src = &_data_lma;
  char *dst = &_data;
  extern char _heap_start;
  // copy data section
  while (dst < &_edata) {
    *dst++ = *src++;
  }

  int ret = main(mainargs);
  halt(ret);
}

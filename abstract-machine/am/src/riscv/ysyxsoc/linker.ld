ENTRY(_start)

MEMORY {
  flash (rx) : ORIGIN = 0x30000000, LENGTH = 16M
  mrom (rx)  : ORIGIN = 0x20000000, LENGTH = 4K
  sram (rwx) : ORIGIN = 0x0f000000, LENGTH = 8K
}

SECTIONS {
  /* _pmem_start and _entry_offset are defined in LDFLAGS */
  . = ORIGIN(flash);
  .text : {
    *(entry)
    *(.text*)
  } > flash AT> flash
  etext = .;
  _etext = .;
  .rodata : {
    *(.rodata*)
    *(.srodata*)
    . = ALIGN(4);
  } > flash AT> flash
  
  .data : {
    _data = .;
    *(.data*)
    *(.sdata*)
    . = ALIGN(4);
    _edata = .;
  } > sram AT> flash
  
  _data_lma = LOADADDR(.data);
  
  .bss (NOLOAD) : {
    _bss_start = .;
    *(.bss*)
    *(.sbss*)
    *(.scommon)
    . = ALIGN(4);
    _end = .;
  } > sram
  
  _stack_top = ORIGIN(sram) + LENGTH(sram);
  . = _stack_top;
  _stack_pointer = .;
  end = .;
  
  _heap_start = ALIGN(0x1000);
}

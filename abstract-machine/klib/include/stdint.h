#ifndef __STDINT_H__
#define __STDINT_H__

// GCC's stdint.h uses include_next which fails in freestanding environment
// without libc. Use stdint-gcc.h which is the freestanding implementation.
#include <stdint-gcc.h>

#endif

#include <am.h>
#include <klib-macros.h>
#include <klib.h>
#define __ITOA_KLIB__
#if !defined(__ISA_NATIVE__) || defined(__NATIVE_USE_KLIB__)
static unsigned long int next = 1;

extern char _pmem_start;
#define PMEM_SIZE (128 * 1024 * 1024)
#define PMEM_END ((uintptr_t)&_pmem_start + PMEM_SIZE)

int rand(void) {
  // RAND_MAX assumed to be 32767
  next = next * 1103515245 + 12345;
  return (unsigned int)(next / 65536) % 32768;
}

void srand(unsigned int seed) { next = seed; }

int abs(int x) { return (x < 0 ? -x : x); }

#ifdef __ITOA_KLIB__

char *itoa_klib(int val, char *buf, int radix, bool sign) {
  char *output = buf;
  char *start = buf;
  uint32_t uval = (uint32_t)val;
  if (sign && val < 0) {
    *buf++ = '-';
    val = -val;
  }
  if (sign) {
    do {
      int digit = val % radix;
      val /= radix;

      if (digit > 9) {
        *buf++ = (char)(digit - 10 + 'a');
      } else {
        *buf++ = (char)(digit + '0');
      }

    } while (val > 0);
    *buf-- = '\0';
    while (start < buf) {
      char temp = *buf;
      *buf-- = *start;
      *start++ = temp;
    }
  } else {
    do {
      int digit = uval % radix;
      uval /= radix;

      if (digit > 9) {
        *buf++ = (char)(digit - 10 + 'a');
      } else {
        *buf++ = (char)(digit + '0');
      }

    } while (uval > 0);
    *buf-- = '\0';
    while (start < buf) {
      char temp = *buf;
      *buf-- = *start;
      *start++ = temp;
    }
  }
  return output;
}

#endif

int atoi(const char *nptr) {
  int x = 0;
  while (*nptr == ' ') {
    nptr++;
  }
  while (*nptr >= '0' && *nptr <= '9') {
    x = x * 10 + *nptr - '0';
    nptr++;
  }
  return x;
}

static char *memptr = 0;
extern char _heap_start;
void *malloc(size_t size) {
  // On native, malloc() will be called during initializaion of C runtime.
  // Therefore do not call panic() here, else it will yield a dead recursion:
  //   panic() -> putchar() -> (glibc) -> malloc() -> panic()
#if !(defined(__ISA_NATIVE__) && defined(__NATIVE_USE_KLIB__))
  size = (size_t)ROUNDUP(size, 8);
  char *old = memptr;
  if (memptr == 0) {
    memptr = heap.start;
    old = memptr;
    // printf("memptr=%d\n", memptr);
  }
  memptr += size;
  assert((uintptr_t)heap.start <= (uintptr_t)memptr &&
         (uintptr_t)memptr < (uintptr_t)heap.end);
  for (uint64_t *p = (uint64_t *)old; p != (uint64_t *)memptr; p++) {
    *p = 0;
  }
  assert((uintptr_t)memptr - (uintptr_t)heap.start <= (uintptr_t)heap.end);
  return old;

  // if(size == 0){
  //   return NULL;
  // }
  // if(memptr == 0){
  //   memptr = _heap_start;
  //   printf("_heap_start=%d\n", _heap_start);
  // }
  // Area heap_t = RANGE(memptr, PMEM_END);
  // memptr = ((memptr+size) / 4 + 1)* 4;//int+1
  // return heap_t.start;
  // // panic("not implemented");
#else
  size = (size_t)ROUNDUP(size, 8);
  char *old = memptr;
  memptr += size;
  assert((uintptr_t)heap.start <= (uintptr_t)memptr &&
         (uintptr_t)memptr < (uintptr_t)heap.end);
  for (uint64_t *p = (uint64_t *)old; p != (uint64_t *)memptr; p++) {
    *p = 0;
  }
  assert((uintptr_t)memptr - (uintptr_t)heap.start <= (uintptr_t)heap.end);
  return old;
#endif
}

void free(void *ptr) {}

#endif
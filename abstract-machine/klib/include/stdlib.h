#ifndef STDLIB_H__
#define STDLIB_H__

#include <am.h>
#include <stddef.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

// #define __NATIVE_USE_KLIB__

// stdlib.h
void   srand     (unsigned int seed);
int    rand      (void);
void  *malloc    (size_t size);
void   free      (void *ptr);
int    abs       (int x);
int    atoi      (const char *nptr);
char*  itoa_klib (int val, char* buf, int radix, bool sign);

#ifdef __cplusplus
}
#endif

#endif

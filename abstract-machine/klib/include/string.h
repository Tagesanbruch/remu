#ifndef __STRING_H__
#define __STRING_H__

#include <klib.h>

char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
size_t strnlen(const char *s, size_t maxlen);

#endif

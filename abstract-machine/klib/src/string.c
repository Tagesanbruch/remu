#include <klib-macros.h>
#include <klib.h>
#include <stdint.h>

#if !defined(__ISA_NATIVE__) || defined(__NATIVE_USE_KLIB__)

size_t strlen(const char *s) {
  size_t i = 0;
  while (*(s + i) != '\0') {
    i++;
  }
  return i;
}

char *strcpy(char *dst, const char *src) {
  assert(dst != NULL && src != NULL);
  char *p = dst;
  while ((*dst++ = *src++) != '\0')
    ;
  return p;
}

char *strncpy(char *dst, const char *src, size_t n) {
  assert(dst != NULL && src != NULL && n >= 0);
  size_t i = 0;
  for (i = 0; i < n && *(src + i) != '\0'; i++) {
    *(dst + i) = *(src + i);
  }
  for (; i < n; i++) {
    *(dst + i) = '\0';
  }
  return dst;
}

char *strcat(char *dst, const char *src) {
  assert(dst != NULL && src != NULL);
  char *pdst = dst;
  size_t lend = strlen(dst);
  size_t lens = strlen(src);

  if ((dst < src) && (src < dst + lend)) {
    // Overlapping case: copy backwards
    for (size_t i = lens; i > 0; i--) {
      *(dst + lend + i) = *(src + i);
    }
    *(dst + lend) = *(src);
  } else {
    // Non-overlapping case: copy forwards
    for (size_t i = 0; i <= lens; i++) {
      *(dst + lend + i) = *(src + i);
    }
  }
  return pdst;
}

int strcmp(const char *s1, const char *s2) {
  assert(s1 != NULL && s2 != NULL);
  size_t i = 0;
  while (*(s1 + i) != '\0') {
    if (*(s1 + i) > *(s2 + i)) {
      return 1;
    } else if (*(s1 + i) < *(s2 + i)) {
      return -1;
    }
    i++;
  }
  if (*(s1 + i) < *(s2 + i)) {
    return -1;
  }
  return 0;
}

int strncmp(const char *s1, const char *s2, size_t n) {
  assert(s1 != NULL && s2 != NULL && n >= 0);
  size_t i = 0;
  signed char res = 0;
  while (n--) {
    if ((res = *(s1 + i) - *(s2 + i)) != 0 || !*(s1 + i)) {
      break;
    }
    i++;
  }
  return res;
}

void *memset(void *s, int c, size_t n) {
  assert(s != NULL);
  assert(n >= 0);
  char *ps = (char *)s;
  for (size_t i = 0; i < n; i++) {
    *(ps + i) = c;
  }
  return s;
}

void *memmove(void *dst, const void *src, size_t n) {
  assert(dst != NULL && src != NULL && n >= 0);
  char *pdst = (char *)dst;
  char *psrc = (char *)src;
  if (pdst > psrc && pdst < psrc + n) {
    for (size_t i = n; i > 0; i--) {
      *(pdst + i - 1) = *(psrc + i - 1);
    }
  } else {
    for (size_t i = 0; i < n; i++) {
      *(pdst + i) = *(psrc + i);
    }
  }
  return dst;
}

void *memcpy(void *out, const void *in, size_t n) {
  assert(in != NULL);
  assert(n >= 0);
  char *pout = (char *)out;
  char *pin = (char *)in;
  // print log info before assert
  // printf("memcpy: pout=0x%u, pin=0x%u, n=%d\n", pout, pin, n);
  assert(!((pout > pin && pout < pin + n) || (pin > pout && pin < pout + n)));
  for (size_t i = 0; i < n; i++) {
    *(pout + i) = *(pin + i);
  }
  return out;
}

int memcmp(const void *s1, const void *s2, size_t n) {
  assert(s1 != NULL && s2 != NULL && n >= 0);
  char *ps1 = (char *)s1;
  char *ps2 = (char *)s2;
  for (size_t i = 0; i < n; i++) {
    if (*(ps1 + i) > *(ps2 + i)) {
      return 1;
    } else if (*(ps1 + i) < *(ps2 + i)) {
      return -1;
    }
  }
  return 0;
}

#endif
#include <am.h>
#include <klib-macros.h>
#include <klib.h>
#include <stdarg.h>

#if !defined(__ISA_NATIVE__) || defined(__NATIVE_USE_KLIB__)

int printf(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int len = 0;
  int d;
  char c;
  char *s;
  char strd[256];
  while (*fmt) {
    switch (*fmt) {
    case '%':
      if (*(fmt + 1) == 's') {
        s = va_arg(ap, char *);
        putstr(s);
        len += strlen(s);
        fmt++;
      } else if (*(fmt + 1) == 'd') {
        d = va_arg(ap, int);
        itoa_klib(d, strd, 10, true);
        int bts = strlen(strd);
        putstr(strd);
        len += bts;
        fmt++;
      } else if (*(fmt + 1) == 'l' && *(fmt + 2) == 'd') {
        d = va_arg(ap, long);
        itoa_klib(d, strd, 10, true);
        int bts = strlen(strd);
        putstr(strd);
        len += bts;
        fmt += 2;
      } else if (*(fmt + 1) == 'x') {
        d = va_arg(ap, int);
        itoa_klib(d, strd, 16, true);
        int bts = strlen(strd);
        putstr(strd);
        len += bts;
        fmt++;
      } else if (*(fmt + 1) == 'u') { // ux
        d = va_arg(ap, uint32_t);
        itoa_klib(d, strd, 16, false);
        int bts = strlen(strd);
        putstr(strd);
        len += bts;
        fmt++;
      } else if (*(fmt + 1) == 'c') {
        c = va_arg(ap, int);
        putch(c);
        len += 1;
        fmt++;
      } else {
        putch('%');
        len++;
      }
      break;
    default:
      putch(*fmt);
      len++;
      break;
    }
    fmt++;
  }
  if (d)
    ;
  va_end(ap);
  // putch('\0');
  return len;
}

int vsprintf(char *out, const char *fmt, va_list ap) {
  int len = 0;
  int d;
  char strd[256];
  while (*fmt) {
    switch (*fmt) {
    case '%':
      if (*(fmt + 1) == 's') {
        char *s = va_arg(ap, char *);
        strcpy(out + len, s);
        len += strlen(s);
        fmt++;
      } else if (*(fmt + 1) == 'd') {
        d = va_arg(ap, int);
        itoa_klib(d, strd, 10, true);
        int bts = strlen(strd);
        strcpy(out + len, strd);
        len += bts;
        fmt++;
      } else {
        *(out + len) = '%';
        len++;
      }
      break;
    default:
      *(out + len) = *fmt;
      len++;
      break;
    }
    fmt++;
  }
  *(out + len) = '\0';
  return len;
}

int sprintf(char *out, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int len = 0;
  int d;
  char strd[256];
  while (*fmt) {
    switch (*fmt) {
    case '%':
      if (*(fmt + 1) == 's') {
        char *s = va_arg(ap, char *);
        strcpy(out + len, s);
        len += strlen(s);
        fmt++;
      } else if (*(fmt + 1) == 'd') {
        d = va_arg(ap, int);
        itoa_klib(d, strd, 10, true);
        int bts = strlen(strd);
        strcpy(out + len, strd);
        len += bts;
        fmt++;
      } else {
        *(out + len) = '%';
        len++;
      }
      break;
    default:
      *(out + len) = *fmt;
      len++;
      break;
    }
    fmt++;
  }
  va_end(ap);
  *(out + len) = '\0';
  return len;
}

int snprintf(char *out, size_t n, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  int len = vsnprintf(out, n, fmt, ap);
  va_end(ap);
  return len;
}

int vsnprintf(char *out, size_t n, const char *fmt, va_list ap) {
  int len = 0;
  int d;
  char strd[256];
  while (*fmt && len < n) {
    switch (*fmt) {
    case '%':
      if (*(fmt + 1) == 's') {
        char *s = va_arg(ap, char *);
        if (len + strlen(s) >= n)
          break;
        strcpy(out + len, s);
        len += strlen(s);
        fmt++;
      } else if (*(fmt + 1) == 'd') {
        d = va_arg(ap, int);
        itoa_klib(d, strd, 10, true);
        int bts = strlen(strd);
        if (len + bts >= n)
          break;
        strcpy(out + len, strd);
        len += bts;
        fmt++;
      } else {
        if (len + 1 >= n)
          break;
        *(out + len) = '%';
        len++;
      }
      break;
    default:
      if (len + 1 >= n)
        break;
      *(out + len) = *fmt;
      len++;
      break;
    }
    fmt++;
  }
  if (len < n)
    *(out + len) = '\0';
  else
    *(out + n - 1) = '\0';
  return len;
}

#endif
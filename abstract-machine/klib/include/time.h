#ifndef __TIME_H__
#define __TIME_H__

#include <sys/types.h>

struct timespec {
  time_t tv_sec;
  long tv_nsec;
};

struct tm {
  int tm_sec;
  int tm_min;
  int tm_hour;
  int tm_mday;
  int tm_mon;
  int tm_year;
  int tm_wday;
  int tm_yday;
  int tm_isdst;
};

typedef long clock_t;

time_t time(time_t *t);
struct tm *gmtime_r(const time_t *timep, struct tm *result);

#endif

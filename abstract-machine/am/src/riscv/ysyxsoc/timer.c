#include <am.h>
#include <ysyxsoc.h>
void __am_timer_init() {
  outl(RTC_ADDR, 0);
  outl(RTC_ADDR + 4, 0);
  // printf("am_timer_init;\n");
}

void __am_timer_uptime(AM_TIMER_UPTIME_T *uptime) {
  uint32_t t1 = 0, t2 = 0;
  //must read high 4bits first!(rtc_io_handler:)
  t2 = inl(RTC_ADDR + 4);
  t1 = inl(RTC_ADDR);
  uptime->us = ((uint64_t)t1) + ((uint64_t)t2 << 32);
  // printf("t1=%d,t2=%d\n", t1, t2);
}
void __am_timer_rtc(AM_TIMER_RTC_T *rtc) {
  rtc->second = 0;
  rtc->minute = 0;
  rtc->hour   = 0;
  rtc->day    = 0;
  rtc->month  = 0;
  rtc->year   = 1900;
}

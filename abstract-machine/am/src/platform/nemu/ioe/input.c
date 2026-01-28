#include <am.h>
#include <nemu.h>

#define KEYDOWN_MASK 0x8000

void __am_input_keybrd(AM_INPUT_KEYBRD_T *kbd) {
  kbd->keydown = 0;
  kbd->keycode = AM_KEY_NONE;
  uint32_t t1 = 0;
  t1 = inl(KBD_ADDR);
  if((t1&KEYDOWN_MASK) != 0){
    kbd->keydown = 1;
    kbd->keycode = t1 & ~KEYDOWN_MASK;
  }
}

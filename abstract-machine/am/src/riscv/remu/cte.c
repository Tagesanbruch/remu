/***************************************************************************************
 * Copyright (c) 2014-2022 Zihao Yu, Nanjing University
 *
 * NEMU is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 *PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 *KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 *NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
 ***************************************************************************************/

#include <am.h>
#include <klib.h>

static Context *(*user_handler)(Event, Context *) = NULL;

#define pdb(...) printf(__VA_ARGS__)

Context *__am_irq_handle(Context *c) {
  if (user_handler) {
    Event ev = {0};
    switch (c->mcause) {
    case 0x8:
    case 0x9:
    case 0xb:
      ev.event = (c->GPR1 == -1) ? EVENT_YIELD : EVENT_SYSCALL;
      c->mepc += 4;
      break;
    default:
      ev.event = EVENT_ERROR;
      break;
    }

    c = user_handler(ev, c);
    assert(c != NULL);
  }

  return c;
}

extern void __am_asm_trap(void);

bool cte_init(Context *(*handler)(Event, Context *)) {
  // initialize exception entry
  asm volatile("csrw mtvec, %0" : : "r"(__am_asm_trap));

  // register event handler
  user_handler = handler;

  return true;
}

Context *kcontext(Area kstack, void (*entry)(void *), void *arg) {
  Context *c = (Context *)kstack.end - 1;
  // Manually initialize all fields to avoid garbage values
  for (int i = 0; i < 32; i++) {
    c->gpr[i] = 0;
  }
  c->mcause = 0;
  c->mstatus = 0x1800;
  c->mepc = (uintptr_t)entry; // FIX: Don't subtract 4!
  c->pdir = NULL;

  // Set the required fields
  c->gpr[10] = (uintptr_t)arg;

  pdb("kcontext: c=%p, entry=%p, c->mepc=0x%x, c->mstatus=0x%x, "
      "c->gpr[10]=0x%x\n",
      c, entry, c->mepc, c->mstatus, c->gpr[10]);
  return c;
}

void yield() {
#ifdef __riscv_e
  asm volatile("li a5, -1; ecall");
#else
  asm volatile("li a7, -1; ecall");
#endif
}

bool ienabled(void) { return false; }

void iset(bool enable) {}

#ifndef __FLASH_H__
#define __FLASH_H__

#include <stdint.h>

#define FLASH_BASE  0x30000000
#define FLASH_SIZE  (16 * 1024 * 1024)  // 16MB

// Flash APIs
void flash_read(uint32_t offset, void *buf, uint32_t len);
uint32_t flash_read32(uint32_t offset);
void *flash_get_base(void);
uint32_t flash_get_size(void);

#endif // __FLASH_H__

/**
 * Flash Driver for REMU Platform
 * 
 * Memory Map:
 *   0x3000_0000 - 0x30FF_FFFF: Flash Storage (16MB)
 */

#include <am.h>
#include <klib.h>

#define FLASH_BASE  0x30000000
#define FLASH_SIZE  (16 * 1024 * 1024)  // 16MB

/**
 * Read from flash (memory-mapped)
 */
void flash_read(uint32_t offset, void *buf, uint32_t len) {
    uint8_t *src = (uint8_t *)(FLASH_BASE + offset);
    uint8_t *dst = (uint8_t *)buf;
    
    for (uint32_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

/**
 * Read 32-bit word from flash
 */
uint32_t flash_read32(uint32_t offset) {
    return *(volatile uint32_t *)(FLASH_BASE + offset);
}

/**
 * Get flash base address for direct memory-mapped access
 */
void *flash_get_base(void) {
    return (void *)FLASH_BASE;
}

/**
 * Get flash size
 */
uint32_t flash_get_size(void) {
    return FLASH_SIZE;
}

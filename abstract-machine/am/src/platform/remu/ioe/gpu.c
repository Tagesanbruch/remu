#include <am.h>
#include <remu.h>

#define SYNC_ADDR (VGACTL_ADDR + 4)
#define N 32
void __am_gpu_init() {
  int i;
  int w = io_read(AM_GPU_CONFIG).width;
  int h = io_read(AM_GPU_CONFIG).height;
  // uint32_t* fb = (uint32_t*)(uintptr_t)FB_ADDR;
  // for (i = 0; i < w * h; i++){
  //   fb[i] = i;
  // }
  outl(SYNC_ADDR, 1);
}

void __am_gpu_config(AM_GPU_CONFIG_T *cfg) {
  *cfg = (AM_GPU_CONFIG_T){.present = true,
                           .has_accel = false,
                           .width = (inl(VGACTL_ADDR) & 0xFFFF0000) >> 16,
                           .height = inl(VGACTL_ADDR) & 0x0000FFFF,
                           .vmemsz = cfg->height * cfg->width};
}

void __am_gpu_fbdraw(AM_GPU_FBDRAW_T *ctl) {
  if (ctl->pixels != NULL) {
    int x = ctl->x;
    int y = ctl->y;
    int w = ctl->w;
    int h = ctl->h;
    int W = io_read(AM_GPU_CONFIG).width;
    int H = io_read(AM_GPU_CONFIG).height;
    uint32_t *fb = (uint32_t *)(uintptr_t)FB_ADDR;

    // int *pixel = malloc(N*N);
    // pixel = memcpy(pixel, ctl->pixels);
    // for(int i = 0;i < N*N;i++){
    //   printf("pixel[i]=%d, ctl->pixels[i]=%d\n",pixel[i],
    //   (int*)ctl->pixels[i]);
    // }
    for (int j = 0; j < h; j++) {
      for (int i = 0; i < w; i++) {
        // printf("x=%d, y=%d, i=%d, j=%d, w=%d, h=%d, W=%d, i + j * w=%d,(x +
        // i) + (y+j) * W=%d, pixel[i + j * w]=%d\n", x, y, i, j, w, h, W, i + j
        // * w,(x + i) + (y+j) * W, pixel[i + j * w]); outl(FB_ADDR + (x + i) +
        // (y+j) * W, pixel[i + j * w]);
        int *pixel_ptr = (int *)ctl->pixels;
        fb[(x + i) + (y + j) * W] = pixel_ptr[i + j * w];
        outl(SYNC_ADDR, 1);
        // ;
      }
    }
    // outl(FB_ADDR + x * w + y, pixel);
  }

  if (ctl->sync) {
    outl(SYNC_ADDR, 1);
  }
}

void __am_gpu_status(AM_GPU_STATUS_T *status) { status->ready = true; }

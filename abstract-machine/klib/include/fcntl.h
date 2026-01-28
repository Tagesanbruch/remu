#ifndef __FCNTL_H__
#define __FCNTL_H__

#ifdef __cplusplus
extern "C" {
#endif

#define O_RDONLY 0
#define O_WRONLY 1
#define O_RDWR 2

#define O_CREAT 0x0200
#define O_APPEND 0x0400
#define O_TRUNC 0x01000
#define O_EXCL 0x0800
#define O_NONBLOCK 0x4000
#define O_DIRECTORY 0x200000
#define O_ACCMODE 0x0003

#define F_GETLK 5
#define F_SETLK 6
#define F_SETLKW 7
#define F_GETFL 3
#define F_SETFL 4

#ifdef __cplusplus
}
#endif

#endif

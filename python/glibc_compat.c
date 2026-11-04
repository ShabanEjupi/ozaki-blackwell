/* glibc_compat.c -- keeps libozaki_blackwell.so loadable on glibc 2.34 (Ubuntu 22.04, Debian 12, RHEL 9, Colab) although it is
 * built on Ubuntu 24.04 (glibc 2.39). Built there, atoi/strtol in our headers and in the static libstdc++ become references to
 * __isoc23_strtol@GLIBC_2.38, and std::random_device to arc4random@GLIBC_2.36 (objdump -T, 4 November 2026). These hidden
 * definitions satisfy those references inside the library with functions every glibc since 2.2.5 / 2.25 has; the wheel tag
 * then follows from the highest version left (GLIBC_2.34: pthread and dl, merged into libc that year). */
#define _GNU_SOURCE
#include <stdint.h>
#include <sys/random.h>

__asm__(".symver ozb_strtol_old, strtol@GLIBC_2.2.5");
__asm__(".symver ozb_strtoul_old, strtoul@GLIBC_2.2.5");
__asm__(".symver ozb_strtoll_old, strtoll@GLIBC_2.2.5");
__asm__(".symver ozb_strtoull_old, strtoull@GLIBC_2.2.5");
long ozb_strtol_old(const char *, char **, int);
unsigned long ozb_strtoul_old(const char *, char **, int);
long long ozb_strtoll_old(const char *, char **, int);
unsigned long long ozb_strtoull_old(const char *, char **, int);

#define HIDDEN __attribute__((visibility("hidden")))
/* C23 accepts 0b prefixes in base 0/2; the old functions do not -- nothing in this library parses binary literals */
HIDDEN long __isoc23_strtol(const char *s, char **e, int b) { return ozb_strtol_old(s, e, b); }
HIDDEN unsigned long __isoc23_strtoul(const char *s, char **e, int b) { return ozb_strtoul_old(s, e, b); }
HIDDEN long long __isoc23_strtoll(const char *s, char **e, int b) { return ozb_strtoll_old(s, e, b); }
HIDDEN unsigned long long __isoc23_strtoull(const char *s, char **e, int b) { return ozb_strtoull_old(s, e, b); }
HIDDEN uint32_t arc4random(void) {
    uint32_t v = 0;
    while (getrandom(&v, sizeof v, 0) != (long)sizeof v) {}
    return v;
}

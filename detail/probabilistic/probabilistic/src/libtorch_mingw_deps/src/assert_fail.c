#ifdef __MINGW32__

#include <assert.h>
#include <assert_fail.h>

void __assert_file(
    const char *assertion,
    const char *file,
    unsigned int line,
    const char *function
) {
    _assert(assertion, file, line);
}

#endif

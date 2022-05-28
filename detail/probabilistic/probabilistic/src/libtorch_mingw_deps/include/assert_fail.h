#if defined(__MINGW32__) && !defined(PROBABILISTIC_ASSERT_FAIL_INCLUDE_GUARD)
#define PROBABILISTIC_ASSERT_FAIL_INCLUDE_GUARD

#ifdef __cplusplus
extern "C" {
#endif

void __assert_fail(
    const char *assertion,
    const char *file,
    unsigned int line,
    const char *function
);

#ifdef __cplusplus
}
#endif

#endif

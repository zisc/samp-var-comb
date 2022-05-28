#ifndef VISIBILITY_H_GUARD
#define VISIBILITY_H_GUARD

// Define a macro for visibility attribution to functions
// accessed by R. See here:
// https://gcc.gnu.org/wiki/Visibility

#if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
#else
    #define DLL_PUBLIC
#endif

#endif


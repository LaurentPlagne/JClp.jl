#ifndef JCLP_BRIDGE_H
#define JCLP_BRIDGE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Returns 1 if JCLP_BRICK_<brick_name> (or JCLP_BRICKS_ALL) is set to "1", else 0. */
int jclp_brick_enabled_raw(const char *brick_name);

/* Lazily dlopen()s libjclp.dylib and resolves `sym_name`. Aborts with diagnostic on failure. */
void *jclp_resolve_symbol(const char *sym_name);

/* Returns 1 if libjclp.dylib has been loaded in this process. */
int jclp_bridge_is_loaded(void);

#ifdef __cplusplus
}
#endif

/* Zero-overhead cached check for hot C/C++ paths (1 static int load after first call). */
#define JCLP_BRICK_ENABLED(NAME_STR)                                     \
    ([]() -> bool {                                                      \
        static int cached = -1;                                          \
        if (__builtin_expect(cached < 0, 0)) {                           \
            cached = jclp_brick_enabled_raw(NAME_STR);                   \
        }                                                                \
        return cached != 0;                                              \
    }())

/* Zero-overhead cached symbol resolver for hot C++ call sites. */
#define JCLP_RESOLVE_FN(FN_TYPE, SYM_NAME)                               \
    ([]() -> FN_TYPE {                                                   \
        static FN_TYPE fn = nullptr;                                     \
        if (__builtin_expect(fn == nullptr, 0)) {                        \
            fn = reinterpret_cast<FN_TYPE>(jclp_resolve_symbol(SYM_NAME)); \
        }                                                                \
        return fn;                                                       \
    }())

#endif /* JCLP_BRIDGE_H */

#include "jclp_bridge.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <string>

#ifdef __APPLE__
#include <crt_externs.h>
#else
extern char **environ;
#endif

namespace {

constexpr const char *kDefaultJClpLibPath =
    "/Users/laurentplagne/Projects/JClp.jl/bridge/build/lib/libjclp.dylib";

std::once_flag g_load_once;
void *g_lib_handle = nullptr;

void load_jclp_library_once() {
    const char *env_path = std::getenv("JCLP_LIB_PATH");
    const char *path = (env_path && env_path[0] != '\0') ? env_path : kDefaultJClpLibPath;
    g_lib_handle = dlopen(path, RTLD_NOW | RTLD_GLOBAL);
    if (!g_lib_handle) {
        std::fprintf(stderr, "[JClp Bridge] FATAL: dlopen('%s') failed: %s\n",
                     path, dlerror());
        std::abort();
    }
    if (std::getenv("JCLP_BRIDGE_VERBOSE")) {
        std::fprintf(stderr, "[JClp Bridge] Loaded %s\n", path);
    }
}

bool env_flag_is_on(const char *name) {
    const char *val = std::getenv(name);
    return val != nullptr && val[0] == '1' && val[1] == '\0';
}

bool any_brick_enabled_in_env() {
    if (env_flag_is_on("JCLP_BRICKS_ALL")) {
        return true;
    }
#ifdef __APPLE__
    char **envp = *_NSGetEnviron();
#else
    char **envp = environ;
#endif
    if (!envp) return false;
    const char *prefix = "JCLP_BRICK_";
    const size_t plen = std::strlen(prefix);
    for (char **p = envp; *p != nullptr; ++p) {
        if (std::strncmp(*p, prefix, plen) == 0) {
            const char *eq = std::strchr(*p + plen, '=');
            if (eq && eq[1] == '1' && eq[2] == '\0') {
                return true;
            }
        }
    }
    return false;
}

/* Preload libjclp.dylib before main() when any JCLP_BRICK_*=1 is active,
   so JuliaC runtime initialization does not pollute Clp's internal solve timer. */
__attribute__((constructor))
static void jclp_bridge_preload_ctor() {
    if (any_brick_enabled_in_env()) {
        std::call_once(g_load_once, load_jclp_library_once);
    }
}

} // namespace

extern "C" {

int jclp_brick_enabled_raw(const char *brick_name) {
    if (env_flag_is_on("JCLP_BRICKS_ALL")) {
        return 1;
    }
    std::string key("JCLP_BRICK_");
    key += brick_name;
    return env_flag_is_on(key.c_str()) ? 1 : 0;
}

void *jclp_resolve_symbol(const char *sym_name) {
    std::call_once(g_load_once, load_jclp_library_once);
    void *sym = dlsym(g_lib_handle, sym_name);
    if (!sym) {
        std::fprintf(stderr, "[JClp Bridge] FATAL: dlsym('%s') failed: %s\n",
                     sym_name, dlerror());
        std::abort();
    }
    return sym;
}

int jclp_bridge_is_loaded(void) {
    return g_lib_handle != nullptr ? 1 : 0;
}

} // extern "C"

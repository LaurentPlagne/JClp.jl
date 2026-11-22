#include "jclp_bridge.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using version_fn_t = int32_t (*)();
using add_fn_t = double (*)(double, double);
using dot_fn_t = double (*)(int32_t, const double *, const double *);

int main() {
    if (!JCLP_BRICK_ENABLED("B0_SMOKE")) {
        std::printf("[test_bridge] B0_SMOKE disabled, loaded=%d\n", jclp_bridge_is_loaded());
        return 0;
    }

    auto t0 = std::chrono::steady_clock::now();
    auto fn_ver = JCLP_RESOLVE_FN(version_fn_t, "jclp_bridge_version");
    auto fn_add = JCLP_RESOLVE_FN(add_fn_t, "jclp_smoke_add");
    auto fn_dot = JCLP_RESOLVE_FN(dot_fn_t, "jclp_smoke_dot");
    auto t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    int32_t ver = fn_ver();
    double sum = fn_add(1.5, 2.25);

    std::vector<double> x = {1.0, -2.0, 3.5, 4.0};
    std::vector<double> y = {2.0, 3.0, -4.0, 0.5};
    double d = fn_dot(static_cast<int32_t>(x.size()), x.data(), y.data());

    // Measure call overhead over 5,000,000 calls
    const int N = 5000000;
    volatile double sink = 0.0;
    auto b0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; ++i) {
        sink = fn_add(sink, 1.0);
    }
    auto b1 = std::chrono::steady_clock::now();
    double ns_per_call = std::chrono::duration<double, std::nano>(b1 - b0).count() / N;

    std::printf("[test_bridge] dlopen+init: %.2f ms | version=%d | add=%.4f | dot=%.4f | call_overhead=%.2f ns/call\n",
                load_ms, ver, sum, d, ns_per_call);

    if (ver != 1 || std::fabs(sum - 3.75) > 0.0 || std::fabs(d - (-16.0)) > 0.0) {
        std::fprintf(stderr, "[test_bridge] FAIL: unexpected result!\n");
        return 1;
    }
    return 0;
}

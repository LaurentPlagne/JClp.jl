#include "ClpSimplex.hpp"
#include "ClpSolve.hpp"
#include "CoinMessageHandler.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char *status_str(int s) {
    switch (s) {
    case 0: return "OPTIMAL";
    case 1: return "PRIMAL_INFEASIBLE";
    case 2: return "DUAL_INFEASIBLE";
    case 3: return "STOPPED_LIMIT";
    case 4: return "STOPPED_ERRORS";
    default: return "UNKNOWN";
    }
}

int main(int argc, char **argv) {
    // Usage: parity_runner [--mode=initial|dual|primal|initial_nopresolve] <mps_file1> [<mps_file2> ...]
    std::string mode = "initial";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--mode=", 7) == 0) {
            mode = argv[i] + 7;
        } else {
            files.emplace_back(argv[i]);
        }
    }
    if (files.empty()) {
        std::fprintf(stderr, "Usage: parity_runner [--mode=initial|dual|primal|initial_nopresolve] <mps>...\n");
        return 2;
    }

    std::printf("model\tstatus\titers\tobjective\tread_ms\tsolve_ms\n");
    for (const auto &path : files) {
        ClpSimplex model;
        model.setLogLevel(0);

        auto tr0 = std::chrono::steady_clock::now();
        int rc = model.readMps(path.c_str(), false, false);
        auto tr1 = std::chrono::steady_clock::now();
        double read_ms = std::chrono::duration<double, std::milli>(tr1 - tr0).count();

        // Extract basename without directory
        size_t slash = path.find_last_of('/');
        std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);

        if (rc != 0) {
            std::printf("%s\tREAD_ERROR(%d)\t-1\t0.0\t%.3f\t0.0\n", name.c_str(), rc, read_ms);
            std::fflush(stdout);
            continue;
        }

        auto ts0 = std::chrono::steady_clock::now();
        if (mode == "initial") {
            model.initialSolve();
        } else if (mode == "initial_nopresolve") {
            ClpSolve opts;
            opts.setPresolveType(ClpSolve::presolveOff);
            model.initialSolve(opts);
        } else if (mode == "dual") {
            model.dual();
        } else if (mode == "primal") {
            model.primal();
        } else {
            std::fprintf(stderr, "Unknown mode: %s\n", mode.c_str());
            return 2;
        }
        auto ts1 = std::chrono::steady_clock::now();
        double solve_ms = std::chrono::duration<double, std::milli>(ts1 - ts0).count();

        std::printf("%s\t%s\t%d\t%.17g\t%.3f\t%.3f\n",
                    name.c_str(),
                    status_str(model.status()),
                    model.numberIterations(),
                    model.objectiveValue(),
                    read_ms,
                    solve_ms);
        std::fflush(stdout);
    }
    return 0;
}

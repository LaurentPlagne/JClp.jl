#!/usr/bin/env julia
# Standalone Pure-Julia (JClp.jl) vs Clean C++ Clp Benchmark (B12c-3).
# Zero C++ Clp calls, zero FFI in JClp.jl!
# Compares pure solve time (excluding MPS I/O), status, iterations, and objective
# against `parity_runner_clean` (C++ Clp -O3 -DNDEBUG, presolve off) on Sample + Netlib.

using JClp
using Printf
using Statistics

const CLP_ROOT = "/Users/laurentplagne/Projects/Clp"
const JCLP_ROOT = "/Users/laurentplagne/Projects/JClp.jl"
const SAMPLE_DIR = joinpath(CLP_ROOT, "parity/data/Sample")
const NETLIB_DIR = joinpath(CLP_ROOT, "parity/data/Netlib_mps")
const RUNNER_CLEAN = joinpath(JCLP_ROOT, "tools/parity/parity_runner_clean")

const BENCH_MODELS = [
    joinpath(SAMPLE_DIR, "afiro.mps"),
    joinpath(SAMPLE_DIR, "brandy.mps"),
    joinpath(SAMPLE_DIR, "e226.mps"),
    joinpath(SAMPLE_DIR, "finnis.mps"),
    joinpath(SAMPLE_DIR, "lseu.mps"),
    joinpath(SAMPLE_DIR, "p0201.mps"),
    joinpath(SAMPLE_DIR, "p0548.mps"),
    joinpath(NETLIB_DIR, "adlittle.mps"),
    joinpath(NETLIB_DIR, "blend.mps"),
    joinpath(NETLIB_DIR, "sc50a.mps"),
    joinpath(NETLIB_DIR, "sc50b.mps"),
    joinpath(NETLIB_DIR, "sc105.mps"),
    joinpath(NETLIB_DIR, "sc205.mps"),
    joinpath(NETLIB_DIR, "share2b.mps"),
    joinpath(NETLIB_DIR, "stocfor1.mps"),
    joinpath(NETLIB_DIR, "israel.mps"),
    joinpath(NETLIB_DIR, "lotfi.mps"),
    joinpath(NETLIB_DIR, "scagr7.mps"),
    joinpath(NETLIB_DIR, "scsd1.mps"),
    joinpath(NETLIB_DIR, "sctap1.mps"),
    joinpath(NETLIB_DIR, "grow15.mps"),
    joinpath(NETLIB_DIR, "stocfor2.mps"),
    joinpath(NETLIB_DIR, "25fv47.mps"),
    joinpath(NETLIB_DIR, "bnl1.mps"),
    joinpath(NETLIB_DIR, "degen2.mps"),
    joinpath(NETLIB_DIR, "fit1p.mps"),
    joinpath(NETLIB_DIR, "scfxm2.mps"),
    joinpath(NETLIB_DIR, "ship08l.mps"),
    joinpath(NETLIB_DIR, "stair.mps"),
    joinpath(NETLIB_DIR, "wood1p.mps"),
]

const SAMPLE_LINEAR_MODELS = [
    "afiro.mps", "app0110.mps", "app0110R.mps", "brandy.mps", "e226.mps",
    "exmip1.5.mps", "exmip1.mps", "finnis.mps", "galenet.mps", "galenetbnds.mps",
    "hello.mps", "input.130.mps", "lseu.mps", "nw460.mps", "p0033.mps",
    "p0201.mps", "p0548.mps", "pack1.mps", "retail3.mps", "scOneInt.mps",
    "share2qp.mps", "tp3.mps", "tp4.mps", "tp5.mps",
]

function collect_all_models()::Vector{String}
    models = String[]
    for fn in SAMPLE_LINEAR_MODELS
        p = joinpath(SAMPLE_DIR, fn)
        isfile(p) && push!(models, p)
    end
    if isdir(NETLIB_DIR)
        for fn in sort(readdir(NETLIB_DIR))
            endswith(fn, ".mps") && push!(models, joinpath(NETLIB_DIR, fn))
        end
    end
    return models
end

function run_clean_batch(models::Vector{String}, mode::String)
    out = read(Cmd([RUNNER_CLEAN, "--mode=" * mode, models...]), String)
    rows = Dict{String,NamedTuple{(:status, :iters, :obj, :solve_ms),Tuple{String,Int,Float64,Float64}}}()
    for line in split(out, '\n')
        isempty(line) && continue
        startswith(lowercase(line), "model\t") && continue
        parts = split(line, '\t')
        length(parts) >= 6 || continue
        rows[String(parts[1])] = (
            status = String(parts[2]),
            iters = parse(Int, parts[3]),
            obj = parse(Float64, parts[4]),
            solve_ms = parse(Float64, parts[6]),
        )
    end
    return rows
end

function run_standalone_bench(mode::String = "dual"; all_models::Bool = false, max_iterations::Int = 200000)
    models = all_models ? collect_all_models() : filter(isfile, BENCH_MODELS)
    cpp = run_clean_batch(models, mode)

    # Warmup JIT on afiro
    let m = JClp.ClpSimplexModel()
        JClp.read_mps!(m, models[1])
        if mode == "initial"
            JClp.initial_solve!(m)
        elseif mode == "dual"
            JClp.dual!(m)
        else
            JClp.primal!(m)
        end
    end

    ps_label = mode == "initial" ? "presolve=on" : "presolve=off"
    println("\n=== Standalone Pure-Julia JClp.jl vs Clean C++ Clp (mode=$(mode), $(ps_label), n=$(length(models))) ===")
    @printf("%-16s | %-7s | %6s %6s | %8s %8s %6s | %8s | %s\n",
            "Model", "Status", "ItCpp", "ItJClp", "Cpp(ms)", "JClp(ms)", "Ratio", "ms/itR", "ObjCheck")
    println("-"^97)

    n_ok = 0
    n_tot = 0
    fails = String[]
    ratios_total = Float64[]
    ratios_per_iter = Float64[]

    for p in models
        fn = basename(p)
        haskey(cpp, fn) || continue
        c = cpp[fn]
        n_tot += 1

        m = JClp.ClpSimplexModel()
        JClp.read_mps!(m, p)
        t0 = time_ns()
        st = if mode == "initial"
            JClp.initial_solve!(m; max_iterations=max_iterations)
        elseif mode == "dual"
            JClp.dual!(m; max_iterations=max_iterations)
        else
            JClp.primal!(m; max_iterations=max_iterations)
        end
        j_ms = (time_ns() - t0) / 1e6

        st_str = st == 0 ? "OPTIMAL" : (st == 1 ? "PRIMAL_INFEASIBLE" : (st == 2 ? "DUAL_INFEASIBLE" : (st == 3 ? "MAXIT" : "ERR($st)")))
        obj_err = abs(m.objective_value - c.obj) / max(1.0, abs(c.obj))
        ok = (st_str == c.status) && (st != 0 || obj_err < 1e-4)
        if ok
            n_ok += 1
        else
            push!(fails, "$(fn): cpp=$(c.status)(it=$(c.iters),obj=$(c.obj)) vs jclp=$(st_str)(it=$(m.number_iterations),obj=$(m.objective_value))")
        end
        ratio = j_ms / max(c.solve_ms, 0.01)
        per_it_r = (m.number_iterations > 0 && c.iters > 0) ?
                   ((j_ms / m.number_iterations) / (c.solve_ms / c.iters)) : ratio
        if ok && c.solve_ms >= 5.0
            push!(ratios_total, ratio)
            push!(ratios_per_iter, per_it_r)
        end
        if !all_models || !ok || c.solve_ms >= 5.0
            obj_tag = ok ? "OK" : @sprintf("DIFF(%.2e)", obj_err)
            @printf("%-16s | %-7s | %6d %6d | %8.2f %8.2f %5.2fx | %7.2fx | %s\n",
                    fn, st_str, c.iters, m.number_iterations, c.solve_ms, j_ms, ratio, per_it_r, obj_tag)
            flush(stdout)
        end
    end
    println("-"^97)
    gm_tot = isempty(ratios_total) ? NaN : exp(mean(log.(ratios_total)))
    gm_pit = isempty(ratios_per_iter) ? NaN : exp(mean(log.(ratios_per_iter)))
    @printf("Summary (%s): %d/%d OK | >=5ms GeoMean: Total=%.3fx | Per-Iter=%.3fx (%d models)\n",
            mode, n_ok, n_tot, gm_tot, gm_pit, length(ratios_total))
    if !isempty(fails)
        println("Failures ($(length(fails))):")
        for f in fails
            println("  - ", f)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    let all_flag = "--all" in ARGS, mode_arg = "both"
        for a in ARGS
            if startswith(a, "--mode=")
                mode_arg = split(a, "=", limit=2)[2]
            end
        end
        if mode_arg == "initial"
            run_standalone_bench("initial"; all_models=all_flag)
        else
            if mode_arg == "dual" || mode_arg == "both"
                run_standalone_bench("dual"; all_models=all_flag)
            end
            if mode_arg == "primal" || mode_arg == "both"
                run_standalone_bench("primal"; all_models=all_flag)
            end
        end
    end
end

#!/usr/bin/env julia
# Parity & Performance Verification Harness for JClp.jl incremental bricks.
# Compares pure C++ baseline (`clp_clean`) vs JuliaC-bridged C++ (`clp_jclp`)
# on:
#   1. Clp's full `unitTest` (`-dirSample ... -unitTest`)
#   2. 23 linear Sample models
#   3. 12 critical large Netlib models (or all 91 with `all_netlib=true`)

using Printf
using Statistics

const CLP_ROOT = "/Users/laurentplagne/Projects/Clp"
const JCLP_ROOT = "/Users/laurentplagne/Projects/JClp.jl"
const SAMPLE_DIR = joinpath(CLP_ROOT, "parity/data/Sample")
const NETLIB_DIR = joinpath(CLP_ROOT, "parity/data/Netlib_mps")

const CLP_CLEAN_BIN = joinpath(CLP_ROOT, "parity/install/bin/clp")
const CLP_JCLP_BIN = joinpath(CLP_ROOT, "parity/install_jclp/bin/clp")
const RUNNER_CLEAN = joinpath(JCLP_ROOT, "tools/parity/parity_runner_clean")
const RUNNER_JCLP = joinpath(JCLP_ROOT, "tools/parity/parity_runner_jclp")
const BASELINE_TSV = joinpath(JCLP_ROOT, "tools/parity/baseline_clean.tsv")

const SAMPLE_LINEAR_MODELS = [
    "afiro.mps", "app0110.mps", "app0110R.mps", "brandy.mps", "e226.mps",
    "exmip1.5.mps", "exmip1.mps", "finnis.mps", "galenet.mps", "galenetbnds.mps",
    "hello.mps", "input.130.mps", "lseu.mps", "nw460.mps", "p0033.mps",
    "p0201.mps", "p0548.mps", "pack1.mps", "retail3.mps", "scOneInt.mps",
    "share2qp.mps", "tp3.mps", "tp4.mps", "tp5.mps",
]

const NETLIB_CRITICAL_12 = [
    "25fv47.mps", "greenbea.mps", "greenbeb.mps", "d2q06c.mps",
    "pilot.mps", "pilot87.mps", "maros-r7.mps", "dfl001.mps",
    "fit2p.mps", "stocfor2.mps", "finnis.mps", "grow15.mps",
]

struct SolveRow
    model::String
    status::String
    iters::Int
    objective::Float64
    read_ms::Float64
    solve_ms::Float64
end

function build_runners()
    src = joinpath(JCLP_ROOT, "tools/parity/parity_runner.cpp")
    for (outbin, prefix, extra_ld) in [
        (RUNNER_CLEAN, joinpath(CLP_ROOT, "parity/install"), String[]),
        (RUNNER_JCLP, joinpath(CLP_ROOT, "parity/install_jclp"),
         ["-L" * joinpath(JCLP_ROOT, "bridge/build/lib"), "-ljclp_bridge"]),
    ]
        inc = joinpath(prefix, "include/coin-or")
        lib = joinpath(prefix, "lib")
        cmd = `clang++ -O3 -DNDEBUG -std=c++17 -I$inc $src -L$lib -Wl,-rpath,$lib -lClp -lCoinUtils $extra_ld -o $outbin`
        run(cmd)
    end
end

function collect_models(; include_sample::Bool=true, all_netlib::Bool=false)
    paths = String[]
    if include_sample
        for f in SAMPLE_LINEAR_MODELS
            p = joinpath(SAMPLE_DIR, f)
            isfile(p) && push!(paths, p)
        end
    end
    if all_netlib
        for f in sort(readdir(NETLIB_DIR))
            endswith(f, ".mps") && push!(paths, joinpath(NETLIB_DIR, f))
        end
    else
        for f in NETLIB_CRITICAL_12
            p = joinpath(NETLIB_DIR, f)
            isfile(p) && push!(paths, p)
        end
    end
    return paths
end

function run_solver_batch(runner::String, models::Vector{String};
                          mode::String="initial",
                          env_overrides::Dict{String,String}=Dict{String,String}())
    env = copy(ENV)
    for (k, v) in env_overrides
        env[k] = v
    end
    cmd = Cmd(`$runner --mode=$mode $models`; env=env)
    out = read(cmd, String)
    rows = SolveRow[]
    for (idx, line) in enumerate(split(chomp(out), '\n'))
        idx == 1 && continue
        parts = split(line, '\t')
        length(parts) == 6 || continue
        push!(rows, SolveRow(
            String(parts[1]),
            String(parts[2]),
            parse(Int, parts[3]),
            parse(Float64, parts[4]),
            parse(Float64, parts[5]),
            parse(Float64, parts[6]),
        ))
    end
    return rows
end

function run_clp_unittest(clp_bin::String;
                          env_overrides::Dict{String,String}=Dict{String,String}())
    env = copy(ENV)
    for (k, v) in env_overrides
        env[k] = v
    end
    return mktempdir() do tmpdir
        cmd = Cmd(`$clp_bin -dirSample $SAMPLE_DIR -unitTest`; env=env, dir=tmpdir)
        io = IOBuffer()
        proc = run(pipeline(cmd; stdout=io, stderr=io); wait=true)
        out = String(take!(io))
        passed = (proc.exitcode == 0) && occursin("All tests completed successfully", out)
        return passed, out
    end
end

function save_rows_tsv(path::String, rows::Vector{SolveRow})
    open(path, "w") do io
        println(io, "model\tstatus\titers\tobjective\tread_ms\tsolve_ms")
        for r in rows
            @printf(io, "%s\t%s\t%d\t%.17g\t%.3f\t%.3f\n",
                    r.model, r.status, r.iters, r.objective, r.read_ms, r.solve_ms)
        end
    end
end

function load_rows_tsv(path::String)
    rows = SolveRow[]
    for (idx, line) in enumerate(eachline(path))
        idx == 1 && continue
        parts = split(line, '\t')
        length(parts) == 6 || continue
        push!(rows, SolveRow(
            String(parts[1]), String(parts[2]),
            parse(Int, parts[3]), parse(Float64, parts[4]),
            parse(Float64, parts[5]), parse(Float64, parts[6]),
        ))
    end
    return rows
end

function verify_parity(; bricks::Vector{String}=String[],
                       all_netlib::Bool=false,
                       mode::String="initial",
                       rebuild_runners::Bool=false,
                       save_baseline::Bool=false,
                       obj_rtol::Float64=1e-12)
    if rebuild_runners || !isfile(RUNNER_CLEAN) || !isfile(RUNNER_JCLP)
        println("[parity] Building parity_runner_clean & parity_runner_jclp...")
        build_runners()
    end

    env_ovr = Dict{String,String}()
    for b in bricks
        env_ovr["JCLP_BRICK_" * b] = "1"
    end

    println("[parity] 1/2 Running Clp -unitTest (bricks = ", isempty(bricks) ? "NONE" : join(bricks, ","), ")...")
    ut_ok, _ = run_clp_unittest(CLP_JCLP_BIN; env_overrides=env_ovr)
    println("  -> Clp -unitTest: ", ut_ok ? "PASSED" : "FAILED")

    models = collect_models(; include_sample=true, all_netlib=all_netlib)
    println("[parity] 2/2 Running ", length(models), " LP models (mode=", mode, ")...")

    base_rows = run_solver_batch(RUNNER_CLEAN, models; mode=mode)
    if save_baseline && !all_netlib && mode == "initial"
        save_rows_tsv(BASELINE_TSV, base_rows)
    end
    jclp_rows = run_solver_batch(RUNNER_JCLP, models; mode=mode, env_overrides=env_ovr)

    @assert length(base_rows) == length(jclp_rows)

    n_fail = 0
    ratios_large = Float64[]
    println()
    @printf("%-16s | %-10s | %8s %8s %5s | %10s %10s %6s | %s\n",
            "Model", "Status", "IterBase", "IterJClp", "dIter", "Base(ms)", "JClp(ms)", "Ratio", "ObjCheck")
    println("-"^95)

    for (b, j) in zip(base_rows, jclp_rows)
        status_ok = (b.status == j.status)
        iter_diff = j.iters - b.iters
        iter_ok = (iter_diff == 0)
        obj_err = abs(j.objective - b.objective) / max(1.0, abs(b.objective))
        obj_ok = obj_err <= obj_rtol
        ok = status_ok && iter_ok && obj_ok
        if !ok
            n_fail += 1
        end
        ratio = b.solve_ms > 0.05 ? (j.solve_ms / b.solve_ms) : 1.0
        if b.solve_ms >= 5.0
            push!(ratios_large, ratio)
        end
        flag = ok ? "OK" : @sprintf("FAIL(s=%d,i=%d,e=%.1e)", status_ok, iter_diff, obj_err)
        if !ok || b.solve_ms >= 5.0 || b.model in NETLIB_CRITICAL_12
            @printf("%-16s | %-10s | %8d %8d %+5d | %10.2f %10.2f %5.2fx | %s\n",
                    b.model, j.status[1:min(10, end)], b.iters, j.iters, iter_diff,
                    b.solve_ms, j.solve_ms, ratio, flag)
        end
    end

    gmean = isempty(ratios_large) ? 1.0 : exp(mean(log.(ratios_large)))
    println("-"^95)
    @printf("Summary: unitTest=%s | LP Parity: %d/%d EXACT | Large-case (>=5ms) GeoMean Ratio: %.3fx\n",
            ut_ok ? "PASS" : "FAIL", length(base_rows) - n_fail, length(base_rows), gmean)

    return (unittest_ok=ut_ok, total=length(base_rows), failures=n_fail, geomean_ratio=gmean)
end

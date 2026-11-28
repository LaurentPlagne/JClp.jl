using JClp
using Test

if !isdefined(@__MODULE__, :check_kkt_optimal)
    include("kkt_utils.jl")
end

@testset "Reoptimization of consecutive LPs" begin
    # max x, x <= b1, x <= b2. The active row alternates between A and B.
    function load_pair!(m)
        return JClp.load_problem!(m, 2, 1, [1, 3], [1, 2], [1.0, 1.0],
            [0.0], [20.0], [-1.0], [-Inf, -Inf], [4.0, 10.0])
    end
    for cold_presolve in (false, true), defer_warm_perturbation in (false, true)
        m = load_pair!(JClp.ClpSimplexModel())
        cache = JClp.BasisCache()
        solve(warm) = JClp.reoptimize!(m; warm_start=warm, cold_presolve,
                                       defer_warm_perturbation, basis_cache=cache)
        @test solve(false) == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ -4.0
        @test check_kkt_optimal(m).ok
        m.row_upper .= (8.0, 6.0)
        @test solve(true) == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ -6.0
        @test check_kkt_optimal(m).ok
        m.row_upper .= (4.0, 10.0)
        @test solve(true) == JClp.CLP_STATUS_OPTIMAL
        @test m.number_iterations == 0
        @test m.objective_value ≈ -4.0
        @test check_kkt_optimal(m).ok

        m.objective[1] = 1.0
        @test solve(true) == JClp.CLP_STATUS_OPTIMAL
        @test iszero(m.objective_value)
        @test length(cache.history) == 1
        for _ in 1:5
            @test solve(true) == JClp.CLP_STATUS_OPTIMAL
        end
        @test length(cache.history) == 4
        m.col_lower[1] = 25.0
        @test solve(false) == JClp.CLP_STATUS_PRIMAL_INFEAS
        @test isempty(cache.history)
        load_pair!(m)
        @test solve(false) == JClp.CLP_STATUS_OPTIMAL
        # Reload a different-sized problem on the same model and explicitly warm-start.
        JClp.load_problem!(m, 1, 1, [1, 2], [1], [2.0], [0.0], [20.0],
                           [-1.0], [-Inf], [6.0])
        @test solve(true) == JClp.CLP_STATUS_OPTIMAL
        @test m.objective_value ≈ -3.0
        @test check_kkt_optimal(m).ok
        @test length(cache.history) == 1
    end

    limited = load_pair!(JClp.ClpSimplexModel())
    limited.max_iterations = 1000
    @test JClp.reoptimize!(limited; warm_start=false, max_iterations=0) == JClp.CLP_STATUS_MAX_ITERATIONS
    @test limited.max_iterations == 1000
    @test JClp.reoptimize!(limited; warm_start=false) == JClp.CLP_STATUS_OPTIMAL

    # Deferring the automatic heuristic must preserve explicitly selected modes.
    for mode in (100, 102)
        m = load_pair!(JClp.ClpSimplexModel())
        @test JClp.dual!(m; perturbation=mode) == JClp.CLP_STATUS_OPTIMAL
        m.row_upper .= (8.0, 6.0)
        reference = deepcopy(m)
        @test JClp.dual!(reference; warm_start=true, perturbation=mode) == JClp.CLP_STATUS_OPTIMAL
        @test JClp.dual!(m; warm_start=true, perturbation=mode,
                        defer_warm_perturbation=true) == JClp.CLP_STATUS_OPTIMAL
        @test m.number_iterations == reference.number_iterations
        @test m.status == reference.status
        @test m.objective_value == reference.objective_value
    end
end

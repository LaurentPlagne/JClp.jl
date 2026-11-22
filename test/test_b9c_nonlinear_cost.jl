using JClp
using Test

@testset "B9c: ClpNonLinearCost (Pure 1-based Julia)" begin
    ntotal = 4
    nlc_status = fill(JClp.CLP_NLC_FEASIBLE | (JClp.CLP_NLC_SAME << 4), ntotal)
    bound = zeros(Float64, ntotal)
    cost2 = Float64[1.0, 2.0, 3.0, 4.0]
    cost = copy(cost2)
    lower = Float64[0.0, 0.0, 0.0, 0.0]
    upper = Float64[1.0, 1.0, 1.0, 1.0]
    sol   = Float64[-0.5, 0.5, 2.5, 0.0]
    pivot_var = Int32[1, 2, 3, 4]
    row_idx = Int32[1, 2, 3]
    upd_work = zeros(Float64, ntotal)
    mstatus = fill(JClp.CLP_STATUS_BASIC, ntotal)

    # check_changed_m2!: var 1 below lower, var 3 above upper
    n_chg, n_inf = JClp.clp_nlc_check_changed_m2!(
        nlc_status, bound, cost2, pivot_var, row_idx, upd_work, 3,
        sol, lower, upper, cost, 1e-7, 100.0, 0,
    )
    @test n_chg == 2
    @test n_inf == 2
    @test row_idx[1:2] == Int32[1, 3]
    @test upd_work[1] ≈ 100.0   # old 1.0 - new (-99.0)
    @test upd_work[3] ≈ -100.0  # old 3.0 - new (103.0)
    @test lower[1] == -floatmax(Float64) && upper[1] == 0.0 && bound[1] == 1.0
    @test lower[3] == 1.0 && upper[3] == floatmax(Float64) && bound[3] == 0.0

    # nearest_m2 on var 1 (stored below lower)
    @test JClp.clp_nlc_nearest_m2(nlc_status, bound, lower, upper, 1, -0.5) == 0.0
    @test JClp.clp_nlc_nearest_m2(nlc_status, bound, lower, upper, 1, 0.8) == 1.0

    # set_one_m2! brings var 1 back to feasible
    diff, n_inf2 = JClp.clp_nlc_set_one_m2!(
        nlc_status, bound, cost2, lower, upper, cost, mstatus,
        1, 0.5, 1e-7, 100.0, n_inf,
    )
    @test n_inf2 == 1
    @test diff ≈ -100.0
    @test lower[1] == 0.0 && upper[1] == 1.0 && cost[1] == 1.0

    # feasible_bounds! restores var 3
    JClp.clp_nlc_feasible_bounds!(nlc_status, bound, cost2, lower, upper, cost, ntotal)
    @test lower[3] == 0.0 && upper[3] == 1.0 && cost[3] == 3.0

    # refresh_one!
    JClp.clp_nlc_refresh_one!(nlc_status, bound, cost2, sol, lower, upper, cost, 3, 1e-7, 50.0)
    @test cost[3] ≈ 53.0
    @test lower[3] == 1.0 && upper[3] == floatmax(Float64)

    # Zero allocations
    @test (@allocated JClp.clp_nlc_feasible_bounds!(nlc_status, bound, cost2, lower, upper, cost, ntotal)) == 0
end

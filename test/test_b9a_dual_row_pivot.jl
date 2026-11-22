using JClp
using Test

@testset "B9a: ClpDualRowDantzig & ClpDualRowSteepest (Pure 1-based Julia)" begin
    @testset "ClpDualRowDantzig" begin
        nrows, ncols = 4, 3
        pivot_var = Int32[1, 2, 4, 5] # 1..3 columns, 4..5 slacks
        solution = Float64[-0.5, 2.0, 0.0, 1.8, 0.0]
        lower    = Float64[ 0.0, 0.0, 0.0, 0.0, 0.0]
        upper    = Float64[ 1.0, 1.0, 1.0, 1.0, 1.0]
        status   = zeros(UInt8, 5)
        # row 1 (col 1): infeas = 0.5 * 1.01 = 0.505
        # row 2 (col 2): infeas = 1.0 * 1.01 = 1.01  (BEST)
        # row 3 (slack 4): infeas = 0.8
        @test JClp.clp_drd_pivot_row(pivot_var, solution, lower, upper, status, nrows, ncols, 1e-7, 0.0) == 2

        # Flag col 2 -> row 3 (0.8) wins
        status[2] = JClp.CLP_STATUS_FLAGGED_BIT
        @test JClp.clp_drd_pivot_row(pivot_var, solution, lower, upper, status, nrows, ncols, 1e-7, 0.0) == 3

        # extract_alpha
        work = Float64[0.5, -2.5, 1.0, 0.0]
        which = Int32[1, 3, 4, 0]
        @test JClp.clp_drd_extract_alpha(work, which, 3, true, 3) == -2.5
        @test JClp.clp_drd_extract_alpha(work, which, 3, true, 2) == 0.0
        @test JClp.clp_drd_extract_alpha(work, which, 3, false, 2) == -2.5

        # update_primal_solution!
        sol2 = copy(solution)
        cost = Float64[2.0, 3.0, 1.0, 4.0, 5.0]
        w_p = Float64[1.0, -2.0, 0.0, 0.0]
        idx_p = Int32[1, 3, 0, 0]
        dobj = JClp.clp_drd_update_primal_solution!(w_p, idx_p, 2, true, pivot_var, sol2, cost, 0.5)
        @test sol2[1] ≈ solution[1] - 0.5 * 1.0
        @test sol2[4] ≈ solution[4] - 0.5 * (-2.0)
        @test dobj ≈ -(0.5 * 1.0 * 2.0) - (0.5 * (-2.0) * 4.0)
        @test w_p[1] == 0.0 && w_p[2] == 0.0
        @test (@allocated JClp.clp_drd_pivot_row(pivot_var, solution, lower, upper, status, nrows, ncols, 1e-7, 0.0)) == 0
    end

    @testset "ClpDualRowSteepest" begin
        nrows = 4
        pivot_var = Int32[1, 2, 3, 4]
        solution = Float64[-1.0, 3.0, 0.5, -2.0]
        lower    = Float64[ 0.0, 0.0, 0.0,  0.0]
        upper    = Float64[ 1.0, 1.0, 1.0,  1.0]
        status   = zeros(UInt8, 4)
        weights  = Float64[1.0, 2.0, 1.0, 10.0]

        infeas = zeros(Float64, nrows)
        infeas_idx = zeros(Int32, nrows)
        nnz = JClp.clp_drs_recompute_infeasibilities!(infeas, infeas_idx, pivot_var, solution, lower, upper, nrows, 1e-7)
        @test nnz == 3
        @test infeas ≈ Float64[1.0, 4.0, 0.0, 4.0]
        @test !JClp.clp_drs_looks_optimal(pivot_var, solution, lower, upper, nrows, 1e-7, 0.0)

        # row 1: 1/1 = 1.0, row 2: 4/2 = 2.0 (BEST), row 4: 4/10 = 0.4
        chosen, nnz2, tc = JClp.clp_drs_pivot_row!(
            infeas, infeas_idx, nnz, weights, pivot_var, solution, lower, upper, status,
            nrows, 0, 1, 10, 1e-7, 0.0, 0.0, 1, -1000, 0.0,
        )
        @test chosen == 2
        @test nnz2 == 3
        @test !tc

        # prep_ftran! & apply_weight_update! & unroll_weights!
        in_val = Float64[2.0, -1.0]
        in_idx = Int32[2, 4]
        perm = Int32[4, 3, 2, 1]
        sp_val = zeros(Float64, nrows)
        sp_idx = zeros(Int32, nrows)
        nrm = JClp.clp_drs_prep_ftran!(in_val, in_idx, 2, perm, sp_val, sp_idx)
        @test nrm ≈ 5.0
        @test sp_val[3] == 2.0 && sp_val[1] == -1.0

        w_before = copy(weights)
        alt_val = zeros(Float64, nrows)
        alt_idx = zeros(Int32, nrows)
        upd_val = Float64[0.5, 2.0]
        upd_idx = Int32[1, 2]
        alpha, nsave = JClp.clp_drs_apply_weight_update!(
            upd_val, upd_idx, 2, sp_val, perm, weights, alt_val, alt_idx,
            2, 2.0, nrm, false,
        )
        @test alpha == 2.0
        @test nsave == 2
        @test weights[2] ≈ 5.0 / 4.0

        JClp.clp_drs_unroll_weights!(weights, alt_val, alt_idx, nsave, true)
        @test weights == w_before
        @test alt_val[1] == 0.0 && alt_val[2] == 0.0

        # update_primal_solution!
        work = Float64[1.0, -0.5, 0.0, 0.0]
        which = Int32[1, 3, 0, 0]
        cost = Float64[1.0, 2.0, 3.0, 4.0]
        sol3 = copy(solution)
        dobj, nnz3 = JClp.clp_drs_update_primal_solution!(
            work, which, 2, true, pivot_var, infeas, infeas_idx, nnz,
            sol3, cost, lower, upper, 1.0, 1e-7, 2,
        )
        @test sol3[1] ≈ -2.0
        @test infeas[1] ≈ 4.0
        @test infeas[2] == JClp.CLP_INDEXED_REALLY_TINY
        @test (@allocated JClp.clp_drs_pivot_row!(
            infeas, infeas_idx, nnz, weights, pivot_var, solution, lower, upper, status,
            nrows, 0, 1, 10, 1e-7, 0.0, 0.0, 1, -1000, 0.0,
        )) == 0
    end
end

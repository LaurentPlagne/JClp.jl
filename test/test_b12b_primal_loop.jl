using Test
using JClp

@testset "B12b-1: ClpSimplexPrimal::pivotResult pure-Julia kernel" begin
    @testset "2x2 primal simplex pivot with 0 allocations" begin
        # min -3x1 - 2x2 s.t. x1 + x2 <= 4 (s1), 2x1 + x2 <= 6 (s2), x >= 0
        # Start at x1=0, x2=0, s1=4, s2=6 (basic slacks). Pivot x1 in -> hits s2=0 at x1=3, s1=1, obj=-9.
        nrows, ncols = 2, 2
        ntotal = 4
        mat_starts = Int32[1, 3, 5]
        mat_lengths = Int32[2, 2]
        mat_indices = Int32[1, 2, 1, 2]
        mat_elements = Float64[1.0, 2.0, 1.0, 1.0]

        cf = JClp.ClpFactorization()
        pivot_var = Int32[3, 4]
        status = UInt8[
            UInt8(JClp.CLP_STATUS_AT_LOWER),
            UInt8(JClp.CLP_STATUS_AT_LOWER),
            UInt8(JClp.CLP_STATUS_BASIC),
            UInt8(JClp.CLP_STATUS_BASIC),
        ]
        pivot_temp = Int32[1, 2]
        JClp.clp_fact_factorize_packed!(
            cf, nrows, ncols, 2, 2, pivot_temp,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            false, nothing, nothing, 0, 1, pivot_var, nrows,
        )
        JClp.clp_fact_reorder_and_finalize!(cf, nrows, ncols, 2, pivot_temp, pivot_var, false)
        cf.lu.max_pivots = 50
        @test cf.lu.status == 0

        # Clp convention: Ax - s = 0, so at x=(0,0), s=(0,0) with s <= (4, 6)
        solution = Float64[0.0, 0.0, 0.0, 0.0]
        lower    = Float64[0.0, 0.0, -1e30, -1e30]
        upper    = Float64[1e30, 1e30, 4.0, 6.0]
        cost     = Float64[-3.0, -2.0, 0.0, 0.0]
        dj       = Float64[-3.0, -2.0, 0.0, 0.0]

        # NonLinearCost method 2: all feasible, currentStatus == CLP_SAME
        nlc_status = fill(UInt8((JClp.CLP_NLC_SAME << 4) | JClp.CLP_NLC_FEASIBLE), ntotal)
        nlc_bound  = zeros(Float64, ntotal)
        nlc_cost2  = copy(cost)

        pcs_weights = ones(Float64, ntotal)
        pcs_ref     = UInt32[0x0000000f]
        pcs_alt_val = zeros(Float64, 10)
        pcs_alt_idx = zeros(Int32, 10)

        prog_in  = zeros(Int32, JClp.CLP_CYCLE)
        prog_out = zeros(Int32, JClp.CLP_CYCLE)
        prog_way = zeros(Int8, JClp.CLP_CYCLE)
        r1_val, r1_idx = zeros(Float64, 10), zeros(Int32, 10)
        r2_val, r2_idx = zeros(Float64, 10), zeros(Int32, 10)
        r3_val, r3_idx = zeros(Float64, 10), zeros(Int32, 10)
        empty_f = Float64[]

        res = JClp.clp_primal_pivot_result!(
            cf, mat_starts, mat_lengths, mat_indices, mat_elements,
            false, empty_f, empty_f,
            pivot_var, solution, lower, upper, cost, dj, status,
            nlc_status, nlc_bound, nlc_cost2,
            true, pcs_weights, pcs_ref, true, pcs_alt_val, pcs_alt_idx, 0,
            1, 0, 0, 1.0,
            prog_in, prog_out, prog_way,
            r1_val, r1_idx, r2_val, r2_idx, r3_val, r3_idx,
            nrows, ncols, 1e-7, 1e-7, 1e10, 1e15, 1.0, 1.0, 0.0, 1e10, 0.0,
            1000, 0, 1, 0,
            -1, 0, 0, 0, -1, -1, 0, 0, 0.0, 1.0,
            0, 1, 0, 1, 0,
            0.0, 0.0, 1e30, -3.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0, 1e-9, 0, UInt32(42),
        )
        @test res[1] == JClp.CLP_PL_EXIT_DONE
        @test res[2] in (-1, -2)
        @test res[4] == 1
        @test solution[1] ≈ 3.0 atol=1e-10
        @test solution[3] ≈ 3.0 atol=1e-10
        @test solution[4] ≈ 6.0 atol=1e-10
        @test res[31] ≈ -9.0 atol=1e-10
        @test pivot_var[2] == Int32(1)

        # Zero-allocation check on a bound-flip pivot (seq_in=2 with upper=0.5)
        upper[2] = 0.5
        function _alloc_pl(cf_, ms, ml, mi, me, ef, pv, sol, lo, hi, co, d, st,
                           ns, nb, nc2, pw, pr, pav, pai, pi_, po_, pw2,
                           r1v, r1i, r2v, r2i, r3v, r3i)
            # Clear r1 from previous pivot
            for k in 1:10; r1v[k] = 0.0; end
            sol[2] = 0.0; st[2] = UInt8(JClp.CLP_STATUS_AT_LOWER)
            JClp.clp_primal_pivot_result!(
                cf_, ms, ml, mi, me, false, ef, ef, pv, sol, lo, hi, co, d, st,
                ns, nb, nc2, true, pw, pr, true, pav, pai, 0, 1, 0, 0, 1.0,
                pi_, po_, pw2, r1v, r1i, r2v, r2i, r3v, r3i,
                2, 2, 1e-7, 1e-7, 1e10, 1e15, 1.0, 1.0, 0.0, 1e10, 0.0, 1000, 0, 1, 0,
                -1, 1, 0, 0, -1, -1, 0, 0, 0.0, 1.0, 0, 2, 0, 1, 0,
                0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -1.0, -9.0, 1e-9, 0, UInt32(42),
            )
            for k in 1:10; r1v[k] = 0.0; end
            sol[2] = 0.0; st[2] = UInt8(JClp.CLP_STATUS_AT_LOWER)
            return @allocated JClp.clp_primal_pivot_result!(
                cf_, ms, ml, mi, me, false, ef, ef, pv, sol, lo, hi, co, d, st,
                ns, nb, nc2, true, pw, pr, true, pav, pai, 0, 1, 0, 0, 1.0,
                pi_, po_, pw2, r1v, r1i, r2v, r2i, r3v, r3i,
                2, 2, 1e-7, 1e-7, 1e10, 1e15, 1.0, 1.0, 0.0, 1e10, 0.0, 1000, 0, 1, 0,
                -1, 1, 0, 0, -1, -1, 0, 0, 0.0, 1.0, 0, 2, 0, 1, 0,
                0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, -1.0, -9.0, 1e-9, 0, UInt32(42),
            )
        end
        @test _alloc_pl(cf, mat_starts, mat_lengths, mat_indices, mat_elements, empty_f,
                        pivot_var, solution, lower, upper, cost, dj, status,
                        nlc_status, nlc_bound, nlc_cost2, pcs_weights, pcs_ref,
                        pcs_alt_val, pcs_alt_idx, prog_in, prog_out, prog_way,
                        r1_val, r1_idx, r2_val, r2_idx, r3_val, r3_idx) == 0
    end

    @testset "2x2 primal whileIterating solves to optimality with 0 allocations" begin
        # min -3x1 - 2x2 s.t. x1 + x2 <= 4, 2x1 + x2 <= 6, x >= 0 -> opt x1=2, x2=2, obj=-10
        nrows, ncols = 2, 2
        ntotal = 4
        mat_starts   = Int32[1, 3, 5]
        mat_lengths  = Int32[2, 2]
        mat_indices  = Int32[1, 2, 1, 2]
        mat_elements = Float64[1.0, 2.0, 1.0, 1.0]
        rc_starts    = Int32[1, 3, 5]
        rc_cols      = Int32[1, 2, 1, 2]
        rc_elements  = Float64[1.0, 1.0, 2.0, 1.0]

        cf = JClp.ClpFactorization()
        cf.lu.max_pivots = 50
        cf.do_statistics = false
        pivot_var = Int32[3, 4]
        status = UInt8[
            UInt8(JClp.CLP_STATUS_AT_LOWER), UInt8(JClp.CLP_STATUS_AT_LOWER),
            UInt8(JClp.CLP_STATUS_BASIC),    UInt8(JClp.CLP_STATUS_BASIC),
        ]
        pivot_temp = Int32[1, 2]
        JClp.clp_fact_factorize_packed!(
            cf, nrows, ncols, 2, 2, pivot_temp,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            false, nothing, nothing, 0, 1, pivot_var, nrows,
        )
        JClp.clp_fact_reorder_and_finalize!(cf, nrows, ncols, 2, pivot_temp, pivot_var, false)

        solution = Float64[0.0, 0.0, 0.0, 0.0]
        lower    = Float64[0.0, 0.0, -1e30, -1e30]
        upper    = Float64[1e30, 1e30, 4.0, 6.0]
        cost     = Float64[-3.0, -2.0, 0.0, 0.0]
        dj       = Float64[-3.0, -2.0, 0.0, 0.0]
        nlc_status = fill(UInt8((JClp.CLP_NLC_SAME << 4) | JClp.CLP_NLC_FEASIBLE), ntotal)
        nlc_bound  = zeros(Float64, ntotal)
        nlc_cost2  = copy(cost)

        pcs_weights    = ones(Float64, ntotal)
        pcs_ref        = UInt32[0x00000003]
        pcs_infeas_val = Float64[9.0, 4.0, 0.0, 0.0]
        pcs_infeas_idx = Int32[1, 2, 0, 0]
        pcs_alt_val    = zeros(Float64, 10)
        pcs_alt_idx    = zeros(Int32, 10)
        prog_in, prog_out, prog_way = fill(Int32(-1), JClp.CLP_CYCLE), fill(Int32(-1), JClp.CLP_CYCLE), zeros(Int8, JClp.CLP_CYCLE)
        r1_val, r1_idx = zeros(Float64, 10), zeros(Int32, 10)
        r2_val, r2_idx = zeros(Float64, 10), zeros(Int32, 10)
        r3_val, r3_idx = zeros(Float64, 10), zeros(Int32, 10)
        c0_val, c0_idx = zeros(Float64, 10), zeros(Int32, 10)
        c1_val, c1_idx = zeros(Float64, 10), zeros(Int32, 10)
        marked_buf = zeros(UInt8, ncols)
        work_col   = zeros(Float64, ncols)
        empty_f    = Float64[]

        res = JClp.clp_primal_while_iterating!(
            cf, mat_starts, mat_lengths, mat_indices, mat_elements,
            mat_starts, mat_indices, mat_elements,
            false, false, empty_f, empty_f,
            true, rc_starts, rc_cols, rc_elements,
            pivot_var, solution, lower, upper, cost, dj, status,
            nlc_status, nlc_bound, nlc_cost2,
            pcs_weights, pcs_ref, true,
            pcs_infeas_val, pcs_infeas_idx, 2,
            pcs_alt_val, pcs_alt_idx, 0,
            1, 0, 4, 0, 1.0,
            prog_in, prog_out, prog_way,
            r1_val, r1_idx, 0, r2_val, r2_idx, r3_val, r3_idx,
            c0_val, c0_idx, c1_val, c1_idx, marked_buf, work_col,
            nrows, ncols, 1e-7, 1e-7, 1e-13, 1e10, 1e15, 1.0, 1.0, 0.0, 1e10, 0.0,
            10, 16384, 1, 0,
            -1, -1, 7, 0, 0, -1, -1, 0, 0, 0.0, 1.0, 0,
            0, 0, 0, 0, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, -1.0, 0.0, 1e-9, 0, UInt32(42),
        )
        @test res[1] == JClp.CLP_PL_EXIT_DONE
        @test res[2] == 0
        @test res[4] == 9
        @test solution[1] ≈ 2.0 atol=1e-10
        @test solution[2] ≈ 2.0 atol=1e-10
        @test res[35] ≈ -10.0 atol=1e-10

        # Zero-allocation check on steady-state call (already optimal -> 0 pivots)
        function _alloc_pwi(cf_, ms, ml, mi, me, rcs, rcc, rce, ef, pv, sol, lo, hi, co, d, st,
                            ns, nb, nc2, pw, pr, piv, pii, pav, pai, pi_, po_, pw2,
                            r1v, r1i, r2v, r2i, r3v, r3i, c0v, c0i, c1v, c1i, mb, wc)
            JClp.clp_primal_while_iterating!(
                cf_, ms, ml, mi, me, ms, mi, me, false, false, ef, ef, true, rcs, rcc, rce,
                pv, sol, lo, hi, co, d, st, ns, nb, nc2, pw, pr, true, piv, pii, 0, pav, pai, 0,
                1, 0, 4, 0, 1.0, pi_, po_, pw2, r1v, r1i, 0, r2v, r2i, r3v, r3i, c0v, c0i, c1v, c1i, mb, wc,
                2, 2, 1e-7, 1e-7, 1e-13, 1e10, 1e15, 1.0, 1.0, 0.0, 1e10, 0.0, 1000, 0, 1, 0,
                -1, -1, 2, 0, 0, -1, -1, 0, 0, 0.0, 1.0, 0, 0, 0, 0, 0, 0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -10.0, 1e-9, 0, UInt32(1),
            )
            return @allocated JClp.clp_primal_while_iterating!(
                cf_, ms, ml, mi, me, ms, mi, me, false, false, ef, ef, true, rcs, rcc, rce,
                pv, sol, lo, hi, co, d, st, ns, nb, nc2, pw, pr, true, piv, pii, 0, pav, pai, 0,
                1, 0, 4, 0, 1.0, pi_, po_, pw2, r1v, r1i, 0, r2v, r2i, r3v, r3i, c0v, c0i, c1v, c1i, mb, wc,
                2, 2, 1e-7, 1e-7, 1e-13, 1e10, 1e15, 1.0, 1.0, 0.0, 1e10, 0.0, 1000, 0, 1, 0,
                -1, -1, 2, 0, 0, -1, -1, 0, 0, 0.0, 1.0, 0, 0, 0, 0, 0, 0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -10.0, 1e-9, 0, UInt32(1),
            )
        end
        @test _alloc_pwi(cf, mat_starts, mat_lengths, mat_indices, mat_elements,
                         rc_starts, rc_cols, rc_elements, empty_f, pivot_var, solution,
                         lower, upper, cost, dj, status, nlc_status, nlc_bound, nlc_cost2,
                         pcs_weights, pcs_ref, pcs_infeas_val, pcs_infeas_idx,
                         pcs_alt_val, pcs_alt_idx, prog_in, prog_out, prog_way,
                         r1_val, r1_idx, r2_val, r2_idx, r3_val, r3_idx,
                         c0_val, c0_idx, c1_val, c1_idx, marked_buf, work_col) == 0
    end
end

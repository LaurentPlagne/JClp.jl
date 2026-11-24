using Test
using JClp

@testset "B12a: ClpSimplexDual::whileIterating pure-Julia inner loop" begin
    @testset "2x2 dual simplex solves to optimality in pure Julia with 0 allocations" begin
        # min 3 x1 + 2 x2  s.t.  x1 + x2 >= 3,  2 x1 + x2 >= 4,  x1, x2 >= 0
        # Optimal: x1 = 1, x2 = 2, obj = 7 (all-slack basis -> 2 dual pivots)
        nrows, ncols = 2, 2
        mat_starts   = Int32[1, 3, 5]
        mat_lengths  = Int32[2, 2]
        mat_indices  = Int32[1, 2, 1, 2]
        mat_elements = Float64[1.0, 2.0, 1.0, 1.0]
        rc_starts    = Int32[1, 3, 5]
        rc_cols      = Int32[1, 2, 1, 2]
        rc_elements  = Float64[1.0, 1.0, 2.0, 1.0]

        cf = JClp.ClpFactorization()
        pivot_var = Int32[3, 4]
        status = UInt8[
            JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_BASIC,    JClp.CLP_STATUS_BASIC,
        ]
        pivot_temp = Int32[1, 2]
        JClp.clp_fact_factorize_packed!(
            cf, nrows, ncols, 2, 2, pivot_temp,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            false, nothing, nothing, 0, 1, pivot_var, nrows,
        )
        JClp.clp_fact_reorder_and_finalize!(cf, nrows, ncols, 2, pivot_temp, pivot_var, false)
        @test cf.lu.status == 0

        solution = Float64[0.0, 0.0, 0.0, 0.0]
        lower    = Float64[0.0, 0.0, 3.0, 4.0]
        upper    = fill(floatmax(Float64), 4)
        cost     = Float64[3.0, 2.0, 0.0, 0.0]
        dj       = Float64[3.0, 2.0, 0.0, 0.0]

        drs_weights    = Float64[1.0, 1.0]
        drs_infeas_val = Float64[9.0, 16.0]
        drs_infeas_idx = Int32[1, 2]
        drs_alt_val    = zeros(Float64, 4)
        drs_alt_idx    = zeros(Int32, 4)

        prog_in  = fill(Int32(-1), JClp.CLP_CYCLE)
        prog_out = fill(Int32(-1), JClp.CLP_CYCLE)
        prog_way = fill(Int8(0), JClp.CLP_CYCLE)

        cap = 8
        r0_val, r0_idx = zeros(cap), zeros(Int32, cap)
        r1_val, r1_idx = zeros(cap), zeros(Int32, cap)
        r2_val, r2_idx = zeros(cap), zeros(Int32, cap)
        r3_val, r3_idx = zeros(cap), zeros(Int32, cap)
        c0_val, c0_idx = zeros(cap), zeros(Int32, cap)
        c1_val, c1_idx = zeros(cap), zeros(Int32, cap)
        marked_buf = zeros(UInt8, ncols)
        work_col = zeros(Float64, ncols)
        empty_f = Float64[]
        cl_o, cu_o = copy(lower[1:2]), copy(upper[1:2])
        rl_o, ru_o = copy(lower[3:4]), copy(upper[3:4])

        res = JClp.clp_dual_while_iterating!(
            cf,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            mat_starts, mat_indices, mat_elements,
            false, false, empty_f, empty_f, empty_f,
            true, rc_starts, rc_cols, rc_elements,
            pivot_var, solution, lower, upper, cost, dj, status,
            cl_o, cu_o, rl_o, ru_o,
            drs_weights, drs_infeas_val, drs_infeas_idx, 2,
            drs_alt_val, drs_alt_idx, 0, 1,
            prog_in, prog_out, prog_way,
            r0_val, r0_idx, r1_val, r1_idx, r2_val, r2_idx, r3_val, r3_idx,
            c0_val, c0_idx, c1_val, c1_idx, marked_buf, work_col,
            nrows, ncols, 1e-7, 1e-7, 1e-13, 1e7, 1e15, 1e-9,
            1.0, 1.0, 0.0, 0.0, 0.0, 1000, 0, 1, 0,
            -1, -1, 0, 0, 0, 8, 0, -1, -1, 0, 0, 0, 2,
            0, 0, 0, 0, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, UInt32(12345),
        )
        @test res[1] == JClp.CLP_DL_EXIT_PIVOT_ROW_NEG
        @test res[4] == 2
        @test solution[1] ≈ 1.0 atol=1e-10
        @test solution[2] ≈ 2.0 atol=1e-10
        @test res[35] ≈ 7.0 atol=1e-10

        # Zero-allocation check on steady-state call (already optimal -> 0 pivots)
        function _alloc_dl(cf_, ms, ml, mi, me, rcs, rcc, rce, pv, sol, lo, hi, co, d, st,
                           clo, cuo, rlo, ruo, dw, div, dii, dav, dai, pi_, po_, pw_,
                           r0v, r0i, r1v, r1i, r2v, r2i, r3v, r3i, c0v, c0i, c1v, c1i, mb, wc, ef)
            JClp.clp_dual_while_iterating!(
                cf_, ms, ml, mi, me, ms, mi, me, false, false, ef, ef, ef, true, rcs, rcc, rce,
                pv, sol, lo, hi, co, d, st, clo, cuo, rlo, ruo, dw, div, dii, 0, dav, dai, 0, 1,
                pi_, po_, pw_, r0v, r0i, r1v, r1i, r2v, r2i, r3v, r3i, c0v, c0i, c1v, c1i, mb, wc,
                2, 2, 1e-7, 1e-7, 1e-13, 1e7, 1e15, 1e-9, 1.0, 1.0, 0.0, 0.0, 0.0, 1000, 0, 1, 0,
                -1, -1, 0, 0, 0, 8, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, UInt32(1),
            )
            return @allocated JClp.clp_dual_while_iterating!(
                cf_, ms, ml, mi, me, ms, mi, me, false, false, ef, ef, ef, true, rcs, rcc, rce,
                pv, sol, lo, hi, co, d, st, clo, cuo, rlo, ruo, dw, div, dii, 0, dav, dai, 0, 1,
                pi_, po_, pw_, r0v, r0i, r1v, r1i, r2v, r2i, r3v, r3i, c0v, c0i, c1v, c1i, mb, wc,
                2, 2, 1e-7, 1e-7, 1e-13, 1e7, 1e15, 1e-9, 1.0, 1.0, 0.0, 0.0, 0.0, 1000, 0, 1, 0,
                -1, -1, 0, 0, 0, 8, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, UInt32(1),
            )
        end
        @test _alloc_dl(cf, mat_starts, mat_lengths, mat_indices, mat_elements,
                        rc_starts, rc_cols, rc_elements, pivot_var, solution, lower, upper,
                        cost, dj, status, cl_o, cu_o, rl_o, ru_o, drs_weights, drs_infeas_val,
                        drs_infeas_idx, drs_alt_val, drs_alt_idx, prog_in, prog_out, prog_way,
                        r0_val, r0_idx, r1_val, r1_idx, r2_val, r2_idx, r3_val, r3_idx,
                        c0_val, c0_idx, c1_val, c1_idx, marked_buf, work_col, empty_f) == 0
    end
end

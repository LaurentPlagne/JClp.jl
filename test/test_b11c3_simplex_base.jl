using Test
using JClp

@testset "B11c-3: housekeeping, deleteRim unscaling, gutsOfSolution accuracy" begin
    @testset "clp_housekeeping! normal pivot, bound flip, refactorize + zero alloc" begin
        nrows, ncols = 2, 3
        pvar = Int32[4, 5]
        sol  = [0.0, 0.0, 0.0, 2.0, 3.0]
        lo   = [0.0, 0.0, 0.0, 0.0, 0.0]
        hi   = [10.0, 10.0, 10.0, 10.0, 10.0]
        st   = UInt8[
            JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_BASIC,
        ]
        pin  = fill(Int32(-1), JClp.CLP_CYCLE)
        pout = fill(Int32(-1), JClp.CLP_CYCLE)
        pway = fill(Int8(0), JClp.CLP_CYCLE)

        # Normal pivot: seq_in=1 enters row 1, seq_out=4 leaves at lower bound
        ret, act, cyc, nit, chg, pflg, aacc, objv, ffac = JClp.clp_housekeeping!(
            pvar, sol, lo, hi, st, pin, pout, pway,
            nrows, ncols, 1, 1, 4, 1, 1,
            2.5, 0.0, 2.0, 1.0, 10.0, -5.0, 1.0, 1.0,
            0, 1000, 0, 0, 0, 1, 1, false,
            1, 200, false, false, -1,
        )
        @test ret == 0 && act == JClp.CLP_HK_ACT_DONE && cyc == 0
        @test nit == 1 && chg == 1
        @test pvar[1] == Int32(1)
        @test sol[1] == 2.5 && sol[4] == 0.0
        @test (st[1] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_BASIC
        @test (st[4] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_LOWER
        @test aacc == 2.0
        @test objv == 5.0

        # Bound flip: seq_in == seq_out == 2 moving to upper bound
        ret2, act2, _, nit2, _, _, _, _, _ = JClp.clp_housekeeping!(
            pvar, sol, lo, hi, st, pin, pout, pway,
            nrows, ncols, 0, 2, 2, 1, -1,
            10.0, 10.0, 1.0, 0.0, 5.0, -1.0, 1.0, 1.0,
            nit, 1000, chg, pflg, 0, 1, 1, false,
            2, 200, false, false, -1,
        )
        @test ret2 == 0 && act2 == JClp.CLP_HK_ACT_DONE && nit2 == 2
        @test (st[2] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_UPPER

        # forceFactorization hit -> returns 1 and relaxes forceFactorization
        ret3, act3, _, _, _, _, _, _, ffac3 = JClp.clp_housekeeping!(
            pvar, sol, lo, hi, st, pin, pout, pway,
            nrows, ncols, 0, 2, 2, -1, 1,
            0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 1.0, 1.0,
            nit2, 1000, chg, pflg, 0, 1, 1, false,
            4, 200, false, false, 4,
        )
        @test ret3 == 1 && act3 == JClp.CLP_HK_ACT_DONE && ffac3 == 5

        function _alloc_hk(pv, s, l, u, st_, pi_, po_, pw_)
            JClp.clp_housekeeping!(pv, s, l, u, st_, pi_, po_, pw_, 2, 3, 0, 2, 2, 1, -1, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 1000, 0, 0, 0, 1, 1, false, 1, 200, false, false, -1)
            return @allocated JClp.clp_housekeeping!(pv, s, l, u, st_, pi_, po_, pw_, 2, 3, 0, 2, 2, 1, -1, 10.0, 10.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0, 1000, 0, 0, 0, 1, 1, false, 1, 200, false, false, -1)
        end
        @test _alloc_hk(pvar, sol, lo, hi, st, pin, pout, pway) == 0
    end

    @testset "clp_delete_rim_unscale!" begin
        nrows, ncols = 1, 2
        col_act = zeros(2)
        col_dj  = zeros(2)
        row_act = zeros(1)
        dual    = [4.0]
        col_act_w = [2.0, 3.0]
        col_lo_w  = [0.0, 0.0]
        col_hi_w  = [10.0, 10.0]
        col_dj_w  = [0.0, 0.0]
        row_act_w = [6.0]
        row_lo_w  = [0.0]
        row_hi_w  = [10.0]
        col_lo = [0.0, 0.0]
        col_hi = [10.0, 10.0]
        row_lo = [0.0]
        row_hi = [10.0]
        rscale = [2.0]
        irscale = [0.5]
        cscale = [0.5, 2.0]
        icscale = [2.0, 0.5]
        uo, ss = JClp.clp_delete_rim_unscale!(
            col_act, col_dj, row_act, dual,
            col_act_w, col_lo_w, col_hi_w, col_dj_w,
            row_act_w, row_lo_w, row_hi_w,
            col_lo, col_hi, row_lo, row_hi,
            nothing, rscale, irscale, cscale, icscale, nothing,
            nrows, ncols, true, 2.0, 1.0, 1e-7, 1e-7, 0, 0,
        )
        @test col_act ≈ [1.0, 6.0]
        @test row_act ≈ [3.0]
        @test dual ≈ [4.0]  # 4.0 * 2.0 / 2.0
        @test uo ≈ 4.0      # min(6-0, 10-6) = 4.0
        @test ss == 3       # row_act=3.0 > row_lo=0.0 with dual=4.0 > dual_tol -> dual unscaled infeasibility
    end

    @testset "clp_guts_of_solution_check_accuracy" begin
        ret, zt, pt, pe, de, ff = JClp.clp_guts_of_solution_check_accuracy(
            200, 50, -1, false, -1,
            1e-13, 0.1, 1e-6, 1e-6, 1e5, 1e-6,
            0.0, 1e5, 1, 100, 5,
        )
        @test ret == -123456789
        @test zt == 1e-18
        @test pt ≈ 0.12
        @test pe == 0.0 && de == 0.0
        @test ff == 5
    end
end

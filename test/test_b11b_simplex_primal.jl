using JClp
using Test

@testset "B11b: ClpSimplexPrimal Hot Kernels & Helpers (Pure 1-based Julia)" begin
    @testset "clp_nlc_change_in_cost_m2!" begin
        nlc_st = UInt8[
            JClp.CLP_NLC_FEASIBLE | (JClp.CLP_NLC_SAME << 4),
            JClp.CLP_NLC_BELOW_LOWER | (JClp.CLP_NLC_SAME << 4),
            JClp.CLP_NLC_ABOVE_UPPER | (JClp.CLP_NLC_SAME << 4),
        ]
        bound = Float64[0.0, 10.0, 2.0]
        lower = Float64[1.0, -floatmax(Float64), 5.0]
        upper = Float64[5.0, 1.0, floatmax(Float64)]

        # 1. Feasible going below lower (alpha > 0) -> rhs = floatmax, curr = BELOW_LOWER
        dc1, rhs1 = JClp.clp_nlc_change_in_cost_m2!(nlc_st, bound, lower, upper, 1, 2.5, 3.0, 100.0)
        @test dc1 == 250.0
        @test rhs1 == floatmax(Float64)
        @test JClp._nlc_curr_status(nlc_st[1]) == JClp.CLP_NLC_BELOW_LOWER
        @test JClp._nlc_orig_status(nlc_st[1]) == JClp.CLP_NLC_FEASIBLE

        # Now same var going back feasible? Feasible going above upper on fresh status:
        nlc_st[1] = JClp.CLP_NLC_FEASIBLE | (JClp.CLP_NLC_SAME << 4)
        dc1b, rhs1b = JClp.clp_nlc_change_in_cost_m2!(nlc_st, bound, lower, upper, 1, -1.5, 2.0, 100.0)
        @test dc1b == 150.0
        @test rhs1b == floatmax(Float64)
        @test JClp._nlc_curr_status(nlc_st[1]) == JClp.CLP_NLC_ABOVE_UPPER

        # 2. Below lower going feasible (alpha < 0) -> rhs += bound - upper
        dc2, rhs2 = JClp.clp_nlc_change_in_cost_m2!(nlc_st, bound, lower, upper, 2, -3.0, 4.0, 100.0)
        @test dc2 == 300.0
        @test rhs2 == 4.0 + (10.0 - 1.0)
        @test JClp._nlc_curr_status(nlc_st[2]) == JClp.CLP_NLC_FEASIBLE

        # 3. Above upper going feasible (alpha > 0) -> rhs += lower - bound
        dc3, rhs3 = JClp.clp_nlc_change_in_cost_m2!(nlc_st, bound, lower, upper, 3, 0.5, 1.0, 100.0)
        @test dc3 == 50.0
        @test rhs3 == 1.0 + (5.0 - 2.0)
        @test JClp._nlc_curr_status(nlc_st[3]) == JClp.CLP_NLC_FEASIBLE

        function _alloc_cic(nlc_st, bound, lower, upper)
            return @allocated JClp.clp_nlc_change_in_cost_m2!(nlc_st, bound, lower, upper, 1, 2.0, 1.0, 100.0)
        end
        _alloc_cic(nlc_st, bound, lower, upper)
        @test _alloc_cic(nlc_st, bound, lower, upper) == 0
    end

    @testset "alwaysOptimal & exactOutgoing helpers" begin
        opts = 0
        @test !JClp.clp_primal_always_optimal(opts)
        @test !JClp.clp_primal_exact_outgoing(opts)
        opts = JClp.clp_primal_always_optimal!(opts, true)
        @test JClp.clp_primal_always_optimal(opts)
        @test opts == 1
        opts = JClp.clp_primal_exact_outgoing!(opts, true)
        @test JClp.clp_primal_exact_outgoing(opts)
        @test opts == 5
        opts = JClp.clp_primal_always_optimal!(opts, false)
        @test !JClp.clp_primal_always_optimal(opts) && JClp.clp_primal_exact_outgoing(opts)
        opts = JClp.clp_primal_exact_outgoing!(opts, false)
        @test opts == 0
    end

    @testset "clp_primal_unflag!" begin
        dj = Float64[1.0, 1e-8, -0.5, 10.0, -0.05]
        st = UInt8[
            JClp.CLP_STATUS_AT_LOWER | JClp.CLP_STATUS_FLAGGED_BIT, # |dj|=1.0 > tol -> counted
            JClp.CLP_STATUS_AT_UPPER | JClp.CLP_STATUS_FLAGGED_BIT, # |dj|=1e-8 <= tol -> not counted
            JClp.CLP_STATUS_AT_LOWER | JClp.CLP_STATUS_FLAGGED_BIT | JClp.CLP_STATUS_ACTIVE_BIT, # |dj|=0.5 > tol -> counted, bit 128 kept
            JClp.CLP_STATUS_BASIC,                                 # not flagged
            JClp.CLP_STATUS_AT_UPPER | JClp.CLP_STATUS_FLAGGED_BIT, # |dj|=0.05 <= relaxed_tol (1e-7 + 0.01 = 0.0100001)? Wait: 0.05 > 0.0100001 -> counted if err>=1e-3
        ]
        # Set largest_dual_err = 0.02 => min(1e-2, 10*0.02) = 1e-2 => relaxed_tol = 1e-2 + 1e-7
        dj[5] = 0.005 # <= 0.01 + 1e-7
        cnt = JClp.clp_primal_unflag!(dj, st, 5, 1e-7, 0.02)
        @test cnt == 2
        @test all(x -> (x & JClp.CLP_STATUS_FLAGGED_BIT) == UInt8(0), st)
        @test (st[1] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_LOWER
        @test (st[3] & JClp.CLP_STATUS_ACTIVE_BIT) == JClp.CLP_STATUS_ACTIVE_BIT

        function _alloc_unflag(dj, st)
            return @allocated JClp.clp_primal_unflag!(dj, st, 5, 1e-7, 0.0)
        end
        _alloc_unflag(dj, st)
        @test _alloc_unflag(dj, st) == 0
    end

    @testset "clp_primal_clear_all!" begin
        # Packed mode
        st = UInt8[
            JClp.CLP_STATUS_BASIC | JClp.CLP_STATUS_ACTIVE_BIT,
            JClp.CLP_STATUS_BASIC | JClp.CLP_STATUS_ACTIVE_BIT | JClp.CLP_STATUS_FLAGGED_BIT,
            JClp.CLP_STATUS_BASIC | JClp.CLP_STATUS_ACTIVE_BIT,
            JClp.CLP_STATUS_BASIC | JClp.CLP_STATUS_ACTIVE_BIT,
        ]
        rval_p = Float64[1.5, -2.5, 0.0, 0.0]
        ridx_p = Int32[1, 3, 0, 0]
        nnz_out = JClp.clp_primal_clear_all!(rval_p, ridx_p, 2, 4, true, st)
        @test nnz_out == 0
        @test rval_p == zeros(Float64, 4)
        @test (st[1] & JClp.CLP_STATUS_ACTIVE_BIT) == UInt8(0)
        @test (st[3] & JClp.CLP_STATUS_ACTIVE_BIT) == UInt8(0)
        @test (st[2] & JClp.CLP_STATUS_ACTIVE_BIT) != UInt8(0)
        @test (st[4] & JClp.CLP_STATUS_ACTIVE_BIT) != UInt8(0)

        # Unpacked sparse (3*nnz < cap)
        st[1] |= JClp.CLP_STATUS_ACTIVE_BIT
        st[3] |= JClp.CLP_STATUS_ACTIVE_BIT
        rval_u = Float64[1.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ridx_u = Int32[1, 3, 0, 0, 0, 0, 0, 0]
        @test JClp.clp_primal_clear_all!(rval_u, ridx_u, 2, 8, false, st) == 0
        @test rval_u == zeros(Float64, 8)
        @test (st[1] & JClp.CLP_STATUS_ACTIVE_BIT) == UInt8(0)
        @test (st[3] & JClp.CLP_STATUS_ACTIVE_BIT) == UInt8(0)

        # Unpacked dense (3*nnz >= cap)
        st[2] |= JClp.CLP_STATUS_ACTIVE_BIT
        rval_d = Float64[1.0, 2.0, 3.0, 4.0]
        ridx_d = Int32[1, 2, 0, 0]
        @test JClp.clp_primal_clear_all!(rval_d, ridx_d, 2, 4, false, st) == 0
        @test rval_d == zeros(Float64, 4)
        @test (st[2] & JClp.CLP_STATUS_ACTIVE_BIT) == UInt8(0)
        @test (st[2] & JClp.CLP_STATUS_FLAGGED_BIT) != UInt8(0)

        function _alloc_ca(rval_p, ridx_p, st)
            return @allocated JClp.clp_primal_clear_all!(rval_p, ridx_p, 2, 4, true, st)
        end
        _alloc_ca(rval_p, ridx_p, st)
        @test _alloc_ca(rval_p, ridx_p, st) == 0
    end

    @testset "clp_primal_ray!" begin
        ncols = 4
        nrows = 3
        pivot_var = Int32[2, 5, 4] # row 1 -> col 2, row 2 -> slack 5 (>ncols), row 3 -> col 4
        ray = ones(Float64, ncols)

        # Packed mode (col 1 entering, dir = 1 => way = -1; row 3 has tiny 1e-13 ignored)
        rval_p = Float64[2.5, -3.0, 1e-13]
        ridx_p = Int32[1, 2, 3]
        JClp.clp_primal_ray!(ray, rval_p, ridx_p, 3, true, pivot_var, ncols, 1, 1)
        @test ray ≈ Float64[1.0, -2.5, 0.0, 0.0]

        # Unpacked mode (incoming slack seq=5 > ncols, dir = -1 => way = +1)
        rval_u = Float64[2.5, -3.0, -4.0]
        ridx_u = Int32[1, 2, 3]
        JClp.clp_primal_ray!(ray, rval_u, ridx_u, 3, false, pivot_var, ncols, 5, -1)
        @test ray ≈ Float64[0.0, 2.5, 0.0, -4.0]

        function _alloc_ray(ray, rval_p, ridx_p, pivot_var)
            return @allocated JClp.clp_primal_ray!(ray, rval_p, ridx_p, 3, true, pivot_var, 4, 1, 1)
        end
        _alloc_ray(ray, rval_p, ridx_p, pivot_var)
        @test _alloc_ray(ray, rval_p, ridx_p, pivot_var) == 0
    end

    @testset "clp_primal_column_tail!" begin
        ntotal = 4
        nlc_st = fill(JClp.CLP_NLC_FEASIBLE | (JClp.CLP_NLC_SAME << 4), ntotal)
        bound = zeros(Float64, ntotal)
        cost2 = Float64[1.0, 2.0, 3.0, 4.0]
        cost = copy(cost2)
        lower = Float64[0.0, 0.0, 0.0, 0.0]
        upper = Float64[10.0, 10.0, 10.0, 10.0]
        sol = Float64[10.0, 0.0, 0.0, 5.0]
        dj = Float64[-2.0, 3.0, -4.0, 1.0]
        st = UInt8[
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_AT_UPPER,
        ]

        # 0 (no incoming)
        @test JClp.clp_primal_column_tail!(
            nlc_st, bound, cost2, sol, lower, upper, cost, dj, st,
            0, true, 1e-7, 100.0, 0,
        ) == (0, 0.0, 0.0, 0.0, 0.0, 0, 0)

        # Normal path without lookBothWays (col 3: AT_LOWER, dj = -4.0 -> dir = 1)
        res_n = JClp.clp_primal_column_tail!(
            nlc_st, bound, cost2, sol, lower, upper, cost, dj, st,
            3, false, 1e-7, 100.0, 0,
        )
        @test res_n == (3, 0.0, -4.0, 0.0, 10.0, 1, 0)

        # lookBothWays flip from AT_UPPER with dj < 0 (col 1: sol=10, dj=-2, weight=100)
        # dualIn -= -100 => 98.0 (> 0 -> dir_in = -1); setOne(1, 10 + 2e-7) puts ABOVE_UPPER (lo=10, hi=dbl_max)
        res_u = JClp.clp_primal_column_tail!(
            nlc_st, bound, cost2, sol, lower, upper, cost, dj, st,
            1, true, 1e-7, 100.0, 0,
        )
        @test res_u[1] == 1
        @test res_u[2] == 10.0
        @test res_u[3] == 98.0
        @test res_u[4] == 10.0
        @test res_u[5] == floatmax(Float64)
        @test res_u[6] == -1
        @test res_u[7] == 1
        @test (st[1] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_LOWER

        # lookBothWays flip from AT_LOWER with dj > 0 (col 2: sol=0, dj=3, weight=100)
        # dualIn -= 100 => -97.0 (< 0 -> dir_in = 1); setOne(2, 0 - 2e-7) puts BELOW_LOWER (lo=-dbl_max, hi=0)
        res_l = JClp.clp_primal_column_tail!(
            nlc_st, bound, cost2, sol, lower, upper, cost, dj, st,
            2, true, 1e-7, 100.0, 1,
        )
        @test res_l[1] == 2
        @test res_l[2] == 0.0
        @test res_l[3] == -97.0
        @test res_l[4] == -floatmax(Float64)
        @test res_l[5] == 0.0
        @test res_l[6] == 1
        @test res_l[7] == 2
        @test (st[2] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_UPPER

        function _alloc_pct(nlc_st, bound, cost2, sol, lower, upper, cost, dj, st)
            return @allocated JClp.clp_primal_column_tail!(
                nlc_st, bound, cost2, sol, lower, upper, cost, dj, st,
                3, true, 1e-7, 100.0, 0,
            )
        end
        _alloc_pct(nlc_st, bound, cost2, sol, lower, upper, cost, dj, st)
        @test _alloc_pct(nlc_st, bound, cost2, sol, lower, upper, cost, dj, st) == 0
    end

    @testset "clp_primal_next_super_basic!" begin
        # 1. Linear scan (super_basic_type = 1)
        ntotal = 6
        cval = zeros(Float64, ntotal)
        cidx = zeros(Int32, ntotal)
        dj = Float64[0.0, 0.0, 0.0, 1e-9, 2.0, 0.0]
        sol = Float64[0.0, 10.0, 5.0, 0.0, 0.0, 3.0]
        lo = Float64[0.0, 0.0, 0.0, -1e25, -1e25, 0.0]
        hi = Float64[10.0, 10.0, 10.0, 1e25, 1e25, 10.0]
        st = UInt8[
            JClp.CLP_STATUS_SUPER_BASIC, # 1: snaps to AT_LOWER (|0-0| <= tol)
            JClp.CLP_STATUS_SUPER_BASIC, # 2: snaps to AT_UPPER (|10-10| <= tol)
            JClp.CLP_STATUS_SUPER_BASIC | JClp.CLP_STATUS_FLAGGED_BIT, # 3: flagged -> skipped
            JClp.CLP_STATUS_SUPER_BASIC, # 4: free, |dj|=1e-9 <= 1e-7 -> set IS_FREE & continue
            JClp.CLP_STATUS_SUPER_BASIC, # 5: free, |dj|=2.0 > 1e-7 -> set IS_FREE & returned!
            JClp.CLP_STATUS_SUPER_BASIC, # 6: superBasic -> returned on next call!
        ]
        r1, ff1, nnz1, mso1 = JClp.clp_primal_next_super_basic!(
            cval, cidx, 0, dj, sol, lo, hi, st, ntotal, 1, 0, 0, 1e-7, 1e-7,
        )
        # With first_free=0 (-1 in C++), init call primes first_free=5 and returns 0 (-1)
        @test r1 == 0 && ff1 == 5 && nnz1 == 0 && mso1 == 0
        @test (st[1] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_LOWER
        @test (st[2] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_UPPER
        @test (st[4] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_IS_FREE
        @test (st[5] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_IS_FREE

        # Next call starting at ff1=5 (IS_FREE, not SUPER_BASIC) returns 0 and advances to 6
        r2, ff2, _, _ = JClp.clp_primal_next_super_basic!(
            cval, cidx, 0, dj, sol, lo, hi, st, ntotal, 1, ff1, 0, 1e-7, 1e-7,
        )
        @test r2 == 0 && ff2 == 6

        # Call at ff2=6 (SUPER_BASIC) returns 6 and exhausts list (ff3 == 0)
        r3, ff3, _, _ = JClp.clp_primal_next_super_basic!(
            cval, cidx, 0, dj, sol, lo, hi, st, ntotal, 1, ff2, 0, 1e-7, 1e-7,
        )
        @test r3 == 6 && ff3 == 0

        # Chain of two SUPER_BASIC vars starting at first_free=1 -> returns 1, advances to 6, then returns 6
        st_chain = UInt8[
            JClp.CLP_STATUS_SUPER_BASIC,
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_SUPER_BASIC | JClp.CLP_STATUS_FLAGGED_BIT,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_SUPER_BASIC,
        ]
        rc1, ffc1, _, _ = JClp.clp_primal_next_super_basic!(
            cval, cidx, 0, dj, Float64[3.0, 0.0, 5.0, 0.0, 0.0, 3.0], lo, hi, st_chain, ntotal, 1, 1, 0, 1e-7, 1e-7,
        )
        @test rc1 == 1 && ffc1 == 6

        # 2. Sorted list (super_basic_type = 3 then 2, TRY_VALUES_PASS == 0)
        lo_s = Float64[0.0, 0.0, 0.0, -1e35, 0.0, 0.0]
        hi_s = Float64[10.0, 10.0, 10.0, 1e35, 10.0, 10.0]
        # distances:
        # var 1: sol=0.0 -> snapped AT_LOWER
        # var 2: sol=2.0 -> -min(2.0, 0.8*8.0) = -2.0
        # var 3: sol=9.0 -> -min(9.0, 0.8*1.0) = -0.8 (closest to bound -> largest key -> popped FIRST!)
        # var 4: free (lo < -1e30 && hi > 1e30) -> set IS_FREE
        # var 5: sol=5.0 -> -min(5.0, 0.8*5.0) = -4.0 (farthest -> popped LAST!)
        # var 6: flagged -> skipped
        sol_s = Float64[1e-9, 2.0, 9.0, 0.0, 5.0, 3.0]
        st_s = UInt8[
            JClp.CLP_STATUS_SUPER_BASIC,
            JClp.CLP_STATUS_SUPER_BASIC,
            JClp.CLP_STATUS_SUPER_BASIC,
            JClp.CLP_STATUS_SUPER_BASIC,
            JClp.CLP_STATUS_SUPER_BASIC,
            JClp.CLP_STATUS_SUPER_BASIC | JClp.CLP_STATUS_FLAGGED_BIT,
        ]
        fill!(cval, 0.0)
        fill!(cidx, Int32(0))
        mso = JClp._SP_SORT_VALUES_BIT
        s1, ff_s1, nnz_s1, mso = JClp.clp_primal_next_super_basic!(
            cval, cidx, 0, dj, sol_s, lo_s, hi_s, st_s, ntotal, 3, 0, mso, 1e-7, 1e-7,
        )
        @test s1 == 3 && ff_s1 == 3 && nnz_s1 == 2
        @test sol_s[1] == 0.0 && (st_s[1] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_AT_LOWER
        @test (st_s[4] & JClp.CLP_STATUS_MASK) == JClp.CLP_STATUS_IS_FREE
        @test cval == zeros(Float64, ntotal)
        st_s[s1] = JClp.CLP_STATUS_BASIC # pivoted into basis

        s2, ff_s2, nnz_s2, mso = JClp.clp_primal_next_super_basic!(
            cval, cidx, nnz_s1, dj, sol_s, lo_s, hi_s, st_s, ntotal, 2, ff_s1, mso, 1e-7, 1e-7,
        )
        @test s2 == 2 && nnz_s2 == 1
        st_s[s2] = JClp.CLP_STATUS_BASIC

        s3, ff_s3, nnz_s3, mso = JClp.clp_primal_next_super_basic!(
            cval, cidx, nnz_s2, dj, sol_s, lo_s, hi_s, st_s, ntotal, 2, ff_s2, mso, 1e-7, 1e-7,
        )
        @test s3 == 5 && nnz_s3 == 0
        st_s[s3] = JClp.CLP_STATUS_BASIC

        s4, ff_s4, nnz_s4, mso = JClp.clp_primal_next_super_basic!(
            cval, cidx, nnz_s3, dj, sol_s, lo_s, hi_s, st_s, ntotal, 2, ff_s3, mso, 1e-7, 1e-7,
        )
        @test s4 == 0 && ff_s4 == 0 && nnz_s4 == 0 && mso == 0

        # Reference check for _sp_coin_sort_2! against pure-Julia sort on n=50 with ties
        n_sort = 50
        k1 = Float64[-(i % 7) * 0.5 for i in 1:n_sort]
        v1 = Int32[i for i in 1:n_sort]
        k2 = copy(k1)
        v2 = copy(v1)
        JClp._sp_coin_sort_2!(k1, v1, n_sort)
        @test issorted(k1[1:n_sort])
        @test k1[1:n_sort] == sort(k2[1:n_sort])
        @test sort!(collect(zip(k1[1:n_sort], v1[1:n_sort]))) == sort!(collect(zip(k2[1:n_sort], v2[1:n_sort])))

        function _alloc_nsb(cval, cidx, dj, sol, lo, hi, st)
            a1 = @allocated JClp.clp_primal_next_super_basic!(
                cval, cidx, 0, dj, sol, lo, hi, st, 6, 1, 0, 0, 1e-7, 1e-7,
            )
            a2 = @allocated JClp.clp_primal_next_super_basic!(
                cval, cidx, 0, dj, sol, lo, hi, st, 6, 3, 0, 0, 1e-7, 1e-7,
            )
            return a1 + a2
        end
        _alloc_nsb(cval, cidx, dj, sol_s, lo_s, hi_s, st_s)
        @test _alloc_nsb(cval, cidx, dj, sol_s, lo_s, hi_s, st_s) == 0
    end

    @testset "clp_primal_row! (all paths)" begin
        # Setup 4 rows, 4 cols => ntotal = 8
        nrows, ncols = 4, 4
        ntotal = nrows + ncols
        pivot_var = Int32[5, 6, 7, 8] # rows hold slacks 5..8, cols are 1..4

        # 1. Normal pivot path (col 1 entering, dir = 1, dualIn = -2.0)
        # Row 1: alpha = 2.0, sol[5]=4.0, lo[5]=0.0 -> hits at theta=2.0
        # Row 2: alpha = -1.0, sol[6]=1.0, hi[6]=4.0 -> hits at theta=3.0
        # Row 3: alpha = 0.5, sol[7]=10.0, lo[7]=0.0 -> hits at theta=20.0
        row_val = Float64[2.0, -1.0, 0.5, 0.0]
        row_idx = Int32[1, 2, 3, 0]
        rhs_val = zeros(Float64, nrows)
        rhs_idx = zeros(Int32, nrows)
        spare_val = zeros(Float64, nrows)
        spare_idx = zeros(Int32, nrows)
        cost = Float64[-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        sol = Float64[0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 10.0, 1.0]
        lo = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hi = Float64[1e25, 1e25, 1e25, 1e25, 10.0, 4.0, 100.0, 10.0]
        dj = Float64[-2.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4]
        status = fill(JClp.CLP_STATUS_BASIC, ntotal)
        status[1] = JClp.CLP_STATUS_AT_LOWER
        nlc_st = fill(JClp.CLP_NLC_FEASIBLE | (JClp.CLP_NLC_SAME << 4), ntotal)
        nlc_bound = zeros(Float64, ntotal)

        res1 = JClp.clp_primal_row!(
            row_val, row_idx, 3,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost, sol, lo, hi, dj, status,
            nlc_st, nlc_bound, 100.0, 0, 1.0,
            0, 1, 1, 1, 0.0, 1e25, 0.0, -2.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-9, 0, 0, 0,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        piv_r, seq_out, dir_in, dir_out, th, al, dj_in, val_out, lo_out, up_out, dj_out, ptol, avg_th = res1
        @test piv_r == 1 && seq_out == 5 && dir_in == 1 && dir_out == 1
        @test th ≈ 2.0 && al == 2.0 && dj_in == -2.0 && val_out == 4.0 && lo_out == 0.0 && dj_out == 0.1
        @test ptol == 1e-7
        @test avg_th ≈ max(2.0, 1e-12)
        @test spare_val == zeros(Float64, nrows)
        @test rhs_val == zeros(Float64, nrows)
        @test all(x -> JClp._nlc_curr_status(x) == JClp.CLP_NLC_SAME, nlc_st)
        @test JClp._sp_is_active(status[1])

        # 2. Bound flip (pivotRow == -2 in C++ => -1 in Julia)
        fill!(status, JClp.CLP_STATUS_BASIC)
        status[1] = JClp.CLP_STATUS_AT_LOWER
        row_val_f = Float64[0.1, 0.0, 0.0, 0.0]
        row_idx_f = Int32[1, 0, 0, 0]
        sol_f = Float64[0.0, 0.0, 0.0, 0.0, 10.0, 1.0, 1.0, 1.0]
        hi_f = Float64[3.0, 1e25, 1e25, 1e25, 100.0, 10.0, 10.0, 10.0]
        # col 1 enters from lower=0 to upper=3 (maxMovement=3.0 < 1e20), row 1 moves by 0.3 <= 10 -> no blocking row -> flip!
        res_flip = JClp.clp_primal_row!(
            row_val_f, row_idx_f, 1,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost, sol_f, lo, hi_f, dj, status,
            nlc_st, nlc_bound, 100.0, 0, 1.0,
            0, 1, 1, 1, 0.0, 3.0, 0.0, -2.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-9, 0, 0, 0,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        @test res_flip[1] == -1 # C++ -2
        @test res_flip[2] == 1  # sequence_out == sequence_in
        @test res_flip[4] == -1 # to upper bound
        @test res_flip[5] == 3.0
        @test res_flip[6] == 0.0

        # 3. Unbounded (pivotRow == -1 in C++ => 0 in Julia), both feasible and infeasible cases
        row_val_u = Float64[-1.0, 0.0, 0.0, 0.0] # alpha=-1 with dir=1 goes up, hi[5]=1e35 -> never blocks!
        hi_u = Float64[1e35, 1e35, 1e35, 1e35, 1e35, 1e35, 1e35, 1e35]
        res_unb = JClp.clp_primal_row!(
            row_val_u, row_idx_f, 1,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost, sol_f, lo, hi_u, dj, status,
            nlc_st, nlc_bound, 100.0, 0, 1.0,
            0, 1, 1, 1, 0.0, 1e35, 0.0, -2.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-9, 0, 0, 0,
            0, 0, 0.0, -1.0, 0.0, 0.0, 0.0,
        )
        @test res_unb[1] == 0 # C++ -1
        @test res_unb[8] == floatmax(Float64)

        # Infeasible unbounded case (nlc_num_infeas > 0)
        row_val_iu = Float64[1.0, 0.0, 0.0, 0.0] # dir=-1, way=-1 -> alpha*way = -1 -> sol[5]=10 going to upper=15 at theta=5!
        # Wait: make maximum_movement > 1e20 (dir=-1, value_in=0, lower_in=-1e35), and no candidates in pass 1:
        # Use row 1 with alpha = -1.0, way = -1 => alpha*way = +1.0 -> going to lo[5] = 0.0 from sol[5] = 5.0? No, that would pivot!
        # Wait: if pivotRow_ < 0 && maximum_movement > 1e20 happens when upper_theta == maximum_movement (e.g. alpha < acceptablePivot)
        row_val_iu[1] = 1e-12 # < acceptablePivot (0.1 * 1e-6 = 1e-7) so won't block!
        sol_iu = Float64[0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 1.0, 1.0]
        res_iu = JClp.clp_primal_row!(
            row_val_iu, row_idx_f, 1,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost, sol_iu, lo, hi_u, dj, status,
            nlc_st, nlc_bound, 100.0, 1, 1.0,
            0, 1, 1, 1, 0.0, 1e35, 0.0, -2.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-6, 0, 0, 0,
            0, 0, 0.0, -1.0, 0.0, 0.0, 0.0,
        )
        @test res_iu[1] == 0
        @test res_iu[8] == floatmax(Float64) # tentative_theta = 5e12 > 1e10 -> stays COIN_DBL_MAX
        # Now with alpha = 1e-12? No, make tentative_theta < 1e10: alpha = 1e-8 (still < acc_piv = 1e-7) and sol[5]=5.0 -> tentative_theta = 5e8 < 1e10!
        row_val_iu[1] = 1e-8
        res_iu2 = JClp.clp_primal_row!(
            row_val_iu, row_idx_f, 1,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost, sol_iu, lo, hi_u, dj, status,
            nlc_st, nlc_bound, 100.0, 1, 1.0,
            0, 1, 1, 1, 0.0, 1e35, 0.0, -2.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-6, 0, 0, 0,
            0, 0, 0.0, -1.0, 0.0, 0.0, 0.0,
        )
        @test res_iu2[1] == 0
        @test res_iu2[8] ≈ 5.0e8

        # 4. Degenerate / small-theta (|theta| <= 1e-6) with special_options=0 vs exactOutgoing (4)
        row_val_d = Float64[1.0, 0.0, 0.0, 0.0]
        sol_d = Float64[0.0, 0.0, 0.0, 0.0, 5e-7, 1.0, 1.0, 1.0] # theta = 5e-7 <= 1e-6, > minimum_theta (1e-12)
        res_deg0 = JClp.clp_primal_row!(
            row_val_d, row_idx_f, 1,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost, sol_d, lo, hi, dj, status,
            nlc_st, nlc_bound, 100.0, 0, 1.0,
            0, 1, 1, 1, 0.0, 1e25, 0.0, -2.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-9, 0, 0, 0,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        @test res_deg0[1] == 1 && res_deg0[5] ≈ 5e-7
        @test res_deg0[9] == 5e-7 - 5e-7 * 1.0 # NOT snapped (new_value)

        # With special_options=4 (exactOutgoing), snaps to lo[5]=0.0 even when |theta| <= 1e-6
        res_deg4 = JClp.clp_primal_row!(
            row_val_d, row_idx_f, 1,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost, sol_d, lo, hi, dj, status,
            nlc_st, nlc_bound, 100.0, 0, 1.0,
            0, 1, 1, 1, 0.0, 1e25, 0.0, -2.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-9, 0, 0, 4,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        @test res_deg4[9] == 0.0

        # 5. CLP_MOVEMENT == 2 re-scan (largestAlpha < 1e-3)
        # nlc_avg_theta = 0.01 => tentative_theta = 0.1
        # Row 1: alpha = 5e-4, sol[5] = 1e-5 => possible with tentative_theta=0.1, largest_alpha = 5e-4 < 1e-3!
        # Row 2: alpha = 1.0, sol[6] = 2.0 => NOT possible at tentative_theta=0.1, BUT possible in re-scan with maximum_movement=100.0!
        # dual_check ~ 100 => Row 1 alone has totalThru*100 = 0.05 < 100, Row 1+Row 2 has totalThru*100 = 100.05 > 100!
        fill!(status, JClp.CLP_STATUS_BASIC)
        status[1] = JClp.CLP_STATUS_AT_LOWER
        row_val_m2 = Float64[5e-4, 1.0, 0.0, 0.0]
        row_idx_m2 = Int32[1, 2, 0, 0]
        sol_m2 = Float64[0.0, 0.0, 0.0, 0.0, 1e-5, 2.0, 1.0, 1.0]
        cost_m2 = Float64[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        res_m2 = JClp.clp_primal_row!(
            row_val_m2, row_idx_m2, 2,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost_m2, sol_m2, lo, hi, dj, status,
            nlc_st, nlc_bound, 100.0, 0, 0.01,
            0, 1, 1, 1, 0.0, 100.0, 0.0, -1.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-9, 0, 0, 0,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        @test res_m2[1] == 2 && res_m2[2] == 6 && res_m2[6] == 1.0 && res_m2[5] ≈ 2.0

        # 6. valuesPass == 1 (recomputes dual_in & direction_in)
        fill!(status, JClp.CLP_STATUS_BASIC)
        status[1] = JClp.CLP_STATUS_SUPER_BASIC
        cost_vp = Float64[1.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0] # dj = 1 - 1*(-2) = +3.0 > dual_tol => direction_in = -1!
        sol_vp = Float64[5.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0]
        lo_vp = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hi_vp = Float64[10.0, 10.0, 10.0, 10.0, 4.0, 10.0, 10.0, 10.0]
        # With dir_in=-1, row 1 alpha=1.0 => way*alpha = -1.0 => sol[5]=2.0 goes to upper=4.0 at theta=-2.0!
        res_vp = JClp.clp_primal_row!(
            Float64[1.0, 0.0, 0.0, 0.0], Int32[1, 0, 0, 0], 1,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost_vp, sol_vp, lo_vp, hi_vp, dj, status,
            nlc_st, nlc_bound, 100.0, 0, 1.0,
            1, 1, 1, 1, 0.0, 10.0, 5.0, 0.0,
            1e-7, 1e-7, 100.0, 1e15, 1e-9, 0, 0, 0,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        @test res_vp[1] == 1 && res_vp[3] == -1 && res_vp[4] == -1 && res_vp[5] ≈ -2.0 && res_vp[7] ≈ 3.0

        # 7. Multi-step iTry bound-flip inside primalRow
        # dual_in = -5.0, nlc_infeas_weight = 2.0, infeas_cost = 2.0
        # Row 1: alpha=1.0, sol[5]=1.0 -> blocks first at theta=1.0, delta_cost=2.0 < 4.95 -> continues to iTry=2!
        # Row 2: alpha=2.0, sol[6]=4.0 -> blocks second at theta=2.0, delta_cost=4.0, total=6.0 >= 4.95 -> picks row 2!
        fill!(status, JClp.CLP_STATUS_BASIC)
        status[1] = JClp.CLP_STATUS_AT_LOWER
        res_itry = JClp.clp_primal_row!(
            Float64[1.0, 2.0, 0.0, 0.0], Int32[1, 2, 0, 0], 2,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, Float64[-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            Float64[0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 1.0, 1.0], lo, hi, dj, status,
            nlc_st, nlc_bound, 2.0, 0, 10.0,
            0, 1, 1, 1, 0.0, 100.0, 0.0, -5.0,
            1e-7, 1e-7, 2.0, 1e15, 1e-9, 0, 0, 0,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        @test res_itry[1] == 2 && res_itry[2] == 6 && res_itry[5] ≈ 2.0 && res_itry[6] == 2.0

        # 8. goBackOne when pivot becomes < 0.1 * best_ever_pivot & < 1e-3
        # Row 1: alpha=1.0, sol[5]=1.0 -> iTry=1 sets last_pivot_row=1, last_theta=1.0, best_ever_pivot=1.0
        # Row 2: alpha=5e-4, sol[6]=0.002 -> iTry=2 sees best_pivot=5e-4 < 0.1*1.0 && < 1e-3 -> goBackOne -> row 1!
        fill!(status, JClp.CLP_STATUS_BASIC)
        status[1] = JClp.CLP_STATUS_AT_LOWER
        res_gbo = JClp.clp_primal_row!(
            Float64[1.0, 5e-4, 0.0, 0.0], Int32[1, 2, 0, 0], 2,
            rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, Float64[-5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            Float64[0.0, 0.0, 0.0, 0.0, 1.0, 0.002, 1.0, 1.0], lo, hi, dj, status,
            nlc_st, nlc_bound, 2.0, 0, 10.0,
            0, 1, 1, 1, 0.0, 100.0, 0.0, -5.0,
            1e-7, 1e-7, 200.0, 1e15, 1e-9, 0, 0, 0,
            0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )
        @test res_gbo[1] == 1 && res_gbo[2] == 5 && res_gbo[5] ≈ 1.0

        # Zero allocation check
        function _alloc_pr(
            row_val, row_idx, rhs_val, rhs_idx, spare_val, spare_idx,
            pivot_var, cost, sol, lo, hi, dj, status, nlc_st, nlc_bound,
        )
            return @allocated JClp.clp_primal_row!(
                row_val, row_idx, 3,
                rhs_val, rhs_idx, spare_val, spare_idx,
                pivot_var, cost, sol, lo, hi, dj, status,
                nlc_st, nlc_bound, 100.0, 0, 1.0,
                0, 1, 1, 1, 0.0, 1e25, 0.0, -2.0,
                1e-7, 1e-7, 100.0, 1e15, 1e-9, 0, 0, 0,
                0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
            )
        end
        _alloc_pr(row_val, row_idx, rhs_val, rhs_idx, spare_val, spare_idx, pivot_var, cost, sol, lo, hi, dj, status, nlc_st, nlc_bound)
        @test _alloc_pr(row_val, row_idx, rhs_val, rhs_idx, spare_val, spare_idx, pivot_var, cost, sol, lo, hi, dj, status, nlc_st, nlc_bound) == 0
    end

    @testset "clp_update_primals_in_primal!" begin
        nrows, ncols = 3, 2
        ntotal = nrows + ncols
        pivot_var = Int32[3, 4, 5]

        # 1. Normal path (values_pass = 0):
        # row 1 (seq 3): active, alpha=2.0, theta=2.0 -> sol=1.0 - 4.0 = -3.0 (< lo=0 -> BELOW_LOWER, diff=100, dj=-100)
        # row 2 (seq 4): NOT active, theta > 0 -> active check skipped
        # row 3 (seq 5): active, alpha=1.0, theta=2.0 -> sol=2.0 - 2.0 = 0.0 (== lo=0, pivot_row=3, feasible -> diff=0, appended!)
        rval = Float64[2.0, 1.0, 1.0]
        ridx = Int32[1, 2, 3]
        sol = Float64[0.0, 0.0, 1.0, 5.0, 2.0]
        lo = zeros(Float64, ntotal)
        hi = fill(10.0, ntotal)
        cost2 = zeros(Float64, ntotal)
        cost = copy(cost2)
        dj = zeros(Float64, ntotal)
        st = fill(JClp.CLP_STATUS_BASIC, ntotal)
        st[1] = JClp.CLP_STATUS_ACTIVE_BIT | JClp.CLP_STATUS_AT_LOWER
        st[3] = JClp.CLP_STATUS_ACTIVE_BIT | JClp.CLP_STATUS_BASIC
        nlc_st = fill(JClp.CLP_NLC_FEASIBLE | (JClp.CLP_NLC_SAME << 4), ntotal)
        nlc_bound = zeros(Float64, ntotal)

        new_nnz, obj_chg, nlc_chg, n_inf = JClp.clp_update_primals_in_primal!(
            rval, ridx, 3, pivot_var,
            sol, lo, hi, cost, dj, st,
            nlc_st, nlc_bound, cost2, 100.0, 0,
            3, 5, -2.5, 2.0, 10.0, 1e-7, 0,
        )
        @test new_nnz == 2
        @test ridx[1:2] == Int32[1, 3]
        @test rval[1] == 100.0
        @test rval[2] == 2.5 # -(-2.5 + (0 - 0))
        @test rval[3] == 0.0
        @test sol[3] ≈ -3.0 && dj[3] == -100.0
        @test nlc_chg ≈ -3.0 * 100.0
        @test obj_chg ≈ 10.0 - 300.0
        @test n_inf == 1
        @test !JClp._sp_is_active(st[1]) && !JClp._sp_is_active(st[3])

        # 2. values_pass == 1 updates even inactive rows and clears active bits
        rval_v = Float64[1.0, -2.0]
        ridx_v = Int32[2, 3]
        sol_v = Float64[0.0, 0.0, 5.0, 1.0, 9.0]
        cost_v = zeros(Float64, ntotal)
        dj_v = zeros(Float64, ntotal)
        st_v = fill(JClp.CLP_STATUS_BASIC, ntotal) # row 2 NOT active!
        nlc_stv = fill(JClp.CLP_NLC_FEASIBLE | (JClp.CLP_NLC_SAME << 4), ntotal)
        nlc_bv = zeros(Float64, ntotal)
        new_nnz_v, obj_v, nlc_v, n_inf_v = JClp.clp_update_primals_in_primal!(
            rval_v, ridx_v, 2, pivot_var,
            sol_v, lo, hi, cost_v, dj_v, st_v,
            nlc_stv, nlc_bv, cost2, 100.0, 0,
            -1, 1, -1.0, 2.0, 0.0, 1e-7, 1,
        )
        @test new_nnz_v == 2 # row 2 -> -1.0 (below 0), row 3 -> 13.0 (above 10)
        @test ridx_v[1:2] == Int32[2, 3]
        @test rval_v[1] == 100.0 && rval_v[2] == -100.0
        @test n_inf_v == 2
        @test nlc_v ≈ (-1.0 * 100.0) + (13.0 * (-100.0))
        @test obj_v ≈ nlc_v

        # 3. Negative theta < 0 processes inactive rows AND pivot_row already in change list subtracts d_in
        rval_neg = Float64[2.0]
        ridx_neg = Int32[1]
        sol_neg = Float64[0.0, 0.0, 1.0, 0.0, 0.0]
        cost_neg = Float64[0.0, 0.0, 3.0, 0.0, 0.0] # will change cost when going below lower: diff=100, new cost=-100
        cost2_neg = zeros(Float64, ntotal)
        dj_neg = zeros(Float64, ntotal)
        st_neg = fill(JClp.CLP_STATUS_BASIC, ntotal) # row 1 NOT active, but theta < 0!
        nlc_stn = fill(JClp.CLP_NLC_FEASIBLE | (JClp.CLP_NLC_SAME << 4), ntotal)
        nlc_bn = zeros(Float64, ntotal)
        # theta = -2.0, alpha = 2.0 => change = -4.0 <= 0 -> going UP: val = 1 - (-4) = 5.0 (in [0,10]). Wait: make sol=7.0 -> val=11.0 > 10 (ABOVE_UPPER, diff=-100)
        sol_neg[3] = 7.0
        new_nnz_n, obj_n, nlc_n, n_inf_n = JClp.clp_update_primals_in_primal!(
            rval_neg, ridx_neg, 1, pivot_var,
            sol_neg, lo, hi, cost_neg, dj_neg, st_neg,
            nlc_stn, nlc_bn, cost2_neg, 100.0, 0,
            1, 3, -1.5, -2.0, 0.0, 1e-7, 0,
        )
        # old_cost = 3.0, cost[3] becomes +100.0, diff = 3.0 - 100.0 = -97.0; d_in = -1.5 + (3.0 - 100.0) = -98.5
        # pivot_pos == 1 -> rval_neg[1] = -97.0 - (-98.5) = +1.5
        @test new_nnz_n == 1 && ridx_neg[1] == Int32(1)
        @test rval_neg[1] ≈ 1.5
        @test n_inf_n == 1

        function _alloc_up(rval, ridx, pivot_var, sol, lo, hi, cost, dj, st, nlc_st, nlc_bound, cost2)
            return @allocated JClp.clp_update_primals_in_primal!(
                rval, ridx, 2, pivot_var,
                sol, lo, hi, cost, dj, st,
                nlc_st, nlc_bound, cost2, 100.0, 0,
                0, 0, -1.0, 1.0, 0.0, 1e-7, 0,
            )
        end
        rval_a = Float64[1.0, 1.0]
        ridx_a = Int32[1, 2]
        _alloc_up(rval_a, ridx_a, pivot_var, sol_v, lo, hi, cost_v, dj_v, st_v, nlc_stv, nlc_bv, cost2)
        @test _alloc_up(rval_a, ridx_a, pivot_var, sol_v, lo, hi, cost_v, dj_v, st_v, nlc_stv, nlc_bv, cost2) == 0
    end
end

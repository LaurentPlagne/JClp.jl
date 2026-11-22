using JClp
using Test

@testset "B11a-1: ClpSimplexDual Hot Kernels (Pure 1-based Julia)" begin
    @testset "clp_dual_column0!" begin
        nrows, ncols = 3, 4
        ntotal = nrows + ncols

        # Fast path: (more_special_options & 8) != 0
        row_val = Float64[0.5, -2.0, 1.0]
        row_idx = Int32[1, 2, 3]
        col_val = Float64[0.8, -0.4, 2.0, 1e-10]
        col_idx = Int32[1, 2, 3, 4]
        spare_val = zeros(Float64, ntotal)
        spare_idx = zeros(Int32, ntotal)

        # Vars: 1..4 cols, 5..7 rows (1-based)
        dj = Float64[1.0, -2.0, 0.5, 0.1, 0.2, -3.0, 0.4]
        status = UInt8[
            JClp.CLP_STATUS_AT_LOWER, # col 1: alpha=0.8>0 -> cand, ratio=(1+tol)/0.8
            JClp.CLP_STATUS_AT_UPPER, # col 2: mult=-1, alpha=0.4>0 -> cand, ratio=(2+tol)/0.4
            JClp.CLP_STATUS_BASIC,    # col 3: skipped
            JClp.CLP_STATUS_AT_LOWER, # col 4: alpha=1e-10 < acceptablePivot
            JClp.CLP_STATUS_AT_LOWER, # row 1 (seq 5): sequence_out -> skipped
            JClp.CLP_STATUS_AT_UPPER, # row 2 (seq 6): mult=-1, alpha=2.0 -> cand, ratio=(3+tol)/2
            JClp.CLP_STATUS_AT_LOWER, # row 3 (seq 7): alpha=1.0 -> cand, ratio=(0.4+tol)/1.0 (smallest!)
        ]
        solution = zeros(Float64, ntotal)
        lower = zeros(Float64, ntotal)
        upper = ones(Float64, ntotal)

        nrem, uth, bfree, seq_in, th, al, nfake = JClp.clp_dual_column0!(
            row_val, row_idx, 3,
            col_val, col_idx, 4,
            spare_val, spare_idx,
            dj, status, solution, lower, upper,
            ncols, 8, 5, 1e-7, 1e7, 1e-6, 0, 0.0, 0.0, 0,
        )
        @test nrem == 5
        @test uth ≈ (0.4 + 1e-7) / 1.0
        @test bfree == 0.0
        @test seq_in == 0
        @test nfake == 0
        @test spare_idx[1:5] == Int32[6, 7, 1, 2, 4]
        @test spare_val[1:5] ≈ Float64[-2.0, 1.0, 0.8, -0.4, 1e-10]

        alloc0 = @allocated JClp.clp_dual_column0!(
            row_val, row_idx, 3,
            col_val, col_idx, 4,
            spare_val, spare_idx,
            dj, status, solution, lower, upper,
            ncols, 8, 5, 1e-7, 1e7, 1e-6, 0, 0.0, 0.0, 0,
        )
        @test alloc0 == 0

        # Full path: (more_special_options & 8) == 0 with free/superBasic + badFree
        fill!(spare_val, 0.0)
        fill!(spare_idx, Int32(0))
        status2 = UInt8[
            JClp.CLP_STATUS_IS_FREE,     # col 1: |dj|=0 <= tol, |alpha|=1e-7 <= 1e-5 -> badFree
            JClp.CLP_STATUS_SUPER_BASIC, # col 2: |dj|=0.5 > tol, |alpha|=0.4 -> selected, fake bound
            JClp.CLP_STATUS_AT_UPPER,    # col 3: alpha=-2.0 -> cand
            JClp.CLP_STATUS_AT_LOWER,    # col 4: alpha=0.5 -> cand
            JClp.CLP_STATUS_IS_FIXED,    # row 1 (seq 5): skipped
            JClp.CLP_STATUS_AT_LOWER,    # row 2 (seq 6): alpha=-1.0 -> not cand
            JClp.CLP_STATUS_AT_UPPER,    # row 3 (seq 7): alpha=1.0 -> not cand
        ]
        dj2 = Float64[0.0, 0.5, -1.0, 0.2, 0.0, 1.0, -1.0]
        sol2 = Float64[0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        lo2 = fill(-1e20, ntotal)
        hi2 = fill(1e20, ntotal)
        col_val2 = Float64[1e-7, 0.4, -2.0, 0.5]
        row_val2 = Float64[1.0, -1.0, 1.0]

        nrem2, uth2, bfree2, seq_in2, th2, al2, nfake2 = JClp.clp_dual_column0!(
            row_val2, row_idx, 3,
            col_val2, col_idx, 4,
            spare_val, spare_idx,
            dj2, status2, sol2, lo2, hi2,
            ncols, 0, 0, 1e-7, 100.0, 1e-8, 0, 0.0, 0.0, 0,
        )
        @test bfree2 == 1e-7
        @test seq_in2 == 2
        @test al2 == 0.4
        @test th2 ≈ 0.5 / 0.4
        @test nfake2 == 1
        @test JClp._sd_get_fake_bound(status2[2]) == JClp.CLP_FAKE_BOTH
        @test JClp._sd_get_status(status2[2]) == JClp.CLP_STATUS_AT_LOWER
        @test lo2[2] == 3.0 && hi2[2] == 103.0
        @test nrem2 == 2
        @test uth2 ≈ min((-1.0 - 1e-7) / (-2.0), (0.2 + 1e-7) / 0.5)
    end

    @testset "clp_dual_column!" begin
        nrows, ncols = 2, 4
        ntotal = nrows + ncols
        row_val = Float64[0.0, 0.0]
        row_idx = Int32[1, 2]
        col_val = Float64[1.0, 2.0, -1.0, 0.5]
        col_idx = Int32[1, 2, 3, 4]
        spare_val = zeros(Float64, ntotal)
        spare_idx = zeros(Int32, ntotal)
        spare2_val = zeros(Float64, ntotal)
        spare2_idx = zeros(Int32, ntotal)

        # Infeasible when no candidates and sequence_in == 0
        dj_inf = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        st_inf = fill(JClp.CLP_STATUS_BASIC, ntotal)
        lo = zeros(Float64, ntotal)
        hi = ones(Float64, ntotal)
        sol = zeros(Float64, ntotal)
        cost = copy(dj_inf)
        res_inf = JClp.clp_dual_column!(
            row_val, row_idx, 0, col_val, col_idx, 4,
            spare_val, spare_idx, spare2_val, spare2_idx,
            dj_inf, cost, sol, lo, hi, st_inf,
            nrows, ncols,
            1e-7, 1e7, -1.0, 0.0, 1.0, 1e15, 0.0,
            0, 8, 0, 0, 1e-8, 1e-8,
            0, 0, 0, 0.0, 0.0, 0.0,
            0, 0.0, 0, 0,
        )
        @test res_inf[1] == 0.0
        @test res_inf[2] == 0

        # Flip-flop ratio test:
        # col 1: at lower, alpha=1.0, dj=0.5, range=1.0 -> flips at theta=0.5 (thru=1.0)
        # col 2: at lower, alpha=2.0, dj=4.0, range=2.0 -> flips at theta=2.0 (thru=4.0)
        # col 3: at upper, alpha=-1.0, dj=-3.0, range=1.0 -> flips at theta=3.0
        # dual_out = -2.5 => |dual_out| = 2.5 -> col 1 flips, col 2 pivots!
        dj = Float64[0.5, 4.0, -3.0, 10.0, 0.0, 0.0]
        cost = copy(dj)
        status = UInt8[
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
        ]
        lo = Float64[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        hi = Float64[1.0, 2.0, 1.0, 1.0, 1.0, 1.0]
        sol = Float64[0.0, 0.0, 1.0, 0.5, 0.0, 0.0]

        res = JClp.clp_dual_column!(
            row_val, row_idx, 0, col_val, col_idx, 3,
            spare_val, spare_idx, spare2_val, spare2_idx,
            dj, cost, sol, lo, hi, status,
            nrows, ncols,
            1e-7, 1e7, -2.5, 0.0, 1.0, 1e15, 100.0,
            0, 8, 0, 0, 1e-8, 1e-8,
            0, 0, 0, 0.0, 0.0, 0.0,
            0, 0.0, 0, 0,
        )
        best_pos, seq_in, dir_in, th, al, lo_in, up_in, val_in, dj_in, acc_mem, nfake, nchg = res
        @test seq_in == 2
        @test dir_in == 1
        @test al == 2.0
        @test th ≈ 2.0
        @test best_pos == 2.0
        @test dj[2] ≈ th * al
        @test all(==(0.0), spare_val)
        @test all(==(0.0), spare2_val)

        # Pre-populated fused dualColumn0 (spare_int0 == -2)
        dj_f = Float64[0.5, 4.0, -3.0, 10.0, 0.0, 0.0]
        cost_f = copy(dj_f)
        spare_val[1:3] .= Float64[1.0, 2.0, -1.0]
        spare_idx[1:3] .= Int32[1, 2, 3]
        res_fused = JClp.clp_dual_column!(
            row_val, row_idx, 0, col_val, col_idx, 3,
            spare_val, spare_idx, spare2_val, spare2_idx,
            dj_f, cost_f, sol, lo, hi, status,
            nrows, ncols,
            1e-7, 1e7, -2.5, 0.0, 1.0, 1e15, 100.0,
            0, 8, 0, 0, 1e-8, 1e-8,
            -2, 0, 3, 0.5 + 1e-7, 0.0, 0.0,
            0, 0.0, 0, 0,
        )
        @test res_fused[2] == 2
        @test res_fused[5] == 2.0
        @test all(==(0.0), spare_val)
        @test all(==(0.0), spare2_val)

        # Pre-selected free variable (spare_int0 == -1) + small alpha bestPossible scan
        dj_s = Float64[0.0, 0.0, -3.0, 10.0, 0.0, 0.0]
        cost_s = copy(dj_s)
        st_s = copy(status)
        st_s[1] = JClp.CLP_STATUS_IS_FREE
        res_free = JClp.clp_dual_column!(
            row_val, row_idx, 0, Float64[1e-7, 2.0, -1.0], col_idx, 3,
            spare_val, spare_idx, spare2_val, spare2_idx,
            dj_s, cost_s, sol, lo, hi, st_s,
            nrows, ncols,
            1e-7, 1e7, -2.5, 0.0, 1.0, 1e15, 100.0,
            0, 0, 0, 0, 1e-8, 1e-8,
            -1, 1, 0, 1e31, 0.0, 1e-7,
            0, 0.0, 0, 0,
        )
        @test res_free[2] == 1
        @test res_free[1] == 2.0 # bestPossible recomputed across col 2 (alpha=2.0)

        # Zero allocation check (inside function to avoid top-level tuple boxing)
        function _alloc_dc(
            row_val, row_idx, col_val, col_idx,
            spare_val, spare_idx, spare2_val, spare2_idx,
            dj, cost, sol, lo, hi, status, nrows, ncols,
        )
            return @allocated JClp.clp_dual_column!(
                row_val, row_idx, 0, col_val, col_idx, 3,
                spare_val, spare_idx, spare2_val, spare2_idx,
                dj, cost, sol, lo, hi, status,
                nrows, ncols,
                1e-7, 1e7, -2.5, 0.0, 1.0, 1e15, 100.0,
                0, 8, 0, 0, 1e-8, 1e-8,
                0, 0, 0, 0.0, 0.0, 0.0,
                0, 0.0, 0, 0,
            )
        end
        dj_a = Float64[0.5, 4.0, -3.0, 10.0, 0.0, 0.0]
        cost_a = copy(dj_a)
        _alloc_dc(row_val, row_idx, col_val, col_idx, spare_val, spare_idx, spare2_val, spare2_idx, dj_a, cost_a, sol, lo, hi, status, nrows, ncols)
        dj_a .= Float64[0.5, 4.0, -3.0, 10.0, 0.0, 0.0]
        cost_a .= dj_a
        @test _alloc_dc(row_val, row_idx, col_val, col_idx, spare_val, spare_idx, spare2_val, spare2_idx, dj_a, cost_a, sol, lo, hi, status, nrows, ncols) == 0

        # MORE_CAREFUL: badFree * theta > 10 * dual_tol with factor_pivots > 0 forces refactorization
        dj_bf = Float64[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        cost_bf = copy(dj_bf)
        st_bf = UInt8[
            JClp.CLP_STATUS_IS_FREE,
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
        ]
        # col 1 has |dj|=1.0 > tol, alpha=0.5 -> theta=2.0; col 2? Wait: make col 2 IS_FREE with |dj|=0, alpha=1e-6 -> badFree=1e-6, theta*badFree=2e-6 > 10*1e-7!
        st_bf[2] = JClp.CLP_STATUS_IS_FREE
        dj_bf[2] = 0.0
        res_bf = JClp.clp_dual_column!(
            row_val, row_idx, 0, Float64[0.5, 1e-6, 1.0], col_idx, 2,
            spare_val, spare_idx, spare2_val, spare2_idx,
            dj_bf, cost_bf, sol, lo, hi, st_bf,
            nrows, ncols,
            1e-7, 1e7, -2.5, 0.0, 1.0, 1e15, 100.0,
            0, 0, 2, 0, 1e-8, 1e-8,
            0, 0, 0, 0.0, 0.0, 0.0,
            0, 0.0, 0, 0,
        )
        @test res_bf[2] == 0       # sequence_in reset to 0 (-1 in C++)
        @test res_bf[10] == -1e-8  # acceptable_pivot_member negated!
        @test res_bf[1] == 0.0     # best_possible == 0.0

        # DONT_MOVE_OBJECTIVE clamping on exact cost modification
        dj_dm = Float64[2.0, 5.0, 0.0, 0.0, 0.0, 0.0]
        cost_dm = copy(dj_dm)
        sol_dm = Float64[100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        st_dm = UInt8[
            JClp.CLP_STATUS_IS_FREE,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
        ]
        # small_move = max(|0.01|, 1e-3) = 0.01; mod_val = theta*alpha - dj
        # Force non-zero mod_val via spare_int0 == -1 with mismatched spare_dbl2/3
        res_dm = JClp.clp_dual_column!(
            row_val, row_idx, 0, Float64[1.0], col_idx, 1,
            spare_val, spare_idx, spare2_val, spare2_idx,
            dj_dm, cost_dm, sol_dm, lo, hi, st_dm,
            nrows, ncols,
            1e-7, 1e7, -2.5, 0.0, 1.0, 1e15, 0.01,
            0, 0, 0, 0, 1e-8, 1e-8,
            -1, 1, 0, 1e31, 3.0, 1.0,
            0, 0.0, 0, 0,
        )
        # raw mod = 3.0*1.0 - 2.0 = 1.0; move_obj = 100.0 > small_move = 0.01 -> scaled to 0.01/100 = 1e-4
        @test cost_dm[1] ≈ 2.0 + 1e-4
        @test res_dm[12] == 1 # number_changed incremented
    end

    @testset "clp_update_duals_in_dual!, clp_update_duals_in_values_pass!, clp_flip_bounds!" begin
        nrows, ncols = 3, 2
        ntotal = nrows + ncols
        # Simple 3x2 matrix: col 1 = [1.0, -2.0, 0.0], col 2 = [0.0, 3.0, 1.0]
        mat_starts = Int32[1, 3, 5]
        mat_lengths = Int32[2, 2]
        mat_indices = Int32[1, 2, 2, 3]
        mat_elements = Float64[1.0, -2.0, 3.0, 1.0]

        # 1. Incremental updateDualsInDual (unscaled, fast mode)
        row_val = Float64[2.0, -0.5]
        row_idx = Int32[1, 3]
        col_val = Float64[1.5, -1.0]
        col_idx = Int32[1, 2]
        out_val = zeros(Float64, nrows)
        out_idx = zeros(Int32, nrows)
        dj = Float64[1.0, -0.5, 0.5, 0.0, -0.2]
        sol = Float64[0.0, 2.0, 1.0, 0.0, 0.0]
        lo = Float64[0.0, 0.0, 0.0, 0.0, 0.0]
        hi = Float64[2.0, 2.0, 1.0, 1.0, 3.0]
        cost = Float64[3.0, -1.0, 2.0, 0.0, 4.0]
        status = UInt8[
            JClp.CLP_STATUS_AT_LOWER, # col 1: 1.0 - 1.0*1.5 = -0.5 < -tol -> flips (mov=+2)
            JClp.CLP_STATUS_AT_UPPER, # col 2: -0.5 - 1.0*(-1.0) = +0.5 > tol -> flips (mov=-2)
            JClp.CLP_STATUS_AT_UPPER, # row 1 (seq 3): 0.5 - 1.0*2.0 = -1.5 -> (-1.5)*(-1)=+1.5 (no flip)
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_AT_LOWER, # row 3 (seq 5): -0.2 - 1.0*(-0.5) = +0.3 (no flip)
        ]

        tot_inf, r_inf, c_inf, o_nnz, d_obj, nfake = JClp.clp_update_duals_in_dual!(
            row_val, row_idx, 2, col_val, col_idx, 2,
            out_val, out_idx, 0,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            nothing, nothing,
            dj, sol, lo, hi, cost, status,
            nrows, ncols, 1.0, 1e-7, 0.0, 1e7, 8, 0, false,
        )
        @test tot_inf == 2 && r_inf == 0 && c_inf == 2
        @test col_idx[1:2] == Int32[1, 2]
        @test row_val[1:2] == [0.0, 0.0]
        @test col_val[1:2] == [0.0, 0.0]
        @test d_obj ≈ 2.0 * 3.0 + (-2.0) * (-1.0)
        # out_val = 2*[1,-2,0] + (-2)*[0,3,1] = [2, -10, -2]
        @test o_nnz == 3
        @test out_val[1:3] ≈ Float64[2.0, -10.0, -2.0]

        # Zero allocation check (unscaled incremental)
        function _alloc_inc(
            row_val, row_idx, col_val, col_idx, out_val, out_idx,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            dj, sol, lo, hi, cost, status, nrows, ncols,
        )
            return @allocated JClp.clp_update_duals_in_dual!(
                row_val, row_idx, 2, col_val, col_idx, 2,
                out_val, out_idx, 0,
                mat_starts, mat_lengths, mat_indices, mat_elements,
                nothing, nothing,
                dj, sol, lo, hi, cost, status,
                nrows, ncols, 1.0, 1e-7, 0.0, 1e7, 8, 0, false,
            )
        end
        _alloc_inc(row_val, row_idx, col_val, col_idx, out_val, out_idx, mat_starts, mat_lengths, mat_indices, mat_elements, dj, sol, lo, hi, cost, status, nrows, ncols)
        @test _alloc_inc(row_val, row_idx, col_val, col_idx, out_val, out_idx, mat_starts, mat_lengths, mat_indices, mat_elements, dj, sol, lo, hi, cost, status, nrows, ncols) == 0

        # 2. Scaled incremental updateDualsInDual with more_special_options = 0 + free var
        fill!(out_val, 0.0)
        row_scale = Float64[2.0, 0.5, 1.0]
        col_scale = Float64[0.5, 2.0]
        row_val2 = Float64[1.0]
        row_idx2 = Int32[1]
        col_val2 = Float64[1.0, 0.5]
        col_idx2 = Int32[1, 2]
        dj2 = Float64[-2.0, 1.0, 2.0, 0.0, 0.0] # col 1 at lower flips, col 2 free updated, row 1 at upper flips
        st2 = UInt8[
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_IS_FREE,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
        ]
        tot_inf_s, r_inf_s, c_inf_s, o_nnz_s, _, _ = JClp.clp_update_duals_in_dual!(
            row_val2, row_idx2, 1, col_val2, col_idx2, 2,
            out_val, out_idx, 0,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            row_scale, col_scale,
            dj2, sol, lo, hi, cost, st2,
            nrows, ncols, 1.0, 1e-7, 0.0, 1e7, 0, 0, false,
        )
        @test tot_inf_s == 2 && r_inf_s == 1 && c_inf_s == 1
        @test dj2[2] ≈ 0.5
        # row 1: mov = 0 - 1 = -1 -> out_val[1] += -(-1)=1; col 1: mov=2 -> col_scale[1]*2=1 -> out_val[1] += 1*2=2 (total 3), out_val[2] += -2*1*0.5=-1
        @test out_val[1] ≈ 3.0 && out_val[2] ≈ -1.0

        function _alloc_scaled(
            row_val2, row_idx2, col_val2, col_idx2, out_val, out_idx,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            row_scale, col_scale, dj2, sol, lo, hi, cost, st2, nrows, ncols,
        )
            return @allocated JClp.clp_update_duals_in_dual!(
                row_val2, row_idx2, 1, col_val2, col_idx2, 2,
                out_val, out_idx, 0,
                mat_starts, mat_lengths, mat_indices, mat_elements,
                row_scale, col_scale,
                dj2, sol, lo, hi, cost, st2,
                nrows, ncols, 1.0, 1e-7, 0.0, 1e7, 0, 0, false,
            )
        end
        _alloc_scaled(row_val2, row_idx2, col_val2, col_idx2, out_val, out_idx, mat_starts, mat_lengths, mat_indices, mat_elements, row_scale, col_scale, dj2, sol, lo, hi, cost, st2, nrows, ncols)
        @test _alloc_scaled(row_val2, row_idx2, col_val2, col_idx2, out_val, out_idx, mat_starts, mat_lengths, mat_indices, mat_elements, row_scale, col_scale, dj2, sol, lo, hi, cost, st2, nrows, ncols) == 0

        # 3. Full recompute with fake bounds creation + swap back + flipBounds
        fill!(out_val, 0.0)
        dj3 = Float64[-1.0, 0.0, 2.0, 0.0, 0.0]
        lo3 = Float64[0.0, -10.0, -1e20, 0.0, 0.0]
        hi3 = Float64[1e20, 0.0, 1.0, 1.0, 1.0]
        sol3 = Float64[0.0, -10.0, 1.0, 0.0, 0.0]
        st3 = UInt8[
            JClp.CLP_STATUS_AT_LOWER,                          # col 1: dj=-1 < -tol, range=1e20 > dualBound -> upperFake!
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_LOWER << 3), # col 2: dj=0 < tol, lowerFake -> swaps to AT_UPPER!
            JClp.CLP_STATUS_AT_UPPER,                          # row 1: dj=2 > tol, range=1e20 > dualBound -> lowerFake!
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_BASIC,
        ]
        tot_full, r_full, c_full, o_full, _, nfake_full = JClp.clp_update_duals_in_dual!(
            row_val, row_idx, 0, col_val, col_idx, 0,
            out_val, out_idx, 0,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            nothing, nothing,
            dj3, sol3, lo3, hi3, cost, st3,
            nrows, ncols, 0.0, 1e-7, 0.0, 50.0, 0, 0, true,
        )
        @test tot_full == 2
        @test r_full == 0 && c_full == 0 # cleared by clp_flip_bounds!
        @test nfake_full == 2
        @test JClp._sd_get_fake_bound(st3[1]) == JClp.CLP_FAKE_UPPER
        @test JClp._sd_get_status(st3[1]) == JClp.CLP_STATUS_AT_UPPER
        @test hi3[1] == 50.0 && sol3[1] == 50.0
        @test JClp._sd_get_status(st3[2]) == JClp.CLP_STATUS_AT_UPPER && sol3[2] == 0.0
        @test JClp._sd_get_fake_bound(st3[3]) == JClp.CLP_FAKE_LOWER
        @test JClp._sd_get_status(st3[3]) == JClp.CLP_STATUS_AT_LOWER
        @test lo3[3] == -49.0 && sol3[3] == -49.0

        # 4. updateDualsInValuesPass
        rv_vp = Float64[2.0, -1.0]
        ri_vp = Int32[1, 3]
        cv_vp = Float64[3.0, -2.0]
        ci_vp = Int32[1, 2]
        dj_vp = Float64[1.0, -0.5, 0.5, 0.0, 1.0]
        st_vp = UInt8[
            JClp.CLP_STATUS_AT_LOWER, # col 1: 1.0 - 1.0*3.0 = -2.0 < -tol -> clamped to 0.0
            JClp.CLP_STATUS_AT_UPPER, # col 2: -0.5 - 1.0*(-2.0) = +1.5 > tol -> clamped to 0.0
            JClp.CLP_STATUS_AT_UPPER, # row 1: 0.5 - 1.0*2.0 = -1.5 -> kept
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_AT_LOWER, # row 3: 1.0 - 1.0*(-1.0) = 2.0 -> kept
        ]
        JClp.clp_update_duals_in_values_pass!(
            rv_vp, ri_vp, 2, cv_vp, ci_vp, 2,
            dj_vp, st_vp, ncols, 1.0, 1e-7,
        )
        @test rv_vp == [0.0, 0.0] && cv_vp == [0.0, 0.0]
        @test dj_vp ≈ Float64[0.0, 0.0, -1.5, 0.0, 2.0]
        alloc_vp = @allocated JClp.clp_update_duals_in_values_pass!(
            rv_vp, ri_vp, 2, cv_vp, ci_vp, 2,
            dj_vp, st_vp, ncols, 1.0, 1e-7,
        )
        @test alloc_vp == 0

        # 5. flipBounds standalone
        sol_fb = zeros(Float64, ntotal)
        lo_fb = Float64[-1.0, -2.0, -3.0, -4.0, -5.0]
        hi_fb = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
        st_fb = UInt8[
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_IS_FREE,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_AT_LOWER,
        ]
        fb_ridx = Int32[1, 2, 3]
        fb_cidx = Int32[1, 2]
        JClp.clp_flip_bounds!(fb_ridx, 3, fb_cidx, 2, sol_fb, lo_fb, hi_fb, st_fb, ncols)
        @test JClp._sd_get_status(st_fb[1]) == JClp.CLP_STATUS_AT_UPPER && sol_fb[1] == 1.0
        @test JClp._sd_get_status(st_fb[2]) == JClp.CLP_STATUS_IS_FREE && sol_fb[2] == 0.0
        @test JClp._sd_get_status(st_fb[3]) == JClp.CLP_STATUS_AT_LOWER && sol_fb[3] == -3.0
        @test JClp._sd_get_status(st_fb[4]) == JClp.CLP_STATUS_BASIC
        @test JClp._sd_get_status(st_fb[5]) == JClp.CLP_STATUS_AT_UPPER && sol_fb[5] == 5.0
        alloc_fb = @allocated JClp.clp_flip_bounds!(fb_ridx, 3, fb_cidx, 2, sol_fb, lo_fb, hi_fb, st_fb, ncols)
        @test alloc_fb == 0
    end
end

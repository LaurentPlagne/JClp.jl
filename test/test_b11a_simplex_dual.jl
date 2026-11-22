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

@testset "B11a-2a: ClpSimplexDual Helpers & Pure Algorithmic Methods" begin
    @testset "clp_dual_next_super_basic!" begin
        dj = Float64[50.0, 0.0, 1e-6, 1e-4, -2e-4, 5.0]
        st = UInt8[
            JClp.CLP_STATUS_IS_FREE,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_IS_FREE,     # |dj| = 1e-6 <= 100 * 1e-7 = 1e-5
            JClp.CLP_STATUS_AT_LOWER,    # not free
            JClp.CLP_STATUS_IS_FREE,     # |dj| = 2e-4 > 1e-5 -> next!
            JClp.CLP_STATUS_SUPER_BASIC, # not IS_FREE
        ]
        @test JClp.clp_dual_next_super_basic!(dj, st, 6, 0, 1e-7) == (0, 0)
        ret1, ff1 = JClp.clp_dual_next_super_basic!(dj, st, 6, 1, 1e-7)
        @test ret1 == 1 && ff1 == 5
        ret2, ff2 = JClp.clp_dual_next_super_basic!(dj, st, 6, ff1, 1e-7)
        @test ret2 == 5 && ff2 == 0
        alloc_nsb = @allocated JClp.clp_dual_next_super_basic!(dj, st, 6, 1, 1e-7)
        @test alloc_nsb == 0
    end

    @testset "clp_dual_row_finish!" begin
        pivot_var = Int32[1, 2, 3, 4, 5]
        sol = Float64[12.0, -3.0, 0.2, 0.8, 0.5]
        lo = Float64[0.0, 0.0, 0.0, 0.0, 0.0]
        hi = Float64[10.0, 10.0, 1.0, 1.0, 1.0]
        dj = Float64[0.0, 0.0, 2.0, -3.0, 0.0]

        @test JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 0, false) == (0, 0.0, 0.0, 0.0, 0, 0.0)

        # Above upper bound
        @test JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 1, false) == (1, 12.0, 0.0, 10.0, -1, 2.0)
        # Below lower bound
        @test JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 2, false) == (2, -3.0, 0.0, 10.0, 1, 3.0)
        # Feasible, closer to lower
        seq3, v3, l3, u3, d3, du3 = JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 3, false)
        @test seq3 == 3 && d3 == 1 && du3 ≈ -0.2
        # Feasible, closer to upper / exact midpoint tie
        seq4, _, _, _, d4, du4 = JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 4, false)
        @test seq4 == 4 && d4 == -1 && du4 ≈ -0.2
        seq5, _, _, _, d5, du5 = JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 5, false)
        @test seq5 == 5 && d5 == -1 && du5 ≈ -0.5

        # Values pass (already_chosen = true)
        @test JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 3, true) == (3, 0.2, 0.0, 1.0, 1, 1.0e-6)
        @test JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 4, true) == (4, 0.8, 0.0, 1.0, -1, 1.0e-6)

        alloc_rf = @allocated JClp.clp_dual_row_finish!(pivot_var, sol, lo, hi, dj, 1, false)
        @test alloc_rf == 0
    end

    @testset "clp_dual_check_fake_bounds & clp_dual_number_at_fake_bound" begin
        st = UInt8[
            JClp.CLP_STATUS_BASIC | (JClp.CLP_FAKE_BOTH << 3),       # ignored
            JClp.CLP_STATUS_IS_FREE | (JClp.CLP_FAKE_BOTH << 3),     # ignored
            JClp.CLP_STATUS_SUPER_BASIC | (JClp.CLP_FAKE_LOWER << 3),# ignored
            JClp.CLP_STATUS_IS_FIXED | (JClp.CLP_FAKE_UPPER << 3),   # ignored
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_NO << 3),      # not active
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_LOWER << 3),   # upper is real -> not active
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_UPPER << 3),   # active!
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_BOTH << 3),    # active!
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),   # lower is real -> not active
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_LOWER << 3),   # active!
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_BOTH << 3),    # active!
        ]
        @test JClp.clp_dual_check_fake_bounds(st, length(st)) == 4
        @test JClp.clp_dual_number_at_fake_bound(st, length(st)) == 4
        @test (@allocated JClp.clp_dual_check_fake_bounds(st, length(st))) == 0
        @test (@allocated JClp.clp_dual_number_at_fake_bound(st, length(st))) == 0
    end

    @testset "clp_dual_original_bound! & clp_dual_change_bound!" begin
        # 2 cols, 2 rows
        nrows, ncols = 2, 2
        ntotal = nrows + ncols
        col_lo = Float64[-10.0, -1.0e60]
        col_hi = Float64[1.0e60, 20.0]
        row_lo = Float64[-5.0, -1.0e55]
        row_hi = Float64[15.0, 1.0e55]

        # 1. Unscaled with rhs_scale = 2.0
        lo = Float64[-10.0, -100.0, -5.0, -50.0]
        hi = Float64[100.0, 20.0, 50.0, 50.0]
        st = UInt8[
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_LOWER << 3),
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),
            JClp.CLP_STATUS_AT_LOWER, # noFake -> no-op
        ]
        nf = 3
        nf = JClp.clp_dual_original_bound!(lo, hi, st, col_lo, col_hi, row_lo, row_hi, nothing, nothing, ncols, 2.0, 1, nf)
        @test nf == 2 && lo[1] == -20.0 && hi[1] == 1.0e60
        @test JClp._sd_get_fake_bound(st[1]) == JClp.CLP_FAKE_NO

        nf = JClp.clp_dual_original_bound!(lo, hi, st, col_lo, col_hi, row_lo, row_hi, nothing, nothing, ncols, 2.0, 3, nf)
        @test nf == 1 && lo[3] == -10.0 && hi[3] == 30.0

        nf_unchanged = JClp.clp_dual_original_bound!(lo, hi, st, col_lo, col_hi, row_lo, row_hi, nothing, nothing, ncols, 2.0, 4, nf)
        @test nf_unchanged == 1 && lo[4] == -50.0

        # 2. Scaled (row_scale, inv_col_scale, rhs_scale)
        lo_s = Float64[-10.0, -100.0, -5.0, -50.0]
        hi_s = Float64[100.0, 20.0, 50.0, 50.0]
        st_s = UInt8[
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_LOWER << 3),
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),
            JClp.CLP_STATUS_AT_LOWER,
        ]
        row_scale = Float64[0.5, 4.0]
        inv_col_scale = Float64[3.0, 0.25]
        nf_s = JClp.clp_dual_original_bound!(lo_s, hi_s, st_s, col_lo, col_hi, row_lo, row_hi, row_scale, inv_col_scale, ncols, 1.5, 2, 3)
        @test nf_s == 2 && lo_s[2] == -1.0e60 && hi_s[2] == 20.0 * ((1.0 * 0.25) * 1.5)
        nf_s = JClp.clp_dual_original_bound!(lo_s, hi_s, st_s, col_lo, col_hi, row_lo, row_hi, row_scale, inv_col_scale, ncols, 1.5, 3, nf_s)
        @test nf_s == 1 && lo_s[3] == -5.0 * (0.5 * 1.5) && hi_s[3] == 15.0 * (0.5 * 1.5)

        # 3. clp_dual_change_bound!
        lo_cb = Float64[-10.0, -100.0, -5.0, -50.0]
        hi_cb = Float64[100.0, 20.0, 50.0, 50.0]
        st_cb = UInt8[
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_LOWER << 3),
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),
            JClp.CLP_STATUS_AT_LOWER,
        ]
        sol_cb = Float64[-10.0, 20.0, -5.0, -50.0]
        # col 1 at old_lower = -10.0, orig upper = 1e60 > -10 + 50 => fake upper at 40.0
        mod1, nf_cb = JClp.clp_dual_change_bound!(lo_cb, hi_cb, st_cb, sol_cb, col_lo, col_hi, row_lo, row_hi, nothing, nothing, ncols, 1.0, 50.0, 1, 3)
        @test mod1 && nf_cb == 3 && lo_cb[1] == -10.0 && hi_cb[1] == 40.0
        @test JClp._sd_get_fake_bound(st_cb[1]) == JClp.CLP_FAKE_UPPER
        # col 2 at old_upper = 20.0, orig lower = -1e60 < 20 - 50 => fake lower at -30.0
        mod2, nf_cb = JClp.clp_dual_change_bound!(lo_cb, hi_cb, st_cb, sol_cb, col_lo, col_hi, row_lo, row_hi, nothing, nothing, ncols, 1.0, 50.0, 2, nf_cb)
        @test mod2 && nf_cb == 3 && lo_cb[2] == -30.0 && hi_cb[2] == 20.0
        @test JClp._sd_get_fake_bound(st_cb[2]) == JClp.CLP_FAKE_LOWER
        # row 1 at -5.0 with fake upper restored to orig upper = 15.0 <= -5.0 + 50.0 => not modified, nf_cb decremented!
        mod3, nf_cb = JClp.clp_dual_change_bound!(lo_cb, hi_cb, st_cb, sol_cb, col_lo, col_hi, row_lo, row_hi, nothing, nothing, ncols, 1.0, 50.0, 3, nf_cb)
        @test !mod3 && nf_cb == 2 && JClp._sd_get_fake_bound(st_cb[3]) == JClp.CLP_FAKE_NO

        function _alloc_ob_cb(lo, hi, st, sol, cl, cu, rl, ru, rs, ics, ncols)
            a1 = @allocated JClp.clp_dual_original_bound!(lo, hi, st, cl, cu, rl, ru, rs, ics, ncols, 1.0, 1, 1)
            a2 = @allocated JClp.clp_dual_change_bound!(lo, hi, st, sol, cl, cu, rl, ru, rs, ics, ncols, 1.0, 50.0, 1, 1)
            return a1 + a2
        end
        _alloc_ob_cb(lo_cb, hi_cb, st_cb, sol_cb, col_lo, col_hi, row_lo, row_hi, nothing, nothing, ncols)
        @test _alloc_ob_cb(lo_cb, hi_cb, st_cb, sol_cb, col_lo, col_hi, row_lo, row_hi, nothing, nothing, ncols) == 0
        _alloc_ob_cb(lo_cb, hi_cb, st_cb, sol_cb, col_lo, col_hi, row_lo, row_hi, row_scale, inv_col_scale, ncols)
        @test _alloc_ob_cb(lo_cb, hi_cb, st_cb, sol_cb, col_lo, col_hi, row_lo, row_hi, row_scale, inv_col_scale, ncols) == 0
    end

    @testset "clp_dual_change_bounds! (modes 0, 1, 2, 3)" begin
        nrows, ncols = 2, 2
        ntotal = nrows + ncols
        mat_starts = Int32[1, 3, 4]
        mat_lengths = Int32[2, 1]
        mat_indices = Int32[1, 2, 2]
        mat_elements = Float64[2.0, -1.0, 3.0]

        col_lo = Float64[0.0, -1e20]
        col_hi = Float64[1e20, 1e20]
        row_lo = Float64[-1e20, 0.0]
        row_hi = Float64[1e20, 100.0]

        # Mode 0: no infeasibilities, tiny dj promotes to SUPER_BASIC & clears bit 8
        dj0 = Float64[1e-12, 0.0, 0.0, 0.0]
        cost0 = Float64[1.0, 2.0, 3.0, 4.0]
        sol0 = Float64[5.0, 0.0, 0.0, 0.0]
        lo0 = Float64[0.0, -10.0, -10.0, 0.0]
        hi0 = Float64[10.0, 10.0, 10.0, 100.0]
        st0 = UInt8[JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3), JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_IS_FREE, JClp.CLP_STATUS_AT_LOWER]
        out_val = zeros(Float64, ntotal)
        out_idx = zeros(Int32, ntotal)
        ret0, onnz0, cc0, db0, nf0, mso0, pstat0, emit0 = JClp.clp_dual_change_bounds!(
            0, out_val, out_idx, 0,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            nothing, nothing, nothing,
            dj0, cost0, sol0, lo0, hi0, st0,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 10.0, 1e15, 1.0, 8, -1, 99.0,
        )
        @test ret0 == -1 && onnz0 == 0 && cc0 == 0.0 && db0 == 10.0 && nf0 == 0 && mso0 == 0 && pstat0 == -1 && !emit0
        @test JClp._sd_get_status(st0[1]) == JClp.CLP_STATUS_SUPER_BASIC
        @test JClp._sd_get_fake_bound(st0[1]) == JClp.CLP_FAKE_NO

        # Mode 0: with infeasibilities + scaled matrix + outputArray
        dj0b = Float64[1.0, -2.0, 3.0, 0.0]
        sol0b = Float64[15.0, 20.0, -30.0, 0.0]
        lo0b = Float64[0.0, -1e20, -1e20, 0.0]
        hi0b = Float64[1e20, 1e20, 1e20, 100.0]
        st0b = UInt8[JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_AT_UPPER, JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_AT_LOWER]
        rs = Float64[2.0, 0.5]
        cs = Float64[0.5, 2.0]
        ics = Float64[2.0, 0.5]
        fill!(out_val, 0.0)
        fill!(out_idx, Int32(0))
        ret0b, onnz0b, cc0b, db0b, nf0b, mso0b, pstat0b, emit0b = JClp.clp_dual_change_bounds!(
            0, out_val, out_idx, 0,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            rs, cs, ics,
            dj0b, cost0, sol0b, lo0b, hi0b, st0b,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 10.0, 1e15, 1.0, 8, -1, 0.0,
        )
        @test ret0b == 3 && db0b == 50.0 && emit0b && nf0b == 4
        @test JClp._sd_get_fake_bound(st0b[1]) == JClp.CLP_FAKE_UPPER
        @test JClp._sd_get_fake_bound(st0b[2]) == JClp.CLP_FAKE_BOTH
        @test JClp._sd_get_fake_bound(st0b[3]) == JClp.CLP_FAKE_BOTH
        @test JClp._sd_get_fake_bound(st0b[4]) == JClp.CLP_FAKE_UPPER
        # Movement != 0 case:
        # col 1 AT_UPPER at val=5, lo=0, hi=1e20 => new_lo=0, new_hi=50 => sol=50 (mov1=+45)
        # row 1 AT_LOWER at val=-5, lo=-1e20, hi=10 => new_hi=10, new_lo=-40 => sol=-40 (mov3=-35)
        sol0c = Float64[5.0, 0.0, -5.0, 0.0]
        lo0c = Float64[0.0, -1e20, -1e20, 0.0]
        hi0c = Float64[1e20, 1e20, 10.0, 100.0]
        st0c = UInt8[JClp.CLP_STATUS_AT_UPPER, JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_BASIC]
        fill!(out_val, 0.0)
        fill!(out_idx, Int32(0))
        ret0c, onnz0c, cc0c, db0c, nf0c, _, _, emit0c = JClp.clp_dual_change_bounds!(
            0, out_val, out_idx, 0,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            rs, cs, ics,
            dj0b, cost0, sol0c, lo0c, hi0c, st0c,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 10.0, 1e15, 1.0, 8, -1, 0.0,
        )
        @test ret0c == 2 && emit0c && sol0c[1] == 50.0 && sol0c[3] == -40.0
        @test onnz0c == 3 # rows 1, 2 from col 1 + row seq 3 via _quick_add!
        mov1 = 45.0
        mov3 = -35.0
        @test out_val[1] ≈ 2.0 * (0.5 * mov1) * 2.0
        @test out_val[2] ≈ (-1.0) * (0.5 * mov1) * 0.5
        @test out_val[3] ≈ -mov3
        @test cc0c ≈ mov1 * cost0[1] + mov3 * cost0[3]

        # Also test Mode 0 with outputArray == nothing
        sol0d = Float64[5.0, 0.0, -5.0, 0.0]
        lo0d = Float64[0.0, -1e20, -1e20, 0.0]
        hi0d = Float64[1e20, 1e20, 10.0, 100.0]
        st0d = UInt8[JClp.CLP_STATUS_AT_UPPER, JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_BASIC]
        ret0d, onnz0d, cc0d, db0d, nf0d, _, _, emit0d = JClp.clp_dual_change_bounds!(
            0, nothing, nothing, 0,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            nothing, nothing, nothing,
            dj0b, cost0, sol0d, lo0d, hi0d, st0d,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 10.0, 1e15, 1.0, 8, -1, 99.0,
        )
        @test ret0d == 2 && onnz0d == 0 && cc0d == 0.0 && db0d == 50.0 && nf0d == 2 && emit0d

        # Mode 1 & Mode 3: initialize == 1, 3 (scaled + unscaled + inverted bound + free + basic gap)
        lo1 = Float64[0.0, -1e20, -1e20, 10.0]
        hi1 = Float64[1e20, 1e20, 100.0, 20.0]
        sol1 = Float64[0.0, 0.0, 0.0, 15.0]
        st1 = UInt8[
            JClp.CLP_STATUS_AT_LOWER,  # upper fake (0 + 25)
            JClp.CLP_STATUS_AT_LOWER,  # free fallback -> [-12.5, 12.5], AT_UPPER, bothFake
            JClp.CLP_STATUS_BASIC,     # gap=1e20 not in (12.5, 50)
            JClp.CLP_STATUS_BASIC,     # gap=10 not in (12.5, 50)
        ]
        # Force basic gap in (0.5*25, 2*25) on row 2 (seq 4): lo=0, hi=30 => restored from row_lo/row_hi
        lo1[4] = 0.0; hi1[4] = 30.0
        ret1, _, cc1, db1, nf1, _, pstat1, emit1 = JClp.clp_dual_change_bounds!(
            1, nothing, nothing, 0,
            nothing, nothing, nothing, nothing,
            nothing, nothing, nothing,
            dj0, cost0, sol1, lo1, hi1, st1,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 25.0, 1e15, 1.0, 8, -1, 12.5,
        )
        @test ret1 == 1 && cc1 == 12.5 && nf1 == 2 && pstat1 == -1 && !emit1
        @test hi1[1] == 25.0 && JClp._sd_get_fake_bound(st1[1]) == JClp.CLP_FAKE_UPPER
        @test lo1[2] == -12.5 && hi1[2] == 12.5 && sol1[2] == 12.5
        @test JClp._sd_get_status(st1[2]) == JClp.CLP_STATUS_AT_UPPER
        @test JClp._sd_get_fake_bound(st1[2]) == JClp.CLP_FAKE_BOTH
        @test lo1[4] == 0.0 && hi1[4] == 100.0 # restored from row_lo/row_hi

        # Mode 3 with scaling
        lo3 = Float64[0.0, 0.0, 0.0, 50.0]
        hi3 = Float64[1e20, 1e20, 100.0, 40.0]
        sol3 = Float64[0.0, 0.0, 0.0, 50.0]
        st3 = UInt8[
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_AT_LOWER, # row 2 overwritten by scaled mode 3 to [0, 75], then lowerFake [50, 75]
        ]
        ret3, _, _, _, nf3, _, pstat3, _ = JClp.clp_dual_change_bounds!(
            3, nothing, nothing, 0,
            nothing, nothing, nothing, nothing,
            rs, cs, ics,
            dj0, cost0, sol3, lo3, hi3, st3,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 25.0, 1e15, 1.5, 8, -1, 0.0,
        )
        @test ret3 == 1 && pstat3 == -1 && hi3[4] == 100.0 * (0.5 * 1.5) && lo3[4] == hi3[4] - 25.0
        @test JClp._sd_get_fake_bound(st3[4]) == JClp.CLP_FAKE_LOWER
        # Test inverted bound with mode 1 directly
        lo_inv = Float64[5.0, 0.0, 0.0, 0.0]
        hi_inv = Float64[3.0, 1.0, 1.0, 1.0]
        st_inv = UInt8[JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_BASIC]
        _, _, _, _, _, _, pstat_inv, _ = JClp.clp_dual_change_bounds!(
            1, nothing, nothing, 0,
            nothing, nothing, nothing, nothing,
            nothing, nothing, nothing,
            dj0, cost0, sol3, lo_inv, hi_inv, st_inv,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 25.0, 1e15, 1.0, 8, -1, 0.0,
        )
        @test pstat_inv == 1 && hi_inv[1] == 5.0

        # Mode 2: restore only altered bounds, keep fake bits
        lo2 = Float64[-99.0, -99.0, -99.0, -99.0]
        hi2 = Float64[99.0, 99.0, 99.0, 99.0]
        st2 = UInt8[
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_UPPER << 3),
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_LOWER << 3),
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_BOTH << 3),
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_NO << 3),
        ]
        ret2, _, _, _, nf2, _, _, _ = JClp.clp_dual_change_bounds!(
            2, nothing, nothing, 0,
            nothing, nothing, nothing, nothing,
            nothing, nothing, nothing,
            dj0, cost0, sol0, lo2, hi2, st2,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 25.0, 1e15, 1.0, 8, -1, 0.0,
        )
        @test ret2 == 0 && nf2 == 0
        @test lo2[1] == -99.0 && hi2[1] == 1e20
        @test lo2[2] == -1e20 && hi2[2] == 99.0
        @test lo2[3] == -1e20 && hi2[3] == 1e20
        @test lo2[4] == -99.0 && hi2[4] == 99.0
        @test JClp._sd_get_fake_bound(st2[1]) == JClp.CLP_FAKE_UPPER

        # Mode 2 scaled (note 1e20 < 1e30 gets scaled by rhs_scale * inv_col_scale)
        lo2s = Float64[-99.0, -99.0, -99.0, -99.0]
        hi2s = Float64[99.0, 99.0, 99.0, 99.0]
        ret2s, _, _, _, _, _, _, _ = JClp.clp_dual_change_bounds!(
            2, nothing, nothing, 0,
            nothing, nothing, nothing, nothing,
            rs, cs, ics,
            dj0, cost0, sol0, lo2s, hi2s, st2,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 25.0, 1e15, 2.0, 8, -1, 0.0,
        )
        @test ret2s == 0 && hi2s[1] == 1e20 * (2.0 * 2.0) && lo2s[2] == -1e20 * (2.0 * 0.5)
        # Test finite scaled bound restoration in mode 2
        st2s_fin = UInt8[
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_LOWER << 3),
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_UPPER << 3),
        ]
        lo2sf = Float64[-99.0, -99.0, -99.0, -99.0]
        hi2sf = Float64[99.0, 99.0, 99.0, 99.0]
        JClp.clp_dual_change_bounds!(
            2, nothing, nothing, 0,
            nothing, nothing, nothing, nothing,
            rs, cs, ics,
            dj0, cost0, sol0, lo2sf, hi2sf, st2s_fin,
            col_lo, col_hi, row_lo, row_hi,
            nrows, ncols, 1e-7, 25.0, 1e15, 2.0, 8, -1, 0.0,
        )
        @test lo2sf[1] == 0.0 && hi2sf[4] == 100.0 * (2.0 * 0.5)

        function _alloc_cb_all(
            out_val, out_idx, mat_starts, mat_lengths, mat_indices, mat_elements,
            rs, cs, ics, dj, cost, sol, lo, hi, st, col_lo, col_hi, row_lo, row_hi, nrows, ncols,
        )
            a0 = @allocated JClp.clp_dual_change_bounds!(
                0, out_val, out_idx, 0, mat_starts, mat_lengths, mat_indices, mat_elements,
                rs, cs, ics, dj, cost, sol, lo, hi, st, col_lo, col_hi, row_lo, row_hi,
                nrows, ncols, 1e-7, 10.0, 1e15, 1.0, 8, -1, 0.0,
            )
            a1 = @allocated JClp.clp_dual_change_bounds!(
                1, nothing, nothing, 0, nothing, nothing, nothing, nothing,
                nothing, nothing, nothing, dj, cost, sol, lo, hi, st, col_lo, col_hi, row_lo, row_hi,
                nrows, ncols, 1e-7, 10.0, 1e15, 1.0, 8, -1, 0.0,
            )
            a2 = @allocated JClp.clp_dual_change_bounds!(
                2, nothing, nothing, 0, nothing, nothing, nothing, nothing,
                rs, cs, ics, dj, cost, sol, lo, hi, st, col_lo, col_hi, row_lo, row_hi,
                nrows, ncols, 1e-7, 10.0, 1e15, 1.0, 8, -1, 0.0,
            )
            a3 = @allocated JClp.clp_dual_change_bounds!(
                3, nothing, nothing, 0, nothing, nothing, nothing, nothing,
                rs, cs, ics, dj, cost, sol, lo, hi, st, col_lo, col_hi, row_lo, row_hi,
                nrows, ncols, 1e-7, 10.0, 1e15, 1.0, 8, -1, 0.0,
            )
            return a0 + a1 + a2 + a3
        end
        _alloc_cb_all(out_val, out_idx, mat_starts, mat_lengths, mat_indices, mat_elements, rs, cs, ics, dj0, cost0, sol0c, lo0c, hi0c, st0c, col_lo, col_hi, row_lo, row_hi, nrows, ncols)
        @test _alloc_cb_all(out_val, out_idx, mat_starts, mat_lengths, mat_indices, mat_elements, rs, cs, ics, dj0, cost0, sol0c, lo0c, hi0c, st0c, col_lo, col_hi, row_lo, row_hi, nrows, ncols) == 0
    end

    @testset "clp_dual_reset_fake_bounds!" begin
        nrows, ncols = 2, 3
        ntotal = nrows + ncols
        col_lo = Float64[0.0, -1e35, -10.0]
        col_hi = Float64[1e35, 10.0, 10.0]
        row_lo = Float64[-1e35, 2.0]
        row_hi = Float64[1e35, 20.0]
        rs = Float64[2.0, 0.5]
        ics = Float64[0.5, 2.0, 1.0]

        lo = zeros(Float64, ntotal)
        hi = zeros(Float64, ntotal)
        sol = Float64[50.0, -15.0, 0.0, 3.0, 4.0]
        st = UInt8[
            JClp.CLP_STATUS_AT_UPPER | (JClp.CLP_FAKE_UPPER << 3),  # upperFake at upper => upper=lo+db=50, sol=50
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_LOWER << 3),  # lowerFake at lower => lower=hi-db, sol=lower
            JClp.CLP_STATUS_BASIC | (JClp.CLP_FAKE_BOTH << 3),      # basic => fake cleared!
            JClp.CLP_STATUS_SUPER_BASIC | (JClp.CLP_FAKE_BOTH << 3),# bothFake superBasic => [val-25, val+25]
            JClp.CLP_STATUS_AT_LOWER | (JClp.CLP_FAKE_BOTH << 3),   # bothFake atLower => [val, val+50]
        ]
        nf, bad = JClp.clp_dual_reset_fake_bounds!(lo, hi, st, sol, col_lo, col_hi, row_lo, row_hi, rs, ics, nrows, ncols, 2.0, 50.0)
        @test bad == 0 && nf == 4
        @test lo[1] == 0.0 && hi[1] == 50.0 && sol[1] == 50.0
        @test hi[2] == 10.0 * (2.0 * 2.0) && lo[2] == hi[2] - 50.0 && sol[2] == lo[2]
        @test JClp._sd_get_fake_bound(st[3]) == JClp.CLP_FAKE_NO && lo[3] == -20.0 && hi[3] == 20.0
        @test lo[4] == 3.0 - 25.0 && hi[4] == 3.0 + 25.0
        @test lo[5] == 4.0 && hi[5] == 54.0

        # Unscaled test + invalid status detection
        nf_u, bad_u = JClp.clp_dual_reset_fake_bounds!(lo, hi, st, sol, col_lo, col_hi, row_lo, row_hi, nothing, nothing, nrows, ncols, 1.0, 50.0)
        @test bad_u == 0 && nf_u == 4 && hi[2] == 10.0 && lo[2] == -40.0

        st_bad = UInt8[JClp.CLP_STATUS_IS_FREE | (JClp.CLP_FAKE_UPPER << 3), JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_BASIC]
        _, bad_seq = JClp.clp_dual_reset_fake_bounds!(lo, hi, st_bad, sol, col_lo, col_hi, row_lo, row_hi, nothing, nothing, nrows, ncols, 1.0, 50.0)
        @test bad_seq == 1

        function _alloc_rfb(lo, hi, st, sol, cl, cu, rl, ru, rs, ics, nrows, ncols)
            a1 = @allocated JClp.clp_dual_reset_fake_bounds!(lo, hi, st, sol, cl, cu, rl, ru, rs, ics, nrows, ncols, 2.0, 50.0)
            a2 = @allocated JClp.clp_dual_reset_fake_bounds!(lo, hi, st, sol, cl, cu, rl, ru, nothing, nothing, nrows, ncols, 1.0, 50.0)
            return a1 + a2
        end
        _alloc_rfb(lo, hi, st, sol, col_lo, col_hi, row_lo, row_hi, rs, ics, nrows, ncols)
        @test _alloc_rfb(lo, hi, st, sol, col_lo, col_hi, row_lo, row_hi, rs, ics, nrows, ncols) == 0
    end

    @testset "clp_dual_check_possible_values_move!" begin
        nrows, ncols = 2, 3
        ntotal = nrows + ncols
        row_val = Float64[1.0, -2.0]
        row_idx = Int32[1, 2]
        col_val = Float64[2.0, -0.5, 0.0]
        col_idx = Int32[1, 2, 3]
        # sequence_out = 3 (col 3), dj_basic = 1.5 > 0 => theta_up = 1.5, change_down = -lo[3] = -2.0
        # col 1 AT_LOWER, alpha = 2.0, dj = 1.0 => theta_down = 0.5, best_alpha_down = 2.0, seq_down = 1
        # col 2 AT_UPPER, alpha = -0.5, dj = -2.0 => theta_down = 4.0 > 0.5 (not tighter)
        # row 1 (seq 4) AT_UPPER, alpha = 1.0, dj = -0.5 => theta_up = 0.5, seq_up = 4
        # row 2 (seq 5) IS_FIXED, alpha = -2.0, upper = 3.0 => change_down += -6.0
        dj = Float64[1.0, -2.0, 1.5, -0.5, 0.0]
        sol = Float64[1.0, 4.0, 2.0, 5.0, 3.0]
        lo = Float64[1.0, 0.0, 2.0, 0.0, 3.0]
        hi = Float64[10.0, 4.0, 8.0, 5.0, 3.0]
        st = UInt8[
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_IS_FIXED,
        ]
        seq_in, dir_in, th, al, lo_in, up_in, val_in, dj_in = JClp.clp_dual_check_possible_values_move!(
            row_val, row_idx, 2, col_val, col_idx, 3,
            dj, sol, lo, hi, st, ncols, 3, 1e-7, 1e-8,
        )
        @test seq_in == 1 && dir_in == 1 && th ≈ 0.5 && al == 2.0
        @test lo_in == 1.0 && up_in == 10.0 && val_in == 1.0 && dj_in == 1.0

        # Free/superBasic candidate overrides with theta = 0
        st_free = copy(st)
        st_free[4] = JClp.CLP_STATUS_IS_FREE
        seq_in_f, dir_in_f, th_f, al_f, _, _, _, _ = JClp.clp_dual_check_possible_values_move!(
            Float64[3.0, -2.0], row_idx, 2, col_val, col_idx, 3,
            dj, sol, lo, hi, st_free, ncols, 3, 1e-7, 1e-8,
        )
        @test seq_in_f == 4 && th_f == 0.0 && al_f == 3.0 && dir_in_f == 1

        alloc_vm = @allocated JClp.clp_dual_check_possible_values_move!(
            row_val, row_idx, 2, col_val, col_idx, 3,
            dj, sol, lo, hi, st, ncols, 3, 1e-7, 1e-8,
        )
        @test alloc_vm == 0
    end

    @testset "clp_dual_check_possible_cleanup!" begin
        nrows, ncols = 2, 3
        row_val = Float64[1.5, 2.5]
        row_idx = Int32[1, 2]
        col_val = Float64[-1.0, 2.0, 0.5]
        col_idx = Int32[1, 2, 3]
        dj = Float64[-1.0, 2.0, 0.0, -3.0, 0.5]
        sol = Float64[4.0, 0.0, 1.0, 5.0, 2.0]
        lo = Float64[0.0, 0.0, 1.0, 0.0, 2.0]
        hi = Float64[4.0, 10.0, 1.0, 5.0, 2.0]
        st = UInt8[
            JClp.CLP_STATUS_AT_UPPER, # col 1: alpha=-1.0 -> theta_down=1.0, best_alpha_down=1.0
            JClp.CLP_STATUS_AT_LOWER, # col 2: alpha=2.0 -> theta_down=1.0, best_alpha_down=2.0
            JClp.CLP_STATUS_IS_FIXED, # col 3: add_seq==0 -> ignored!
            JClp.CLP_STATUS_AT_UPPER, # row 1 (seq 4): alpha=1.5 -> theta_up=2.0, best_alpha_up=1.5
            JClp.CLP_STATUS_IS_FIXED, # row 2 (seq 5): add_seq!=0 -> alpha=2.5, best_alpha_down=2.5, theta_down=0.2!
        ]
        seq_in, dir_in, th, al, lo_in, up_in, val_in, dj_in = JClp.clp_dual_check_possible_cleanup!(
            row_val, row_idx, 2, col_val, col_idx, 3,
            dj, sol, lo, hi, st, ncols, 1e-8, 1e-7,
        )
        @test seq_in == 5 && dir_in == 1 && th ≈ 0.2 && al == 2.5 && lo_in == 2.0 && up_in == 2.0

        # No acceptable candidate (all alphas below 10 * acceptable_pivot)
        seq_none, _, th_none, al_none, _, _, _, _ = JClp.clp_dual_check_possible_cleanup!(
            Float64[1e-8, 1e-8], row_idx, 2, Float64[1e-8, 1e-8, 1e-8], col_idx, 3,
            dj, sol, lo, hi, st, ncols, 1e-6, 1e-7,
        )
        @test seq_none == 0 && th_none == 0.0 && al_none == 0.0

        alloc_cu = @allocated JClp.clp_dual_check_possible_cleanup!(
            row_val, row_idx, 2, col_val, col_idx, 3,
            dj, sol, lo, hi, st, ncols, 1e-8, 1e-7,
        )
        @test alloc_cu == 0
    end
end

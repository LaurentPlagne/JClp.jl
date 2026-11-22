using Test
using JClp

@testset "B11c-2: createRim1, createRim4, computePrimals/computeDuals kernels" begin
    @testset "clp_create_rim1!" begin
        nrows, ncols = 2, 3
        lower = zeros(nrows + ncols)
        upper = zeros(nrows + ncols)
        col_lo = [1.0, -1e25, -1e-9]
        col_hi = [1.0 + 1e-9, 1e25, 1e-9]
        row_lo = [-1e25, 2.0]
        row_hi = [5.0, 1e25]
        JClp.clp_create_rim1!(
            lower, upper, col_lo, col_hi, row_lo, row_hi,
            nothing, nothing, nrows, 0, ncols, nrows, ncols,
            0, false, 1.0, 1e-7,
        )
        @test lower[1] == 1.0 && upper[1] == 1.0          # tiny positive gap snapped to lower
        @test lower[2] == -floatmax(Float64) && upper[2] == floatmax(Float64)  # free
        @test lower[3] == 0.0 && upper[3] == 0.0          # tiny gap straddling 0 snapped to 0
        @test lower[4] == -floatmax(Float64) && upper[4] == 5.0
        @test lower[5] == 2.0 && upper[5] == floatmax(Float64)

        # Scaled + !initial
        rscale = [2.0, 0.5]
        ics    = [0.25, 1.0, 1.0]
        JClp.clp_create_rim1!(
            lower, upper, col_lo, col_hi, row_lo, row_hi,
            rscale, ics, nrows, 0, ncols, nrows, ncols,
            0, false, 4.0, 1e-7,
        )
        @test lower[1] ≈ 1.0 && upper[1] ≈ 1.0            # 1.0 * (4.0 * 0.25)
        @test upper[4] ≈ 40.0                             # 5.0 * (4.0 * 2.0)
        @test lower[5] ≈ 4.0                              # 2.0 * (4.0 * 0.5)
    end

    @testset "clp_create_rim4!" begin
        nrows, ncols = 2, 3
        cost = ones(nrows + ncols)
        obj = [1.0, -2.0, 3.0]
        rscale = [2.0, 4.0]
        cscale = [0.5, 1.0, 2.0]
        JClp.clp_create_rim4!(
            cost, obj, nothing, rscale, cscale,
            nrows, 0, ncols, nrows, ncols, 0, false, -1.0,
        )
        @test cost == [-0.5, 2.0, -6.0, 0.0, 0.0]
    end

    @testset "computePrimals sub-kernels + zero alloc" begin
        nrows, ncols = 3, 2
        sol = [10.0, 20.0, 30.0, 40.0, 50.0]
        pvar = Int32[1, 4, 5]
        JClp.clp_zero_basic_solution!(sol, pvar, nrows)
        @test sol == [0.0, 20.0, 30.0, 0.0, 0.0]

        dst_val = zeros(3)
        dst_idx = zeros(Int32, 3)
        a = [1.0, -2.0, 3.0]
        b = [-1.0, 5.0, 0.0]
        nnz = JClp.clp_pack_sum_nonzero!(dst_val, dst_idx, a, b, nrows)
        @test nnz == 2
        @test dst_idx[1:2] == Int32[2, 3]
        @test dst_val == [0.0, 3.0, 3.0]

        JClp.clp_scatter_basic_solution!(sol, pvar, dst_val, dst_idx, nnz, nrows, false)
        @test sol[4] == 3.0 && sol[5] == 3.0 && sol[1] == 0.0

        work = [1e-12, -2e-11, 0.0]
        row_act = [0.0, 0.0, 0.0]
        err = JClp.clp_primal_residual_scale!(work, row_act, nrows, 131072.0)
        @test err ≈ 2e-11
        @test work[2] ≈ -2e-11 * 131072.0

        r_val = zeros(3)
        r_idx = zeros(Int32, 3)
        rnnz = JClp.clp_refine_prepare!(r_val, r_idx, work, nrows, 1.0)
        @test rnnz == 2
        @test all(==(0.0), work)

        prev = [1.0, 2.0, 0.0]
        cnnz = JClp.clp_refine_combine!(r_val, r_idx, prev, nrows, 1.0 / 131072.0)
        @test cnnz == 2
        @test r_val[1] ≈ 1.0 + 1e-12
        @test r_val[2] ≈ 2.0 - 2e-11

        function _alloc_prim(w, ra)
            JClp.clp_primal_residual_scale!(w, ra, 3, 1.0)
            return @allocated JClp.clp_primal_residual_scale!(w, ra, 3, 1.0)
        end
        @test _alloc_prim(work, row_act) == 0
    end

    @testset "computeDuals sub-kernels + zero alloc" begin
        nrows, ncols = 3, 2
        cost = [1.5, 0.0, -2.0, 0.0, 3.0]
        pvar = Int32[1, 3, 5]
        status = fill(JClp.CLP_STATUS_BASIC, 5)
        status[2] = JClp.CLP_STATUS_AT_LOWER
        arr_val = zeros(3)
        arr_idx = zeros(Int32, 3)
        nnz = JClp.clp_compute_duals_setup!(arr_val, arr_idx, cost, pvar, nothing, status, nrows)
        @test nnz == 3
        @test arr_val == [1.5, -2.0, 3.0]

        sidx = zeros(Int32, 3)
        ns = JClp.clp_compute_duals_collect_structurals!(sidx, pvar, nrows, ncols)
        @test ns == 1 && sidx[1] == Int32(1)

        work = zeros(3)
        row_obj = [-2.0, 0.0, 3.0]
        col_obj = [1.5, 0.0]
        work2 = [1.4, 0.0, 0.0]
        arr_btran = [2.0, 0.0, -2.95]
        derr = JClp.clp_compute_duals_residual!(work, pvar, arr_btran, row_obj, col_obj, work2, nothing, nrows, ncols)
        @test derr ≈ 0.1
        @test work[1] ≈ 0.1
        @test work[2] ≈ 0.0
        @test work[3] ≈ 0.05

        dual = zeros(3)
        row_dj = zeros(3)
        col_dj = zeros(2)
        which_col = zeros(Int32, 2)
        n_nb = JClp.clp_compute_duals_finalize_prep!(dual, row_dj, col_dj, which_col, arr_btran, row_obj, col_obj, status, nrows, ncols)
        @test n_nb == 1 && which_col[1] == Int32(2)
        @test dual == arr_btran
        @test row_dj ≈ [0.0, 0.0, 0.05]
        @test col_dj == [0.0, 0.0]

        function _alloc_dual(w, pv, ab, ro, co, w2)
            JClp.clp_compute_duals_residual!(w, pv, ab, ro, co, w2, nothing, 3, 2)
            return @allocated JClp.clp_compute_duals_residual!(w, pv, ab, ro, co, w2, nothing, 3, 2)
        end
        @test _alloc_dual(work, pvar, arr_btran, row_obj, col_obj, work2) == 0
    end
end

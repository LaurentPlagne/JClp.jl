using JClp
using Test

@testset "B9b: ClpPrimalColumnDantzig & ClpPrimalColumnSteepest (Pure 1-based Julia)" begin
    @testset "ClpPrimalColumnDantzig" begin
        nrows, ncols = 2, 3
        rc = Float64[-0.5, 2.0, -3.0, -3.0, 0.1]
        status = UInt8[
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_AT_LOWER, # slack 1: -3.0 * 1.01 = -3.03 -> wins!
            JClp.CLP_STATUS_BASIC,
        ]
        @test JClp.clp_pcd_price(rc, status, nrows, ncols, 1e-7) == 4

        # Flag slack 1 -> col 3 (-3.0) wins
        status[4] |= JClp.CLP_STATUS_FLAGGED_BIT
        @test JClp.clp_pcd_price(rc, status, nrows, ncols, 1e-7) == 3

        # Free variable bias: free dj > 0.1 * bestDj
        status[1] = JClp.CLP_STATUS_IS_FREE
        @test JClp.clp_pcd_price(rc, status, nrows, ncols, 1e-7) == 1

        # update_djs!
        upd_val = Float64[1.0, -0.5]
        upd_idx = Int32[1, 3]
        JClp.clp_pcd_update_djs!(upd_val, upd_idx, 2, rc)
        @test rc[1] ≈ -1.5 && rc[3] ≈ -2.5
        @test upd_val[1] == 0.0 && upd_val[2] == 0.0
        @test (@allocated JClp.clp_pcd_price(rc, status, nrows, ncols, 1e-7)) == 0
    end

    @testset "ClpPrimalColumnSteepest" begin
        ntotal = 5
        rc = Float64[-2.0, 0.0, 3.0, -1.0, 0.5]
        status = UInt8[
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_BASIC,
            JClp.CLP_STATUS_AT_UPPER,
            JClp.CLP_STATUS_AT_LOWER,
            JClp.CLP_STATUS_IS_FIXED,
        ]
        infeas = zeros(Float64, ntotal)
        infeas_idx = zeros(Int32, ntotal)
        nnz = JClp.clp_pcs_redo_infeasibilities!(infeas, infeas_idx, rc, status, ntotal, 1e-7)
        @test nnz == 3
        @test infeas ≈ Float64[4.0, 0.0, 9.0, 1.0, 0.0]
        @test !JClp.clp_pcs_looks_optimal_simple(rc, status, ntotal, 1e-7)

        weights = Float64[1.0, 1.0, 10.0, 0.2, 1.0]
        # col 1: 4/1 = 4, col 3: 9/10 = 0.9, col 4: 1/0.2 = 5 (BEST!)
        best_seq, best_dj = JClp.clp_pcs_price_scan(infeas, infeas_idx, nnz, weights, status, 2, 3, 100, 1e-14, 0.0)
        @test best_seq == 4
        @test best_dj ≈ 5.0

        # Mode 1 full scan
        best_seq1, best_dj1 = JClp.clp_pcs_price_scan(infeas, infeas_idx, nnz, weights, status, 1, 1, 100, 1e-14, 0.0)
        @test best_seq1 == 4
        @test best_dj1 ≈ 5.0

        # Dantzig mode (switchType == 5): col 3 (9.0) wins
        best_seq5, _ = JClp.clp_pcs_price_scan(infeas, infeas_idx, nnz, weights, status, 5, 5, 100, 1e-14, 0.0)
        @test best_seq5 == 3

        # just_djs_section!
        upd_val = Float64[1.0]
        upd_idx = Int32[1]
        nnz2 = JClp.clp_pcs_just_djs_section!(upd_val, upd_idx, 1, rc, status, infeas, infeas_idx, nnz, 0, 1.0, 1e-7)
        @test rc[1] ≈ -3.0
        @test infeas[1] ≈ 9.0
        @test upd_val[1] == 0.0

        # devex_section!
        ref_bits = UInt32[0xff]
        upd_val2 = Float64[2.0]
        upd_idx2 = Int32[1]
        JClp.clp_pcs_devex_section!(upd_val2, upd_idx2, 1, rc, status, weights, ref_bits, infeas, infeas_idx, nnz2, 0, 1.0, 2.0, 1.0, 1e-7)
        @test weights[1] ≈ max(0.99 * 1.0, 2.0 * 2.0 * 2.0 + 1.0)

        # unroll_weights!
        saved_val = Float64[1.5, 0.0, 0.0, 0.0, 0.0]
        saved_idx = Int32[1]
        JClp.clp_pcs_unroll_weights!(weights, saved_val, saved_idx, 1)
        @test weights[1] == 1.5 && saved_val[1] == 0.0

        @test (@allocated JClp.clp_pcs_price_scan(infeas, infeas_idx, nnz, weights, status, 2, 3, 100, 1e-14, 0.0)) == 0
    end
end

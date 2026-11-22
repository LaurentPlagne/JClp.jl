using JClp
using Test

@testset "B8: ClpPackedMatrix (Pure 1-based Julia)" begin
    # Construct a 4x5 sparse matrix A:
    #   [ 2.0   0.0  -1.0   0.0   3.0 ]
    #   [ 0.0   4.0   0.0   1.5   0.0 ]
    #   [-3.0   1.0   0.0   0.0   2.0 ]
    #   [ 0.0   0.0   5.0  -2.0   0.0 ]
    nrows, ncols = 4, 5
    A_dense = [
        2.0  0.0 -1.0  0.0  3.0;
        0.0  4.0  0.0  1.5  0.0;
       -3.0  1.0  0.0  0.0  2.0;
        0.0  0.0  5.0 -2.0  0.0
    ]
    starts = Int32[1, 3, 5, 7, 9, 11]
    lengths = Int32[2, 2, 2, 2, 2]
    indices = Int32[1, 3, 2, 3, 1, 4, 2, 4, 1, 3]
    elements = Float64[2.0, -3.0, 4.0, 1.0, -1.0, 5.0, 1.5, -2.0, 3.0, 2.0]

    # CSR (row-major) copy of A:
    r_starts = Int32[1, 4, 6, 9, 11]
    r_cols = Int32[1, 3, 5, 2, 4, 1, 2, 5, 3, 4]
    r_elems = Float64[2.0, -1.0, 3.0, 4.0, 1.5, -3.0, 1.0, 2.0, 5.0, -2.0]

    rs = Float64[0.5, 2.0, 0.25, 4.0]
    cs = Float64[1.5, 0.5, 2.0, 0.25, 3.0]
    A_scaled = Diagonal_mul(rs, A_dense, cs) = (rs .* A_dense) .* cs'

    @testset "times (unscaled & scaled, with/without gaps)" begin
        x = Float64[1.0, -2.0, 0.5, 0.0, 3.0]
        y = Float64[10.0, 20.0, 30.0, 40.0]
        JClp.clp_mat_times_unscaled!(starts, nothing, false, indices, elements, ncols, 2.5, x, y)
        @test y ≈ Float64[10.0, 20.0, 30.0, 40.0] .+ 2.5 .* (A_dense * x)

        y2 = Float64[10.0, 20.0, 30.0, 40.0]
        JClp.clp_mat_times_unscaled!(starts, lengths, true, indices, elements, ncols, 2.5, x, y2)
        @test y2 ≈ y

        ys = Float64[1.0, 2.0, 3.0, 4.0]
        JClp.clp_mat_times_scaled!(starts, nothing, false, indices, elements, ncols, -1.5, x, ys, rs, cs)
        @test ys ≈ Float64[1.0, 2.0, 3.0, 4.0] .- 1.5 .* (A_scaled(rs, A_dense, cs) * x)

        ys2 = Float64[1.0, 2.0, 3.0, 4.0]
        JClp.clp_mat_times_scaled!(starts, lengths, true, indices, elements, ncols, -1.5, x, ys2, rs, cs)
        @test ys2 ≈ ys

        # Zero allocations on hot path
        @test (@allocated JClp.clp_mat_times_unscaled!(starts, nothing, false, indices, elements, ncols, 2.5, x, y)) == 0
        @test (@allocated JClp.clp_mat_times_scaled!(starts, nothing, false, indices, elements, ncols, -1.5, x, ys, rs, cs)) == 0
    end

    @testset "transposeTimes & transposeTimesSubset" begin
        x = Float64[1.5, -0.5, 2.0, 3.0]
        y = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
        JClp.clp_mat_transpose_times_unscaled!(starts, nothing, false, indices, elements, ncols, -1.0, x, y)
        @test y ≈ Float64[1.0, 2.0, 3.0, 4.0, 5.0] .- (A_dense' * x)

        y2 = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
        JClp.clp_mat_transpose_times_unscaled!(starts, nothing, false, indices, elements, ncols, 2.0, x, y2)
        @test y2 ≈ Float64[1.0, 2.0, 3.0, 4.0, 5.0] .+ 2.0 .* (A_dense' * x)

        ys = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
        JClp.clp_mat_transpose_times_scaled!(starts, nothing, false, indices, elements, nrows, ncols, -1.0, x, ys, rs, cs, nothing)
        @test ys ≈ Float64[1.0, 2.0, 3.0, 4.0, 5.0] .- (A_scaled(rs, A_dense, cs)' * x)

        ys_sp = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
        spare = zeros(Float64, nrows)
        JClp.clp_mat_transpose_times_scaled!(starts, nothing, false, indices, elements, nrows, ncols, -1.0, x, ys_sp, rs, cs, spare)
        @test ys_sp ≈ ys

        which = Int32[2, 5, 1]
        ysub = Float64[10.0, 20.0, 30.0, 40.0, 50.0]
        JClp.clp_mat_transpose_times_subset!(starts, indices, elements, nrows, 3, which, x, ysub, rs, cs, spare)
        expected_sub = Float64[10.0, 20.0, 30.0, 40.0, 50.0]
        full_prod = A_scaled(rs, A_dense, cs)' * x
        for c in which
            expected_sub[c] -= full_prod[c]
        end
        @test ysub ≈ expected_sub
        @test (@allocated JClp.clp_mat_transpose_times_unscaled!(starts, nothing, false, indices, elements, ncols, -1.0, x, y)) == 0
    end

    @testset "gutsOfTransposeTimes (unscaled, scaled, status, dual)" begin
        pi = Float64[1.0, -2.0, 0.5, 4.0]
        out_idx = zeros(Int32, ncols)
        out_val = zeros(Float64, ncols)
        nnz = JClp.clp_mat_guts_tt_unscaled!(starts, indices, elements, ncols, pi, out_idx, out_val, 1e-13)
        ref = A_dense' * pi
        @test nnz == ncols
        @test out_idx[1:nnz] == Int32[1, 2, 3, 4, 5]
        @test out_val[1:nnz] ≈ ref

        status = UInt8[0, 1, 2, 3, 1] # cols 2 and 5 are basic ((status & 3) == 1) -> skipped
        nnz_st = JClp.clp_mat_guts_tt_unscaled_status!(starts, indices, elements, ncols, pi, out_idx, out_val, status, 1e-13)
        @test nnz_st == 3
        @test out_idx[1:3] == Int32[1, 3, 4]
        @test out_val[1:3] ≈ [ref[1], ref[3], ref[4]]

        nnz_sc = JClp.clp_mat_guts_tt_scaled_status!(starts, indices, elements, ncols, pi, cs, out_idx, out_val, status, 1e-13)
        @test nnz_sc == 3
        @test out_val[1:3] ≈ [ref[1] * cs[1], ref[3] * cs[3], ref[4] * cs[4]]

        # Dual pre-scan variant
        spare_idx = zeros(Int32, ncols)
        spare_val = zeros(Float64, ncols)
        rc = Float64[0.0, 0.0, -1.0, 2.0, 0.0]
        nnz_d, nrem, ut = JClp.clp_mat_guts_tt_unscaled_dual!(
            starts, indices, elements, ncols, pi, out_idx, out_val, status,
            spare_idx, spare_val, rc, 1e31, 1e-7, 1e-7, 0, 1e-13,
        )
        @test nnz_d == 3
        @test nrem >= 0
        @test (@allocated JClp.clp_mat_guts_tt_unscaled_status!(starts, indices, elements, ncols, pi, out_idx, out_val, status, 1e-13)) == 0
    end

    @testset "transposeTimesByRow (EQ1, EQ2, GE3, GEK, unpacked)" begin
        out_idx = zeros(Int32, ncols)
        out_val = zeros(Float64, ncols)
        marked = zeros(UInt8, ncols)
        lookup = zeros(Int32, ncols)
        work = zeros(Float64, ncols)

        # EQ1: row 1 with pi0 = 2.0, scalar = -1.0
        n1 = JClp.clp_mat_tt_by_row_eq1_packed!(r_starts, r_cols, r_elems, 1, 2.0, out_idx, out_val, 1e-13, -1.0)
        @test n1 == 3
        @test out_idx[1:3] == Int32[1, 3, 5]
        @test out_val[1:3] ≈ [-4.0, 2.0, -6.0]

        # EQ2: rows 1 and 3
        fill!(out_val, 0.0)
        n2 = JClp.clp_mat_tt_by_row_eq2_packed!(r_starts, r_cols, r_elems, 1, 3, 2.0, 1.0, out_idx, out_val, marked, lookup, 1e-13, 1.0)
        ref2 = 2.0 .* A_dense[1, :] .+ 1.0 .* A_dense[3, :]
        got2 = zeros(Float64, ncols)
        for k in 1:n2
            got2[out_idx[k]] = out_val[k]
        end
        @test got2 ≈ ref2
        @test all(==(UInt8(0)), marked)

        # GE3: rows 1, 2, 3
        fill!(out_val, 0.0)
        which_r = Int32[1, 2, 3]
        pi_p = Float64[1.0, -2.0, 3.0]
        n3 = JClp.clp_mat_tt_by_row_ge3_packed!(r_starts, r_cols, r_elems, which_r, pi_p, 3, out_idx, out_val, work, 1e-13, -1.0)
        ref3 = -1.0 .* (1.0 .* A_dense[1, :] .- 2.0 .* A_dense[2, :] .+ 3.0 .* A_dense[3, :])
        got3 = zeros(Float64, ncols)
        for k in 1:n3
            got3[out_idx[k]] = out_val[k]
        end
        @test got3 ≈ ref3
        @test all(==(0.0), work)

        # GEK: same 3 rows
        fill!(out_val, 0.0)
        nk = JClp.clp_mat_tt_by_row_gek_packed!(r_starts, r_cols, r_elems, which_r, pi_p, 3, ncols, out_idx, out_val, 1e-13, -1.0)
        gotk = zeros(Float64, ncols)
        for k in 1:nk
            gotk[out_idx[k]] = out_val[k]
        end
        @test gotk ≈ ref3

        # Unpacked n=3
        fill!(out_val, 0.0)
        pi_full = Float64[1.0, -2.0, 3.0, 0.0]
        nu = JClp.clp_mat_tt_by_row_unpacked!(r_starts, r_cols, r_elems, which_r, pi_full, 3, out_idx, out_val, marked, 1e-13, -1.0)
        @test out_val ≈ ref3
        @test all(==(UInt8(0)), marked)
    end

    @testset "subsetTransposeTimes, unpack, unpackPacked, add" begin
        pi = Float64[1.0, -2.0, 3.0, -4.0]
        which = Int32[5, 2, 1]
        out_s = zeros(Float64, 3)
        JClp.clp_mat_subset_tt!(starts, nothing, false, indices, elements, which, 3, pi, rs, cs, out_s)
        full_sc = A_scaled(rs, A_dense, cs)' * pi
        @test out_s ≈ [full_sc[5], full_sc[2], full_sc[1]]

        # unpack & unpackPacked
        u_idx = zeros(Int32, nrows)
        u_val = zeros(Float64, nrows)
        nnz_u = JClp.clp_mat_unpack!(starts, lengths, indices, elements, 1, rs, cs, u_idx, u_val, 0)
        @test nnz_u == 2
        @test u_val ≈ A_scaled(rs, A_dense, cs)[:, 1]

        up_idx = zeros(Int32, nrows)
        up_val = zeros(Float64, nrows)
        nnz_up = JClp.clp_mat_unpack_packed!(starts, lengths, indices, elements, 1, rs, cs, up_idx, up_val)
        @test nnz_up == 2
        @test up_idx[1:2] == Int32[1, 3]
        @test up_val[1:2] ≈ [A_scaled(rs, A_dense, cs)[1, 1], A_scaled(rs, A_dense, cs)[3, 1]]

        # add_dense & add_indexed
        arr = zeros(Float64, nrows)
        JClp.clp_mat_add_dense!(starts, lengths, indices, elements, 1, 2.0, nothing, nothing, arr)
        @test arr ≈ 2.0 .* A_dense[:, 1]
        JClp.clp_mat_add_dense!(starts, lengths, indices, elements, 3, -1.0, rs, cs, arr)
        @test arr ≈ 2.0 .* A_dense[:, 1] .- A_scaled(rs, A_dense, cs)[:, 3]
    end

    @testset "countBasis, fillBasis, rangeOfElements, scaleRowCopy, reallyScale, partialPricing" begin
        which_b = Int32[1, 3, 5]
        @test JClp.clp_mat_count_basis(lengths, which_b, 3) == 6

        index_row_u = zeros(Int32, 10)
        start_u = Int32[2, 0, 0, 0] # suppose 1 slack already at pos 1
        row_count = zeros(Int32, nrows)
        col_count = zeros(Int32, 3)
        element_u = zeros(Float64, 10)
        JClp.clp_mat_fill_basis!(
            starts, lengths, indices, elements, false, which_b, 3,
            nothing, nothing, index_row_u, start_u, row_count, col_count, element_u,
        )
        @test start_u == Int32[2, 4, 6, 8]
        @test col_count == Int32[2, 2, 2]
        @test row_count == Int32[3, 0, 2, 1]
        @test index_row_u[2:7] == Int32[1, 3, 1, 4, 1, 3]
        @test element_u[2:7] == Float64[2.0, -3.0, -1.0, 5.0, 3.0, 2.0]

        sn, ln, sp, lp = JClp.clp_mat_range_of_elements(starts, lengths, elements, ncols)
        @test sn == -1.0
        @test ln == -3.0
        @test sp == 1.0
        @test lp == 5.0

        # scaleRowCopy & reallyScale
        r_el_copy = copy(r_elems)
        JClp.clp_mat_scale_row_copy!(r_starts, r_cols, r_el_copy, nrows, rs, cs)
        c_el_copy = copy(elements)
        JClp.clp_mat_really_scale!(starts, lengths, indices, c_el_copy, ncols, rs, cs)
        # Check entry (1,1): 2.0 * rs[1] * cs[1] = 1.5
        @test r_el_copy[1] ≈ 2.0 * rs[1] * cs[1]
        @test c_el_copy[1] ≈ 2.0 * rs[1] * cs[1]

        # partialPricing (unscaled)
        duals = Float64[1.0, 0.0, 0.0, 0.0]
        cost = Float64[0.0, 0.0, 0.0, 0.0, 0.0]
        rc = zeros(Float64, ncols)
        status = UInt8[JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_BASIC, JClp.CLP_STATUS_AT_UPPER, JClp.CLP_STATUS_AT_LOWER, JClp.CLP_STATUS_AT_LOWER]
        # col 1: dj = 0 - 1*2 = -2 -> atLower -> -dj = +2
        # col 3: dj = 0 - 1*(-1) = +1 -> atUpper -> +dj = +1
        # col 5: dj = 0 - 1*3 = -3 -> atLower -> -dj = +3 (BEST!)
        best_seq, n_wanted, saved_dj = JClp.clp_mat_partial_pricing!(
            starts, lengths, indices, elements, ncols, 0.0, 1.0,
            nothing, nothing, duals, cost, rc, status,
            0, 0, 1e-7, 10, 10, -1, -1, 1e-7,
        )
        @test best_seq == 5
        @test saved_dj ≈ -3.0
        @test rc[5] ≈ -3.0
    end
end

using JClp
using Test

@testset "B10: ClpFactorization (pure 1-based Julia)" begin
    @testset "Factorize packed (unscaled + FTRAN + BTRAN + FT replace)" begin
        # 4x4 LP matrix A (5 structural columns + 4 rows)
        # Basis: 1 slack on row 1 (col 1 of B = -e_1), plus structural cols 1, 2, 3
        # Col 1: [2, 1, 0, 0]
        # Col 2: [0, 3, 1, 0]
        # Col 3: [1, 0, 2, 4]
        # Col 4 (incoming): [0, 1, 1, 2]
        # Col 5: [1, 1, 1, 1]
        nrows = 4
        ncols_model = 5
        mat_starts  = Int32[1, 3, 5, 8, 11, 15]
        mat_lengths = Int32[2, 2, 3, 3, 4]
        mat_indices = Int32[1, 2,  2, 3,  1, 3, 4,  2, 3, 4,  1, 2, 3, 4]
        mat_elems   = Float64[2.0, 1.0,  3.0, 1.0,  1.0, 2.0, 4.0,  1.0, 1.0, 2.0,  1.0, 1.0, 1.0, 1.0]

        # pivot_temp: [slack_row_1, col_1, col_2, col_3] (1-based)
        pivot_temp = Int32[1, 1, 2, 3]
        pivot_var = zeros(Int32, nrows)

        cf = JClp.ClpFactorization()
        st = JClp.clp_fact_factorize_packed!(
            cf, nrows, ncols_model, 1, 4,
            pivot_temp, mat_starts, mat_lengths, mat_indices, mat_elems,
            false, nothing, nothing, 0, 1, pivot_var, nrows,
        )
        @test st == 0
        @test cf.lu.nrows == 4
        @test cf.lu.npivots == 0
        @test cf.last_number_pivots == 0
        @test JClp.clp_fact_space_for_ft(cf)

        # Build dense B in pivot_var order and check B * x == b
        B = zeros(Float64, 4, 4)
        for j in 1:4
            v = Int(pivot_var[j])
            if v > ncols_model
                B[v - ncols_model, j] = -1.0
            else
                s = Int(mat_starts[v])
                e = s + Int(mat_lengths[v]) - 1
                for k in s:e
                    B[Int(mat_indices[k]), j] = mat_elems[k]
                end
            end
        end

        # FTRAN: solve B * x = [1, 2, 3, 4]
        w_val = zeros(Float64, 16)
        w_idx = zeros(Int32, 16)
        r_val = zeros(Float64, 16)
        r_idx = zeros(Int32, 16)
        b = [1.0, 2.0, 3.0, 4.0]
        for i in 1:4
            r_val[i] = b[i]
            r_idx[i] = Int32(i)
        end
        nnz_out = JClp.clp_fact_update_column!(cf, w_val, w_idx, 0, r_val, r_idx, 4, false, false)
        @test nnz_out > 0
        x = copy(r_val[1:4])
        @test B * x ≈ b atol=1e-12

        # Zero-allocation check on FTRAN
        fill!(r_val, 0.0)
        for i in 1:4
            r_val[i] = b[i]
            r_idx[i] = Int32(i)
        end
        alloc_ftran = @allocated JClp.clp_fact_update_column!(cf, w_val, w_idx, 0, r_val, r_idx, 4, false, false)
        @test alloc_ftran == 0

        # BTRAN: solve B' * y = [1, -1, 2, 0.5]
        fill!(r_val, 0.0)
        c = [1.0, -1.0, 2.0, 0.5]
        for i in 1:4
            r_val[i] = c[i]
            r_idx[i] = Int32(i)
        end
        nnz_b = JClp.clp_fact_update_column_transpose!(cf, w_val, w_idx, 0, false, r_val, r_idx, 4, false)
        @test nnz_b > 0
        y = copy(r_val[1:4])
        @test B' * y ≈ c atol=1e-12

        # Zero-allocation check on BTRAN
        fill!(r_val, 0.0)
        for i in 1:4
            r_val[i] = c[i]
            r_idx[i] = Int32(i)
        end
        alloc_btran = @allocated JClp.clp_fact_update_column_transpose!(cf, w_val, w_idx, 0, false, r_val, r_idx, 4, false)
        @test alloc_btran == 0

        # UpdateColumnForDebug preserves stats
        nf_before = cf.lu.nftran_counts
        fill!(r_val, 0.0)
        r_val[1] = 1.0; r_idx[1] = Int32(1)
        JClp.clp_fact_update_column_for_debug!(cf, w_val, w_idx, 0, r_val, r_idx, 1, false, false)
        @test cf.lu.nftran_counts == nf_before

        # FT update + replace column 2 with structural col 4 = [0, 1, 1, 2]
        fill!(r_val, 0.0)
        r_val[2] = 1.0; r_val[3] = 1.0; r_val[4] = 2.0
        r_idx[1] = Int32(2); r_idx[2] = Int32(3); r_idx[3] = Int32(4)
        res_ft = JClp.clp_fact_update_column_ft!(cf, w_val, w_idx, r_val, r_idx, 3, false)
        @test res_ft > 0
        # Find a pivot row with non-zero entry
        piv_row = 0
        piv_val = 0.0
        for i in 1:4
            if abs(r_val[i]) > 1e-4
                piv_row = i
                piv_val = r_val[i]
                break
            end
        end
        @test piv_row > 0
        rc = JClp.clp_fact_replace_column!(
            cf, w_val, w_idx, r_val, r_idx, res_ft, false,
            piv_row, piv_val, false, 3, 2,
        )
        @test rc == 0
        @test cf.lu.npivots == 1

        # Verify updated basis B_new * x == b
        B[:, piv_row] .= [0.0, 1.0, 1.0, 2.0]
        fill!(r_val, 0.0)
        for i in 1:4
            r_val[i] = b[i]
            r_idx[i] = Int32(i)
        end
        JClp.clp_fact_update_column!(cf, w_val, w_idx, 0, r_val, r_idx, 4, false, false)
        @test B * r_val[1:4] ≈ b atol=1e-11

        # timeToRefactorize (zero-alloc)
        @test JClp.clp_fact_time_to_refactorize!(cf) == false
        @test (@allocated JClp.clp_fact_time_to_refactorize!(cf)) == 0

        # goSparse, getWeights, cleanUp
        JClp.clp_fact_go_sparse!(cf)
        weights = zeros(Int32, 4)
        JClp.clp_fact_get_weights!(cf, weights)
        @test all(w -> w >= 0, weights)
        JClp.clp_fact_clean_up!(cf)
        @test cf.lu.nftran_counts == 0
    end

    @testset "Factorize packed (scaled + singular detection)" begin
        nrows = 3
        ncols_model = 3
        mat_starts  = Int32[1, 3, 5, 7]
        mat_lengths = Int32[2, 2, 2]
        mat_indices = Int32[1, 2,  2, 3,  1, 3]
        mat_elems   = Float64[2.0, 1.0,  3.0, 1.0,  1.0, 2.0]
        row_scale   = Float64[2.0, 0.5, 4.0]
        col_scale   = Float64[0.5, 2.0, 0.25]
        pivot_temp  = Int32[1, 2, 3]
        pivot_var   = zeros(Int32, 3)

        cf = JClp.ClpFactorization()
        st = JClp.clp_fact_factorize_packed!(
            cf, nrows, ncols_model, 0, 3,
            pivot_temp, mat_starts, mat_lengths, mat_indices, mat_elems,
            false, row_scale, col_scale, 0, 1, pivot_var, nrows,
        )
        @test st == 0

        # Singular basis (col 2 == 2 * col 1)
        mat_elems_sing = Float64[1.0, 2.0,  2.0, 4.0,  1.0, 1.0]
        mat_indices_sing = Int32[1, 2,  1, 2,  1, 3]
        st_sing = JClp.clp_fact_factorize_packed!(
            cf, nrows, ncols_model, 0, 3,
            pivot_temp, mat_starts, mat_lengths, mat_indices_sing, mat_elems_sing,
            false, nothing, nothing, 0, 1, pivot_var, nrows,
        )
        @test st_sing == -1
    end

    @testset "Tolerances and copy" begin
        cf = JClp.ClpFactorization()
        JClp.clp_fact_safer_tolerances!(cf, 1.0e-14, 0.25)
        @test cf.lu.zero_tol == 1.0e-14
        @test cf.lu.pivot_tol == 0.25
        JClp.clp_fact_safer_tolerances!(cf, 1.0e-12, -2.0)
        @test cf.lu.zero_tol == 1.0e-14
        @test cf.lu.pivot_tol == 0.5

        cf2 = JClp.copy_clp_factorization(cf)
        @test cf2.lu.zero_tol == cf.lu.zero_tol
        @test cf2.lu.pivot_tol == cf.lu.pivot_tol
    end
end

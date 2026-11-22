@testset "B4 CoinDenseFactorization" begin
    @testset "Factor, FTRAN, BTRAN (unpacked & packed) + multi-pivot PFI updates" begin
        # Non-trivial pivoted 5x5 basis
        B = Float64[
            0.0  2.0  0.0  1.0 -1.0;
            3.0 -1.0  2.0  0.0  0.5;
            1.0  0.0 -4.0  2.0  1.0;
            0.0  1.5  1.0  3.0  0.0;
            -2.0 0.5  0.0 -1.0  4.0
        ]
        n = 5
        df = JClp.CoinDenseFactorization(20)
        @test JClp.factor_dense!(df, B) == Int32(0)
        @test df.number_good_u == Int32(n)

        # Note: in CoinDenseFactorization, pivotVariable[pivotRow_[i+n]] = i (column i is mapped to row pivotRow_[i+n])
        # So solve solves P * B_eff * x = b where column j of the basis has index pivotRow_[j + n] in the returned vector!
        # Let's build the exact effective basis matrix B_perm whose column pivotRow_[j + n] + 1 is B[:, j + 1]:
        B_eff = zeros(Float64, n, n)
        for j in 0:(n - 1)
            B_eff[:, Int(df.pivot_row[j + n + 1]) + 1] = B[:, j + 1]
        end

        r1 = JClp.CoinIndexedVector(n)
        r2 = JClp.CoinIndexedVector(n)

        # Helper: solve B_eff * x = rhs (FTRAN) in unpacked and packed modes
        function test_ftran_btran(Bcur::Matrix{Float64}, rhs::Vector{Float64})
            # Unpacked FTRAN
            JClp.clear!(r2)
            for i in 1:n
                rhs[i] != 0.0 && JClp.quick_insert!(r2, Int32(i - 1), rhs[i])
            end
            JClp.update_column!(df, r1, r2)
            @test JClp.check_clear(r1)
            @test r2.elements ≈ Bcur \ rhs atol=1e-12

            # Packed FTRAN
            JClp.clear!(r2)
            nz_i = Int32[i - 1 for i in 1:n if rhs[i] != 0.0]
            nz_v = Float64[rhs[i] for i in 1:n if rhs[i] != 0.0]
            JClp.create_packed!(r2, Int32(length(nz_i)), nz_i, nz_v)
            JClp.update_column!(df, r1, r2)
            @test JClp.check_clear(r1)
            @test JClp.packed_mode(r2)
            x_packed = zeros(Float64, n)
            for k in 1:Int(r2.n_elements)
                x_packed[Int(r2.indices[k]) + 1] = r2.elements[k]
            end
            @test x_packed ≈ Bcur \ rhs atol=1e-12

            # Unpacked BTRAN
            JClp.clear!(r2)
            for i in 1:n
                rhs[i] != 0.0 && JClp.quick_insert!(r2, Int32(i - 1), rhs[i])
            end
            JClp.update_column_transpose!(df, r1, r2)
            @test JClp.check_clear(r1)
            @test r2.elements ≈ Bcur' \ rhs atol=1e-12

            # Packed BTRAN
            JClp.clear!(r2)
            JClp.create_packed!(r2, Int32(length(nz_i)), nz_i, nz_v)
            JClp.update_column_transpose!(df, r1, r2)
            @test JClp.check_clear(r1)
            y_packed = zeros(Float64, n)
            for k in 1:Int(r2.n_elements)
                y_packed[Int(r2.indices[k]) + 1] = r2.elements[k]
            end
            @test y_packed ≈ Bcur' \ rhs atol=1e-12
        end

        test_ftran_btran(B_eff, Float64[1.0, -2.5, 3.0, 0.5, -1.0])

        # Perform 3 successive basis column replacements (PFI updates) and verify FTRAN & BTRAN after each!
        new_cols = [
            (Int32(2), Float64[1.0, 2.0, -1.0, 0.5, 3.0]),
            (Int32(0), Float64[-0.5, 1.0, 2.0, -3.0, 1.5]),
            (Int32(4), Float64[2.0, 0.0, 1.5, 1.0, -2.5]),
        ]
        for (piv_row0, a_new) in new_cols
            # Compute alpha = B_eff \ a_new via FTRAN (packed)
            JClp.clear!(r2)
            nz_i = Int32[i - 1 for i in 1:n if a_new[i] != 0.0]
            nz_v = Float64[a_new[i] for i in 1:n if a_new[i] != 0.0]
            JClp.create_packed!(r2, Int32(length(nz_i)), nz_i, nz_v)
            JClp.update_column!(df, r1, r2)
            piv_val = 0.0
            for k in 1:Int(r2.n_elements)
                if r2.indices[k] == piv_row0
                    piv_val = r2.elements[k]
                end
            end
            @test abs(piv_val) > 1e-6
            @test JClp.replace_column!(df, r2, piv_row0, piv_val) == Int32(0)
            B_eff[:, Int(piv_row0) + 1] = a_new
            test_ftran_btran(B_eff, Float64[-1.5, 0.75, 2.0, -0.5, 4.0])
        end
        @test df.number_pivots == Int32(3)
    end

    @testset "Singular matrix detection & zero-alloc hot FTRAN/BTRAN" begin
        S = Float64[1.0 2.0; 2.0 4.0]
        df = JClp.CoinDenseFactorization(5)
        @test JClp.factor_dense!(df, S) == Int32(-1)
        @test df.number_good_u == Int32(1)

        # Zero-allocation check on C ABI update_column / update_column_transpose
        B = Float64[2.0 1.0 0.0; 0.5 3.0 -1.0; -1.0 0.5 4.0]
        df2 = JClp.CoinDenseFactorization(10)
        @test JClp.factor_dense!(df2, B) == Int32(0)
        r1 = zeros(Float64, 3)
        r2 = Float64[1.0, 2.0, 3.0]
        idx = Int32[0, 1, 2]
        function _bench_cdf(df2, r1, r2, idx)
            r2 .= (1.0, 2.0, 3.0); idx .= (Int32(0), Int32(1), Int32(2))
            a1 = GC.@preserve df2 r1 r2 idx @allocated JClp.jclp_cdf_update_column(
                Cint(3), Cint(0), Cint(0), pointer(df2.elements), pointer(df2.pivot_row), 1e-13,
                pointer(r1), pointer(r2), pointer(idx), Cint(3), Cint(0), Cint(0))
            r2 .= (1.0, 2.0, 3.0); idx .= (Int32(0), Int32(1), Int32(2))
            a2 = GC.@preserve df2 r1 r2 idx @allocated JClp.jclp_cdf_update_column_transpose(
                Cint(3), Cint(0), Cint(0), pointer(df2.elements), pointer(df2.pivot_row), 1e-13,
                pointer(r1), pointer(r2), pointer(idx), Cint(3), Cint(0))
            return a1 + a2
        end
        _bench_cdf(df2, r1, r2, idx)
        @test _bench_cdf(df2, r1, r2, idx) == 0

        # LAPACK Accelerate mode (solve_mode = 1) + PFI updates
        df_lap = JClp.CoinDenseFactorization(10; solve_mode = 1)
        @test JClp.factor_dense!(df_lap, B) == Int32(0)
        @test df_lap.solve_mode == Int32(1)
        v1 = JClp.CoinIndexedVector(3)
        v2 = JClp.CoinIndexedVector(3)
        rhs = Float64[1.5, -2.0, 3.5]
        for i in 1:3
            JClp.quick_insert!(v2, Int32(i - 1), rhs[i])
        end
        JClp.update_column!(df_lap, v1, v2)
        @test v2.elements ≈ B \ rhs atol=1e-12
        JClp.clear!(v2)
        for i in 1:3
            JClp.quick_insert!(v2, Int32(i - 1), rhs[i])
        end
        JClp.update_column_transpose!(df_lap, v1, v2)
        @test v2.elements ≈ B' \ rhs atol=1e-12
    end
end

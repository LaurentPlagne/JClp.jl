@testset "B3 CoinPackedMatrix" begin
    # Helper to construct dense matrix from CoinPackedMatrix
    function to_dense(m::JClp.CoinPackedMatrix)
        nr = Int(JClp.get_num_rows(m))
        nc = Int(JClp.get_num_cols(m))
        A = zeros(Float64, nr, nc)
        for i in 1:Int(m.major_dim)
            s = Int(m.start[i])
            l = Int(m.length[i])
            for k in 1:l
                idx = Int(m.index[s + k]) + 1
                val = m.element[s + k]
                if m.col_ordered
                    A[idx, i] += val
                else
                    A[i, idx] += val
                end
            end
        end
        return A
    end

    @testset "reverse_ordered_copy_of!, reverse_ordering!, times!, transpose_times!" begin
        # 3x4 col-ordered matrix with an intentional gap between col 1 and col 2
        # A = [1  0  5  0;
        #      0  3  0  7;
        #      2  4  6  0]
        el = Float64[1.0, 2.0, 999.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ind = Int32[0, 2, 0, 1, 2, 0, 2, 1]
        st = Int32[0, 3, 5, 7, 8]
        len = Int32[2, 2, 2, 1]
        M = JClp.CoinPackedMatrix(true, 3, 4, el, ind, st, len)

        A_expected = Float64[
            1.0 0.0 5.0 0.0;
            0.0 3.0 0.0 7.0;
            2.0 4.0 6.0 0.0
        ]
        @test to_dense(M) == A_expected

        # count_ortho_length!
        ortho = Int32[]
        JClp.count_ortho_length!(ortho, M)
        @test ortho == Int32[2, 2, 3]

        # reverse_ordered_copy_of! (CSC -> CSR)
        R = JClp.CoinPackedMatrix()
        JClp.reverse_ordered_copy_of!(R, M)
        @test !JClp.is_col_ordered(R)
        @test JClp.get_num_rows(R) == Int32(3)
        @test JClp.get_num_cols(R) == Int32(4)
        @test JClp.get_num_elements(R) == Int32(7)
        @test to_dense(R) == A_expected

        # reverse_ordering! in-place roundtrip
        R2 = JClp.CoinPackedMatrix()
        JClp.reverse_ordered_copy_of!(R2, R)
        @test JClp.is_col_ordered(R2)
        @test to_dense(R2) == A_expected
        JClp.reverse_ordering!(R2)
        @test !JClp.is_col_ordered(R2)
        @test to_dense(R2) == A_expected

        # times! and transpose_times! on both CSC and CSR
        x = Float64[1.5, -2.0, 3.0, 0.5]
        y_csc = zeros(Float64, 3)
        y_csr = zeros(Float64, 3)
        JClp.times!(y_csc, M, x)
        JClp.times!(y_csr, R, x)
        @test y_csc ≈ A_expected * x
        @test y_csr ≈ A_expected * x

        w = Float64[2.0, -1.0, 4.0]
        yt_csc = zeros(Float64, 4)
        yt_csr = zeros(Float64, 4)
        JClp.transpose_times!(yt_csc, M, w)
        JClp.transpose_times!(yt_csr, R, w)
        @test yt_csc ≈ A_expected' * w
        @test yt_csr ≈ A_expected' * w
    end

    @testset "remove_gaps!, eliminate_duplicates!, clean_matrix!" begin
        # Matrix with gaps, duplicates, tiny entries, and unsorted indices
        el = Float64[2.0, 1.0, -1.0, 0.0, 5.0, 3.0, 1e-15, -2.0]
        ind = Int32[2, 0, 0, 1, 1, 2, 0, 1]
        st = Int32[0, 3, 4, 8]
        len = Int32[3, 0, 4]
        M = JClp.CoinPackedMatrix(true, 3, 3, el, ind, st, len)
        M_copy = JClp.CoinPackedMatrix(true, 3, 3, el, ind, st, len)

        # remove_gaps!(remove_value = -1.0) compacts gaps without dropping entries
        JClp.remove_gaps!(M, -1.0)
        @test M.start == Int32[0, 3, 3, 7]
        @test M.size == Int32(7)

        # remove_gaps!(remove_value = 1e-12) drops tiny entries too
        JClp.remove_gaps!(M, 1e-12)
        @test M.size == Int32(6)

        # clean_matrix! merges duplicates (row 0: 1.0 + (-1.0) = 0 -> dropped), drops 1e-15, sorts indices!
        elim = JClp.clean_matrix!(M_copy, 1e-12)
        @test elim == Int32(4) # two cancelling row-0 + one 1e-15 + one merged dup row-1
        @test M_copy.size == Int32(3)
        @test M_copy.start == Int32[0, 1, 1, 3]
        @test M_copy.index == Int32[2, 1, 2]
        @test to_dense(M_copy) ≈ Float64[0.0 0.0 0.0; 0.0 0.0 3.0; 2.0 0.0 3.0]
    end

    @testset "Zero-allocation checks on B3 C ABI kernels" begin
        function _bench_cpm_alloc(st, len, ind, el, x, y, ortho)
            a1 = GC.@preserve st len ind ortho @allocated JClp.jclp_cpm_count_ortho_length(
                Cint(4), Cint(3), pointer(st), pointer(len), pointer(ind), pointer(ortho))
            a2 = GC.@preserve st len ind el x y @allocated JClp.jclp_cpm_times_major(
                Cint(4), Cint(3), pointer(st), pointer(len), pointer(ind), pointer(el), pointer(x), pointer(y))
            a3 = GC.@preserve st len ind el y x @allocated JClp.jclp_cpm_times_minor(
                Cint(4), pointer(st), pointer(len), pointer(ind), pointer(el), pointer(y), pointer(x))
            return a1 + a2 + a3
        end
        el = Float64[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        ind = Int32[0, 2, 1, 2, 0, 2, 1]
        st = Int32[0, 2, 4, 6, 7]
        len = Int32[2, 2, 2, 1]
        x = Float64[1.0, 2.0, 3.0, 4.0]
        y = zeros(Float64, 3)
        ortho = zeros(Int32, 3)
        _bench_cpm_alloc(st, len, ind, el, x, y, ortho)
        @test _bench_cpm_alloc(st, len, ind, el, x, y, ortho) == 0
    end
end

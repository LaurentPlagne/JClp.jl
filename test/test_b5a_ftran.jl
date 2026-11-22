@testset "B5a CoinFactorization FTRAN (Densish/Sparsish/Sparse, updateColumn, updateColumnFT, TwoColumnsFT, 0-alloc)" begin
    # Same 4x4 synthetic LU basis B = L * (I + U_strict) * D
    n = 4
    L = Float64[
        1.0  0.0 0.0 0.0;
        2.0  1.0 0.0 0.0;
        0.0 -1.0 1.0 0.0;
        1.0  0.0 2.0 1.0
    ]
    U_strict = Float64[
        0.0 -0.5  0.0  1.5;
        0.0  0.0  0.25 -0.5;
        0.0  0.0  0.0 -0.3;
        0.0  0.0  0.0  0.0
    ]
    d_diag = Float64[2.0, 4.0, -3.0, 5.0]
    D = zeros(Float64, n, n)
    for i in 1:n; D[i, i] = d_diag[i]; end
    B = L * (Float64[i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n] + U_strict) * D

    # U by column (extra room at end for FT saved column!)
    # col 0: empty
    # col 1: (0, -0.5)
    # col 2: (1, 0.25)
    # col 3: (0, 1.5), (1, -0.5), (2, -0.3)
    elementU = zeros(Float64, 32)
    indexRowU = zeros(Int32, 32)
    elementU[1:5] = Float64[-0.5, 0.25, 1.5, -0.5, -0.3]
    indexRowU[1:5] = Int32[0, 1, 0, 1, 2]
    startColumnU = zeros(Int32, 17) # maximumColumnsExtra = 16 -> index 17 is startColumnU[maxColsExtra]
    startColumnU[1:5] = Int32[0, 0, 1, 2, 5]
    startColumnU[17] = Int32(5)
    numberInColumn = zeros(Int32, 17)
    numberInColumn[1:4] = Int32[0, 1, 1, 3]

    # L by column
    elementL = Float64[2.0, 1.0, -1.0, 2.0]
    indexRowL = Int32[1, 3, 2, 3]
    startColumnL = Int32[0, 2, 3, 4, 4]

    pivotRegion = 1.0 ./ d_diag
    permute = Int32[0, 1, 2, 3]
    pivotColumnBack = Int32[0, 1, 2, 3]
    sparseArray = zeros(Int32, 4 * 16)

    cnt_ftran = Ref{Int32}(0)
    c_in = Ref{Float64}(0.0)
    c_l = Ref{Float64}(0.0)
    c_r = Ref{Float64}(0.0)
    c_u = Ref{Float64}(0.0)

    rhs = Float64[1.0, -2.0, 0.5, 3.0]
    x_expected = B \ rhs

    configs = [
        (Int32(0),   Int32(0),   0.0,  0.0,  0.0),  # L densish, U densish
        (Int32(1),   Int32(100), 1.0,  1.0,  1.0),  # L sparsish, U sparsish
        (Int32(100), Int32(200), 1.0,  1.0,  1.0),  # L sparse DFS, U sparse DFS
    ]

    for (st1, st2, avgL, avgR, avgU) in configs
        for packed in (false, true)
            r1 = JClp.CoinIndexedVector(n)
            r2 = JClp.CoinIndexedVector(n)
            if packed
                JClp.create_packed!(r2, Int32(n), Int32[0, 1, 2, 3], rhs)
            else
                for i in 1:n
                    JClp.quick_insert!(r2, Int32(i - 1), rhs[i])
                end
            end

            GC.@preserve elementU indexRowU startColumnU numberInColumn elementL indexRowL startColumnL pivotRegion permute pivotColumnBack sparseArray cnt_ftran c_in c_l c_r c_u r1 r2 begin
                cf = JClp.CoinFactorizationPodView(
                    pointer(elementU), pointer(indexRowU), pointer(startColumnU), pointer(numberInColumn), C_NULL,
                    C_NULL, C_NULL, C_NULL, C_NULL,
                    pointer(elementL), pointer(indexRowL), pointer(startColumnL),
                    C_NULL, C_NULL, C_NULL,
                    C_NULL, C_NULL, C_NULL,
                    pointer(pivotRegion), C_NULL, pointer(pivotColumnBack),
                    pointer(permute), C_NULL, C_NULL, C_NULL,
                    pointer(sparseArray), C_NULL, C_NULL,
                    C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                    Base.unsafe_convert(Ptr{Int32}, cnt_ftran),
                    Base.unsafe_convert(Ptr{Float64}, c_in),
                    Base.unsafe_convert(Ptr{Float64}, c_l),
                    Base.unsafe_convert(Ptr{Float64}, c_r),
                    Base.unsafe_convert(Ptr{Float64}, c_u),
                    C_NULL, C_NULL,
                    1e-13, -1.0, 0.0, 0.0, 0.0, avgL, avgR, avgU,
                    Int32(n), Int32(n), Int32(16), Int32(n), Int32(16), Int32(n), Int32(n), Int32(n), Int32(0),
                    Int32(0), Int32(0), Int32(0), Int32(32), Int32(0), Int32(0), Int32(0), Int32(4),
                    st1, st2, Int32(1), Int32(1), Int32(0),
                )
                cf_ref = Ref(cf)
                r1_pod = Ref(JClp.CoinIndexedVecPod(pointer(r1.elements), pointer(r1.indices), r1.n_elements, r1.capacity, Int32(r1.packed_mode), Int32(0)))
                r2_pod = Ref(JClp.CoinIndexedVecPod(pointer(r2.elements), pointer(r2.indices), r2.n_elements, r2.capacity, Int32(r2.packed_mode), Int32(0)))
                nnz = JClp.jclp_cf_update_column(
                    Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod),
                    Cint(0),
                )
                JClp.set_num_elements!(r1, Int32(0))
                JClp.set_num_elements!(r2, Int32(nnz))
            end

            @test JClp.check_clear(r1)
            @test all(==(Int32(0)), sparseArray[(3 * 16 + 1):end])
            x_got = zeros(Float64, n)
            if packed
                for k in 1:Int(r2.n_elements)
                    x_got[Int(r2.indices[k]) + 1] = r2.elements[k]
                end
            else
                x_got .= r2.elements
            end
            @test x_got ≈ x_expected atol=1e-12
        end
    end

    # Test updateColumnFT (saves L^{-1} rhs into U column `numberColumnsExtra` and returns +nnz)
    startColumnU[17] = Int32(5)
    r1 = JClp.CoinIndexedVector(n)
    r2 = JClp.CoinIndexedVector(n)
    JClp.create_packed!(r2, Int32(n), Int32[0, 1, 2, 3], rhs)
    GC.@preserve elementU indexRowU startColumnU numberInColumn elementL indexRowL startColumnL pivotRegion permute pivotColumnBack sparseArray cnt_ftran c_in c_l c_r c_u r1 r2 begin
        cf = JClp.CoinFactorizationPodView(
            pointer(elementU), pointer(indexRowU), pointer(startColumnU), pointer(numberInColumn), C_NULL,
            C_NULL, C_NULL, C_NULL, C_NULL,
            pointer(elementL), pointer(indexRowL), pointer(startColumnL),
            C_NULL, C_NULL, C_NULL,
            C_NULL, C_NULL, C_NULL,
            pointer(pivotRegion), C_NULL, pointer(pivotColumnBack),
            pointer(permute), C_NULL, C_NULL, C_NULL,
            pointer(sparseArray), C_NULL, C_NULL,
            C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
            Base.unsafe_convert(Ptr{Int32}, cnt_ftran),
            Base.unsafe_convert(Ptr{Float64}, c_in),
            Base.unsafe_convert(Ptr{Float64}, c_l),
            Base.unsafe_convert(Ptr{Float64}, c_r),
            Base.unsafe_convert(Ptr{Float64}, c_u),
            C_NULL, C_NULL,
            1e-13, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
            Int32(n), Int32(n), Int32(16), Int32(n), Int32(16), Int32(n), Int32(n), Int32(n), Int32(0),
            Int32(0), Int32(0), Int32(0), Int32(32), Int32(0), Int32(0), Int32(0), Int32(4),
            Int32(100), Int32(200), Int32(1), Int32(1), Int32(0),
        )
        cf_ref = Ref(cf)
        r1_pod = Ref(JClp.CoinIndexedVecPod(pointer(r1.elements), pointer(r1.indices), Int32(0), Int32(n), Int32(0), Int32(0)))
        r2_pod = Ref(JClp.CoinIndexedVecPod(pointer(r2.elements), pointer(r2.indices), Int32(n), Int32(n), Int32(1), Int32(0)))
        rc = JClp.jclp_cf_update_column_ft(
            Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
            Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
            Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod),
        )
        @test rc == Cint(4)
        JClp.set_num_elements!(r2, Int32(rc))
        x_ft = zeros(Float64, n)
        for k in 1:Int(r2.n_elements)
            x_ft[Int(r2.indices[k]) + 1] = r2.elements[k]
        end
        @test x_ft ≈ x_expected atol=1e-12
        # Check saved FT column matches L \ rhs!
        linv_rhs = L \ rhs
        saved = zeros(Float64, n)
        s = Int(startColumnU[n + 1])
        len_saved = Int(numberInColumn[n + 1])
        for k in 1:len_saved
            saved[Int(indexRowU[s + k]) + 1] = elementU[s + k]
        end
        @test saved ≈ linv_rhs atol=1e-12

        # Zero-allocation check on jclp_cf_update_column
        function _bench_ftran(cf_ref, r1_pod, r2_pod, r2, rhs)
            for i in 1:length(rhs)
                r2.elements[i] = rhs[i]
                r2.indices[i] = Int32(i - 1)
            end
            return GC.@preserve cf_ref r1_pod r2_pod r2 @allocated JClp.jclp_cf_update_column(
                Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod),
                Cint(0),
            )
        end
        _bench_ftran(cf_ref, r1_pod, r2_pod, r2, rhs)
        @test _bench_ftran(cf_ref, r1_pod, r2_pod, r2, rhs) == 0
    end
end

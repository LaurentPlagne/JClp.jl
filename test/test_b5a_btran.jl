@testset "B5a CoinFactorization BTRAN (all modes: Densish/Sparsish/Sparse/ByRow/R/0-alloc)" begin
    # Build a synthetic 4x4 sparse LU factorization B = L * U (with diagonal D stored inverted in pivotRegion)
    # L = [1  0  0  0;     U = [2  -1  0  3;
    #      2  1  0  0;          0   4  1 -2;
    #      0 -1  1  0;          0   0 -3  1;
    #      1  0  2  1]          0   0  0  5]
    # Note CoinFactorization convention:
    # - U off-diagonals stored in elementU (by col and row-to-col conversion), diagonal inverse in pivotRegion = [1/2, 1/4, -1/3, 1/5]
    # - Actually in CoinFactorization: B^T y = rhs => U^T L^T y = rhs:
    #   First region[i] *= pivotRegion[i]  (or in U: row i of U has off-diagonal entries j > i with coeff U[i,j]/U[i,i] OR raw U[i,j]?)
    #   Check CoinFactorization4.cpp:1248 & 1705:
    #     Step 1: region[i] *= pivotRegion[i]  (before U!) -> wait: in U, U_eff has 1 on diag and u_{i,j} = U[i,j]/U[j,j]?
    #     In BTRAN U: region[j] -= element[i,j] * region[i], where region[i] ALREADY multiplied by pivotRegion[i] (= 1/U[i,i])!
    #     Wait: that means (D^{-1} + U_strict)^T -> actually U_true = (I + U_strict) * D  where D = diag(1/pivotRegion),
    #     so U_true^{-T} = (I + U_strict)^{-T} * D^{-1}!
    #     Let's verify: (U_true^T z)_i = D_i * (z_i + sum_{k<i} U_strict[k,i] z_k) = rhs_i
    #     Wait: first divide rhs_i by D_i => w_i = rhs_i * pivotRegion[i] = z_i + sum_{k<i} U_strict[k,i] z_k
    #     Wait: if region[iRow] -= element[i, iRow] * pivotValue where pivotValue = region[i], then:
    #     z_0 = w_0; z_1 = w_1 - U[0,1]*z_0; z_2 = w_2 - U[0,2]*z_0 - U[1,2]*z_1 ...
    #     That solves (I + U_strict^T) z = D^{-1} rhs => D (I + U_strict^T) z = rhs => ((I + U_strict) D)^T z = rhs!
    #     And then L^T y = z solves (I + L_strict^T) y = z (L has unit diagonal)!
    #     So B = L * (I + U_strict) * D! Let's verify ALL 4 BTRAN modes (densish, sparsish, sparse, byRow) match B' \ rhs!

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
    U_full = (Float64[i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n] + U_strict) * D
    B = L * U_full

    # Build CoinFactorization arrays for U (by row) and L (by col and by row)
    # U strict nonzeros: (0,1):-0.5, (0,3):1.5, (1,2):0.25, (1,3):-0.5, (2,3):-0.3
    elementU = Float64[-0.5, 1.5, 0.25, -0.5, -0.3]
    indexColumnU = Int32[1, 3, 2, 3, 3]
    convertRowToColumnU = Int32[0, 1, 2, 3, 4]
    startRowU = Int32[0, 2, 4, 5, 5]
    numberInRow = Int32[2, 2, 1, 0]

    # L strict by column: col 0 -> (1,2.0),(3,1.0); col 1 -> (2,-1.0); col 2 -> (3,2.0); col 3 -> empty
    elementL = Float64[2.0, 1.0, -1.0, 2.0]
    indexRowL = Int32[1, 3, 2, 3]
    startColumnL = Int32[0, 2, 3, 4, 4]

    # L strict by row: row 0 -> empty; row 1 -> (0,2.0); row 2 -> (1,-1.0); row 3 -> (0,1.0),(2,2.0)
    elementByRowL = Float64[2.0, -1.0, 1.0, 2.0]
    indexColumnL = Int32[0, 1, 0, 2]
    startRowL = Int32[0, 0, 1, 2, 4]

    pivotRegion = 1.0 ./ d_diag
    pivotColumn = Int32[0, 1, 2, 3]
    pivotColumnBack = Int32[0, 1, 2, 3]
    permute = Int32[0, 1, 2, 3]
    sparseArray = zeros(Int32, 4 * 16) # >= 4 * maximumRowsExtra (16)

    cnt_btran = Ref{Int32}(0)
    c_in = Ref{Float64}(0.0)
    c_u = Ref{Float64}(0.0)
    c_r = Ref{Float64}(0.0)
    c_l = Ref{Float64}(0.0)

    rhs = Float64[1.0, -2.0, 0.5, 3.0]
    y_expected = B' \ rhs

    # Test ALL dispatch modes:
    # (sparseThreshold=0 -> U densish + L densish by col),
    # (sparseThreshold=1, sparseThreshold2=2, avg=10 -> U densish + L by row),
    # (sparseThreshold=1, sparseThreshold2=100, avg=1.0 -> U sparsish + L sparsish),
    # (sparseThreshold=100, sparseThreshold2=200, avg=1.0 -> U sparse DFS + L sparse DFS)
    configs = [
        (Int32(0),   Int32(0),   0.0,  0.0),  # U densish, L densish (by col)
        (Int32(1),   Int32(2),   10.0, 10.0), # U densish, L by row
        (Int32(1),   Int32(100), 1.0,  1.0),  # U sparsish, L sparsish
        (Int32(100), Int32(200), 1.0,  1.0),  # U sparse DFS, L sparse DFS
    ]

    for (st1, st2, avgU, avgL) in configs
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

            GC.@preserve elementU indexColumnU convertRowToColumnU startRowU numberInRow elementL indexRowL startColumnL elementByRowL indexColumnL startRowL pivotRegion pivotColumn pivotColumnBack permute sparseArray cnt_btran c_in c_u c_r c_l r1 r2 begin
                cf = JClp.CoinFactorizationPodView(
                    pointer(elementU), C_NULL, C_NULL, C_NULL, C_NULL,
                    pointer(startRowU), pointer(numberInRow), pointer(indexColumnU), pointer(convertRowToColumnU),
                    pointer(elementL), pointer(indexRowL), pointer(startColumnL),
                    pointer(elementByRowL), pointer(indexColumnL), pointer(startRowL),
                    C_NULL, C_NULL, C_NULL,
                    pointer(pivotRegion), pointer(pivotColumn), pointer(pivotColumnBack),
                    pointer(permute), C_NULL, C_NULL, C_NULL,
                    pointer(sparseArray), C_NULL, C_NULL,
                    Base.unsafe_convert(Ptr{Int32}, cnt_btran),
                    Base.unsafe_convert(Ptr{Float64}, c_in),
                    Base.unsafe_convert(Ptr{Float64}, c_u),
                    Base.unsafe_convert(Ptr{Float64}, c_r),
                    Base.unsafe_convert(Ptr{Float64}, c_l),
                    C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                    C_NULL, C_NULL,
                    1e-13, -1.0, avgU, 0.0, avgL, 0.0, 0.0, 0.0,
                    Int32(n), Int32(n), Int32(16), Int32(n), Int32(16), Int32(n), Int32(n), Int32(n), Int32(0),
                    Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
                    st1, st2, Int32(1), Int32(1), Int32(0),
                )
                cf_ref = Ref(cf)
                r1_pod = Ref(JClp.CoinIndexedVecPod(pointer(r1.elements), pointer(r1.indices), r1.n_elements, r1.capacity, Int32(r1.packed_mode), Int32(0)))
                r2_pod = Ref(JClp.CoinIndexedVecPod(pointer(r2.elements), pointer(r2.indices), r2.n_elements, r2.capacity, Int32(r2.packed_mode), Int32(0)))
                nnz = JClp.jclp_cf_update_column_transpose(
                    Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod),
                )
                JClp.set_num_elements!(r1, Int32(0))
                JClp.set_num_elements!(r2, Int32(nnz))
            end

            @test JClp.check_clear(r1)
            # Check mark bytes are completely zeroed after sparse/sparsish!
            @test all(==(Int32(0)), sparseArray[(3 * 16 + 1):end])
            y_got = zeros(Float64, n)
            if packed
                for k in 1:Int(r2.n_elements)
                    y_got[Int(r2.indices[k]) + 1] = r2.elements[k]
                end
            else
                y_got .= r2.elements
            end
            @test y_got ≈ y_expected atol=1e-12
        end
    end

    # Zero-allocation check on hot jclp_cf_update_column_transpose
    function _bench_btran(cf_ref, r1_pod, r2_pod, r2, rhs)
        for i in 1:length(rhs)
            r2.elements[i] = rhs[i]
            r2.indices[i] = Int32(i - 1)
        end
        return GC.@preserve cf_ref r1_pod r2_pod r2 @allocated JClp.jclp_cf_update_column_transpose(
            Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
            Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
            Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod),
        )
    end
    r1 = JClp.CoinIndexedVector(n)
    r2 = JClp.CoinIndexedVector(n)
    GC.@preserve elementU indexColumnU convertRowToColumnU startRowU numberInRow elementL indexRowL startColumnL elementByRowL indexColumnL startRowL pivotRegion pivotColumn pivotColumnBack permute sparseArray cnt_btran c_in c_u c_r c_l r1 r2 begin
        cf = JClp.CoinFactorizationPodView(
            pointer(elementU), C_NULL, C_NULL, C_NULL, C_NULL,
            pointer(startRowU), pointer(numberInRow), pointer(indexColumnU), pointer(convertRowToColumnU),
            pointer(elementL), pointer(indexRowL), pointer(startColumnL),
            pointer(elementByRowL), pointer(indexColumnL), pointer(startRowL),
            C_NULL, C_NULL, C_NULL,
            pointer(pivotRegion), pointer(pivotColumn), pointer(pivotColumnBack),
            pointer(permute), C_NULL, C_NULL, C_NULL,
            pointer(sparseArray), C_NULL, C_NULL,
            Base.unsafe_convert(Ptr{Int32}, cnt_btran),
            Base.unsafe_convert(Ptr{Float64}, c_in),
            Base.unsafe_convert(Ptr{Float64}, c_u),
            Base.unsafe_convert(Ptr{Float64}, c_r),
            Base.unsafe_convert(Ptr{Float64}, c_l),
            C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
            C_NULL, C_NULL,
            1e-13, -1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
            Int32(n), Int32(n), Int32(16), Int32(n), Int32(16), Int32(n), Int32(n), Int32(n), Int32(0),
            Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
            Int32(100), Int32(200), Int32(1), Int32(1), Int32(0),
        )
        cf_ref = Ref(cf)
        r1_pod = Ref(JClp.CoinIndexedVecPod(pointer(r1.elements), pointer(r1.indices), Int32(0), Int32(n), Int32(0), Int32(0)))
        r2_pod = Ref(JClp.CoinIndexedVecPod(pointer(r2.elements), pointer(r2.indices), Int32(n), Int32(n), Int32(0), Int32(0)))
        _bench_btran(cf_ref, r1_pod, r2_pod, r2, rhs) # warmup
        @test _bench_btran(cf_ref, r1_pod, r2_pod, r2, rhs) == 0
    end
end

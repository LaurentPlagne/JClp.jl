@testset "B5b CoinFactorization Forrest-Tomlin replaceColumn" begin
    @testset "checkPivot thresholds across row counts & relaxCheck" begin
        @test JClp.cf_check_pivot(2.0, 2.0, Int32(10), Int32(10), 1.0) == Int32(0)
        @test JClp.cf_check_pivot(2.0 * (1.0 + 5e-6), 2.0, Int32(10), Int32(11), 1.0) == Int32(0)
        @test JClp.cf_check_pivot(2.0 * (1.0 + 5e-6), 2.0, Int32(10), Int32(25), 1.0) == Int32(2)
        @test JClp.cf_check_pivot(2.0 * (1.0 + 5e-9), 2.0, Int32(10), Int32(70), 1.0) == Int32(1)
        @test JClp.cf_check_pivot(2.0 * (1.0 + 5e-9), 2.0, Int32(10), Int32(70), 100.0) == Int32(0)
        # Small pivot <= 1e-9
        @test JClp.cf_check_pivot(1e-10, 1e-10, Int32(10), Int32(10), 1.0) == Int32(0)
        @test JClp.cf_check_pivot(1e-10, 2e-10, Int32(10), Int32(10), 1.0) == Int32(2)
    end

    @testset "Full FTRAN-FT -> replaceColumn -> FTRAN & BTRAN over 3 Forrest-Tomlin pivots" begin
        # Start from 4x4 basis B = L * (I + U_strict) * D, perform 3 basis column replacements,
        # and verify B_new \ rhs (FTRAN) and B_new' \ rhs (BTRAN) match exact linear algebra!
        n = 4
        max_pivs = 8
        mre = n + max_pivs # 12
        mce = n + max_pivs # 12
        len_area_u = 128
        len_area_r = 128

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

        # U by column
        elementU = zeros(Float64, len_area_u)
        indexRowU = zeros(Int32, len_area_u)
        elementU[1:5] = Float64[-0.5, 0.25, 1.5, -0.5, -0.3]
        indexRowU[1:5] = Int32[0, 1, 0, 1, 2]
        startColumnU = zeros(Int32, mce + 1)
        startColumnU[1:5] = Int32[0, 0, 1, 2, 5]
        startColumnU[mce + 1] = Int32(5)
        numberInColumn = zeros(Int32, mce + 1)
        numberInColumn[1:4] = Int32[0, 1, 1, 3]
        numberInColumnPlus = zeros(Int32, mce + 1)

        # U by row (matches elementU[1:5]: col1->row0(pos0), col2->row1(pos1), col3->rows 0,1,2(pos 2,3,4))
        indexColumnU = zeros(Int32, len_area_u)
        convertRowToColumnU = zeros(Int32, len_area_u)
        startRowU = zeros(Int32, mre + 1)
        numberInRow = zeros(Int32, mre + 1)
        # row 0: (col 1, pos 0), (col 3, pos 2) -> start 0, len 2
        # row 1: (col 2, pos 1), (col 3, pos 3) -> start 2, len 2
        # row 2: (col 3, pos 4)                 -> start 4, len 1
        # row 3: empty                          -> start 5, len 0
        indexColumnU[1:5] = Int32[1, 3, 2, 3, 3]
        convertRowToColumnU[1:5] = Int32[0, 2, 1, 3, 4]
        startRowU[1:5] = Int32[0, 2, 4, 5, 5]
        startRowU[mre + 1] = Int32(5)
        numberInRow[1:4] = Int32[2, 2, 1, 0]

        # Doubly-linked row list: sentinel = mre (12), order: 0 -> 1 -> 2 -> 3 -> 12
        nextRow = zeros(Int32, mre + 1)
        lastRow = zeros(Int32, mre + 1)
        nextRow[mre + 1] = Int32(0); lastRow[1] = Int32(mre)
        nextRow[1] = Int32(1); lastRow[2] = Int32(0)
        nextRow[2] = Int32(2); lastRow[3] = Int32(1)
        nextRow[3] = Int32(3); lastRow[4] = Int32(2)
        nextRow[4] = Int32(mre); lastRow[mre + 1] = Int32(3)

        # Doubly-linked column list for transpose R (sentinel = mce, order: 0 -> 1 -> 2 -> 3 -> mce)
        nextColumn = zeros(Int32, mce + 1)
        lastColumn = zeros(Int32, mce + 1)
        nextColumn[mce + 1] = Int32(0); lastColumn[1] = Int32(mce)
        nextColumn[1] = Int32(1); lastColumn[2] = Int32(0)
        nextColumn[2] = Int32(2); lastColumn[3] = Int32(1)
        nextColumn[3] = Int32(3); lastColumn[4] = Int32(2)
        nextColumn[4] = Int32(mce); lastColumn[mce + 1] = Int32(3)

        # L by col & by row
        elementL = Float64[2.0, 1.0, -1.0, 2.0]
        indexRowL = Int32[1, 3, 2, 3]
        startColumnL = Int32[0, 2, 3, 4, 4]
        elementByRowL = Float64[2.0, -1.0, 1.0, 2.0]
        indexColumnL = Int32[0, 1, 0, 2]
        startRowL = Int32[0, 0, 1, 2, 4]

        # R area (2 * len_area_r: first half row-by-row R, second half column-wise R)
        elementR = zeros(Float64, 2 * len_area_r)
        indexRowR = zeros(Int32, 2 * len_area_r)
        startColumnR = zeros(Int32, (max_pivs + 1) + (mce + 1))

        pivotRegion = zeros(Float64, mre + 1)
        pivotRegion[1:4] = 1.0 ./ d_diag
        pivotColumn = zeros(Int32, mre + 1)
        pivotColumnBack = zeros(Int32, mre + 1)
        permute = zeros(Int32, mre + 2)
        permuteBack = fill(Int32(-1), mre + 1)
        for i in 0:(n - 1)
            pivotColumn[i + 1] = Int32(i)
            pivotColumnBack[i + 1] = Int32(i)
            permute[i + 1] = Int32(i)
            permuteBack[i + 1] = Int32(i)
        end

        sparseArray = zeros(Int32, 4 * mre)
        r1 = JClp.CoinIndexedVector(mre)
        r2 = JClp.CoinIndexedVector(mre)

        nre = Int32(n); nce = Int32(n); nu = Int32(n); ngu = Int32(n)
        np = Int32(0); nr_cnt = Int32(0); lr = Int32(0); lu = Int32(5)
        max_u = Int32(5); tot_el = Int32(9); n_comp = Int32(0); cf_st = Int32(0)

        function make_cf()
            return JClp.CoinFactorizationPodView(
                pointer(elementU), pointer(indexRowU), pointer(startColumnU),
                pointer(numberInColumn), pointer(numberInColumnPlus),
                pointer(startRowU), pointer(numberInRow), pointer(indexColumnU), pointer(convertRowToColumnU),
                pointer(elementL), pointer(indexRowL), pointer(startColumnL),
                pointer(elementByRowL), pointer(indexColumnL), pointer(startRowL),
                pointer(elementR), pointer(indexRowR), pointer(startColumnR),
                pointer(pivotRegion), pointer(pivotColumn), pointer(pivotColumnBack),
                pointer(permute), pointer(permuteBack), pointer(nextColumn), pointer(lastColumn),
                pointer(sparseArray), C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL,
                1e-13, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                Int32(n), nre, Int32(mre), nce, Int32(mce), ngu, nu, Int32(n), Int32(0),
                nr_cnt, lr, Int32(len_area_r), Int32(len_area_u), Int32(0), Int32(0), np, Int32(max_pivs),
                Int32(100), Int32(200), Int32(1), Int32(0), Int32(0),
            )
        end

        pivots = [
            (Int32(1), Float64[1.0, 2.0, -1.0, 3.0]),
            (Int32(3), Float64[-0.5, 1.5, 2.0, 1.0]),
            (Int32(0), Float64[2.5, -1.0, 0.5, -2.0]),
        ]

        GC.@preserve elementU indexRowU startColumnU numberInColumn numberInColumnPlus startRowU numberInRow indexColumnU convertRowToColumnU nextRow lastRow nextColumn lastColumn elementL indexRowL startColumnL elementByRowL indexColumnL startRowL elementR indexRowR startColumnR pivotRegion pivotColumn pivotColumnBack permute permuteBack sparseArray r1 r2 begin
            # First test singular pivot rejection & rollback (pivotCheck far from true alpha)
            JClp.clear!(r1); JClp.clear!(r2)
            a0 = Float64[1.0, 2.0, -1.0, 3.0]
            JClp.create_packed!(r2, Int32(n), Int32[0, 1, 2, 3], a0)
            cf_ref = Ref(make_cf())
            r1_pod = Ref(JClp.CoinIndexedVecPod(pointer(r1.elements), pointer(r1.indices), Int32(0), Int32(mre), Int32(0), Int32(0)))
            r2_pod = Ref(JClp.CoinIndexedVecPod(pointer(r2.elements), pointer(r2.indices), Int32(n), Int32(mre), Int32(1), Int32(0)))
            rc_ft = JClp.jclp_cf_update_column_ft(
                Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod),
            )
            @test rc_ft > 0
            st_ref = Ref(JClp.CoinFactorizationReplaceStatePod(
                pointer(nextRow), pointer(lastRow), 1.0,
                nre, nce, nu, ngu, np, nr_cnt, lr, lu, max_u, tot_el, tot_el, Int32(16), n_comp, cf_st, Int32(0), Int32(0),
            ))
            rc_bad = JClp.jclp_cf_replace_column(
                Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
                Base.unsafe_convert(Ptr{JClp.CoinFactorizationReplaceStatePod}, st_ref),
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                Cint(1), 999.0, Cint(1),
            )
            @test rc_bad == Cint(2) # rejected!
            @test st_ref[].numberPivots == Int32(0)
            @test JClp.check_clear(r1)

            # Now perform 3 valid Forrest-Tomlin pivots and verify FTRAN & BTRAN after each!
            for (piv_row0, a_new) in pivots
                JClp.clear!(r1); JClp.clear!(r2)
                JClp.create_packed!(r2, Int32(n), Int32[0, 1, 2, 3], a_new)
                cf_ref = Ref(make_cf())
                r1_pod = Ref(JClp.CoinIndexedVecPod(pointer(r1.elements), pointer(r1.indices), Int32(0), Int32(mre), Int32(0), Int32(0)))
                r2_pod = Ref(JClp.CoinIndexedVecPod(pointer(r2.elements), pointer(r2.indices), Int32(n), Int32(mre), Int32(1), Int32(0)))
                rc_ft = JClp.jclp_cf_update_column_ft(
                    Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod),
                )
                @test rc_ft > 0
                alpha = zeros(Float64, n)
                for k in 1:Int(rc_ft)
                    alpha[Int(r2.indices[k]) + 1] = r2.elements[k]
                end
                piv_check = alpha[Int(piv_row0) + 1]

                st_ref = Ref(JClp.CoinFactorizationReplaceStatePod(
                    pointer(nextRow), pointer(lastRow), 1.0,
                    nre, nce, nu, ngu, np, nr_cnt, lr, lu, max_u, tot_el, Int32(9), Int32(16), n_comp, cf_st, Int32(0), Int32(0),
                ))
                rc_rep = JClp.jclp_cf_replace_column(
                    Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref),
                    Base.unsafe_convert(Ptr{JClp.CoinFactorizationReplaceStatePod}, st_ref),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                    Cint(piv_row0), piv_check, Cint(1),
                )
                @test rc_rep == Cint(0)
                st_out = st_ref[]
                nre = st_out.numberRowsExtra; nce = st_out.numberColumnsExtra
                nu = st_out.numberU; ngu = st_out.numberGoodU; np = st_out.numberPivots
                nr_cnt = st_out.numberR; lr = st_out.lengthR; lu = st_out.lengthU
                max_u = st_out.maximumU; tot_el = st_out.totalElements
                n_comp = st_out.numberCompressions; cf_st = st_out.status

                B[:, Int(piv_row0) + 1] = a_new

                # Verify FTRAN (B \ rhs)
                rhs = Float64[1.5, -2.0, 0.75, 3.25]
                JClp.clear!(r1); JClp.clear!(r2)
                JClp.create_packed!(r2, Int32(n), Int32[0, 1, 2, 3], rhs)
                cf_ref2 = Ref(make_cf())
                r2_pod2 = Ref(JClp.CoinIndexedVecPod(pointer(r2.elements), pointer(r2.indices), Int32(n), Int32(mre), Int32(1), Int32(0)))
                nnz_f = JClp.jclp_cf_update_column(
                    Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref2),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod2),
                    Cint(0),
                )
                x_got = zeros(Float64, n)
                for k in 1:Int(nnz_f); x_got[Int(r2.indices[k]) + 1] = r2.elements[k]; end
                @test x_got ≈ B \ rhs atol=1e-11

                # Verify BTRAN (B' \ rhs)
                JClp.clear!(r1); JClp.clear!(r2)
                JClp.create_packed!(r2, Int32(n), Int32[0, 1, 2, 3], rhs)
                nnz_b = JClp.jclp_cf_update_column_transpose(
                    Base.unsafe_convert(Ptr{JClp.CoinFactorizationPodView}, cf_ref2),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r1_pod),
                    Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, r2_pod2),
                )
                y_got = zeros(Float64, n)
                for k in 1:Int(nnz_b); y_got[Int(r2.indices[k]) + 1] = r2.elements[k]; end
                @test y_got ≈ B' \ rhs atol=1e-11
            end
            @test np == Int32(3)
        end
    end
end

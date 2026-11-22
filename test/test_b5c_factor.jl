using Test
using JClp
using LinearAlgebra
using Random

# Helper: allocate all CoinFactorization arrays for an n x n matrix given as 0-based triplets,
# run B5c (preProcess -> factorSparse -> optional factorDense -> cleanup -> optional goSparse),
# and solve A*x = b (FTRAN) & A'*y = c (BTRAN) via B5a kernels to verify mathematical exactness.
function _run_b5c_factor_and_solve(
    A::Matrix{Float64};
    dense_threshold::Int32 = Int32(71),
    bias_lu::Int32 = Int32(2),
    length_area_u_override::Int = 0,
    length_area_l_override::Int = 0,
    force_large_mark::Bool = false,
    do_go_sparse::Bool = false,
)
    n = size(A, 1)
    @assert size(A, 2) == n
    rows_tri = Int32[]
    cols_tri = Int32[]
    vals_tri = Float64[]
    for j in 1:n, i in 1:n
        v = A[i, j]
        if v != 0.0
            push!(rows_tri, Int32(i - 1))
            push!(cols_tri, Int32(j - 1))
            push!(vals_tri, v)
        end
    end
    nnz = length(vals_tri)
    max_pivots = 50
    mre = n + max_pivots
    mce = n + max_pivots
    lau = length_area_u_override > 0 ? length_area_u_override : max(4 * nnz + 4 * n + 256, 2 * n * n + 64)
    lal = length_area_l_override > 0 ? length_area_l_override : max(4 * nnz + 4 * n + 256, 2 * n * n + 64)

    elementU = zeros(Float64, lau + 4)
    indexRowU = zeros(Int32, lau + 4)
    indexColumnU = zeros(Int32, lau + 4)
    convertRowToColumnU = zeros(Int32, lau + 4)
    startColumnU = zeros(Int32, mce + 1)
    startRowU = zeros(Int32, mre + 1)
    numberInRow = zeros(Int32, mre + 1)
    numberInColumn = zeros(Int32, mce + 1)
    numberInColumnPlus = zeros(Int32, mce + 1)

    elementL = zeros(Float64, lal)
    indexRowL = zeros(Int32, lal)
    startColumnL = zeros(Int32, n + 1)
    elementByRowL = zeros(Float64, lal)
    indexColumnL = zeros(Int32, lal)
    startRowL = zeros(Int32, n + 1)

    pivotColumn = zeros(Int32, mce + 1)
    pivotColumnBack = zeros(Int32, mre + 1)
    permute = zeros(Int32, mre + 1)
    permuteBack = zeros(Int32, mre + 1)
    pivotRegion = zeros(Float64, mre + 1)

    firstCount = zeros(Int32, max(n + 2, mre + 1))
    nextCount = zeros(Int32, 2 * n)
    lastCount = zeros(Int32, 2 * n)
    nextColumn = zeros(Int32, mce + 1)
    lastColumn = zeros(Int32, mce + 1)
    nextRow = zeros(Int32, mre + 1)
    lastRow = zeros(Int32, mre + 1)
    saveColumn = zeros(Int32, n)
    markRow = zeros(Int32, n)

    workArea = zeros(Float64, n)
    workArea2 = zeros(UInt32, max(1000, n * n))
    denseArea = zeros(Float64, max(1, n * n + 8 * n + 64))
    densePermute = zeros(Int32, max(1, n))
    sparseArray = zeros(Int32, 4 * mre + 16)
    startColumnR = zeros(Int32, max_pivots + 1 + mce + 1)

    for k in 1:nnz
        indexRowU[k] = rows_tri[k]
        indexColumnU[k] = cols_tri[k]
        elementU[k] = vals_tri[k]
    end

    st_ref = Ref(JClp.CoinFactorizationFactorStatePod(
        pointer(elementU), pointer(indexRowU), pointer(indexColumnU),
        pointer(startColumnU), pointer(startRowU), pointer(convertRowToColumnU),
        pointer(numberInRow), pointer(numberInColumn), pointer(numberInColumnPlus),
        pointer(elementL), pointer(indexRowL), pointer(startColumnL),
        pointer(elementByRowL), pointer(indexColumnL), pointer(startRowL),
        pointer(pivotColumn), pointer(pivotColumnBack),
        pointer(permute), pointer(permuteBack), pointer(pivotRegion),
        pointer(firstCount), pointer(nextCount), pointer(lastCount),
        pointer(nextColumn), pointer(lastColumn),
        pointer(nextRow), pointer(lastRow),
        pointer(saveColumn), pointer(markRow),
        pointer(workArea), pointer(workArea2),
        pointer(denseArea), pointer(densePermute), pointer(sparseArray),
        C_NULL, C_NULL,
        0.1, 1.0e-13, -1.0, 1.0, -1.0,
        Int32(n), Int32(n), Int32(mre), Int32(mce), Int32(n),
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(nnz),
        Int32(nnz), Int32(lau), Int32(0), Int32(0), Int32(0), Int32(lal),
        Int32(0), Int32(0), Int32(0), Int32(nnz), Int32(0),
        Int32(0), dense_threshold, Int32(4), bias_lu,
        Int32(0), Int32(-1), Int32(max_pivots), Int32(length(workArea2)),
        Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
    ))

    status = Int32(0)
    st_final = nothing
    GC.@preserve elementU indexRowU indexColumnU convertRowToColumnU startColumnU startRowU numberInRow numberInColumn numberInColumnPlus elementL indexRowL startColumnL elementByRowL indexColumnL startRowL pivotColumn pivotColumnBack permute permuteBack pivotRegion firstCount nextCount lastCount nextColumn lastColumn nextRow lastRow saveColumn markRow workArea workArea2 denseArea densePermute sparseArray startColumnR st_ref begin
        st_ptr = Base.unsafe_convert(Ptr{JClp.CoinFactorizationFactorStatePod}, st_ref)
        JClp.cf_pre_process!(st_ptr, Int32(0))
        if force_large_mark
            status = JClp.cf_factor_sparse_impl!(Int32, st_ptr, JClp.CF_LARGE_SET, JClp.CF_LARGE_UNSET)
        else
            status = JClp.cf_factor_sparse!(st_ptr)
        end
        if status == Int32(0)
            st_mid = st_ref[]
            if st_mid.numberGoodU < Int32(n)
                status = JClp.cf_factor_singular!(st_ptr)
                # Mirror C++ permute_.swap(nextRow_)
                permute, nextRow = nextRow, permute
                return (status=status, state=st_ref[], permute=permute, pivotColumn=pivotColumn)
            end
        elseif status == Int32(2)
            status = JClp.cf_factor_dense!(st_ptr)
        end
        if status != Int32(0)
            return (status=status, state=st_ref[], permute=permute, pivotColumn=pivotColumn)
        end
        JClp.cf_cleanup!(st_ptr)
        # Mirror C++ swaps: numberInColumn_.swap(numberInColumnPlus_), permute_.swap(nextRow_)
        numberInColumn, numberInColumnPlus = numberInColumnPlus, numberInColumn
        permute, nextRow = nextRow, permute

        # Mirror factorize() post-cleanup permutation setup:
        # pivotColumn[0..n-1] = permute[0..n-1], pivotColumnBack[0..n-1] = permuteBack[0..n-1]
        col_perm = [permuteBack[pivotColumnBack[i] + 1] for i in 1:n]
        for i in 1:n
            pivotColumn[i] = permute[i]
            pivotColumnBack[i] = permuteBack[i]
        end

        st_final = st_ref[]
        if do_go_sparse
            # Force threshold positive if requested for testing even on smaller n
            st_gs = JClp.CoinFactorizationFactorStatePod(
                pointer(elementU), pointer(indexRowU), pointer(indexColumnU),
                pointer(startColumnU), pointer(startRowU), pointer(convertRowToColumnU),
                pointer(numberInRow), pointer(numberInColumn), pointer(numberInColumnPlus),
                pointer(elementL), pointer(indexRowL), pointer(startColumnL),
                pointer(elementByRowL), pointer(indexColumnL), pointer(startRowL),
                pointer(pivotColumn), pointer(pivotColumnBack),
                pointer(permute), pointer(permuteBack), pointer(pivotRegion),
                pointer(firstCount), pointer(nextCount), pointer(lastCount),
                pointer(nextColumn), pointer(lastColumn),
                pointer(nextRow), pointer(lastRow),
                pointer(saveColumn), pointer(markRow),
                pointer(workArea), pointer(workArea2),
                pointer(denseArea), pointer(densePermute), pointer(sparseArray),
                C_NULL, C_NULL,
                st_final.pivotTolerance, st_final.zeroTolerance, st_final.slackValue,
                st_final.areaFactor, st_final.timeLimit,
                st_final.numberRows, st_final.numberColumns,
                st_final.maximumRowsExtra, st_final.maximumColumnsExtra, st_final.biggerDimension,
                st_final.numberGoodU, st_final.numberGoodL, st_final.numberSlacks,
                st_final.numberU, st_final.maximumU, st_final.lengthU, st_final.lengthAreaU,
                st_final.numberL, st_final.baseL, st_final.lengthL, st_final.lengthAreaL,
                st_final.numberR, st_final.lengthR, st_final.lengthAreaR,
                st_final.totalElements, st_final.factorElements,
                st_final.numberDense, st_final.denseThreshold, st_final.numberTrials,
                st_final.biasLU, st_final.numberCompressions, st_final.status,
                st_final.maximumPivots, st_final.workArea2Size,
                Int32(max(10, div(n, 2))), Int32(max(10, div(n, 2))),
                st_final.useExtraR, st_final.freeDense, Int32(0),
            )
            st_gs_ref = Ref(st_gs)
            GC.@preserve st_gs_ref begin
                gs_ptr = Base.unsafe_convert(Ptr{JClp.CoinFactorizationFactorStatePod}, st_gs_ref)
                JClp.cf_go_sparse_build!(gs_ptr)
                st_final = st_gs_ref[]
            end
        end

        # Build CoinFactorizationPodView and verify FTRAN & BTRAN on random RHS
        rng = MersenneTwister(12345 + n)
        x_true = randn(rng, n)
        b_rhs = A * x_true
        y_true = randn(rng, n)
        c_rhs = A' * y_true

        # FTRAN solves B * z = b where B has columns permuted by col_perm (col j of A sits at pivot col col_perm[j])
        # So B[:, col_perm[j]+1] == A[:, j], hence z[col_perm[j]+1] == x_true[j]
        n_btran = Ref(Int32(0)); b_in = Ref(0.0); b_u = Ref(0.0); b_r = Ref(0.0); b_l = Ref(0.0)
        n_ftran = Ref(Int32(0)); f_in = Ref(0.0); f_l = Ref(0.0); f_r = Ref(0.0); f_u = Ref(0.0)
        len_r_ref = Ref(Int32(0)); num_r_ref = Ref(Int32(0))

        cf = JClp.CoinFactorizationPodView(
            pointer(elementU), pointer(indexRowU), pointer(startColumnU),
            pointer(numberInColumn), st_final.useExtraR != Int32(0) ? pointer(numberInColumnPlus) : Ptr{Int32}(C_NULL),
            pointer(startRowU), pointer(numberInRow), pointer(indexColumnU), pointer(convertRowToColumnU),
            pointer(elementL), pointer(indexRowL), pointer(startColumnL),
            do_go_sparse ? pointer(elementByRowL) : Ptr{Float64}(C_NULL),
            do_go_sparse ? pointer(indexColumnL) : Ptr{Int32}(C_NULL),
            do_go_sparse ? pointer(startRowL) : Ptr{Int32}(C_NULL),
            pointer(elementL) + Int(st_final.lengthL) * sizeof(Float64),
            pointer(indexRowL) + Int(st_final.lengthL) * sizeof(Int32),
            pointer(startColumnR),
            pointer(pivotRegion), pointer(pivotColumn), pointer(pivotColumnBack),
            pointer(permute), pointer(permuteBack), pointer(nextColumn), pointer(lastColumn),
            pointer(sparseArray), pointer(denseArea), pointer(densePermute),
            Base.unsafe_convert(Ptr{Int32}, n_btran),
            Base.unsafe_convert(Ptr{Float64}, b_in), Base.unsafe_convert(Ptr{Float64}, b_u),
            Base.unsafe_convert(Ptr{Float64}, b_r), Base.unsafe_convert(Ptr{Float64}, b_l),
            Base.unsafe_convert(Ptr{Int32}, n_ftran),
            Base.unsafe_convert(Ptr{Float64}, f_in), Base.unsafe_convert(Ptr{Float64}, f_l),
            Base.unsafe_convert(Ptr{Float64}, f_r), Base.unsafe_convert(Ptr{Float64}, f_u),
            Base.unsafe_convert(Ptr{Int32}, len_r_ref), Base.unsafe_convert(Ptr{Int32}, num_r_ref),
            1.0e-13, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            Int32(n), Int32(n), Int32(mre), Int32(n), Int32(mce),
            st_final.numberGoodU, st_final.numberU, st_final.numberL, st_final.baseL,
            Int32(0), Int32(0), st_final.lengthAreaR, Int32(lau),
            st_final.numberDense, st_final.numberSlacks, Int32(0), Int32(max_pivots),
            st_final.sparseThreshold, st_final.sparseThreshold2, Int32(1), Int32(0), Int32(0),
        )

        # FTRAN
        r1_el = zeros(Float64, mre + 1); r1_ix = zeros(Int32, mre + 1)
        r2_el = zeros(Float64, mre + 1); r2_ix = zeros(Int32, mre + 1)
        for i in 1:n
            r2_el[i] = b_rhs[i]
            r2_ix[i] = Int32(i - 1)
        end
        reg1 = Ref(JClp.CoinIndexedVecPod(pointer(r1_el), pointer(r1_ix), Int32(0), Int32(mre), Int32(0), Int32(0)))
        reg2 = Ref(JClp.CoinIndexedVecPod(pointer(r2_el), pointer(r2_ix), Int32(n), Int32(mre), Int32(0), Int32(0)))
        GC.@preserve r1_el r1_ix r2_el r2_ix reg1 reg2 n_btran b_in b_u b_r b_l n_ftran f_in f_l f_r f_u len_r_ref num_r_ref begin
            JClp.cf_update_column!(
                cf,
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, reg1),
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, reg2),
                false,
            )
        end
        x_got = [r2_el[col_perm[j] + 1] for j in 1:n]
        ftran_err = maximum(abs.(x_got .- x_true))

        # BTRAN: solves B' * y = c_perm where c_perm[col_perm[j]+1] = c_rhs[j]
        fill!(r1_el, 0.0); fill!(r2_el, 0.0)
        for j in 1:n
            r2_el[col_perm[j] + 1] = c_rhs[j]
            r2_ix[j] = Int32(j - 1)
        end
        reg1[] = JClp.CoinIndexedVecPod(pointer(r1_el), pointer(r1_ix), Int32(0), Int32(mre), Int32(0), Int32(0))
        reg2[] = JClp.CoinIndexedVecPod(pointer(r2_el), pointer(r2_ix), Int32(n), Int32(mre), Int32(0), Int32(0))
        GC.@preserve r1_el r1_ix r2_el r2_ix reg1 reg2 n_btran b_in b_u b_r b_l n_ftran f_in f_l f_r f_u len_r_ref num_r_ref begin
            JClp.cf_update_column_transpose!(
                cf,
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, reg1),
                Base.unsafe_convert(Ptr{JClp.CoinIndexedVecPod}, reg2),
            )
        end
        btran_err = maximum(abs.(r2_el[1:n] .- y_true))

        return (status=status, state=st_final, ftran_err=ftran_err, btran_err=btran_err)
    end
end

@testset "B5c: CoinFactorization Markowitz Sparse LU" begin
    @testset "ABI layout" begin
        @test sizeof(JClp.CoinFactorizationFactorStatePod) == 464
        @test JClp.jclp_cf_factor_state_sizeof() == Cint(464)
    end

    @testset "separateLinks" begin
        nr = Int32(5)
        fc = fill(Int32(-1), 6)
        ncnt = fill(Int32(-2), 10)
        lcnt = fill(Int32(-2), 10)
        GC.@preserve fc ncnt lcnt begin
            pfc = pointer(fc); pnc = pointer(ncnt); plc = pointer(lcnt)
            # Add mixed rows (0, 2) and cols (5, 7) with count 1
            for idx in Int32[0, 5, 2, 7]
                JClp._cf_add_link!(pfc, pnc, plc, idx, Int32(1))
            end
            # Columns first (rows_first = false)
            JClp.cf_separate_links!(pfc, pnc, plc, nr, Int32(1), false)
            order = Int32[]
            cur = fc[2]
            while cur >= 0
                push!(order, cur)
                cur = ncnt[cur + 1]
            end
            @test length(order) == 4
            @test order[1] >= nr && order[2] >= nr
            @test order[3] < nr && order[4] < nr

            # Rows first (rows_first = true)
            JClp.cf_separate_links!(pfc, pnc, plc, nr, Int32(1), true)
            order2 = Int32[]
            cur = fc[2]
            while cur >= 0
                push!(order2, cur)
                cur = ncnt[cur + 1]
            end
            @test length(order2) == 4
            @test order2[1] < nr && order2[2] < nr
            @test order2[3] >= nr && order2[4] >= nr
        end
    end

    @testset "Slacks, row singletons, column singletons" begin
        # Mix of -1 slacks + upper/lower triangular singletons
        A = zeros(Float64, 10, 10)
        for i in 1:4
            A[i, i] = -1.0  # slacks
        end
        # Row singletons & column singletons on 5..10
        for i in 5:10
            A[i, i] = 2.0 + 0.3 * i
        end
        A[5, 1] = 1.5; A[6, 5] = -0.7; A[7, 6] = 1.2
        A[8, 9] = 0.5; A[9, 10] = -0.8; A[10, 2] = 0.4
        res = _run_b5c_factor_and_solve(A)
        @test res.status == Int32(0)
        @test res.state.numberSlacks >= Int32(2)
        @test res.ftran_err < 1e-12
        @test res.btran_err < 1e-12
    end

    @testset "pivotOneOtherRow (2-nnz columns with fill-in and cancellation)" begin
        # Tridiagonal + cancellation entries to exercise pivotOneOtherRow thoroughly
        n = 16
        A = zeros(Float64, n, n)
        for i in 1:n
            A[i, i] = 4.0
            if i < n
                A[i + 1, i] = -1.0
                A[i, i + 1] = 0.5
            end
        end
        # Force an exact numerical cancellation after first pivot
        A[2, 3] = 0.5
        A[1, 3] = 2.0
        res = _run_b5c_factor_and_solve(A; dense_threshold=Int32(0))
        @test res.status == Int32(0)
        @test res.ftran_err < 1e-12
        @test res.btran_err < 1e-12
    end

    @testset "General Markowitz pivot (Small UInt16 & Large Int32) + compression" begin
        rng = MersenneTwister(2026)
        n = 40
        A = zeros(Float64, n, n)
        for i in 1:n
            A[i, i] = 5.0 + rand(rng)
        end
        # Random sparse off-diagonals (~5 per column)
        for j in 1:n
            for _ in 1:5
                i = rand(rng, 1:n)
                A[i, j] += randn(rng)
            end
        end
        # Normal Small (UInt16) path with tight U space to trigger getColumnSpace & getRowSpace compressions
        res_small = _run_b5c_factor_and_solve(A; dense_threshold=Int32(0), length_area_u_override=650)
        @test res_small.status == Int32(0)
        @test res_small.ftran_err < 1e-11
        @test res_small.btran_err < 1e-11

        # Large (Int32) markRow path
        res_large = _run_b5c_factor_and_solve(A; dense_threshold=Int32(0), force_large_mark=true)
        @test res_large.status == Int32(0)
        @test res_large.ftran_err < 1e-11
        @test res_large.btran_err < 1e-11

        # biasLU = 0 (U bias) and biasLU = 3 (L bias, no slack pass)
        for b_lu in Int32[0, 1, 3]
            res_b = _run_b5c_factor_and_solve(A; dense_threshold=Int32(0), bias_lu=b_lu)
            @test res_b.status == Int32(0)
            @test res_b.ftran_err < 1e-11
            @test res_b.btran_err < 1e-11
        end
    end

    @testset "factorDense via LAPACK Accelerate dgetrf_" begin
        rng = MersenneTwister(99)
        n = 32 # multiple of 8 for COIN_ALIGN_DENSE == 2
        A = randn(rng, n, n) + 6.0 * I
        # Dense threshold = 8 -> triggers factorDense with LAPACK dgetrf_
        res = _run_b5c_factor_and_solve(A; dense_threshold=Int32(8))
        @test res.status == Int32(0)
        @test res.state.numberDense > Int32(0)
        @test res.ftran_err < 1e-11
        @test res.btran_err < 1e-11
    end

    @testset "Singular matrix handling" begin
        A = [
            2.0 1.0 0.0 0.0;
            1.0 3.0 0.0 0.0;
            0.0 0.0 2.0 4.0;
            0.0 0.0 1.0 2.0
        ]
        res = _run_b5c_factor_and_solve(A; dense_threshold=Int32(0))
        @test res.status == Int32(-1)
        @test count(x -> x < 0, res.permute[1:4]) == 1
        @test count(x -> x < 0, res.pivotColumn[1:4]) == 1
    end

    @testset "goSparse row-copy of L & hypersparse solve" begin
        rng = MersenneTwister(777)
        n = 50
        A = zeros(Float64, n, n)
        for i in 1:n
            A[i, i] = 4.0 + rand(rng)
        end
        for j in 1:n, _ in 1:4
            A[rand(rng, 1:n), j] += 0.5 * randn(rng)
        end
        res = _run_b5c_factor_and_solve(A; dense_threshold=Int32(0), do_go_sparse=true)
        @test res.status == Int32(0)
        @test res.ftran_err < 1e-11
        @test res.btran_err < 1e-11

        # Threshold logic for small / medium / large n
        for (nr_test, exp_t, exp_t2) in [
            (Int32(100), Int32(0), Int32(0)),
            (Int32(600), Int32(100), Int32(300)),
            (Int32(4000), Int32(500), Int32(2000)),
            (Int32(20000), Int32(500), Int32(2500)),
        ]
            st_r = Ref(JClp.CoinFactorizationFactorStatePod(
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.1, 1e-13, -1.0, 1.0, -1.0,
                nr_test, nr_test, nr_test, nr_test, nr_test,
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(4), Int32(2), Int32(0), Int32(0),
                Int32(200), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
            ))
            GC.@preserve st_r begin
                p = Base.unsafe_convert(Ptr{JClp.CoinFactorizationFactorStatePod}, st_r)
                t = JClp.cf_go_sparse_thresholds!(p)
                @test t == exp_t
                @test st_r[].sparseThreshold == exp_t
                @test st_r[].sparseThreshold2 == exp_t2
            end
        end
    end

    @testset "Zero allocations on hot B5c kernels" begin
        rng = MersenneTwister(42)
        n = 24
        A = zeros(Float64, n, n)
        for i in 1:n
            A[i, i] = 5.0
        end
        for j in 1:n, _ in 1:3
            A[rand(rng, 1:n), j] += 0.4 * randn(rng)
        end
        rows_tri = Int32[]; cols_tri = Int32[]; vals_tri = Float64[]
        for j in 1:n, i in 1:n
            v = A[i, j]
            if v != 0.0
                push!(rows_tri, Int32(i - 1)); push!(cols_tri, Int32(j - 1)); push!(vals_tri, v)
            end
        end
        nnz = length(vals_tri)
        mre = n + 20; mce = n + 20; lau = 2000; lal = 2000
        elementU = zeros(Float64, lau + 4); indexRowU = zeros(Int32, lau + 4)
        indexColumnU = zeros(Int32, lau + 4); convertRowToColumnU = zeros(Int32, lau + 4)
        startColumnU = zeros(Int32, mce + 1); startRowU = zeros(Int32, mre + 1)
        numberInRow = zeros(Int32, mre + 1); numberInColumn = zeros(Int32, mce + 1)
        numberInColumnPlus = zeros(Int32, mce + 1)
        elementL = zeros(Float64, lal); indexRowL = zeros(Int32, lal); startColumnL = zeros(Int32, n + 1)
        elementByRowL = zeros(Float64, lal); indexColumnL = zeros(Int32, lal); startRowL = zeros(Int32, n + 1)
        pivotColumn = zeros(Int32, mce + 1); pivotColumnBack = zeros(Int32, mre + 1)
        permute = zeros(Int32, mre + 1); permuteBack = zeros(Int32, mre + 1); pivotRegion = zeros(Float64, mre + 1)
        firstCount = zeros(Int32, mre + 1); nextCount = zeros(Int32, 2 * n); lastCount = zeros(Int32, 2 * n)
        nextColumn = zeros(Int32, mce + 1); lastColumn = zeros(Int32, mce + 1)
        nextRow = zeros(Int32, mre + 1); lastRow = zeros(Int32, mre + 1)
        saveColumn = zeros(Int32, n); markRow = zeros(Int32, n)
        workArea = zeros(Float64, n); workArea2 = zeros(UInt32, 2000)
        denseArea = zeros(Float64, 1); densePermute = zeros(Int32, 1); sparseArray = zeros(Int32, 4 * mre + 16)

        function _run_once!(st_ptr)
            for k in 1:nnz
                indexRowU[k] = rows_tri[k]; indexColumnU[k] = cols_tri[k]; elementU[k] = vals_tri[k]
            end
            JClp._fs_store_i32!(st_ptr, JClp._FS_OFF_numberGoodU, Int32(0))
            JClp._fs_store_i32!(st_ptr, JClp._FS_OFF_numberGoodL, Int32(0))
            JClp._fs_store_i32!(st_ptr, JClp._FS_OFF_lengthL, Int32(0))
            JClp._fs_store_i32!(st_ptr, JClp._FS_OFF_numberCompressions, Int32(0))
            JClp._fs_store_i32!(st_ptr, JClp._FS_OFF_status, Int32(0))
            JClp.cf_pre_process!(st_ptr, Int32(0))
            JClp.cf_factor_sparse!(st_ptr)
            JClp.cf_cleanup!(st_ptr)
            JClp.cf_go_sparse_build!(st_ptr)
            return nothing
        end

        st_ref = Ref(JClp.CoinFactorizationFactorStatePod(
            pointer(elementU), pointer(indexRowU), pointer(indexColumnU),
            pointer(startColumnU), pointer(startRowU), pointer(convertRowToColumnU),
            pointer(numberInRow), pointer(numberInColumn), pointer(numberInColumnPlus),
            pointer(elementL), pointer(indexRowL), pointer(startColumnL),
            pointer(elementByRowL), pointer(indexColumnL), pointer(startRowL),
            pointer(pivotColumn), pointer(pivotColumnBack),
            pointer(permute), pointer(permuteBack), pointer(pivotRegion),
            pointer(firstCount), pointer(nextCount), pointer(lastCount),
            pointer(nextColumn), pointer(lastColumn),
            pointer(nextRow), pointer(lastRow),
            pointer(saveColumn), pointer(markRow),
            pointer(workArea), pointer(workArea2),
            pointer(denseArea), pointer(densePermute), pointer(sparseArray),
            C_NULL, C_NULL,
            0.1, 1e-13, -1.0, 1.0, -1.0,
            Int32(n), Int32(n), Int32(mre), Int32(mce), Int32(n),
            Int32(0), Int32(0), Int32(0), Int32(0), Int32(nnz),
            Int32(nnz), Int32(lau), Int32(0), Int32(0), Int32(0), Int32(lal),
            Int32(0), Int32(0), Int32(0), Int32(nnz), Int32(0),
            Int32(0), Int32(0), Int32(4), Int32(2), Int32(0), Int32(0),
            Int32(20), Int32(2000), Int32(10), Int32(10), Int32(0), Int32(0), Int32(0),
        ))
        GC.@preserve elementU indexRowU indexColumnU convertRowToColumnU startColumnU startRowU numberInRow numberInColumn numberInColumnPlus elementL indexRowL startColumnL elementByRowL indexColumnL startRowL pivotColumn pivotColumnBack permute permuteBack pivotRegion firstCount nextCount lastCount nextColumn lastColumn nextRow lastRow saveColumn markRow workArea workArea2 denseArea densePermute sparseArray st_ref begin
            p = Base.unsafe_convert(Ptr{JClp.CoinFactorizationFactorStatePod}, st_ref)
            _run_once!(p) # warmup
            allocs = @allocated _run_once!(p)
            @test allocs == 0
        end
    end
end

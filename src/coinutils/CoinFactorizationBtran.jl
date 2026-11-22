# Faithful translation of CoinUtils/src/CoinFactorization4.cpp — BTRAN (`updateColumnTranspose*`).
# Zero-allocation, hypersparse DFS + bitmask + dense LAPACK Accelerate, --trim=safe compatible.

# Hardware FMA matching Apple clang++ -O3 arm64 `fmsub` contraction of `c -= a * b`.
@inline _fmsub(a::Float64, b::Float64, c::Float64)::Float64 = fma(-a, b, c)

# --- BTRAN PFI (CoinFactorization4.cpp:3220-3261) ---
function cf_update_column_transpose_pfi!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    index::Ptr{Int32},
    number_nonzero_in::Int32,
)::Int32
    nr = Int(cf.numberRows)
    piv_col = cf.pivotColumn + nr * sizeof(Int32)
    piv_reg = cf.pivotRegion + nr * sizeof(Float64)
    tol = cf.zeroTolerance
    start_col = cf.startColumnU + nr * sizeof(Int32)
    idx_row = cf.indexRowU
    el = cf.elementU

    nnz = Int(number_nonzero_in)
    i = Int(cf.numberPivots) - 1
    while i >= 0
        prow = Int(unsafe_load(piv_col, i + 1))
        pval = unsafe_load(region, prow + 1) * unsafe_load(piv_reg, i + 1)
        js = Int(unsafe_load(start_col, i + 1))
        je = Int(unsafe_load(start_col, i + 2))
        j = js
        while j < je
            ir = Int(unsafe_load(idx_row, j + 1))
            pval = _fmsub(unsafe_load(el, j + 1), unsafe_load(region, ir + 1), pval)
            j += 1
        end
        if abs(pval) > tol
            if unsafe_load(region, prow + 1) == 0.0
                nnz += 1
                unsafe_store!(index, Int32(prow), nnz)
            end
            unsafe_store!(region, pval, prow + 1)
        else
            if unsafe_load(region, prow + 1) != 0.0
                unsafe_store!(region, COIN_INDEXED_REALLY_TINY_ELEMENT, prow + 1)
            end
        end
        i -= 1
    end
    return Int32(nnz)
end

# --- BTRAN U Densish (CoinFactorization4.cpp:1673-1721) ---
function cf_update_column_transpose_u_densish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    smallest_index::Int32,
)::Int32
    tol = cf.zeroTolerance
    start_row = cf.startRowU
    conv = cf.convertRowToColumnU
    idx_col = cf.indexColumnU
    el = cf.elementU
    num_in_row = cf.numberInRow
    last = Int(cf.numberU)

    nnz = 0
    i = Int(smallest_index)
    while i < last
        pval = unsafe_load(region, i + 1)
        if abs(pval) > tol
            s = Int(unsafe_load(start_row, i + 1))
            nin = Int(unsafe_load(num_in_row, i + 1))
            e_even = s + (nin & ~1)
            j = s
            while j < e_even
                ir0 = Int(unsafe_load(idx_col, j + 1))
                ir1 = Int(unsafe_load(idx_col, j + 2))
                ge0 = Int(unsafe_load(conv, j + 1))
                ge1 = Int(unsafe_load(conv, j + 2))
                v0 = unsafe_load(el, ge0 + 1)
                v1 = unsafe_load(el, ge1 + 1)
                unsafe_store!(region, _fmsub(v0, pval, unsafe_load(region, ir0 + 1)), ir0 + 1)
                unsafe_store!(region, _fmsub(v1, pval, unsafe_load(region, ir1 + 1)), ir1 + 1)
                j += 2
            end
            if (nin & 1) != 0
                ir = Int(unsafe_load(idx_col, e_even + 1))
                ge = Int(unsafe_load(conv, e_even + 1))
                v = unsafe_load(el, ge + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
            end
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
        else
            unsafe_store!(region, 0.0, i + 1)
        end
        i += 1
    end
    return Int32(nnz)
end

# --- BTRAN U Sparsish (CoinFactorization4.cpp:1724-1829) ---
function cf_update_column_transpose_u_sparsish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
    smallest_index::Int32,
)::Int32
    tol = cf.zeroTolerance
    start_row = cf.startRowU
    conv = cf.convertRowToColumnU
    idx_col = cf.indexColumnU
    el = cf.elementU
    num_in_row = cf.numberInRow
    last = Int(cf.numberU)

    mark = reinterpret(Ptr{UInt8}, cf.sparseArray + 3 * Int(cf.maximumRowsExtra) * sizeof(Int32))
    n_in = Int(nnz_in)
    i = 1
    while i <= n_in
        ipiv = Int(unsafe_load(region_index, i))
        iword = ipiv >> 3
        ibit = ipiv - (iword << 3)
        old_m = unsafe_load(mark, iword + 1)
        unsafe_store!(mark, old_m | (UInt8(1) << ibit), iword + 1)
        i += 1
    end

    nnz = 0
    k_start = Int(smallest_index) >> 3
    k_last = last >> 3

    k = k_start
    while k < k_last
        imark = unsafe_load(mark, k + 1)
        if imark != UInt8(0)
            ii = k << 3
            i_last = ii + 8
            while ii < i_last
                pval = unsafe_load(region, ii + 1)
                if abs(pval) > tol
                    s = Int(unsafe_load(start_row, ii + 1))
                    e = s + Int(unsafe_load(num_in_row, ii + 1))
                    j = s
                    while j < e
                        ir = Int(unsafe_load(idx_col, j + 1))
                        ge = Int(unsafe_load(conv, j + 1))
                        v = unsafe_load(el, ge + 1)
                        iw = ir >> 3
                        ib = ir - (iw << 3)
                        unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
                        unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                        j += 1
                    end
                    nnz += 1
                    unsafe_store!(region_index, Int32(ii), nnz)
                else
                    unsafe_store!(region, 0.0, ii + 1)
                end
                ii += 1
            end
            unsafe_store!(mark, UInt8(0), k + 1)
        end
        k += 1
    end

    unsafe_store!(mark, UInt8(0), k_last + 1)
    ii = k_last << 3
    while ii < last
        pval = unsafe_load(region, ii + 1)
        if abs(pval) > tol
            s = Int(unsafe_load(start_row, ii + 1))
            e = s + Int(unsafe_load(num_in_row, ii + 1))
            j = s
            while j < e
                ir = Int(unsafe_load(idx_col, j + 1))
                ge = Int(unsafe_load(conv, j + 1))
                v = unsafe_load(el, ge + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j += 1
            end
            nnz += 1
            unsafe_store!(region_index, Int32(ii), nnz)
        else
            unsafe_store!(region, 0.0, ii + 1)
        end
        ii += 1
    end
    return Int32(nnz)
end

# --- BTRAN U Sparse (hypersparse DFS, CoinFactorization4.cpp:1832-1931) ---
function cf_update_column_transpose_u_sparse!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    start_row = cf.startRowU
    conv = cf.convertRowToColumnU
    idx_col = cf.indexColumnU
    el = cf.elementU
    num_in_row = cf.numberInRow

    mre = Int(cf.maximumRowsExtra)
    stack = cf.sparseArray
    list = stack + mre * sizeof(Int32)
    next = list + mre * sizeof(Int32)
    mark = reinterpret(Ptr{UInt8}, next + mre * sizeof(Int32))

    n_list = 0
    n_in = Int(nnz_in)
    k = 1
    while k <= n_in
        kpiv = Int(unsafe_load(region_index, k))
        unsafe_store!(stack, Int32(kpiv), 1)
        j = Int(unsafe_load(start_row, kpiv + 1)) + Int(unsafe_load(num_in_row, kpiv + 1)) - 1
        unsafe_store!(next, Int32(j), 1)
        n_stack = 1
        while n_stack > 0
            kpiv = Int(unsafe_load(stack, n_stack))
            n_stack -= 1
            if unsafe_load(mark, kpiv + 1) != UInt8(1)
                j = Int(unsafe_load(next, n_stack + 1))
                if j >= Int(unsafe_load(start_row, kpiv + 1))
                    next_kpiv = Int(unsafe_load(idx_col, j + 1))
                    j -= 1
                    n_stack += 1
                    unsafe_store!(next, Int32(j), n_stack)
                    if unsafe_load(mark, next_kpiv + 1) == UInt8(0)
                        j2 = Int(unsafe_load(start_row, next_kpiv + 1)) + Int(unsafe_load(num_in_row, next_kpiv + 1)) - 1
                        n_stack += 1
                        unsafe_store!(stack, Int32(next_kpiv), n_stack)
                        unsafe_store!(mark, UInt8(2), next_kpiv + 1)
                        unsafe_store!(next, Int32(j2), n_stack)
                    end
                else
                    n_list += 1
                    unsafe_store!(list, Int32(kpiv), n_list)
                    unsafe_store!(mark, UInt8(1), kpiv + 1)
                end
            end
        end
        k += 1
    end

    nnz = 0
    i = n_list
    while i >= 1
        ipiv = Int(unsafe_load(list, i))
        unsafe_store!(mark, UInt8(0), ipiv + 1)
        pval = unsafe_load(region, ipiv + 1)
        if abs(pval) > tol
            s = Int(unsafe_load(start_row, ipiv + 1))
            e = s + Int(unsafe_load(num_in_row, ipiv + 1))
            j = s
            while j < e
                ir = Int(unsafe_load(idx_col, j + 1))
                ge = Int(unsafe_load(conv, j + 1))
                v = unsafe_load(el, ge + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j += 1
            end
            nnz += 1
            unsafe_store!(region_index, Int32(ipiv), nnz)
        else
            unsafe_store!(region, 0.0, ipiv + 1)
        end
        i -= 1
    end
    return Int32(nnz)
end

# --- BTRAN U Dispatcher (CoinFactorization4.cpp:1935-1978) ---
function cf_update_column_transpose_u!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
    smallest_index::Int32,
)::Int32
    go_sparse = Int32(0)
    if cf.sparseThreshold > Int32(0)
        if cf.btranAverageAfterU != 0.0
            new_num = Float64(nnz_in) * cf.btranAverageAfterU
            if new_num < Float64(cf.sparseThreshold)
                go_sparse = Int32(2)
            elseif new_num < Float64(cf.sparseThreshold2)
                go_sparse = Int32(1)
            else
                go_sparse = Int32(0)
            end
        else
            go_sparse = nnz_in < cf.sparseThreshold ? Int32(2) : Int32(0)
        end
    else
        go_sparse = Int32(0)
    end
    if go_sparse == Int32(0)
        return cf_update_column_transpose_u_densish!(cf, region, region_index, smallest_index)
    elseif go_sparse == Int32(1)
        return cf_update_column_transpose_u_sparsish!(cf, region, region_index, nnz_in, smallest_index)
    else
        return cf_update_column_transpose_u_sparse!(cf, region, region_index, nnz_in)
    end
end

# --- BTRAN R Densish (CoinFactorization4.cpp:2506-2538) ---
function cf_update_column_transpose_r_densish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
)::Nothing
    nr = Int(cf.numberRows)
    last = Int(cf.numberRowsExtra) - 1
    idx_row = cf.indexRowR
    el = cf.elementR
    start_col = cf.startColumnR # indexed by (i - nr)
    perm = cf.permute

    i = last
    while i >= nr
        put_row = Int(unsafe_load(perm, i + 1))
        pval = unsafe_load(region, i + 1)
        unsafe_store!(region, 0.0, i + 1)
        if pval != 0.0
            js = Int(unsafe_load(start_col, i - nr + 1))
            je = Int(unsafe_load(start_col, i - nr + 2))
            j = js
            while j < je
                v = unsafe_load(el, j + 1)
                ir = Int(unsafe_load(idx_row, j + 1))
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j += 1
            end
            unsafe_store!(region, pval, put_row + 1)
        end
        i -= 1
    end
    return nothing
end

# --- BTRAN R Sparse (CoinFactorization4.cpp:2540-2600) ---
function cf_update_column_transpose_r_sparse!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    nr = Int(cf.numberRows)
    last = Int(cf.numberRowsExtra) - 1
    idx_row = cf.indexRowR
    el = cf.elementR
    start_col = cf.startColumnR
    perm = cf.permute
    spare = cf.sparseArray

    nnz = Int(nnz_in)
    i = 0
    while i < nnz
        ri = Int(unsafe_load(region_index, i + 1))
        unsafe_store!(spare, Int32(i), ri + 1)
        i += 1
    end

    i = last
    while i >= nr
        put_row = Int(unsafe_load(perm, i + 1))
        pval = unsafe_load(region, i + 1)
        unsafe_store!(region, 0.0, i + 1)
        if pval != 0.0
            js = Int(unsafe_load(start_col, i - nr + 1))
            je = Int(unsafe_load(start_col, i - nr + 2))
            j = js
            while j < je
                v = unsafe_load(el, j + 1)
                ir = Int(unsafe_load(idx_row, j + 1))
                old_v = unsafe_load(region, ir + 1)
                new_v = _fmsub(v, pval, old_v)
                if old_v != 0.0
                    if new_v != 0.0
                        unsafe_store!(region, new_v, ir + 1)
                    else
                        unsafe_store!(region, COIN_INDEXED_REALLY_TINY_ELEMENT, ir + 1)
                    end
                elseif abs(new_v) > tol
                    unsafe_store!(region, new_v, ir + 1)
                    unsafe_store!(spare, Int32(nnz), ir + 1)
                    nnz += 1
                    unsafe_store!(region_index, Int32(ir), nnz)
                end
                j += 1
            end
            unsafe_store!(region, pval, put_row + 1)
            pos = Int(unsafe_load(spare, i + 1))
            unsafe_store!(region_index, Int32(put_row), pos + 1)
            unsafe_store!(spare, Int32(pos), put_row + 1)
        end
        i -= 1
    end
    return Int32(nnz)
end

# --- BTRAN R Dispatcher (CoinFactorization4.cpp:2603-2623) ---
function cf_update_column_transpose_r!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    if cf.numberRowsExtra == cf.numberRows
        return nnz_in
    end
    if nnz_in == Int32(0)
        return Int32(0)
    end
    if nnz_in < (cf.sparseThreshold << 2) || (cf.numberL == Int32(0) && cf.sparseArray != C_NULL)
        nnz = cf_update_column_transpose_r_sparse!(cf, region, region_index, nnz_in)
        if cf.collectStatistics != Int32(0)
            unsafe_store!(cf.p_btranCountAfterR, unsafe_load(cf.p_btranCountAfterR) + Float64(nnz))
        end
        return nnz
    else
        cf_update_column_transpose_r_densish!(cf, region)
        if cf.collectStatistics != Int32(0)
            est = min(Int(nnz_in) << 1, Int(cf.numberRows))
            unsafe_store!(cf.p_btranCountAfterR, unsafe_load(cf.p_btranCountAfterR) + Float64(est))
        end
        return cf.numberRows + Int32(1)
    end
end

# --- BTRAN L Densish by column (CoinFactorization4.cpp:1982-2062) ---
function cf_update_column_transpose_l_densish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
)::Int32
    tol = cf.zeroTolerance
    nr = Int(cf.numberRows)
    nnz = 0
    first = nr - 1
    while first >= 0
        if unsafe_load(region, first + 1) != 0.0
            break
        end
        first -= 1
    end
    if first >= 0
        base = Int(cf.baseL)
        start_col = cf.startColumnL
        idx_row = cf.indexRowL
        el = cf.elementL
        last = base + Int(cf.numberL)
        if first >= last
            first = last - 1
        end
        i = first
        while i >= base
            pval = unsafe_load(region, i + 1)
            js = Int(unsafe_load(start_col, i + 1))
            je = Int(unsafe_load(start_col, i + 2))
            j = js
            while j < je
                ir = Int(unsafe_load(idx_row, j + 1))
                pval = _fmsub(unsafe_load(el, j + 1), unsafe_load(region, ir + 1), pval)
                j += 1
            end
            if abs(pval) > tol
                unsafe_store!(region, pval, i + 1)
                nnz += 1
                unsafe_store!(region_index, Int32(i), nnz)
            else
                unsafe_store!(region, 0.0, i + 1)
            end
            i -= 1
        end
        if first < base
            base = first + 1
        end
        i = base - 1
        while i >= 0
            pval = unsafe_load(region, i + 1)
            if abs(pval) > tol
                unsafe_store!(region, pval, i + 1)
                nnz += 1
                unsafe_store!(region_index, Int32(i), nnz)
            else
                unsafe_store!(region, 0.0, i + 1)
            end
            i -= 1
        end
    end
    return Int32(nnz)
end

# --- BTRAN L By Row (CoinFactorization4.cpp:2065-2098) ---
function cf_update_column_transpose_l_by_row!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
)::Int32
    tol = cf.zeroTolerance
    nr = Int(cf.numberRows)
    el = cf.elementByRowL
    start_row = cf.startRowL
    col = cf.indexColumnL

    first = nr - 1
    while first >= 0
        if unsafe_load(region, first + 1) != 0.0
            break
        end
        first -= 1
    end
    nnz = 0
    i = first
    while i >= 0
        pval = unsafe_load(region, i + 1)
        if abs(pval) > tol
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
            j = Int(unsafe_load(start_row, i + 2)) - 1
            js = Int(unsafe_load(start_row, i + 1))
            while j >= js
                ir = Int(unsafe_load(col, j + 1))
                v = unsafe_load(el, j + 1)
                unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, ir + 1)), ir + 1)
                j -= 1
            end
        else
            unsafe_store!(region, 0.0, i + 1)
        end
        i -= 1
    end
    return Int32(nnz)
end

# --- BTRAN L Sparsish by Row (CoinFactorization4.cpp:2100-2193) ---
function cf_update_column_transpose_l_sparsish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    nr = Int(cf.numberRows)
    el = cf.elementByRowL
    start_row = cf.startRowL
    col = cf.indexColumnL

    mark = reinterpret(Ptr{UInt8}, cf.sparseArray + 3 * Int(cf.maximumRowsExtra) * sizeof(Int32))
    n_in = Int(nnz_in)
    i = 1
    while i <= n_in
        ipiv = Int(unsafe_load(region_index, i))
        iw = ipiv >> 3
        ib = ipiv - (iw << 3)
        unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
        i += 1
    end

    nnz = 0
    j_last = ((nr - 1) >> 3) << 3
    ii = nr - 1
    while ii >= j_last
        pval = unsafe_load(region, ii + 1)
        if abs(pval) > tol
            nnz += 1
            unsafe_store!(region_index, Int32(ii), nnz)
            j = Int(unsafe_load(start_row, ii + 2)) - 1
            js = Int(unsafe_load(start_row, ii + 1))
            while j >= js
                ir = Int(unsafe_load(col, j + 1))
                v = unsafe_load(el, j + 1)
                iw = ir >> 3
                ib = ir - (iw << 3)
                unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
                unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, ir + 1)), ir + 1)
                j -= 1
            end
        else
            unsafe_store!(region, 0.0, ii + 1)
        end
        ii -= 1
    end

    j_word = j_last >> 3
    unsafe_store!(mark, UInt8(0), j_word + 1)
    k = j_word - 1
    while k >= 0
        if unsafe_load(mark, k + 1) != UInt8(0)
            i_last = k << 3
            ii = i_last + 7
            while ii >= i_last
                pval = unsafe_load(region, ii + 1)
                if abs(pval) > tol
                    nnz += 1
                    unsafe_store!(region_index, Int32(ii), nnz)
                    j = Int(unsafe_load(start_row, ii + 2)) - 1
                    js = Int(unsafe_load(start_row, ii + 1))
                    while j >= js
                        ir = Int(unsafe_load(col, j + 1))
                        v = unsafe_load(el, j + 1)
                        iw = ir >> 3
                        ib = ir - (iw << 3)
                        unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
                        unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, ir + 1)), ir + 1)
                        j -= 1
                    end
                else
                    unsafe_store!(region, 0.0, ii + 1)
                end
                ii -= 1
            end
            unsafe_store!(mark, UInt8(0), k + 1)
        end
        k -= 1
    end
    return Int32(nnz)
end

# --- BTRAN L Sparse (hypersparse DFS, CoinFactorization4.cpp:2196-2278) ---
function cf_update_column_transpose_l_sparse!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    el = cf.elementByRowL
    start_row = cf.startRowL
    col = cf.indexColumnL

    mre = Int(cf.maximumRowsExtra)
    stack = cf.sparseArray
    list = stack + mre * sizeof(Int32)
    next = list + mre * sizeof(Int32)
    mark = reinterpret(Ptr{UInt8}, next + mre * sizeof(Int32))

    n_list = 0
    n_in = Int(nnz_in)
    k = 1
    while k <= n_in
        kpiv = Int(unsafe_load(region_index, k))
        if unsafe_load(mark, kpiv + 1) == UInt8(0) && unsafe_load(region, kpiv + 1) != 0.0
            unsafe_store!(stack, Int32(kpiv), 1)
            j = Int(unsafe_load(start_row, kpiv + 2)) - 1
            n_stack = 0
            while n_stack >= 0
                if j >= Int(unsafe_load(start_row, kpiv + 1))
                    jpiv = Int(unsafe_load(col, j + 1))
                    j -= 1
                    unsafe_store!(next, Int32(j), n_stack + 1)
                    if unsafe_load(mark, jpiv + 1) == UInt8(0)
                        kpiv = jpiv
                        j = Int(unsafe_load(start_row, kpiv + 2)) - 1
                        n_stack += 1
                        unsafe_store!(stack, Int32(kpiv), n_stack + 1)
                        unsafe_store!(mark, UInt8(1), kpiv + 1)
                        unsafe_store!(next, Int32(j), n_stack + 1)
                    end
                else
                    n_list += 1
                    unsafe_store!(list, Int32(kpiv), n_list)
                    unsafe_store!(mark, UInt8(1), kpiv + 1)
                    n_stack -= 1
                    if n_stack >= 0
                        kpiv = Int(unsafe_load(stack, n_stack + 1))
                        j = Int(unsafe_load(next, n_stack + 1))
                    end
                end
            end
        end
        k += 1
    end

    nnz = 0
    i = n_list
    while i >= 1
        ipiv = Int(unsafe_load(list, i))
        unsafe_store!(mark, UInt8(0), ipiv + 1)
        pval = unsafe_load(region, ipiv + 1)
        if abs(pval) > tol
            nnz += 1
            unsafe_store!(region_index, Int32(ipiv), nnz)
            js = Int(unsafe_load(start_row, ipiv + 1))
            je = Int(unsafe_load(start_row, ipiv + 2))
            j = js
            while j < je
                ir = Int(unsafe_load(col, j + 1))
                v = unsafe_load(el, j + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j += 1
            end
        else
            unsafe_store!(region, 0.0, ipiv + 1)
        end
        i -= 1
    end
    return Int32(nnz)
end

# --- BTRAN L Dispatcher + Dense LAPACK block (CoinFactorization4.cpp:2280-2391) ---
function cf_update_column_transpose_l!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
    cap1::Int32,
)::Int32
    nnz = nnz_in
    if cf.numberL == Int32(0) && cf.numberDense == Int32(0)
        if cf.sparseArray != C_NULL || nnz < cf.numberRows
            return nnz
        end
    end
    go_sparse = Int32(-1)
    if cf.sparseThreshold > Int32(0)
        if cf.btranAverageAfterL != 0.0
            new_num = Float64(nnz) * cf.btranAverageAfterL
            if new_num < Float64(cf.sparseThreshold)
                go_sparse = Int32(2)
            elseif new_num < Float64(cf.sparseThreshold2)
                go_sparse = Int32(1)
            else
                go_sparse = Int32(0)
            end
        else
            go_sparse = nnz < cf.sparseThreshold ? Int32(2) : Int32(0)
        end
    else
        go_sparse = Int32(-1)
    end

    if cf.numberDense > Int32(0)
        last_sparse = cf.numberRows - cf.numberDense
        number = Int(nnz)
        do_dense = false
        if number <= Int(cf.numberRows)
            i = 0
            while i < number
                ir = unsafe_load(region_index, i + 1)
                if ir >= last_sparse
                    do_dense = true
                    number -= 1
                    unsafe_store!(region_index, unsafe_load(region_index, number + 1), i + 1)
                else
                    i += 1
                end
            end
        else
            i = Int(cf.numberRows) - 1
            while i >= Int(last_sparse)
                if unsafe_load(region, i + 1) != 0.0
                    do_dense = true
                    number = Int(civ_scan_tol_ptr!(region, region_index, Int32(0), cap1, Int32(0), last_sparse, cf.zeroTolerance))
                    break
                end
                i -= 1
            end
            go_sparse = cf.sparseThreshold != Int32(0) ? Int32(0) : Int32(-1)
        end
        if do_dense
            nnz = Int32(number)
            _lapack_dgetrs!(
                UInt8('T'), cf.numberDense, Int32(1),
                cf.denseAreaAddress, cf.numberDense, cf.densePermute,
                region + Int(last_sparse) * sizeof(Float64), cf.numberDense,
            )
            if go_sparse > Int32(0) || cf.numberL == Int32(0)
                added = civ_scan_tol_ptr!(region, region_index, nnz, cap1, last_sparse, cf.numberRows, cf.zeroTolerance)
                nnz += added
            end
        end
        if cf.numberL == Int32(0)
            if Int(nnz) > Int(cf.numberRows)
                nnz = civ_scan_tol_ptr!(region, region_index, Int32(0), cap1, Int32(0), cf.numberRows, cf.zeroTolerance)
            end
            return nnz
        end
    end

    if go_sparse > Int32(0) && nnz > cf.numberRows
        go_sparse = Int32(0)
    end

    if go_sparse == Int32(-1)
        return cf_update_column_transpose_l_densish!(cf, region, region_index)
    elseif go_sparse == Int32(0)
        return cf_update_column_transpose_l_by_row!(cf, region, region_index)
    elseif go_sparse == Int32(1)
        return cf_update_column_transpose_l_sparsish!(cf, region, region_index, nnz)
    else
        return cf_update_column_transpose_l_sparse!(cf, region, region_index, nnz)
    end
end

# --- Top-level BTRAN: CoinFactorization::updateColumnTranspose (CoinFactorization4.cpp:1193-1309) ---
function cf_update_column_transpose!(
    cf::CoinFactorizationPodView,
    reg1_ptr::Ptr{CoinIndexedVecPod},
    reg2_ptr::Ptr{CoinIndexedVecPod},
)::Int32
    r1 = unsafe_load(reg1_ptr)
    r2 = unsafe_load(reg2_ptr)

    # regionSparse->clear()
    civ_clear_ptr!(r1.elements, r1.indices, r1.n_elements, r1.capacity, r1.packed_mode != Int32(0))
    region = r1.elements
    region_index = r1.indices
    vector = r2.elements
    index = r2.indices
    nnz = Int(r2.n_elements)
    packed = r2.packed_mode != Int32(0)
    piv_col = cf.pivotColumn

    if packed
        i = 1
        while i <= nnz
            ir_in = Int(unsafe_load(index, i))
            val = unsafe_load(vector, i)
            ir = Int(unsafe_load(piv_col, ir_in + 1))
            unsafe_store!(vector, 0.0, i)
            unsafe_store!(region, val, ir + 1)
            unsafe_store!(region_index, Int32(ir), i)
            i += 1
        end
    else
        i = 1
        while i <= nnz
            ir_in = Int(unsafe_load(index, i))
            val = unsafe_load(vector, ir_in + 1)
            unsafe_store!(vector, 0.0, ir_in + 1)
            ir = Int(unsafe_load(piv_col, ir_in + 1))
            unsafe_store!(region, val, ir + 1)
            unsafe_store!(region_index, Int32(ir), i)
            i += 1
        end
    end

    nnz_i32 = Int32(nnz)
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_numberBtranCounts, unsafe_load(cf.p_numberBtranCounts) + Int32(1))
        unsafe_store!(cf.p_btranCountInput, unsafe_load(cf.p_btranCountInput) + Float64(nnz_i32))
    end

    if cf.doForrestTomlin == Int32(0)
        nnz_i32 = cf_update_column_transpose_pfi!(cf, region, region_index, nnz_i32)
    end

    # Apply pivotRegion and find smallestIndex
    piv_reg = cf.pivotRegion
    smallest_idx = cf.numberRowsExtra
    j = 1
    n_cur = Int(nnz_i32)
    while j <= n_cur
        ir = unsafe_load(region_index, j)
        if ir < smallest_idx
            smallest_idx = ir
        end
        ir1 = Int(ir) + 1
        unsafe_store!(region, unsafe_load(region, ir1) * unsafe_load(piv_reg, ir1), ir1)
        j += 1
    end

    # U^T solve
    nnz_i32 = cf_update_column_transpose_u!(cf, region, region_index, nnz_i32, smallest_idx)
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_btranCountAfterU, unsafe_load(cf.p_btranCountAfterU) + Float64(nnz_i32))
    end

    # R^T solve
    nnz_i32 = cf_update_column_transpose_r!(cf, region, region_index, nnz_i32)

    # L^T solve (+ dense LAPACK block)
    nnz_i32 = cf_update_column_transpose_l!(cf, region, region_index, nnz_i32, r1.capacity)
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_btranCountAfterL, unsafe_load(cf.p_btranCountAfterL) + Float64(nnz_i32))
    end

    # Permute back into regionSparse2 and zero regionSparse
    perm_back = cf.pivotColumnBack
    n_out = Int(nnz_i32)
    number = 0
    if packed
        i = 1
        while i <= n_out
            ir = Int(unsafe_load(region_index, i))
            val = unsafe_load(region, ir + 1)
            unsafe_store!(region, 0.0, ir + 1)
            out_row = unsafe_load(perm_back, ir + 1)
            number += 1
            unsafe_store!(vector, val, number)
            unsafe_store!(index, out_row, number)
            i += 1
        end
    else
        i = 1
        while i <= n_out
            ir = Int(unsafe_load(region_index, i))
            val = unsafe_load(region, ir + 1)
            unsafe_store!(region, 0.0, ir + 1)
            out_row = unsafe_load(perm_back, ir + 1)
            unsafe_store!(vector, val, Int(out_row) + 1)
            number += 1
            unsafe_store!(index, out_row, number)
            i += 1
        end
    end

    return Int32(number)
end

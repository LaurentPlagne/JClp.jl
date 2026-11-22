# Faithful translation of CoinUtils/src/CoinFactorization3.cpp + updateColumnPFI
# Zero-allocation hypersparse FTRAN (`updateColumn`, `updateColumnFT`, `updateTwoColumnsFT`),
# hardware arm64 FMA (`_fmsub`), Accelerate LAPACK `dgetrs_`, `--trim=safe` compatible.

# --- Permute back (CoinFactorization3.cpp:176-234) ---
function cf_permute_back!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    old_number::Int32,
    out::Ptr{Float64},
    out_index::Ptr{Int32},
    out_packed::Bool,
)::Int32
    perm_back = cf.pivotColumnBack
    tol = cf.zeroTolerance
    n_old = Int(old_number)
    number = 0
    if out_packed
        j = 1
        while j <= n_old
            ir = Int(unsafe_load(region_index, j))
            val = unsafe_load(region, ir + 1)
            unsafe_store!(region, 0.0, ir + 1)
            if abs(val) > tol
                out_row = unsafe_load(perm_back, ir + 1)
                number += 1
                unsafe_store!(out_index, out_row, number)
                unsafe_store!(out, val, number)
            end
            j += 1
        end
    else
        j = 1
        while j <= n_old
            ir = Int(unsafe_load(region_index, j))
            val = unsafe_load(region, ir + 1)
            unsafe_store!(region, 0.0, ir + 1)
            if abs(val) > tol
                out_row = unsafe_load(perm_back, ir + 1)
                number += 1
                unsafe_store!(out_index, out_row, number)
                unsafe_store!(out, val, Int(out_row) + 1)
            end
            j += 1
        end
    end
    return Int32(number)
end

# --- FTRAN L Densish (CoinFactorization3.cpp:322-380) ---
function cf_update_column_l_densish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    start_col = cf.startColumnL
    idx_row = cf.indexRowL
    el = cf.elementL
    nr = Int(cf.numberRows)
    base_l = Int(cf.baseL)
    last = nr - Int(cf.numberDense)

    nnz = 0
    smallest = Int(cf.numberRowsExtra)
    n_in = Int(nnz_in)
    k = 1
    while k <= n_in
        ipiv = Int(unsafe_load(region_index, k))
        if ipiv >= base_l
            if ipiv < smallest
                smallest = ipiv
            end
        else
            nnz += 1
            unsafe_store!(region_index, Int32(ipiv), nnz)
        end
        k += 1
    end

    i = smallest
    while i < last
        pval = unsafe_load(region, i + 1)
        if abs(pval) > tol
            js = Int(unsafe_load(start_col, i + 1))
            je = Int(unsafe_load(start_col, i + 2))
            j = js
            while j < je
                ir = Int(unsafe_load(idx_row, j + 1))
                v = unsafe_load(el, j + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j += 1
            end
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
        else
            unsafe_store!(region, 0.0, i + 1)
        end
        i += 1
    end

    i = last
    while i < nr
        pval = unsafe_load(region, i + 1)
        if abs(pval) > tol
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
        else
            unsafe_store!(region, 0.0, i + 1)
        end
        i += 1
    end
    return Int32(nnz)
end

# --- FTRAN L Sparsish (CoinFactorization3.cpp:382-528) ---
function cf_update_column_l_sparsish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    start_col = cf.startColumnL
    idx_row = cf.indexRowL
    el = cf.elementL
    nr = Int(cf.numberRows)
    base_l = Int(cf.baseL)
    last = nr - Int(cf.numberDense)

    mark = reinterpret(Ptr{UInt8}, cf.sparseArray + 3 * Int(cf.maximumRowsExtra) * sizeof(Int32))
    nnz = 0
    smallest = Int(cf.numberRowsExtra)
    n_in = Int(nnz_in)
    k = 1
    while k <= n_in
        ipiv = Int(unsafe_load(region_index, k))
        if ipiv < base_l
            nnz += 1
            unsafe_store!(region_index, Int32(ipiv), nnz)
        else
            if ipiv < smallest
                smallest = ipiv
            end
            iw = ipiv >> 3
            ib = ipiv - (iw << 3)
            unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
        end
        k += 1
    end

    j_last = ((smallest + 7) >> 3) << 3
    if j_last > last
        j_last = last
    end
    i = smallest
    while i < j_last
        pval = unsafe_load(region, i + 1)
        js = Int(unsafe_load(start_col, i + 1))
        je = Int(unsafe_load(start_col, i + 2))
        if abs(pval) > tol
            j = js
            while j < je
                ir = Int(unsafe_load(idx_row, j + 1))
                v = unsafe_load(el, j + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                iw = ir >> 3
                ib = ir - (iw << 3)
                unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
                j += 1
            end
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
        else
            unsafe_store!(region, 0.0, i + 1)
        end
        i += 1
    end

    k_last = last >> 3
    if j_last < last
        k = j_last >> 3
        while k < k_last
            if unsafe_load(mark, k + 1) != UInt8(0)
                ii = k << 3
                i_last = ii + 8
                while ii < i_last
                    pval = unsafe_load(region, ii + 1)
                    js = Int(unsafe_load(start_col, ii + 1))
                    je = Int(unsafe_load(start_col, ii + 2))
                    if abs(pval) > tol
                        j = js
                        while j < je
                            ir = Int(unsafe_load(idx_row, j + 1))
                            v = unsafe_load(el, j + 1)
                            unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                            iw = ir >> 3
                            ib = ir - (iw << 3)
                            unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
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
        i = k_last << 3
    end

    while i < last
        pval = unsafe_load(region, i + 1)
        js = Int(unsafe_load(start_col, i + 1))
        je = Int(unsafe_load(start_col, i + 2))
        if abs(pval) > tol
            j = js
            while j < je
                ir = Int(unsafe_load(idx_row, j + 1))
                v = unsafe_load(el, j + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j += 1
            end
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
        else
            unsafe_store!(region, 0.0, i + 1)
        end
        i += 1
    end

    while i < nr
        pval = unsafe_load(region, i + 1)
        if abs(pval) > tol
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
        else
            unsafe_store!(region, 0.0, i + 1)
        end
        i += 1
    end

    unsafe_store!(mark, UInt8(0), (smallest >> 3) + 1)
    kk_last = (nr + 7) >> 3
    if kk_last > k_last
        coin_zero_n!(mark + k_last, kk_last - k_last)
    end
    return Int32(nnz)
end

# --- FTRAN L Sparse (hypersparse DFS, CoinFactorization3.cpp:530-625) ---
function cf_update_column_l_sparse!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    start_col = cf.startColumnL
    idx_row = cf.indexRowL
    el = cf.elementL
    base_l = Int(cf.baseL)

    mre = Int(cf.maximumRowsExtra)
    stack = cf.sparseArray
    list = stack + mre * sizeof(Int32)
    next = list + mre * sizeof(Int32)
    mark = reinterpret(Ptr{UInt8}, next + mre * sizeof(Int32))

    nnz = 0
    n_list = 0
    n_in = Int(nnz_in)
    k = 1
    while k <= n_in
        kpiv = Int(unsafe_load(region_index, k))
        if kpiv >= base_l
            if unsafe_load(mark, kpiv + 1) == UInt8(0)
                unsafe_store!(stack, Int32(kpiv), 1)
                j = Int(unsafe_load(start_col, kpiv + 2)) - 1
                n_stack = 0
                while n_stack >= 0
                    if j >= Int(unsafe_load(start_col, kpiv + 1))
                        jpiv = Int(unsafe_load(idx_row, j + 1))
                        j -= 1
                        unsafe_store!(next, Int32(j), n_stack + 1)
                        if unsafe_load(mark, jpiv + 1) == UInt8(0)
                            kpiv = jpiv
                            j = Int(unsafe_load(start_col, kpiv + 2)) - 1
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
        else
            nnz += 1
            unsafe_store!(region_index, Int32(kpiv), nnz)
        end
        k += 1
    end

    i = n_list
    while i >= 1
        ipiv = Int(unsafe_load(list, i))
        unsafe_store!(mark, UInt8(0), ipiv + 1)
        pval = unsafe_load(region, ipiv + 1)
        if abs(pval) > tol
            nnz += 1
            unsafe_store!(region_index, Int32(ipiv), nnz)
            js = Int(unsafe_load(start_col, ipiv + 1))
            je = Int(unsafe_load(start_col, ipiv + 2))
            j = js
            while j < je
                ir = Int(unsafe_load(idx_row, j + 1))
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

# --- FTRAN L Dispatcher + Dense LAPACK (CoinFactorization3.cpp:236-320) ---
function cf_update_column_l!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    nnz = nnz_in
    if cf.numberL > Int32(0)
        go_sparse = Int32(0)
        if cf.sparseThreshold > Int32(0)
            if cf.ftranAverageAfterL != 0.0
                new_num = Float64(nnz) * cf.ftranAverageAfterL
                if new_num < Float64(cf.sparseThreshold) && Float64(cf.numberL << 2) > new_num
                    go_sparse = Int32(2)
                elseif new_num < Float64(cf.sparseThreshold2) && Float64(cf.numberL << 1) > new_num
                    go_sparse = Int32(1)
                else
                    go_sparse = Int32(0)
                end
            else
                if nnz < cf.sparseThreshold && (cf.numberL << 2) > nnz
                    go_sparse = Int32(2)
                else
                    go_sparse = Int32(0)
                end
            end
        end
        if go_sparse == Int32(0)
            nnz = cf_update_column_l_densish!(cf, region, region_index, nnz)
        elseif go_sparse == Int32(1)
            nnz = cf_update_column_l_sparsish!(cf, region, region_index, nnz)
        else
            nnz = cf_update_column_l_sparse!(cf, region, region_index, nnz)
        end
    end

    if cf.numberDense > Int32(0)
        last_sparse = cf.numberRows - cf.numberDense
        number = Int(nnz)
        i = 0
        do_dense = false
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
        if do_dense
            _lapack_dgetrs!(
                UInt8('N'), cf.numberDense, Int32(1),
                cf.denseAreaAddress, cf.numberDense, cf.densePermute,
                region + Int(last_sparse) * sizeof(Float64), cf.numberDense,
            )
            nr = Int(cf.numberRows)
            ii = Int(last_sparse)
            while ii < nr
                val = unsafe_load(region, ii + 1)
                if val != 0.0
                    if abs(val) >= 1.0e-15
                        number += 1
                        unsafe_store!(region_index, Int32(ii), number)
                    else
                        unsafe_store!(region, 0.0, ii + 1)
                    end
                end
                ii += 1
            end
            nnz = Int32(number)
        end
    end
    return nnz
end

# --- Helper: Choose R method (CoinFactorization3.cpp:1676-1711 / 1996-2035) ---
@inline function _choose_ftran_r_method(cf::CoinFactorizationPodView, nnz::Int, for_ft::Bool)::Int32
    size_r = Float64(unsafe_load(cf.startColumnR, Int(cf.numberR) + 1))
    avg_r = size_r / Float64(cf.numberRowsExtra)
    final_w = Float64(nnz)
    mt1 = Float64(cf.numberPivots) * (2.0 + (Float64(nnz) / Float64(cf.numberRows)) * avg_r)
    mt1 += Float64(nnz) * (1.0 + avg_r)
    mt0 = mt1 + Float64(nnz + Int(cf.numberPivots)) * 0.1
    mt1 += Float64(nnz) * final_w
    mt2 = size_r + Float64(cf.numberPivots) * 2.0 + Float64(nnz) * final_w
    if cf.numberInColumnPlus == C_NULL
        mt0 = 1.0e100
        mt1 = 1.0e100
    elseif cf.sparseArray == C_NULL
        mt0 = 1.0e100
    end
    if for_ft
        mt1 += final_w
    end
    best = 1.0e100
    method = Int32(-1)
    if mt0 < best
        best = mt0; method = Int32(0)
    end
    if mt1 < best
        best = mt1; method = Int32(1)
    end
    if mt2 < best
        method = Int32(2)
    end
    return method
end

# --- FTRAN R (CoinFactorization3.cpp:1660-1977, non-STACK active branch) ---
function cf_update_column_r!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    if cf.numberR == Int32(0)
        return nnz_in
    end
    tol = cf.zeroTolerance
    nr = Int(cf.numberRows)
    nre = Int(cf.numberRowsExtra)
    perm = cf.permute
    nnz = Int(nnz_in)
    method = _choose_ftran_r_method(cf, nnz, false)

    if method == Int32(0)
        mre = Int(cf.maximumRowsExtra)
        mark = reinterpret(Ptr{UInt8}, cf.sparseArray + 3 * mre * sizeof(Int32))
        i = nr
        while i < nre
            ir = Int(unsafe_load(perm, i + 1))
            unsafe_store!(mark, UInt8(1), ir + 1)
            i += 1
        end
        el_r = cf.elementR + Int(cf.lengthAreaR) * sizeof(Float64)
        idx_r = cf.indexRowR + Int(cf.lengthAreaR) * sizeof(Int32)
        start_r = cf.startColumnR + (Int(cf.maximumPivots) + 1) * sizeof(Int32)
        nicp = cf.numberInColumnPlus

        new_num = 0
        i = 1
        while i <= nnz
            ir = Int(unsafe_load(region_index, i))
            if unsafe_load(mark, ir + 1) == UInt8(0)
                new_num += 1
                unsafe_store!(region_index, Int32(ir), new_num)
            end
            num = Int(unsafe_load(nicp, ir + 1))
            if num != 0
                pval = unsafe_load(region, ir + 1)
                s = Int(unsafe_load(start_r, ir + 1))
                e = s + num
                j = s
                while j < e
                    v = unsafe_load(el_r, j + 1)
                    jr = Int(unsafe_load(idx_r, j + 1))
                    unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, jr + 1)), jr + 1)
                    j += 1
                end
            end
            i += 1
        end
        nnz = new_num
        i = nr
        while i < nre
            ir = Int(unsafe_load(perm, i + 1))
            pval = unsafe_load(region, ir + 1) + unsafe_load(region, i + 1)
            unsafe_store!(region, 0.0, ir + 1)
            if abs(pval) > tol
                unsafe_store!(region, pval, i + 1)
                if unsafe_load(mark, i + 1) == UInt8(0)
                    nnz += 1
                    unsafe_store!(region_index, Int32(i), nnz)
                end
                num = Int(unsafe_load(nicp, i + 1))
                s = Int(unsafe_load(start_r, i + 1))
                e = s + num
                j = s
                while j < e
                    v = unsafe_load(el_r, j + 1)
                    jr = Int(unsafe_load(idx_r, j + 1))
                    unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, jr + 1)), jr + 1)
                    j += 1
                end
            else
                unsafe_store!(region, 0.0, i + 1)
            end
            unsafe_store!(mark, UInt8(0), ir + 1)
            i += 1
        end
    elseif method == Int32(1)
        el_r = cf.elementR + Int(cf.lengthAreaR) * sizeof(Float64)
        idx_r = cf.indexRowR + Int(cf.lengthAreaR) * sizeof(Int32)
        start_r = cf.startColumnR + (Int(cf.maximumPivots) + 1) * sizeof(Int32)
        nicp = cf.numberInColumnPlus

        i = 1
        while i <= nnz
            ir = Int(unsafe_load(region_index, i))
            num = Int(unsafe_load(nicp, ir + 1))
            if num != 0
                pval = unsafe_load(region, ir + 1)
                s = Int(unsafe_load(start_r, ir + 1))
                e = s + num
                j = s
                while j < e
                    v = unsafe_load(el_r, j + 1)
                    jr = Int(unsafe_load(idx_r, j + 1))
                    unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, jr + 1)), jr + 1)
                    j += 1
                end
            end
            i += 1
        end
        i = nr
        while i < nre
            ir = Int(unsafe_load(perm, i + 1))
            pval = unsafe_load(region, ir + 1) + unsafe_load(region, i + 1)
            unsafe_store!(region, 0.0, ir + 1)
            if abs(pval) > tol
                unsafe_store!(region, pval, i + 1)
                nnz += 1
                unsafe_store!(region_index, Int32(i), nnz)
                num = Int(unsafe_load(nicp, i + 1))
                s = Int(unsafe_load(start_r, i + 1))
                e = s + num
                j = s
                while j < e
                    v = unsafe_load(el_r, j + 1)
                    jr = Int(unsafe_load(idx_r, j + 1))
                    unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, jr + 1)), jr + 1)
                    j += 1
                end
            else
                unsafe_store!(region, 0.0, i + 1)
            end
            i += 1
        end
    else
        el = cf.elementR
        idx_row = cf.indexRowR
        start_col = cf.startColumnR
        s = Int(unsafe_load(start_col, 1))
        i = nr
        while i < nre
            e = Int(unsafe_load(start_col, i - nr + 2))
            ir = Int(unsafe_load(perm, i + 1))
            pval = unsafe_load(region, ir + 1)
            unsafe_store!(region, 0.0, ir + 1)
            j = s
            while j < e
                v = unsafe_load(el, j + 1)
                jr = Int(unsafe_load(idx_row, j + 1))
                v *= unsafe_load(region, jr + 1)
                pval -= v
                j += 1
            end
            s = e
            if abs(pval) > tol
                unsafe_store!(region, pval, i + 1)
                nnz += 1
                unsafe_store!(region_index, Int32(i), nnz)
            else
                unsafe_store!(region, 0.0, i + 1)
            end
            i += 1
        end
    end

    if method != Int32(0)
        n = nnz
        nnz = 0
        i = 1
        while i <= n
            idx_val = unsafe_load(region_index, i)
            if unsafe_load(region, Int(idx_val) + 1) != 0.0
                nnz += 1
                unsafe_store!(region_index, idx_val, nnz)
            end
            i += 1
        end
    end
    return Int32(nnz)
end

# --- FTRAN R FT (CoinFactorization3.cpp:1979-2244) ---
function cf_update_column_r_ft!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    start_col_u = cf.startColumnU
    num_in_col = cf.numberInColumn
    icol = Int(cf.numberColumnsExtra)
    max_cols_extra = Int(cf.maximumColumnsExtra)
    nnz = Int(nnz_in)

    if cf.numberR > Int32(0)
        tol = cf.zeroTolerance
        nr = Int(cf.numberRows)
        nre = Int(cf.numberRowsExtra)
        perm = cf.permute
        method = _choose_ftran_r_method(cf, nnz, true)

        if method == Int32(0)
            mre = Int(cf.maximumRowsExtra)
            mark = reinterpret(Ptr{UInt8}, cf.sparseArray + 3 * mre * sizeof(Int32))
            i = nr
            while i < nre
                ir = Int(unsafe_load(perm, i + 1))
                unsafe_store!(mark, UInt8(1), ir + 1)
                i += 1
            end
            el_r = cf.elementR + Int(cf.lengthAreaR) * sizeof(Float64)
            idx_r = cf.indexRowR + Int(cf.lengthAreaR) * sizeof(Int32)
            start_r = cf.startColumnR + (Int(cf.maximumPivots) + 1) * sizeof(Int32)
            nicp = cf.numberInColumnPlus

            s_u = unsafe_load(start_col_u, max_cols_extra + 1)
            unsafe_store!(start_col_u, s_u, icol + 1)
            put_el = cf.elementU + Int(s_u) * sizeof(Float64)

            new_num = 0
            i = 1
            while i <= nnz
                ir = Int(unsafe_load(region_index, i))
                pval = unsafe_load(region, ir + 1)
                if unsafe_load(mark, ir + 1) == UInt8(0)
                    new_num += 1
                    unsafe_store!(put_el, pval, new_num)
                    unsafe_store!(region_index, Int32(ir), new_num)
                end
                num = Int(unsafe_load(nicp, ir + 1))
                if num != 0
                    s = Int(unsafe_load(start_r, ir + 1))
                    e = s + num
                    j = s
                    while j < e
                        v = unsafe_load(el_r, j + 1)
                        jr = Int(unsafe_load(idx_r, j + 1))
                        unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, jr + 1)), jr + 1)
                        j += 1
                    end
                end
                i += 1
            end
            nnz = new_num
            i = nr
            while i < nre
                ir = Int(unsafe_load(perm, i + 1))
                pval = unsafe_load(region, ir + 1) + unsafe_load(region, i + 1)
                unsafe_store!(region, 0.0, ir + 1)
                if abs(pval) > tol
                    unsafe_store!(region, pval, i + 1)
                    if unsafe_load(mark, i + 1) == UInt8(0)
                        nnz += 1
                        unsafe_store!(put_el, pval, nnz)
                        unsafe_store!(region_index, Int32(i), nnz)
                    end
                    num = Int(unsafe_load(nicp, i + 1))
                    s = Int(unsafe_load(start_r, i + 1))
                    e = s + num
                    j = s
                    while j < e
                        v = unsafe_load(el_r, j + 1)
                        jr = Int(unsafe_load(idx_r, j + 1))
                        unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, jr + 1)), jr + 1)
                        j += 1
                    end
                else
                    unsafe_store!(region, 0.0, i + 1)
                end
                unsafe_store!(mark, UInt8(0), ir + 1)
                i += 1
            end
            unsafe_store!(num_in_col, Int32(nnz), icol + 1)
            unsafe_store!(start_col_u, s_u + Int32(nnz), max_cols_extra + 1)
        elseif method == Int32(1)
            el_r = cf.elementR + Int(cf.lengthAreaR) * sizeof(Float64)
            idx_r = cf.indexRowR + Int(cf.lengthAreaR) * sizeof(Int32)
            start_r = cf.startColumnR + (Int(cf.maximumPivots) + 1) * sizeof(Int32)
            nicp = cf.numberInColumnPlus

            i = 1
            while i <= nnz
                ir = Int(unsafe_load(region_index, i))
                num = Int(unsafe_load(nicp, ir + 1))
                if num != 0
                    pval = unsafe_load(region, ir + 1)
                    s = Int(unsafe_load(start_r, ir + 1))
                    e = s + num
                    j = s
                    while j < e
                        v = unsafe_load(el_r, j + 1)
                        jr = Int(unsafe_load(idx_r, j + 1))
                        unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, jr + 1)), jr + 1)
                        j += 1
                    end
                end
                i += 1
            end
            i = nr
            while i < nre
                ir = Int(unsafe_load(perm, i + 1))
                pval = unsafe_load(region, ir + 1) + unsafe_load(region, i + 1)
                unsafe_store!(region, 0.0, ir + 1)
                if abs(pval) > tol
                    unsafe_store!(region, pval, i + 1)
                    nnz += 1
                    unsafe_store!(region_index, Int32(i), nnz)
                    num = Int(unsafe_load(nicp, i + 1))
                    s = Int(unsafe_load(start_r, i + 1))
                    e = s + num
                    j = s
                    while j < e
                        v = unsafe_load(el_r, j + 1)
                        jr = Int(unsafe_load(idx_r, j + 1))
                        unsafe_store!(region, _fmsub(pval, v, unsafe_load(region, jr + 1)), jr + 1)
                        j += 1
                    end
                else
                    unsafe_store!(region, 0.0, i + 1)
                end
                i += 1
            end
        else
            el = cf.elementR
            idx_row = cf.indexRowR
            start_col = cf.startColumnR
            s = Int(unsafe_load(start_col, 1))
            i = nr
            while i < nre
                e = Int(unsafe_load(start_col, i - nr + 2))
                ir = Int(unsafe_load(perm, i + 1))
                pval = unsafe_load(region, ir + 1)
                unsafe_store!(region, 0.0, ir + 1)
                j = s
                while j < e
                    v = unsafe_load(el, j + 1)
                    jr = Int(unsafe_load(idx_row, j + 1))
                    v *= unsafe_load(region, jr + 1)
                    pval -= v
                    j += 1
                end
                s = e
                if abs(pval) > tol
                    unsafe_store!(region, pval, i + 1)
                    nnz += 1
                    unsafe_store!(region_index, Int32(i), nnz)
                else
                    unsafe_store!(region, 0.0, i + 1)
                end
                i += 1
            end
        end

        if method != Int32(0)
            n = nnz
            nnz = 0
            s_u = unsafe_load(start_col_u, icol + 1)
            put_idx = cf.indexRowU + Int(s_u) * sizeof(Int32)
            put_el = cf.elementU + Int(s_u) * sizeof(Float64)
            i = 1
            while i <= n
                idx_val = unsafe_load(region_index, i)
                val = unsafe_load(region, Int(idx_val) + 1)
                if val != 0.0
                    nnz += 1
                    unsafe_store!(put_idx, idx_val, nnz)
                    unsafe_store!(put_el, val, nnz)
                    unsafe_store!(region_index, idx_val, nnz)
                end
                i += 1
            end
            unsafe_store!(num_in_col, Int32(nnz), icol + 1)
            unsafe_store!(start_col_u, s_u + Int32(nnz), max_cols_extra + 1)
        end
        return Int32(nnz)
    else
        s_u = unsafe_load(start_col_u, icol + 1)
        unsafe_store!(num_in_col, Int32(nnz), icol + 1)
        unsafe_store!(start_col_u, s_u + Int32(nnz), max_cols_extra + 1)
        put_idx = cf.indexRowU + Int(s_u) * sizeof(Int32)
        put_el = cf.elementU + Int(s_u) * sizeof(Float64)
        i = 1
        while i <= nnz
            idx_val = unsafe_load(region_index, i)
            val = unsafe_load(region, Int(idx_val) + 1)
            unsafe_store!(put_idx, idx_val, i)
            unsafe_store!(put_el, val, i)
            i += 1
        end
        return Int32(nnz)
    end
end

# --- FTRAN U Densish (CoinFactorization3.cpp:1102-1217) ---
function cf_update_column_u_densish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
)::Int32
    tol = cf.zeroTolerance
    start_col = cf.startColumnU
    idx_row = cf.indexRowU
    el = cf.elementU
    num_in_col = cf.numberInColumn
    piv_reg = cf.pivotRegion
    n_slacks = Int(cf.numberSlacks)

    nnz = 0
    i = Int(cf.numberU) - 1
    while i >= n_slacks
        pval = unsafe_load(region, i + 1)
        if pval != 0.0
            unsafe_store!(region, 0.0, i + 1)
            if abs(pval) > tol
                s = Int(unsafe_load(start_col, i + 1))
                j = Int(unsafe_load(num_in_col, i + 1)) - 1
                while j >= 0
                    ir = Int(unsafe_load(idx_row, s + j + 1))
                    v = unsafe_load(el, s + j + 1)
                    unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                    j -= 1
                end
                pval *= unsafe_load(piv_reg, i + 1)
                unsafe_store!(region, pval, i + 1)
                nnz += 1
                unsafe_store!(region_index, Int32(i), nnz)
            end
        end
        i -= 1
    end

    if cf.slackValue == -1.0
        i = n_slacks - 1
        while i >= 0
            val = unsafe_load(region, i + 1)
            if val != 0.0
                if abs(val) > tol
                    unsafe_store!(region, -val, i + 1)
                    nnz += 1
                    unsafe_store!(region_index, Int32(i), nnz)
                else
                    unsafe_store!(region, 0.0, i + 1)
                end
            end
            i -= 1
        end
    else
        i = n_slacks - 1
        while i >= 0
            val = unsafe_load(region, i + 1)
            if val != 0.0
                if abs(val) > tol
                    unsafe_store!(region, val, i + 1)
                    nnz += 1
                    unsafe_store!(region_index, Int32(i), nnz)
                else
                    unsafe_store!(region, 0.0, i + 1)
                end
            end
            i -= 1
        end
    end
    return Int32(nnz)
end

# --- FTRAN U Sparse (hypersparse DFS, CoinFactorization3.cpp:1225-1382) ---
function cf_update_column_u_sparse!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    index_in::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    start_col = cf.startColumnU
    idx_row = cf.indexRowU
    el = cf.elementU
    piv_reg = cf.pivotRegion
    num_in_col = cf.numberInColumn
    n_slacks = Int(cf.numberSlacks)

    mre = Int(cf.maximumRowsExtra)
    stack = cf.sparseArray
    list = stack + mre * sizeof(Int32)
    next = list + mre * sizeof(Int32)
    mark = reinterpret(Ptr{UInt8}, next + mre * sizeof(Int32))

    # Slacks stored backwards from end of stack array (1-based offset mre down)
    put_pos = mre
    n_list = 0
    n_in = Int(nnz_in)
    i = 1
    while i <= n_in
        kpiv = Int(unsafe_load(index_in, i))
        unsafe_store!(stack, Int32(kpiv), 1)
        j = Int(unsafe_load(start_col, kpiv + 1)) + Int(unsafe_load(num_in_col, kpiv + 1)) - 1
        n_stack = 1
        unsafe_store!(next, Int32(j), 1)
        while n_stack > 0
            kpiv = Int(unsafe_load(stack, n_stack))
            n_stack -= 1
            if unsafe_load(mark, kpiv + 1) != UInt8(1)
                j = Int(unsafe_load(next, n_stack + 1))
                if j >= Int(unsafe_load(start_col, kpiv + 1))
                    next_kpiv = Int(unsafe_load(idx_row, j + 1))
                    j -= 1
                    n_stack += 1
                    unsafe_store!(next, Int32(j), n_stack)
                    if unsafe_load(mark, next_kpiv + 1) == UInt8(0)
                        nin = Int(unsafe_load(num_in_col, next_kpiv + 1))
                        if nin != 0
                            j2 = Int(unsafe_load(start_col, next_kpiv + 1)) + nin - 1
                            n_stack += 1
                            unsafe_store!(stack, Int32(next_kpiv), n_stack)
                            unsafe_store!(mark, UInt8(2), next_kpiv + 1)
                            unsafe_store!(next, Int32(j2), n_stack)
                        else
                            unsafe_store!(mark, UInt8(1), next_kpiv + 1)
                            if next_kpiv >= n_slacks
                                n_list += 1
                                unsafe_store!(list, Int32(next_kpiv), n_list)
                            else
                                unsafe_store!(stack, Int32(next_kpiv), put_pos)
                                put_pos -= 1
                            end
                        end
                    end
                else
                    unsafe_store!(mark, UInt8(1), kpiv + 1)
                    if kpiv >= n_slacks
                        n_list += 1
                        unsafe_store!(list, Int32(kpiv), n_list)
                    else
                        unsafe_store!(stack, Int32(kpiv), put_pos)
                        put_pos -= 1
                    end
                end
            end
        end
        i += 1
    end

    nnz = 0
    i = n_list
    while i >= 1
        ipiv = Int(unsafe_load(list, i))
        unsafe_store!(mark, UInt8(0), ipiv + 1)
        pval = unsafe_load(region, ipiv + 1)
        unsafe_store!(region, 0.0, ipiv + 1)
        if abs(pval) > tol
            s = Int(unsafe_load(start_col, ipiv + 1))
            num = Int(unsafe_load(num_in_col, ipiv + 1))
            j = s
            je = s + num
            while j < je
                v = unsafe_load(el, j + 1)
                ir = Int(unsafe_load(idx_row, j + 1))
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j += 1
            end
            pval *= unsafe_load(piv_reg, ipiv + 1)
            unsafe_store!(region, pval, ipiv + 1)
            nnz += 1
            unsafe_store!(region_index, Int32(ipiv), nnz)
        end
        i -= 1
    end

    # Slacks from put_pos + 1 .. mre
    p = put_pos + 1
    if cf.slackValue == 1.0
        while p <= mre
            ipiv = Int(unsafe_load(stack, p))
            unsafe_store!(mark, UInt8(0), ipiv + 1)
            pval = unsafe_load(region, ipiv + 1)
            unsafe_store!(region, 0.0, ipiv + 1)
            if abs(pval) > tol
                unsafe_store!(region, pval, ipiv + 1)
                nnz += 1
                unsafe_store!(region_index, Int32(ipiv), nnz)
            end
            p += 1
        end
    else
        while p <= mre
            ipiv = Int(unsafe_load(stack, p))
            unsafe_store!(mark, UInt8(0), ipiv + 1)
            pval = unsafe_load(region, ipiv + 1)
            unsafe_store!(region, 0.0, ipiv + 1)
            if abs(pval) > tol
                unsafe_store!(region, -pval, ipiv + 1)
                nnz += 1
                unsafe_store!(region_index, Int32(ipiv), nnz)
            end
            p += 1
        end
    end
    return Int32(nnz)
end

# --- FTRAN U Sparsish (CoinFactorization3.cpp:1398-1658) ---
function cf_update_column_u_sparsish!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    index_in::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    start_col = cf.startColumnU
    idx_row = cf.indexRowU
    el = cf.elementU
    piv_reg = cf.pivotRegion
    num_in_col = cf.numberInColumn
    n_u = Int(cf.numberU)
    n_slacks = Int(cf.numberSlacks)

    mre = Int(cf.maximumRowsExtra)
    mark = reinterpret(Ptr{UInt8}, cf.sparseArray + 3 * mre * sizeof(Int32))

    n_in = Int(nnz_in)
    ii = 1
    while ii <= n_in
        ipiv = Int(unsafe_load(index_in, ii))
        iw = ipiv >> 3
        ib = ipiv - (iw << 3)
        unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
        ii += 1
    end

    nnz = 0
    j_last = ((n_u - 1) >> 3) << 3
    if j_last < n_slacks
        j_last = n_slacks
    end
    i = n_u - 1
    while i >= j_last
        pval = unsafe_load(region, i + 1)
        unsafe_store!(region, 0.0, i + 1)
        if abs(pval) > tol
            s = Int(unsafe_load(start_col, i + 1))
            j = Int(unsafe_load(num_in_col, i + 1)) - 1
            while j >= 0
                ir = Int(unsafe_load(idx_row, s + j + 1))
                v = unsafe_load(el, s + j + 1)
                iw = ir >> 3
                ib = ir - (iw << 3)
                unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j -= 1
            end
            pval *= unsafe_load(piv_reg, i + 1)
            unsafe_store!(region, pval, i + 1)
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
        end
        i -= 1
    end

    k_last = (n_slacks + 7) >> 3
    if j_last > n_slacks
        k = (j_last >> 3) - 1
        while k >= k_last
            if unsafe_load(mark, k + 1) != UInt8(0)
                i_last = k << 3
                ii2 = i_last + 7
                while ii2 >= i_last
                    pval = unsafe_load(region, ii2 + 1)
                    if pval != 0.0
                        unsafe_store!(region, 0.0, ii2 + 1)
                        if abs(pval) > tol
                            s = Int(unsafe_load(start_col, ii2 + 1))
                            j = Int(unsafe_load(num_in_col, ii2 + 1)) - 1
                            while j >= 0
                                ir = Int(unsafe_load(idx_row, s + j + 1))
                                v = unsafe_load(el, s + j + 1)
                                iw = ir >> 3
                                ib = ir - (iw << 3)
                                unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
                                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                                j -= 1
                            end
                            pval *= unsafe_load(piv_reg, ii2 + 1)
                            unsafe_store!(region, pval, ii2 + 1)
                            nnz += 1
                            unsafe_store!(region_index, Int32(ii2), nnz)
                        end
                    end
                    ii2 -= 1
                end
                unsafe_store!(mark, UInt8(0), k + 1)
            end
            k -= 1
        end
        i = (k_last << 3) - 1
    end

    while i >= n_slacks
        pval = unsafe_load(region, i + 1)
        unsafe_store!(region, 0.0, i + 1)
        if abs(pval) > tol
            s = Int(unsafe_load(start_col, i + 1))
            j = Int(unsafe_load(num_in_col, i + 1)) - 1
            while j >= 0
                ir = Int(unsafe_load(idx_row, s + j + 1))
                v = unsafe_load(el, s + j + 1)
                iw = ir >> 3
                ib = ir - (iw << 3)
                unsafe_store!(mark, unsafe_load(mark, iw + 1) | (UInt8(1) << ib), iw + 1)
                unsafe_store!(region, _fmsub(v, pval, unsafe_load(region, ir + 1)), ir + 1)
                j -= 1
            end
            pval *= unsafe_load(piv_reg, i + 1)
            unsafe_store!(region, pval, i + 1)
            nnz += 1
            unsafe_store!(region_index, Int32(i), nnz)
        end
        i -= 1
    end

    if n_slacks > 0
        jl = ((n_slacks - 1) >> 3) << 3
        sgn = cf.slackValue == 1.0 ? 1.0 : -1.0
        ii3 = n_slacks - 1
        while ii3 >= jl
            val = unsafe_load(region, ii3 + 1)
            if val != 0.0
                unsafe_store!(region, 0.0, ii3 + 1)
                if abs(val) > tol
                    unsafe_store!(region, sgn * val, ii3 + 1)
                    nnz += 1
                    unsafe_store!(region_index, Int32(ii3), nnz)
                end
            end
            ii3 -= 1
        end
        k = (jl >> 3) - 1
        while k >= 0
            if unsafe_load(mark, k + 1) != UInt8(0)
                i_last = k << 3
                ii4 = i_last + 7
                while ii4 >= i_last
                    val = unsafe_load(region, ii4 + 1)
                    if val != 0.0
                        unsafe_store!(region, 0.0, ii4 + 1)
                        if abs(val) > tol
                            unsafe_store!(region, sgn * val, ii4 + 1)
                            nnz += 1
                            unsafe_store!(region_index, Int32(ii4), nnz)
                        end
                    end
                    ii4 -= 1
                end
                unsafe_store!(mark, UInt8(0), k + 1)
            end
            k -= 1
        end
        unsafe_store!(mark, UInt8(0), ((n_slacks - 1) >> 3) + 1)
    end
    if n_u > 0
        unsafe_store!(mark, UInt8(0), ((n_u - 1) >> 3) + 1)
    end
    unsafe_store!(mark, UInt8(0), (n_slacks >> 3) + 1)
    return Int32(nnz)
end

# --- FTRAN U Dispatcher (CoinFactorization3.cpp:1019-1092) ---
function cf_update_column_u!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    index_in::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    go_sparse = Int32(0)
    if cf.sparseThreshold > Int32(0)
        if cf.ftranAverageAfterR != 0.0
            new_num = Float64(nnz_in) * cf.ftranAverageAfterU
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
    end
    nnz = if go_sparse == Int32(0)
        cf_update_column_u_densish!(cf, region, region_index)
    elseif go_sparse == Int32(1)
        cf_update_column_u_sparsish!(cf, region, region_index, index_in, nnz_in)
    else
        cf_update_column_u_sparse!(cf, region, region_index, index_in, nnz_in)
    end
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_ftranCountAfterU, unsafe_load(cf.p_ftranCountAfterU) + Float64(nnz))
    end
    return nnz
end

# --- FTRAN PFI (CoinFactorization4.cpp:3253-3303) ---
function cf_update_column_pfi!(
    cf::CoinFactorizationPodView,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    nnz_in::Int32,
)::Int32
    tol = cf.zeroTolerance
    nr = Int(cf.numberRows)
    start_col = cf.startColumnU + nr * sizeof(Int32)
    idx_row = cf.indexRowU
    el = cf.elementU
    piv_reg = cf.pivotRegion + nr * sizeof(Float64)
    piv_col = cf.pivotColumn + nr * sizeof(Int32)

    nnz = Int(nnz_in)
    np = Int(cf.numberPivots)
    i = 0
    while i < np
        prow = Int(unsafe_load(piv_col, i + 1))
        pval = unsafe_load(region, prow + 1)
        if pval != 0.0
            if abs(pval) > tol
                js = Int(unsafe_load(start_col, i + 1))
                je = Int(unsafe_load(start_col, i + 2))
                j = js
                while j < je
                    ir = Int(unsafe_load(idx_row, j + 1))
                    old_v = unsafe_load(region, ir + 1)
                    val = _fmsub(pval, unsafe_load(el, j + 1), old_v)
                    if old_v == 0.0
                        if abs(val) > tol
                            unsafe_store!(region, val, ir + 1)
                            nnz += 1
                            unsafe_store!(region_index, Int32(ir), nnz)
                        end
                    else
                        if abs(val) > tol
                            unsafe_store!(region, val, ir + 1)
                        else
                            unsafe_store!(region, COIN_INDEXED_REALLY_TINY_ELEMENT, ir + 1)
                        end
                    end
                    j += 1
                end
                unsafe_store!(region, pval * unsafe_load(piv_reg, i + 1), prow + 1)
            else
                unsafe_store!(region, COIN_INDEXED_REALLY_TINY_ELEMENT, prow + 1)
            end
        end
        i += 1
    end
    return Int32(nnz)
end

# --- FTRAN Two-column U Densish (CoinFactorization3.cpp:857-1008) ---
function cf_update_two_columns_u_densish!(
    cf::CoinFactorizationPodView,
    region1::Ptr{Float64},
    index1::Ptr{Int32},
    region2::Ptr{Float64},
    index2::Ptr{Int32},
)::Tuple{Int32,Int32}
    tol = cf.zeroTolerance
    start_col = cf.startColumnU
    idx_row = cf.indexRowU
    el = cf.elementU
    num_in_col = cf.numberInColumn
    piv_reg = cf.pivotRegion
    n_slacks = Int(cf.numberSlacks)

    nnz_a = 0
    nnz_b = 0
    i = Int(cf.numberU) - 1
    while i >= n_slacks
        pval2 = unsafe_load(region2, i + 1)
        do2 = abs(pval2) > tol
        unsafe_store!(region2, 0.0, i + 1)
        pval1 = unsafe_load(region1, i + 1)
        do1 = abs(pval1) > tol
        unsafe_store!(region1, 0.0, i + 1)
        if do2
            s = Int(unsafe_load(start_col, i + 1))
            n = Int(unsafe_load(num_in_col, i + 1))
            if !do1
                j = 0
                while j < n
                    ir = Int(unsafe_load(idx_row, s + j + 1))
                    v = unsafe_load(el, s + j + 1)
                    unsafe_store!(region2, _fmsub(v, pval2, unsafe_load(region2, ir + 1)), ir + 1)
                    j += 1
                end
                pval2 *= unsafe_load(piv_reg, i + 1)
                unsafe_store!(region2, pval2, i + 1)
                nnz_b += 1
                unsafe_store!(index2, Int32(i), nnz_b)
            else
                j = 0
                while j < n
                    ir = Int(unsafe_load(idx_row, s + j + 1))
                    v = unsafe_load(el, s + j + 1)
                    unsafe_store!(region1, _fmsub(v, pval1, unsafe_load(region1, ir + 1)), ir + 1)
                    unsafe_store!(region2, _fmsub(v, pval2, unsafe_load(region2, ir + 1)), ir + 1)
                    j += 1
                end
                pr = unsafe_load(piv_reg, i + 1)
                unsafe_store!(region1, pval1 * pr, i + 1)
                nnz_a += 1
                unsafe_store!(index1, Int32(i), nnz_a)
                unsafe_store!(region2, pval2 * pr, i + 1)
                nnz_b += 1
                unsafe_store!(index2, Int32(i), nnz_b)
            end
        elseif do1
            s = Int(unsafe_load(start_col, i + 1))
            n = Int(unsafe_load(num_in_col, i + 1))
            j = 0
            while j < n
                ir = Int(unsafe_load(idx_row, s + j + 1))
                v = unsafe_load(el, s + j + 1)
                unsafe_store!(region1, _fmsub(v, pval1, unsafe_load(region1, ir + 1)), ir + 1)
                j += 1
            end
            pval1 *= unsafe_load(piv_reg, i + 1)
            unsafe_store!(region1, pval1, i + 1)
            nnz_a += 1
            unsafe_store!(index1, Int32(i), nnz_a)
        end
        i -= 1
    end

    i = n_slacks - 1
    while i >= 0
        v2 = unsafe_load(region2, i + 1)
        v1 = unsafe_load(region1, i + 1)
        if abs(v2) > tol
            unsafe_store!(region2, -v2, i + 1)
            nnz_b += 1
            unsafe_store!(index2, Int32(i), nnz_b)
        else
            unsafe_store!(region2, 0.0, i + 1)
        end
        if v1 != 0.0
            if abs(v1) > tol
                unsafe_store!(region1, -v1, i + 1)
                nnz_a += 1
                unsafe_store!(index1, Int32(i), nnz_a)
            else
                unsafe_store!(region1, 0.0, i + 1)
            end
        end
        i -= 1
    end
    return Int32(nnz_a), Int32(nnz_b)
end

# --- Top-level FTRAN: CoinFactorization::updateColumn (CoinFactorization3.cpp:89-174) ---
function cf_update_column!(
    cf::CoinFactorizationPodView,
    reg1_ptr::Ptr{CoinIndexedVecPod},
    reg2_ptr::Ptr{CoinIndexedVecPod},
    no_permute::Bool,
)::Int32
    r1 = unsafe_load(reg1_ptr)
    r2 = unsafe_load(reg2_ptr)
    region = r1.elements
    region_index = r1.indices
    perm = cf.permute

    nnz = Int32(0)
    if !no_permute
        nnz = r2.n_elements
        idx2 = r2.indices
        arr2 = r2.elements
        n_in = Int(nnz)
        if r2.packed_mode != Int32(0)
            j = 1
            while j <= n_in
                ir_in = Int(unsafe_load(idx2, j))
                val = unsafe_load(arr2, j)
                unsafe_store!(arr2, 0.0, j)
                ir = unsafe_load(perm, ir_in + 1)
                unsafe_store!(region, val, Int(ir) + 1)
                unsafe_store!(region_index, ir, j)
                j += 1
            end
        else
            j = 1
            while j <= n_in
                ir_in = Int(unsafe_load(idx2, j))
                val = unsafe_load(arr2, ir_in + 1)
                unsafe_store!(arr2, 0.0, ir_in + 1)
                ir = unsafe_load(perm, ir_in + 1)
                unsafe_store!(region, val, Int(ir) + 1)
                unsafe_store!(region_index, ir, j)
                j += 1
            end
        end
    else
        nnz = r1.n_elements
    end

    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_numberFtranCounts, unsafe_load(cf.p_numberFtranCounts) + Int32(1))
        unsafe_store!(cf.p_ftranCountInput, unsafe_load(cf.p_ftranCountInput) + Float64(nnz))
    end

    nnz = cf_update_column_l!(cf, region, region_index, nnz)
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_ftranCountAfterL, unsafe_load(cf.p_ftranCountAfterL) + Float64(nnz))
    end
    nnz = cf_update_column_r!(cf, region, region_index, nnz)
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_ftranCountAfterR, unsafe_load(cf.p_ftranCountAfterR) + Float64(nnz))
    end
    nnz = cf_update_column_u!(cf, region, region_index, region_index, nnz)
    if cf.doForrestTomlin == Int32(0)
        nnz = cf_update_column_pfi!(cf, region, region_index, nnz)
    end

    if !no_permute
        return cf_permute_back!(cf, region, region_index, nnz, r2.elements, r2.indices, r2.packed_mode != Int32(0))
    else
        return nnz
    end
end

# --- Top-level FTRAN FT: CoinFactorization::updateColumnFT (CoinFactorization3.cpp:2249-2364) ---
function cf_update_column_ft!(
    cf::CoinFactorizationPodView,
    reg1_ptr::Ptr{CoinIndexedVecPod},
    reg2_ptr::Ptr{CoinIndexedVecPod},
)::Int32
    r1 = unsafe_load(reg1_ptr)
    r2 = unsafe_load(reg2_ptr)
    region = r1.elements
    region_index = r1.indices
    perm = cf.permute
    start_col_u = cf.startColumnU

    do_ft = cf.doForrestTomlin != Int32(0)
    if do_ft
        icol = Int(cf.numberColumnsExtra)
        max_ce = Int(cf.maximumColumnsExtra)
        s = unsafe_load(start_col_u, max_ce + 1)
        unsafe_store!(start_col_u, s, icol + 1)
        space = cf.lengthAreaU - (s + cf.numberRowsExtra)
        do_ft = space >= Int32(0)
        if do_ft
            region_index = cf.indexRowU + Int(s) * sizeof(Int32)
        else
            unsafe_store!(start_col_u, cf.lengthAreaU + Int32(1), max_ce + 1)
        end
    end

    nnz = r2.n_elements
    n_in = Int(nnz)
    idx2 = r2.indices
    arr2 = r2.elements
    if r2.packed_mode != Int32(0)
        j = 1
        while j <= n_in
            ir_in = Int(unsafe_load(idx2, j))
            val = unsafe_load(arr2, j)
            unsafe_store!(arr2, 0.0, j)
            ir = unsafe_load(perm, ir_in + 1)
            unsafe_store!(region, val, Int(ir) + 1)
            unsafe_store!(region_index, ir, j)
            j += 1
        end
    else
        j = 1
        while j <= n_in
            ir_in = Int(unsafe_load(idx2, j))
            val = unsafe_load(arr2, ir_in + 1)
            unsafe_store!(arr2, 0.0, ir_in + 1)
            ir = unsafe_load(perm, ir_in + 1)
            unsafe_store!(region, val, Int(ir) + 1)
            unsafe_store!(region_index, ir, j)
            j += 1
        end
    end

    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_numberFtranCounts, unsafe_load(cf.p_numberFtranCounts) + Int32(1))
        unsafe_store!(cf.p_ftranCountInput, unsafe_load(cf.p_ftranCountInput) + Float64(nnz))
    end

    nnz = cf_update_column_l!(cf, region, region_index, nnz)
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_ftranCountAfterL, unsafe_load(cf.p_ftranCountAfterL) + Float64(nnz))
    end

    if do_ft
        nnz = cf_update_column_r_ft!(cf, region, region_index, nnz)
    else
        nnz = cf_update_column_r!(cf, region, r1.indices, nnz)
    end
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_ftranCountAfterR, unsafe_load(cf.p_ftranCountAfterR) + Float64(nnz))
    end

    nnz = cf_update_column_u!(cf, region, r1.indices, region_index, nnz)
    if cf.doForrestTomlin == Int32(0)
        nnz = cf_update_column_pfi!(cf, region, r1.indices, nnz)
    end

    out_n = cf_permute_back!(cf, region, r1.indices, nnz, r2.elements, r2.indices, r2.packed_mode != Int32(0))
    return do_ft ? out_n : -out_n
end

# --- Top-level Two-Column FTRAN FT: CoinFactorization::updateTwoColumnsFT (CoinFactorization3.cpp:631-855) ---
# Writes updated n_elements for region3 into out_n3_ptr, returns n_elements of region2.
function cf_update_two_columns_ft!(
    cf::CoinFactorizationPodView,
    reg1_ptr::Ptr{CoinIndexedVecPod},
    reg2_ptr::Ptr{CoinIndexedVecPod},
    reg3_ptr::Ptr{CoinIndexedVecPod},
    no_permute_reg3::Bool,
    out_n3_ptr::Ptr{Int32},
)::Int32
    r1 = unsafe_load(reg1_ptr)
    r2 = unsafe_load(reg2_ptr)
    r3 = unsafe_load(reg3_ptr)
    perm = cf.permute

    # Map regionFT and regionUpdate pointers cleanly
    ft_el = no_permute_reg3 ? r1.elements : r3.elements
    ft_idx = no_permute_reg3 ? r1.indices : r3.indices
    upd_el = no_permute_reg3 ? r3.elements : r1.elements
    upd_idx = no_permute_reg3 ? r3.indices : r1.indices

    nnz_upd = Int32(0)
    if !no_permute_reg3
        nnz_upd = r3.n_elements
        n3 = Int(nnz_upd)
        j = 1
        while j <= n3
            ir_in = Int(unsafe_load(r3.indices, j))
            val = unsafe_load(r3.elements, ir_in + 1)
            unsafe_store!(r3.elements, 0.0, ir_in + 1)
            ir = unsafe_load(perm, ir_in + 1)
            unsafe_store!(upd_el, val, Int(ir) + 1)
            unsafe_store!(upd_idx, ir, j)
            j += 1
        end
    else
        nnz_upd = r3.n_elements
    end

    # Permute regionSparse2 (packed) into regionFT, storing indices into indexRowU + start
    start_col_u = cf.startColumnU
    s_u = unsafe_load(start_col_u, Int(cf.maximumColumnsExtra) + 1)
    unsafe_store!(start_col_u, s_u, Int(cf.numberColumnsExtra) + 1)
    region_index_ft = cf.indexRowU + Int(s_u) * sizeof(Int32)

    nnz_ft = r2.n_elements
    n2 = Int(nnz_ft)
    j = 1
    while j <= n2
        ir_in = Int(unsafe_load(r2.indices, j))
        val = unsafe_load(r2.elements, j)
        unsafe_store!(r2.elements, 0.0, j)
        ir = unsafe_load(perm, ir_in + 1)
        unsafe_store!(ft_el, val, Int(ir) + 1)
        unsafe_store!(region_index_ft, ir, j)
        j += 1
    end

    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_numberFtranCounts, unsafe_load(cf.p_numberFtranCounts) + Int32(2))
        unsafe_store!(cf.p_ftranCountInput, unsafe_load(cf.p_ftranCountInput) + Float64(nnz_ft + nnz_upd))
    end

    # L
    nnz_ft = cf_update_column_l!(cf, ft_el, region_index_ft, nnz_ft)
    nnz_upd = cf_update_column_l!(cf, upd_el, upd_idx, nnz_upd)
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_ftranCountAfterL, unsafe_load(cf.p_ftranCountAfterL) + Float64(nnz_ft + nnz_upd))
    end

    # R
    nnz_ft = cf_update_column_r_ft!(cf, ft_el, region_index_ft, nnz_ft)
    nnz_upd = cf_update_column_r!(cf, upd_el, upd_idx, nnz_upd)
    if cf.collectStatistics != Int32(0)
        unsafe_store!(cf.p_ftranCountAfterR, unsafe_load(cf.p_ftranCountAfterR) + Float64(nnz_ft + nnz_upd))
    end

    # U
    go_sparse = Int32(0)
    if cf.sparseThreshold > Int32(0)
        avg_nnz = (nnz_upd + nnz_ft) >> 1
        if cf.ftranAverageAfterR != 0.0
            new_num = Float64(avg_nnz) * cf.ftranAverageAfterU
            if new_num < Float64(cf.sparseThreshold)
                go_sparse = Int32(2)
            elseif new_num < Float64(cf.sparseThreshold2)
                go_sparse = Int32(1)
            end
        else
            if avg_nnz < cf.sparseThreshold
                go_sparse = Int32(2)
            end
        end
    end

    if go_sparse == Int32(0) && cf.numberRows < Int32(1000)
        nnz_ft, nnz_upd = cf_update_two_columns_u_densish!(cf, ft_el, ft_idx, upd_el, upd_idx)
        if cf.collectStatistics != Int32(0)
            unsafe_store!(cf.p_ftranCountAfterU, unsafe_load(cf.p_ftranCountAfterU) + Float64(nnz_ft + nnz_upd))
        end
    else
        nnz_ft = cf_update_column_u!(cf, ft_el, ft_idx, region_index_ft, nnz_ft)
        nnz_upd = cf_update_column_u!(cf, upd_el, upd_idx, upd_idx, nnz_upd)
    end

    out_n2 = cf_permute_back!(cf, ft_el, ft_idx, nnz_ft, r2.elements, r2.indices, r2.packed_mode != Int32(0))
    if !no_permute_reg3
        out_n3 = cf_permute_back!(cf, upd_el, upd_idx, nnz_upd, r3.elements, r3.indices, r3.packed_mode != Int32(0))
        unsafe_store!(out_n3_ptr, out_n3)
    else
        unsafe_store!(out_n3_ptr, nnz_upd)
    end
    return out_n2
end

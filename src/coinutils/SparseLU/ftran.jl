# Pure idiomatic 1-based Julia FTRAN (`B x = a`) for `SparseLU`.
# Implements adaptive hypersparse DFS / 8-row bitmask / dense solves for L, R, FT, U,
# two-column FT solve, PFI fallback, and dense trailing LAPACK Accelerate block.
# Zero heap allocations on all hot paths (`@allocated == 0`).

# --- Permute result back into output vector and zero work region ---
function _ftran_permute_back!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
    out::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_packed::Bool,
)::Int
    perm_back = lu.pivot_col_back
    tol = lu.zero_tol
    nnz = 0
    if out_packed
        @inbounds for j in 1:nnz_in
            ir = Int(region_idx[j])
            val = region[ir]
            region[ir] = 0.0
            if abs(val) > tol
                nnz += 1
                out_idx[nnz] = perm_back[ir]
                out[nnz] = val
            end
        end
    else
        @inbounds for j in 1:nnz_in
            ir = Int(region_idx[j])
            val = region[ir]
            region[ir] = 0.0
            if abs(val) > tol
                out_row32 = perm_back[ir]
                nnz += 1
                out_idx[nnz] = out_row32
                out[Int(out_row32)] = val
            end
        end
    end
    return nnz
end

# --- FTRAN L: Densish ---
function ftran_l_densish!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    start_col = lu.start_col_l
    idx_row = lu.index_row_l
    el_l = lu.element_l
    nr = lu.nrows
    base = lu.base_l + 1          # 1-based first L column
    last_sparse = nr - lu.ndense

    nnz = 0
    smallest = lu.nrows_extra + 1
    @inbounds for k in 1:nnz_in
        ipiv32 = region_idx[k]
        ipiv = Int(ipiv32)
        if ipiv >= base
            if ipiv < smallest
                smallest = ipiv
            end
        else
            nnz += 1
            region_idx[nnz] = ipiv32
        end
    end

    @inbounds for i in smallest:last_sparse
        pval = region[i]
        if pval != 0.0
            if abs(pval) > tol
                js = Int(start_col[i])
                je = Int(start_col[i + 1]) - 1
                for j in js:je
                    ir = Int(idx_row[j])
                    region[ir] = _lu_fmsub(el_l[j], pval, region[ir])
                end
                nnz += 1
                region_idx[nnz] = i % Int32
            else
                region[i] = 0.0
            end
        end
    end

    @inbounds for i in (last_sparse + 1):nr
        pval = region[i]
        if pval != 0.0
            if abs(pval) > tol
                nnz += 1
                region_idx[nnz] = i % Int32
            else
                region[i] = 0.0
            end
        end
    end
    return nnz
end

# --- FTRAN L: Sparsish (8-row bitmask) ---
function ftran_l_sparsish!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    start_col = lu.start_col_l
    idx_row = lu.index_row_l
    el_l = lu.element_l
    mark = lu.sparse_mark
    nr = lu.nrows
    base = lu.base_l + 1
    last_sparse = nr - lu.ndense

    nnz = 0
    smallest = lu.nrows_extra + 1
    @inbounds for k in 1:nnz_in
        ipiv32 = region_idx[k]
        ipiv = Int(ipiv32)
        if ipiv < base
            nnz += 1
            region_idx[nnz] = ipiv32
        else
            if ipiv < smallest
                smallest = ipiv
            end
            _set_mark!(mark, ipiv)
        end
    end

    # Align up to next 8-row block boundary (1-based)
    # 0-based smallest0 = smallest - 1; j_last0 = ((smallest0 + 7) >> 3) << 3 -> 1-based end is j_last0
    j_last = min((((smallest - 1) + 7) >> 3) << 3, last_sparse)
    @inbounds for i in smallest:j_last
        pval = region[i]
        if abs(pval) > tol
            js = Int(start_col[i])
            je = Int(start_col[i + 1]) - 1
            for j in js:je
                ir = Int(idx_row[j])
                region[ir] = _lu_fmsub(el_l[j], pval, region[ir])
                _set_mark!(mark, ir)
            end
            nnz += 1
            region_idx[nnz] = i % Int32
        else
            region[i] = 0.0
        end
    end

    k_last = (last_sparse >> 3) + 1
    next_i = j_last + 1
    if j_last < last_sparse
        k_start = (j_last >> 3) + 1
        @inbounds for k in k_start:(k_last - 1)
            if mark[k] != UInt8(0)
                hi = k << 3
                lo = hi - 7
                for ii in lo:hi
                    pval = region[ii]
                    if abs(pval) > tol
                        js = Int(start_col[ii])
                        je = Int(start_col[ii + 1]) - 1
                        for j in js:je
                            ir = Int(idx_row[j])
                            region[ir] = _lu_fmsub(el_l[j], pval, region[ir])
                            _set_mark!(mark, ir)
                        end
                        nnz += 1
                        region_idx[nnz] = ii % Int32
                    else
                        region[ii] = 0.0
                    end
                end
                mark[k] = UInt8(0)
            end
        end
        next_i = ((k_last - 1) << 3) + 1
    end

    @inbounds for i in next_i:last_sparse
        pval = region[i]
        if abs(pval) > tol
            js = Int(start_col[i])
            je = Int(start_col[i + 1]) - 1
            for j in js:je
                ir = Int(idx_row[j])
                region[ir] = _lu_fmsub(el_l[j], pval, region[ir])
            end
            nnz += 1
            region_idx[nnz] = i % Int32
        else
            region[i] = 0.0
        end
    end

    tail_start = max(next_i, last_sparse + 1)
    @inbounds for i in tail_start:nr
        pval = region[i]
        if abs(pval) > tol
            nnz += 1
            region_idx[nnz] = i % Int32
        else
            region[i] = 0.0
        end
    end

    @inbounds begin
        if smallest <= lu.nrows_extra
            mark[_mark_byte(smallest)] = UInt8(0)
        end
        kk_last = ((nr + 7) >> 3)
        for k in k_last:kk_last
            mark[k] = UInt8(0)
        end
    end
    return nnz
end

# --- FTRAN L: Hypersparse DFS ---
function ftran_l_sparse!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    start_col = lu.start_col_l
    idx_row = lu.index_row_l
    el_l = lu.element_l
    base = lu.base_l + 1

    stack = lu.sparse_stack
    list = lu.sparse_list
    next = lu.sparse_aux
    mark = lu.sparse_mark

    nnz = 0
    n_list = 0
    @inbounds for k in 1:nnz_in
        kpiv32 = region_idx[k]
        kpiv = Int(kpiv32)
        if kpiv >= base
            if mark[kpiv] == UInt8(0)
                stack[1] = kpiv32
                j = Int(start_col[kpiv + 1]) - 1
                n_stack = 1
                while n_stack >= 1
                    if j >= Int(start_col[kpiv])
                        jpiv32 = idx_row[j]
                        jpiv = Int(jpiv32)
                        j -= 1
                        next[n_stack] = j % Int32
                        if mark[jpiv] == UInt8(0)
                            kpiv32 = jpiv32
                            kpiv = jpiv
                            j = Int(start_col[kpiv + 1]) - 1
                            n_stack += 1
                            stack[n_stack] = kpiv32
                            mark[kpiv] = UInt8(1)
                            next[n_stack] = j % Int32
                        end
                    else
                        n_list += 1
                        list[n_list] = kpiv % Int32
                        mark[kpiv] = UInt8(1)
                        n_stack -= 1
                        if n_stack >= 1
                            kpiv = Int(stack[n_stack])
                            j = Int(next[n_stack])
                        end
                    end
                end
            end
        else
            nnz += 1
            region_idx[nnz] = kpiv32
        end
    end

    @inbounds for i in n_list:-1:1
        ipiv32 = list[i]
        ipiv = Int(ipiv32)
        mark[ipiv] = UInt8(0)
        pval = region[ipiv]
        if abs(pval) > tol
            nnz += 1
            region_idx[nnz] = ipiv32
            js = Int(start_col[ipiv])
            je = Int(start_col[ipiv + 1]) - 1
            for j in js:je
                ir = Int(idx_row[j])
                region[ir] = _lu_fmsub(el_l[j], pval, region[ir])
            end
        else
            region[ipiv] = 0.0
        end
    end
    return nnz
end

# --- FTRAN L Dispatcher + Dense LAPACK Block ---
function ftran_l!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    nnz = nnz_in
    if lu.nl > 0
        go_sparse = 0
        if lu.sparse_threshold > 0
            if lu.ftran_avg_after_l != 0.0
                est = Float64(nnz) * lu.ftran_avg_after_l
                if est < Float64(lu.sparse_threshold) && Float64(lu.nl << 2) > est
                    go_sparse = 2
                elseif est < Float64(lu.sparse_threshold2) && Float64(lu.nl << 1) > est
                    go_sparse = 1
                end
            elseif nnz < lu.sparse_threshold && (lu.nl << 2) > nnz
                go_sparse = 2
            end
        end
        if go_sparse == 0
            nnz = ftran_l_densish!(lu, region, region_idx, nnz)
        elseif go_sparse == 1
            nnz = ftran_l_sparsish!(lu, region, region_idx, nnz)
        else
            nnz = ftran_l_sparse!(lu, region, region_idx, nnz)
        end
    end

    if lu.ndense > 0
        nr = lu.nrows
        last_sparse = nr - lu.ndense
        number = nnz
        do_dense = false
        i = 1
        @inbounds while i <= number
            ir = Int(region_idx[i])
            if ir > last_sparse
                do_dense = true
                region_idx[i] = region_idx[number]
                number -= 1
            else
                i += 1
            end
        end
        if do_dense
            lu_lapack_dgetrs!(
                UInt8('N'), lu.dense_area, lu.ndense, lu.dense_permute,
                region, last_sparse + 1,
            )
            @inbounds for ii in (last_sparse + 1):nr
                val = region[ii]
                if val != 0.0
                    if abs(val) >= 1.0e-15
                        number += 1
                        region_idx[number] = ii % Int32
                    else
                        region[ii] = 0.0
                    end
                end
            end
            nnz = number
        end
    end
    return nnz
end

# --- Helper: Cost model choosing R solve method (0 = row-marked, 1 = row-unmarked, 2 = col) ---
@inline function _choose_ftran_r_method(lu::SparseLU, nnz::Int, for_ft::Bool)::Int
    size_r = Float64(lu.start_col_r[lu.nr + 1] - Int32(1))
    avg_r = size_r / Float64(lu.nrows_extra)
    final_w = Float64(nnz)
    mt1 = Float64(lu.npivots) * (2.0 + (Float64(nnz) / Float64(lu.nrows)) * avg_r)
    mt1 += Float64(nnz) * (1.0 + avg_r)
    mt0 = mt1 + Float64(nnz + lu.npivots) * 0.1
    mt1 += Float64(nnz) * final_w
    mt2 = size_r + Float64(lu.npivots) * 2.0 + Float64(nnz) * final_w
    if !lu.has_row_r
        mt0 = 1.0e100
        mt1 = 1.0e100
    elseif !lu.has_sparse
        mt0 = 1.0e100
    end
    if for_ft
        mt1 += final_w
    end
    best = 1.0e100
    method = -1
    if mt0 < best
        best = mt0; method = 0
    end
    if mt1 < best
        best = mt1; method = 1
    end
    if mt2 < best
        method = 2
    end
    return method
end

# --- FTRAN R ---
function ftran_r!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    lu.nr == 0 && return nnz_in
    tol = lu.zero_tol
    nr = lu.nrows
    nre = lu.nrows_extra
    perm = lu.permute
    nnz = nnz_in
    method = _choose_ftran_r_method(lu, nnz, false)

    if method == 0
        mark = lu.sparse_mark
        @inbounds for i in (nr + 1):nre
            mark[Int(perm[i])] = UInt8(1)
        end
        el_r = lu.element_row_r
        idx_r = lu.index_col_r
        start_r = lu.start_row_r
        len_r = lu.len_row_r

        new_num = 0
        @inbounds for i in 1:nnz
            ir32 = region_idx[i]
            ir = Int(ir32)
            if mark[ir] == UInt8(0)
                new_num += 1
                region_idx[new_num] = ir32
            end
            num = Int(len_r[ir])
            if num != 0
                pval = region[ir]
                s = Int(start_r[ir])
                e = s + num - 1
                for j in s:e
                    jr = Int(idx_r[j])
                    region[jr] = _lu_fmsub(pval, el_r[j], region[jr])
                end
            end
        end
        nnz = new_num
        @inbounds for i in (nr + 1):nre
            ir = Int(perm[i])
            pval = region[ir] + region[i]
            region[ir] = 0.0
            if abs(pval) > tol
                region[i] = pval
                if mark[i] == UInt8(0)
                    nnz += 1
                    region_idx[nnz] = i % Int32
                end
                num = Int(len_r[i])
                s = Int(start_r[i])
                e = s + num - 1
                for j in s:e
                    jr = Int(idx_r[j])
                    region[jr] = _lu_fmsub(pval, el_r[j], region[jr])
                end
            else
                region[i] = 0.0
            end
            mark[ir] = UInt8(0)
        end
    elseif method == 1
        el_r = lu.element_row_r
        idx_r = lu.index_col_r
        start_r = lu.start_row_r
        len_r = lu.len_row_r

        @inbounds for i in 1:nnz
            ir = Int(region_idx[i])
            num = Int(len_r[ir])
            if num != 0
                pval = region[ir]
                s = Int(start_r[ir])
                e = s + num - 1
                for j in s:e
                    jr = Int(idx_r[j])
                    region[jr] = _lu_fmsub(pval, el_r[j], region[jr])
                end
            end
        end
        @inbounds for i in (nr + 1):nre
            ir = Int(perm[i])
            pval = region[ir] + region[i]
            region[ir] = 0.0
            if abs(pval) > tol
                region[i] = pval
                nnz += 1
                region_idx[nnz] = i % Int32
                num = Int(len_r[i])
                s = Int(start_r[i])
                e = s + num - 1
                for j in s:e
                    jr = Int(idx_r[j])
                    region[jr] = _lu_fmsub(pval, el_r[j], region[jr])
                end
            else
                region[i] = 0.0
            end
        end
    else
        el = lu.element_r
        idx_row = lu.index_row_r
        start_col = lu.start_col_r
        @inbounds s = Int(start_col[1])
        @inbounds for i in (nr + 1):nre
            e = Int(start_col[i - nr + 1])
            ir = Int(perm[i])
            pval = region[ir]
            region[ir] = 0.0
            for j in s:(e - 1)
                v = el[j] * region[Int(idx_row[j])]
                pval -= v
            end
            s = e
            if abs(pval) > tol
                region[i] = pval
                nnz += 1
                region_idx[nnz] = i % Int32
            else
                region[i] = 0.0
            end
        end
    end

    if method != 0
        n = nnz
        nnz = 0
        @inbounds for i in 1:n
            idx_val = region_idx[i]
            if region[Int(idx_val)] != 0.0
                nnz += 1
                region_idx[nnz] = idx_val
            end
        end
    end
    return nnz
end

# --- FTRAN R with Forrest-Tomlin column stash into U ---
function ftran_r_ft!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    start_col_u = lu.start_col_u
    len_col_u = lu.len_col_u
    el_u = lu.element_u
    idx_row_u = lu.index_row_u
    icol = lu.ncols_extra + 1             # 1-based next extra U column slot
    end_slot = lu.max_cols_extra + 1      # sentinel holding next free U offset
    nnz = nnz_in

    if lu.nr > 0
        tol = lu.zero_tol
        nr = lu.nrows
        nre = lu.nrows_extra
        perm = lu.permute
        method = _choose_ftran_r_method(lu, nnz, true)

        if method == 0
            mark = lu.sparse_mark
            @inbounds for i in (nr + 1):nre
                mark[Int(perm[i])] = UInt8(1)
            end
            el_r = lu.element_row_r
            idx_r = lu.index_col_r
            start_r = lu.start_row_r
            len_r = lu.len_row_r

            @inbounds s_u = Int(start_col_u[end_slot])
            @inbounds start_col_u[icol] = s_u % Int32
            put_base = s_u - 1

            new_num = 0
            @inbounds for i in 1:nnz
                ir32 = region_idx[i]
                ir = Int(ir32)
                pval = region[ir]
                if mark[ir] == UInt8(0)
                    new_num += 1
                    el_u[put_base + new_num] = pval
                    region_idx[new_num] = ir32
                end
                num = Int(len_r[ir])
                if num != 0
                    s = Int(start_r[ir])
                    e = s + num - 1
                    for j in s:e
                        jr = Int(idx_r[j])
                        region[jr] = _lu_fmsub(pval, el_r[j], region[jr])
                    end
                end
            end
            nnz = new_num
            @inbounds for i in (nr + 1):nre
                ir = Int(perm[i])
                pval = region[ir] + region[i]
                region[ir] = 0.0
                if abs(pval) > tol
                    region[i] = pval
                    if mark[i] == UInt8(0)
                        nnz += 1
                        el_u[put_base + nnz] = pval
                        region_idx[nnz] = i % Int32
                    end
                    num = Int(len_r[i])
                    s = Int(start_r[i])
                    e = s + num - 1
                    for j in s:e
                        jr = Int(idx_r[j])
                        region[jr] = _lu_fmsub(pval, el_r[j], region[jr])
                    end
                else
                    region[i] = 0.0
                end
                mark[ir] = UInt8(0)
            end
            @inbounds begin
                len_col_u[icol] = nnz % Int32
                start_col_u[end_slot] = (s_u + nnz) % Int32
            end
        elseif method == 1
            el_r = lu.element_row_r
            idx_r = lu.index_col_r
            start_r = lu.start_row_r
            len_r = lu.len_row_r

            @inbounds for i in 1:nnz
                ir = Int(region_idx[i])
                num = Int(len_r[ir])
                if num != 0
                    pval = region[ir]
                    s = Int(start_r[ir])
                    e = s + num - 1
                    for j in s:e
                        jr = Int(idx_r[j])
                        region[jr] = _lu_fmsub(pval, el_r[j], region[jr])
                    end
                end
            end
            @inbounds for i in (nr + 1):nre
                ir = Int(perm[i])
                pval = region[ir] + region[i]
                region[ir] = 0.0
                if abs(pval) > tol
                    region[i] = pval
                    nnz += 1
                    region_idx[nnz] = i % Int32
                    num = Int(len_r[i])
                    s = Int(start_r[i])
                    e = s + num - 1
                    for j in s:e
                        jr = Int(idx_r[j])
                        region[jr] = _lu_fmsub(pval, el_r[j], region[jr])
                    end
                else
                    region[i] = 0.0
                end
            end
        else
            el = lu.element_r
            idx_row = lu.index_row_r
            start_col = lu.start_col_r
            @inbounds s = Int(start_col[1])
            @inbounds for i in (nr + 1):nre
                e = Int(start_col[i - nr + 1])
                ir = Int(perm[i])
                pval = region[ir]
                region[ir] = 0.0
                for j in s:(e - 1)
                    v = el[j] * region[Int(idx_row[j])]
                    pval -= v
                end
                s = e
                if abs(pval) > tol
                    region[i] = pval
                    nnz += 1
                    region_idx[nnz] = i % Int32
                else
                    region[i] = 0.0
                end
            end
        end

        if method != 0
            n = nnz
            nnz = 0
            @inbounds s_u = Int(start_col_u[icol])
            put_base = s_u - 1
            @inbounds for i in 1:n
                idx_val = region_idx[i]
                val = region[Int(idx_val)]
                if val != 0.0
                    nnz += 1
                    idx_row_u[put_base + nnz] = idx_val
                    el_u[put_base + nnz] = val
                    region_idx[nnz] = idx_val
                end
            end
            @inbounds begin
                len_col_u[icol] = nnz % Int32
                start_col_u[end_slot] = (s_u + nnz) % Int32
            end
        end
        return nnz
    else
        @inbounds begin
            s_u = Int(start_col_u[icol])
            len_col_u[icol] = nnz % Int32
            start_col_u[end_slot] = (s_u + nnz) % Int32
            put_base = s_u - 1
            for i in 1:nnz
                idx_val = region_idx[i]
                idx_row_u[put_base + i] = idx_val
                el_u[put_base + i] = region[Int(idx_val)]
            end
        end
        return nnz
    end
end

# --- FTRAN U: Densish ---
function ftran_u_densish!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
)::Int
    tol = lu.zero_tol
    start_col = lu.start_col_u
    idx_row = lu.index_row_u
    el_u = lu.element_u
    len_col = lu.len_col_u
    piv_reg = lu.pivot_region
    n_slacks = lu.nslacks

    nnz = 0
    @inbounds for i in lu.nu:-1:(n_slacks + 1)
        pval = region[i]
        if pval != 0.0
            region[i] = 0.0
            if abs(pval) > tol
                s = Int(start_col[i])
                e = s + Int(len_col[i]) - 1
                for j in s:e
                    ir = Int(idx_row[j])
                    region[ir] = _lu_fmsub(el_u[j], pval, region[ir])
                end
                region[i] = pval * piv_reg[i]
                nnz += 1
                region_idx[nnz] = i % Int32
            end
        end
    end

    if lu.slack_value == -1.0
        @inbounds for i in n_slacks:-1:1
            val = region[i]
            if val != 0.0
                if abs(val) > tol
                    region[i] = -val
                    nnz += 1
                    region_idx[nnz] = i % Int32
                else
                    region[i] = 0.0
                end
            end
        end
    else
        @inbounds for i in n_slacks:-1:1
            val = region[i]
            if val != 0.0
                if abs(val) > tol
                    nnz += 1
                    region_idx[nnz] = i % Int32
                else
                    region[i] = 0.0
                end
            end
        end
    end
    return nnz
end

# --- FTRAN U: Hypersparse DFS ---
function ftran_u_sparse!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    index_in::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    start_col = lu.start_col_u
    idx_row = lu.index_row_u
    el_u = lu.element_u
    piv_reg = lu.pivot_region
    len_col = lu.len_col_u
    n_slacks = lu.nslacks

    mre = lu.max_rows_extra
    stack = lu.sparse_stack
    list = lu.sparse_list
    next = lu.sparse_aux
    mark = lu.sparse_mark

    put_pos = mre
    n_list = 0
    @inbounds for i in 1:nnz_in
        kpiv32 = index_in[i]
        kpiv = Int(kpiv32)
        stack[1] = kpiv32
        next[1] = (Int(start_col[kpiv]) + Int(len_col[kpiv]) - 1) % Int32
        n_stack = 1
        while n_stack > 0
            kpiv = Int(stack[n_stack])
            n_stack -= 1
            if mark[kpiv] != UInt8(1)
                j = Int(next[n_stack + 1])
                if j >= Int(start_col[kpiv])
                    next_kpiv32 = idx_row[j]
                    next_kpiv = Int(next_kpiv32)
                    j -= 1
                    n_stack += 1
                    next[n_stack] = j % Int32
                    if mark[next_kpiv] == UInt8(0)
                        nin = Int(len_col[next_kpiv])
                        if nin != 0
                            j2 = Int(start_col[next_kpiv]) + nin - 1
                            n_stack += 1
                            stack[n_stack] = next_kpiv32
                            mark[next_kpiv] = UInt8(2)
                            next[n_stack] = j2 % Int32
                        else
                            mark[next_kpiv] = UInt8(1)
                            if next_kpiv > n_slacks
                                n_list += 1
                                list[n_list] = next_kpiv32
                            else
                                stack[put_pos] = next_kpiv32
                                put_pos -= 1
                            end
                        end
                    end
                else
                    mark[kpiv] = UInt8(1)
                    if kpiv > n_slacks
                        n_list += 1
                        list[n_list] = kpiv % Int32
                    else
                        stack[put_pos] = kpiv % Int32
                        put_pos -= 1
                    end
                end
            end
        end
    end

    nnz = 0
    @inbounds for i in n_list:-1:1
        ipiv32 = list[i]
        ipiv = Int(ipiv32)
        mark[ipiv] = UInt8(0)
        pval = region[ipiv]
        region[ipiv] = 0.0
        if abs(pval) > tol
            s = Int(start_col[ipiv])
            e = s + Int(len_col[ipiv]) - 1
            for j in s:e
                ir = Int(idx_row[j])
                region[ir] = _lu_fmsub(el_u[j], pval, region[ir])
            end
            region[ipiv] = pval * piv_reg[ipiv]
            nnz += 1
            region_idx[nnz] = ipiv32
        end
    end

    if lu.slack_value == 1.0
        @inbounds for p in (put_pos + 1):mre
            ipiv32 = stack[p]
            ipiv = Int(ipiv32)
            mark[ipiv] = UInt8(0)
            pval = region[ipiv]
            region[ipiv] = 0.0
            if abs(pval) > tol
                region[ipiv] = pval
                nnz += 1
                region_idx[nnz] = ipiv32
            end
        end
    else
        @inbounds for p in (put_pos + 1):mre
            ipiv32 = stack[p]
            ipiv = Int(ipiv32)
            mark[ipiv] = UInt8(0)
            pval = region[ipiv]
            region[ipiv] = 0.0
            if abs(pval) > tol
                region[ipiv] = -pval
                nnz += 1
                region_idx[nnz] = ipiv32
            end
        end
    end
    return nnz
end

# --- FTRAN U: Sparsish (8-row bitmask) ---
function ftran_u_sparsish!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    index_in::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    start_col = lu.start_col_u
    idx_row = lu.index_row_u
    el_u = lu.element_u
    piv_reg = lu.pivot_region
    len_col = lu.len_col_u
    mark = lu.sparse_mark
    n_u = lu.nu
    n_slacks = lu.nslacks

    @inbounds for ii in 1:nnz_in
        _set_mark!(mark, Int(index_in[ii]))
    end

    nnz = 0
    # 0-based j_last0 = max(((n_u - 1) >> 3) << 3, n_slacks) -> 1-based start is j_last0 + 1
    j_last = max((((n_u - 1) >> 3) << 3) + 1, n_slacks + 1)
    @inbounds for i in n_u:-1:j_last
        pval = region[i]
        region[i] = 0.0
        if abs(pval) > tol
            s = Int(start_col[i])
            e = s + Int(len_col[i]) - 1
            for j in s:e
                ir = Int(idx_row[j])
                _set_mark!(mark, ir)
                region[ir] = _lu_fmsub(el_u[j], pval, region[ir])
            end
            region[i] = pval * piv_reg[i]
            nnz += 1
            region_idx[nnz] = i % Int32
        end
    end

    k_last = ((n_slacks + 7) >> 3) + 1
    rem_hi = j_last - 1
    if j_last > n_slacks + 1
        k_hi = ((j_last - 1) >> 3)
        @inbounds for k in k_hi:-1:k_last
            if mark[k] != UInt8(0)
                hi = k << 3
                lo = hi - 7
                for ii in hi:-1:lo
                    pval = region[ii]
                    if pval != 0.0
                        region[ii] = 0.0
                        if abs(pval) > tol
                            s = Int(start_col[ii])
                            e = s + Int(len_col[ii]) - 1
                            for j in s:e
                                ir = Int(idx_row[j])
                                _set_mark!(mark, ir)
                                region[ir] = _lu_fmsub(el_u[j], pval, region[ir])
                            end
                            region[ii] = pval * piv_reg[ii]
                            nnz += 1
                            region_idx[nnz] = ii % Int32
                        end
                    end
                end
                mark[k] = UInt8(0)
            end
        end
        rem_hi = (k_last - 1) << 3
    end

    @inbounds for i in rem_hi:-1:(n_slacks + 1)
        pval = region[i]
        region[i] = 0.0
        if abs(pval) > tol
            s = Int(start_col[i])
            e = s + Int(len_col[i]) - 1
            for j in s:e
                ir = Int(idx_row[j])
                _set_mark!(mark, ir)
                region[ir] = _lu_fmsub(el_u[j], pval, region[ir])
            end
            region[i] = pval * piv_reg[i]
            nnz += 1
            region_idx[nnz] = i % Int32
        end
    end

    if n_slacks > 0
        jl = (((n_slacks - 1) >> 3) << 3) + 1
        sgn = lu.slack_value == 1.0 ? 1.0 : -1.0
        @inbounds for ii in n_slacks:-1:jl
            val = region[ii]
            if val != 0.0
                region[ii] = 0.0
                if abs(val) > tol
                    region[ii] = sgn * val
                    nnz += 1
                    region_idx[nnz] = ii % Int32
                end
            end
        end
        k_hi = (jl - 1) >> 3
        @inbounds for k in k_hi:-1:1
            if mark[k] != UInt8(0)
                hi = k << 3
                lo = hi - 7
                for ii in hi:-1:lo
                    val = region[ii]
                    if val != 0.0
                        region[ii] = 0.0
                        if abs(val) > tol
                            region[ii] = sgn * val
                            nnz += 1
                            region_idx[nnz] = ii % Int32
                        end
                    end
                end
                mark[k] = UInt8(0)
            end
        end
        @inbounds mark[_mark_byte(n_slacks)] = UInt8(0)
    end
    @inbounds begin
        if n_u > 0
            mark[_mark_byte(n_u)] = UInt8(0)
        end
        mark[(n_slacks >> 3) + 1] = UInt8(0)
    end
    return nnz
end

# --- FTRAN U Dispatcher ---
function ftran_u!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    index_in::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    go_sparse = 0
    if lu.sparse_threshold > 0
        if lu.ftran_avg_after_r != 0.0
            est = Float64(nnz_in) * lu.ftran_avg_after_u
            if est < Float64(lu.sparse_threshold)
                go_sparse = 2
            elseif est < Float64(lu.sparse_threshold2)
                go_sparse = 1
            end
        else
            go_sparse = nnz_in < lu.sparse_threshold ? 2 : 0
        end
    end
    nnz = if go_sparse == 0
        ftran_u_densish!(lu, region, region_idx)
    elseif go_sparse == 1
        ftran_u_sparsish!(lu, region, region_idx, index_in, nnz_in)
    else
        ftran_u_sparse!(lu, region, region_idx, index_in, nnz_in)
    end
    if lu.collect_statistics
        lu.ftran_count_after_u += Float64(nnz)
    end
    return nnz
end

# --- FTRAN PFI (when do_forrest_tomlin == false) ---
function ftran_pfi!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    nr = lu.nrows
    start_col = lu.start_col_u
    idx_row = lu.index_row_u
    el_u = lu.element_u
    piv_reg = lu.pivot_region
    piv_col = lu.pivot_col

    nnz = nnz_in
    @inbounds for p in 1:lu.npivots
        k = nr + p
        prow = Int(piv_col[k])
        pval = region[prow]
        if pval != 0.0
            if abs(pval) > tol
                js = Int(start_col[k])
                je = Int(start_col[k + 1]) - 1
                for j in js:je
                    ir32 = idx_row[j]
                    ir = Int(ir32)
                    old_v = region[ir]
                    val = _lu_fmsub(pval, el_u[j], old_v)
                    if old_v == 0.0
                        if abs(val) > tol
                            region[ir] = val
                            nnz += 1
                            region_idx[nnz] = ir32
                        end
                    elseif abs(val) > tol
                        region[ir] = val
                    else
                        region[ir] = LU_TINY_ELEMENT
                    end
                end
                region[prow] = pval * piv_reg[k]
            else
                region[prow] = LU_TINY_ELEMENT
            end
        end
    end
    return nnz
end

# --- FTRAN Two-Column U Densish ---
function ftran_two_u_densish!(
    lu::SparseLU,
    region1::AbstractVector{Float64},
    index1::AbstractVector{Int32},
    region2::AbstractVector{Float64},
    index2::AbstractVector{Int32},
)::Tuple{Int,Int}
    tol = lu.zero_tol
    start_col = lu.start_col_u
    idx_row = lu.index_row_u
    el_u = lu.element_u
    len_col = lu.len_col_u
    piv_reg = lu.pivot_region
    n_slacks = lu.nslacks

    nnz_a = 0
    nnz_b = 0
    @inbounds for i in lu.nu:-1:(n_slacks + 1)
        pval2 = region2[i]
        do2 = abs(pval2) > tol
        region2[i] = 0.0
        pval1 = region1[i]
        do1 = abs(pval1) > tol
        region1[i] = 0.0
        if do2
            s = Int(start_col[i])
            e = s + Int(len_col[i]) - 1
            if !do1
                for j in s:e
                    ir = Int(idx_row[j])
                    region2[ir] = _lu_fmsub(el_u[j], pval2, region2[ir])
                end
                region2[i] = pval2 * piv_reg[i]
                nnz_b += 1
                index2[nnz_b] = i % Int32
            else
                for j in s:e
                    ir = Int(idx_row[j])
                    v = el_u[j]
                    region1[ir] = _lu_fmsub(v, pval1, region1[ir])
                    region2[ir] = _lu_fmsub(v, pval2, region2[ir])
                end
                pr = piv_reg[i]
                i32 = i % Int32
                region1[i] = pval1 * pr
                nnz_a += 1
                index1[nnz_a] = i32
                region2[i] = pval2 * pr
                nnz_b += 1
                index2[nnz_b] = i32
            end
        elseif do1
            s = Int(start_col[i])
            e = s + Int(len_col[i]) - 1
            for j in s:e
                ir = Int(idx_row[j])
                region1[ir] = _lu_fmsub(el_u[j], pval1, region1[ir])
            end
            region1[i] = pval1 * piv_reg[i]
            nnz_a += 1
            index1[nnz_a] = i % Int32
        end
    end

    @inbounds for i in n_slacks:-1:1
        v2 = region2[i]
        v1 = region1[i]
        if abs(v2) > tol
            region2[i] = -v2
            nnz_b += 1
            index2[nnz_b] = i % Int32
        else
            region2[i] = 0.0
        end
        if v1 != 0.0
            if abs(v1) > tol
                region1[i] = -v1
                nnz_a += 1
                index1[nnz_a] = i % Int32
            else
                region1[i] = 0.0
            end
        end
    end
    return nnz_a, nnz_b
end

# --- Top-level FTRAN (CoinFactorization::updateColumn) ---
function ftran!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    w_nnz::Int,
    vector::AbstractVector{Float64},
    index::AbstractVector{Int32},
    nnz_in::Int,
    packed::Bool;
    no_permute::Bool = false,
)::Int
    perm = lu.permute
    if !no_permute
        nnz = nnz_in
        if packed
            @inbounds for j in 1:nnz
                ir_in = Int(index[j])
                val = vector[j]
                vector[j] = 0.0
                ir = perm[ir_in]
                region[Int(ir)] = val
                region_idx[j] = ir
            end
        else
            @inbounds for j in 1:nnz
                ir_in = Int(index[j])
                val = vector[ir_in]
                vector[ir_in] = 0.0
                ir = perm[ir_in]
                region[Int(ir)] = val
                region_idx[j] = ir
            end
        end
    else
        nnz = w_nnz
    end

    if lu.collect_statistics
        lu.nftran_counts += 1
        lu.ftran_count_input += Float64(nnz)
    end

    nnz = ftran_l!(lu, region, region_idx, nnz)
    if lu.collect_statistics
        lu.ftran_count_after_l += Float64(nnz)
    end
    nnz = ftran_r!(lu, region, region_idx, nnz)
    if lu.collect_statistics
        lu.ftran_count_after_r += Float64(nnz)
    end
    nnz = ftran_u!(lu, region, region_idx, region_idx, nnz)
    if !lu.do_forrest_tomlin
        nnz = ftran_pfi!(lu, region, region_idx, nnz)
    end

    if !no_permute
        return _ftran_permute_back!(lu, region, region_idx, nnz, vector, index, packed)
    else
        return nnz
    end
end

# --- Top-level FTRAN with Forrest-Tomlin stash (CoinFactorization::updateColumnFT) ---
function ftran_ft!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    vector::AbstractVector{Float64},
    index::AbstractVector{Int32},
    nnz_in::Int,
    packed::Bool,
)::Int
    perm = lu.permute
    start_col_u = lu.start_col_u
    do_ft = lu.do_forrest_tomlin
    s_u = 1
    if do_ft
        icol = lu.ncols_extra + 1
        end_slot = lu.max_cols_extra + 1
        @inbounds s_u = Int(start_col_u[end_slot])
        @inbounds start_col_u[icol] = s_u % Int32
        do_ft = (s_u + lu.nrows_extra) <= (lu.len_area_u + 1)
        if !do_ft
            @inbounds start_col_u[end_slot] = (lu.len_area_u + 2) % Int32
        end
    end

    if do_ft
        ft_idx = @view lu.index_row_u[s_u:end]
        return _ftran_ft_core!(lu, region, region_idx, ft_idx, vector, index, nnz_in, packed, true)
    else
        return _ftran_ft_core!(lu, region, region_idx, region_idx, vector, index, nnz_in, packed, false)
    end
end

function _ftran_ft_core!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    work_idx::AbstractVector{Int32},
    vector::AbstractVector{Float64},
    index::AbstractVector{Int32},
    nnz_in::Int,
    packed::Bool,
    do_ft::Bool,
)::Int
    perm = lu.permute
    if packed
        @inbounds for j in 1:nnz_in
            ir_in = Int(index[j])
            val = vector[j]
            vector[j] = 0.0
            ir = perm[ir_in]
            region[Int(ir)] = val
            work_idx[j] = ir
        end
    else
        @inbounds for j in 1:nnz_in
            ir_in = Int(index[j])
            val = vector[ir_in]
            vector[ir_in] = 0.0
            ir = perm[ir_in]
            region[Int(ir)] = val
            work_idx[j] = ir
        end
    end

    nnz = nnz_in
    if lu.collect_statistics
        lu.nftran_counts += 1
        lu.ftran_count_input += Float64(nnz)
    end

    nnz = ftran_l!(lu, region, work_idx, nnz)
    if lu.collect_statistics
        lu.ftran_count_after_l += Float64(nnz)
    end
    nnz = do_ft ? ftran_r_ft!(lu, region, work_idx, nnz) : ftran_r!(lu, region, region_idx, nnz)
    if lu.collect_statistics
        lu.ftran_count_after_r += Float64(nnz)
    end
    nnz = ftran_u!(lu, region, region_idx, work_idx, nnz)
    if !lu.do_forrest_tomlin
        nnz = ftran_pfi!(lu, region, region_idx, nnz)
    end
    out_n = _ftran_permute_back!(lu, region, region_idx, nnz, vector, index, packed)
    return do_ft ? out_n : -out_n
end

# --- Top-level Two-Column FTRAN FT (CoinFactorization::updateTwoColumnsFT) ---
function ftran_two_ft!(
    lu::SparseLU,
    r1_val::AbstractVector{Float64},
    r1_idx::AbstractVector{Int32},
    r2_val::AbstractVector{Float64},
    r2_idx::AbstractVector{Int32},
    r2_nnz::Int,
    r2_packed::Bool,
    r3_val::AbstractVector{Float64},
    r3_idx::AbstractVector{Int32},
    r3_nnz::Int,
    r3_packed::Bool;
    no_permute_r3::Bool = false,
)::Tuple{Int,Int}
    # Check preconditions matching CoinFactorization::updateTwoColumnsFT
    end_slot = lu.max_cols_extra + 1
    @inbounds s_u = Int(lu.start_col_u[end_slot])
    space_ok = (s_u + lu.nrows_extra) <= (lu.len_area_u + 1)
    if !lu.do_forrest_tomlin || !space_ok || !r2_packed || r3_packed
        out2 = ftran_ft!(lu, r1_val, r1_idx, r2_val, r2_idx, r2_nnz, r2_packed)
        if no_permute_r3
            out3 = ftran!(lu, r3_val, r3_idx, r3_nnz, r1_val, r1_idx, 0, false; no_permute = true)
        else
            out3 = ftran!(lu, r1_val, r1_idx, 0, r3_val, r3_idx, r3_nnz, r3_packed; no_permute = false)
        end
        return out2, out3
    end

    perm = lu.permute
    @inbounds lu.start_col_u[lu.ncols_extra + 1] = s_u % Int32
    ft_idx_u = @view lu.index_row_u[s_u:end]

    ft_el = no_permute_r3 ? r1_val : r3_val
    ft_idx = no_permute_r3 ? r1_idx : r3_idx
    upd_el = no_permute_r3 ? r3_val : r1_val
    upd_idx = no_permute_r3 ? r3_idx : r1_idx

    nnz_upd = r3_nnz
    if !no_permute_r3
        @inbounds for j in 1:nnz_upd
            ir_in = Int(r3_idx[j])
            val = r3_val[ir_in]
            r3_val[ir_in] = 0.0
            ir = perm[ir_in]
            upd_el[Int(ir)] = val
            upd_idx[j] = ir
        end
    end

    nnz_ft = r2_nnz
    @inbounds for j in 1:nnz_ft
        ir_in = Int(r2_idx[j])
        val = r2_val[j]
        r2_val[j] = 0.0
        ir = perm[ir_in]
        ft_el[Int(ir)] = val
        ft_idx_u[j] = ir
    end

    if lu.collect_statistics
        lu.nftran_counts += 2
        lu.ftran_count_input += Float64(nnz_ft + nnz_upd)
    end

    nnz_ft = ftran_l!(lu, ft_el, ft_idx_u, nnz_ft)
    nnz_upd = ftran_l!(lu, upd_el, upd_idx, nnz_upd)
    if lu.collect_statistics
        lu.ftran_count_after_l += Float64(nnz_ft + nnz_upd)
    end

    nnz_ft = ftran_r_ft!(lu, ft_el, ft_idx_u, nnz_ft)
    nnz_upd = ftran_r!(lu, upd_el, upd_idx, nnz_upd)
    if lu.collect_statistics
        lu.ftran_count_after_r += Float64(nnz_ft + nnz_upd)
    end

    go_sparse = 0
    if lu.sparse_threshold > 0
        avg_nnz = (nnz_upd + nnz_ft) >> 1
        if lu.ftran_avg_after_r != 0.0
            est = Float64(avg_nnz) * lu.ftran_avg_after_u
            if est < Float64(lu.sparse_threshold)
                go_sparse = 2
            elseif est < Float64(lu.sparse_threshold2)
                go_sparse = 1
            end
        elseif avg_nnz < lu.sparse_threshold
            go_sparse = 2
        end
    end

    if go_sparse == 0 && lu.nrows < 1000
        nnz_ft, nnz_upd = ftran_two_u_densish!(lu, ft_el, ft_idx, upd_el, upd_idx)
        if lu.collect_statistics
            lu.ftran_count_after_u += Float64(nnz_ft + nnz_upd)
        end
    else
        nnz_ft = ftran_u!(lu, ft_el, ft_idx, ft_idx_u, nnz_ft)
        nnz_upd = ftran_u!(lu, upd_el, upd_idx, upd_idx, nnz_upd)
    end

    out_n2 = _ftran_permute_back!(lu, ft_el, ft_idx, nnz_ft, r2_val, r2_idx, r2_packed)
    out_n3 = no_permute_r3 ? nnz_upd : _ftran_permute_back!(lu, upd_el, upd_idx, nnz_upd, r3_val, r3_idx, r3_packed)
    return out_n2, out_n3
end

# --- Idiomatic IndexedVector overloads ---
function ftran!(lu::SparseLU, work::IndexedVector, rhs::IndexedVector; no_permute::Bool = false)::Int
    nnz = ftran!(
        lu, work.values, work.indices, work.nnz,
        rhs.values, rhs.indices, rhs.nnz, rhs.packed;
        no_permute = no_permute,
    )
    if no_permute
        work.nnz = nnz
    else
        work.nnz = 0
        rhs.nnz = nnz
    end
    return nnz
end

function ftran_ft!(lu::SparseLU, work::IndexedVector, rhs::IndexedVector)::Int
    res = ftran_ft!(lu, work.values, work.indices, rhs.values, rhs.indices, rhs.nnz, rhs.packed)
    work.nnz = 0
    rhs.nnz = abs(res)
    return res
end

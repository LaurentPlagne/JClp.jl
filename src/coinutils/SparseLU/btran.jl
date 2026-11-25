# Pure idiomatic 1-based Julia BTRAN (`B^T y = c`) for `SparseLU`.
# Implements adaptive hypersparse DFS / 8-row bitmask / dense solves for U^T, R^T, L^T,
# PFI fallback, and dense trailing LAPACK Accelerate block.
# Zero heap allocations on all hot paths (`@allocated == 0`).

@inline function _scan_range_tol!(
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz::Int,
    lo::Int,
    hi::Int,
    tol::Float64,
)::Int
    count = 0
    @inbounds for i in lo:hi
        v = region[i]
        if v != 0.0
            if abs(v) > tol
                count += 1
                indices[nnz + count] = i % Int32
            else
                region[i] = 0.0
            end
        end
    end
    return count
end

@inline function _clear_vec!(
    values::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz::Int,
    packed::Bool,
)::Nothing
    if packed
        @inbounds for i in 1:nnz
            values[i] = 0.0
        end
    else
        @inbounds for i in 1:nnz
            values[Int(indices[i])] = 0.0
        end
    end
    return nothing
end

# --- BTRAN PFI (when do_forrest_tomlin == false) ---
function btran_pfi!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    nr = lu.nrows
    piv_col = lu.pivot_col
    piv_reg = lu.pivot_region
    start_col = lu.start_col_u
    idx_row = lu.index_row_u
    el_u = lu.element_u
    tol = lu.zero_tol

    nnz = nnz_in
    @inbounds for p in lu.npivots:-1:1
        k = nr + p
        prow32 = piv_col[k]
        prow = Int(prow32)
        pval = region[prow] * piv_reg[k]
        js = Int(start_col[k])
        je = Int(start_col[k + 1]) - 1
        for j in js:je
            pval = _lu_fmsub(el_u[j], region[Int(idx_row[j])], pval)
        end
        if abs(pval) > tol
            if region[prow] == 0.0
                nnz += 1
                indices[nnz] = prow32
            end
            region[prow] = pval
        elseif region[prow] != 0.0
            region[prow] = LU_TINY_ELEMENT
        end
    end
    return nnz
end

# --- BTRAN U: Densish ---
function btran_u_densish!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    smallest::Int,
)::Int
    tol = lu.zero_tol
    start_row = lu.start_row_u
    len_row = lu.len_row_u
    idx_col = lu.index_col_u
    row_to_col = lu.row_to_col_u
    el_u = lu.element_u
    last_u = lu.nu

    nnz = 0
    @inbounds for i in smallest:last_u
        pval = region[i]
        if pval != 0.0
            if abs(pval) > tol
                s = Int(start_row[i])
                nin = Int(len_row[i])
                e = s + nin - 1
                for j in s:e
                    ir = Int(idx_col[j])
                    ge = Int(row_to_col[j])
                    region[ir] = _lu_fmsub(el_u[ge], pval, region[ir])
                end
                nnz += 1
                indices[nnz] = i % Int32
            else
                region[i] = 0.0
            end
        end
    end
    return nnz
end

# --- BTRAN U: Sparsish (8-row bitmask) ---
function btran_u_sparsish!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
    smallest::Int,
)::Int
    tol = lu.zero_tol
    start_row = lu.start_row_u
    len_row = lu.len_row_u
    idx_col = lu.index_col_u
    row_to_col = lu.row_to_col_u
    el_u = lu.element_u
    mark = lu.sparse_mark
    last_u = lu.nu

    @inbounds for i in 1:nnz_in
        _set_mark!(mark, Int(indices[i]))
    end

    nnz = 0
    k_start = _mark_byte(smallest)
    k_last = (last_u >> 3) + 1

    @inbounds for k in k_start:(k_last - 1)
        if mark[k] != UInt8(0)
            hi = k << 3
            lo = hi - 7
            for ii in lo:hi
                pval = region[ii]
                if abs(pval) > tol
                    s = Int(start_row[ii])
                    e = s + Int(len_row[ii]) - 1
                    for j in s:e
                        ir = Int(idx_col[j])
                        ge = Int(row_to_col[j])
                        _set_mark!(mark, ir)
                        region[ir] = _lu_fmsub(el_u[ge], pval, region[ir])
                    end
                    nnz += 1
                    indices[nnz] = ii % Int32
                else
                    region[ii] = 0.0
                end
            end
            mark[k] = UInt8(0)
        end
    end

    @inbounds begin
        mark[k_last] = UInt8(0)
        tail_lo = ((k_last - 1) << 3) + 1
        for ii in tail_lo:last_u
            pval = region[ii]
            if abs(pval) > tol
                s = Int(start_row[ii])
                e = s + Int(len_row[ii]) - 1
                for j in s:e
                    ir = Int(idx_col[j])
                    ge = Int(row_to_col[j])
                    region[ir] = _lu_fmsub(el_u[ge], pval, region[ir])
                end
                nnz += 1
                indices[nnz] = ii % Int32
            else
                region[ii] = 0.0
            end
        end
    end
    return nnz
end

# --- BTRAN U: Hypersparse DFS ---
function btran_u_sparse!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    start_row = lu.start_row_u
    len_row = lu.len_row_u
    idx_col = lu.index_col_u
    row_to_col = lu.row_to_col_u
    el_u = lu.element_u

    stack = lu.sparse_stack
    list = lu.sparse_list
    next = lu.sparse_aux
    mark = lu.sparse_mark

    n_list = 0
    @inbounds for k in 1:nnz_in
        kpiv32 = indices[k]
        kpiv = Int(kpiv32)
        stack[1] = kpiv32
        next[1] = (Int(start_row[kpiv]) + Int(len_row[kpiv]) - 1) % Int32
        n_stack = 1
        while n_stack > 0
            kpiv = Int(stack[n_stack])
            n_stack -= 1
            if mark[kpiv] != UInt8(1)
                j = Int(next[n_stack + 1])
                if j >= Int(start_row[kpiv])
                    next_kpiv32 = idx_col[j]
                    next_kpiv = Int(next_kpiv32)
                    j -= 1
                    n_stack += 1
                    next[n_stack] = j % Int32
                    if mark[next_kpiv] == UInt8(0)
                        j2 = Int(start_row[next_kpiv]) + Int(len_row[next_kpiv]) - 1
                        n_stack += 1
                        stack[n_stack] = next_kpiv32
                        mark[next_kpiv] = UInt8(2)
                        next[n_stack] = j2 % Int32
                    end
                else
                    n_list += 1
                    list[n_list] = kpiv % Int32
                    mark[kpiv] = UInt8(1)
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
        if abs(pval) > tol
            s = Int(start_row[ipiv])
            e = s + Int(len_row[ipiv]) - 1
            for j in s:e
                ir = Int(idx_col[j])
                ge = Int(row_to_col[j])
                region[ir] = _lu_fmsub(el_u[ge], pval, region[ir])
            end
            nnz += 1
            indices[nnz] = ipiv32
        else
            region[ipiv] = 0.0
        end
    end
    return nnz
end

# --- BTRAN U Dispatcher ---
function btran_u!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
    smallest::Int,
)::Int
    go_sparse = 0
    if lu.sparse_threshold > 0
        if lu.btran_avg_after_u != 0.0
            est = Float64(nnz_in) * lu.btran_avg_after_u
            if est < Float64(lu.sparse_threshold)
                go_sparse = 2
            elseif est < Float64(lu.sparse_threshold2)
                go_sparse = 1
            end
        else
            go_sparse = nnz_in < lu.sparse_threshold ? 2 : 0
        end
    end
    if go_sparse == 0
        return btran_u_densish!(lu, region, indices, smallest)
    elseif go_sparse == 1
        return btran_u_sparsish!(lu, region, indices, nnz_in, smallest)
    else
        return btran_u_sparse!(lu, region, indices, nnz_in)
    end
end

# --- BTRAN R: Densish ---
function btran_r_densish!(lu::SparseLU, region::AbstractVector{Float64})::Nothing
    nr = lu.nrows
    idx_row = lu.index_row_r
    el_r = lu.element_r
    start_col = lu.start_col_r
    perm = lu.permute

    @inbounds for i in lu.nrows_extra:-1:(nr + 1)
        put_row = Int(perm[i])
        pval = region[i]
        region[i] = 0.0
        if pval != 0.0
            k = i - nr
            js = Int(start_col[k])
            je = Int(start_col[k + 1]) - 1
            for j in js:je
                ir = Int(idx_row[j])
                region[ir] = _lu_fmsub(el_r[j], pval, region[ir])
            end
            region[put_row] = pval
        end
    end
    return nothing
end

# --- BTRAN R: Sparse ---
function btran_r_sparse!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    nr = lu.nrows
    idx_row = lu.index_row_r
    el_r = lu.element_r
    start_col = lu.start_col_r
    perm = lu.permute
    pos_map = lu.sparse_stack

    nnz = nnz_in
    @inbounds for i in 1:nnz
        pos_map[Int(indices[i])] = i % Int32
    end

    @inbounds for i in lu.nrows_extra:-1:(nr + 1)
        put_row32 = perm[i]
        put_row = Int(put_row32)
        pval = region[i]
        region[i] = 0.0
        if pval != 0.0
            k = i - nr
            js = Int(start_col[k])
            je = Int(start_col[k + 1]) - 1
            for j in js:je
                ir32 = idx_row[j]
                ir = Int(ir32)
                old_v = region[ir]
                new_v = _lu_fmsub(el_r[j], pval, old_v)
                if old_v != 0.0
                    region[ir] = new_v != 0.0 ? new_v : LU_TINY_ELEMENT
                elseif abs(new_v) > tol
                    region[ir] = new_v
                    nnz += 1
                    pos_map[ir] = nnz % Int32
                    indices[nnz] = ir32
                end
            end
            region[put_row] = pval
            pos = Int(pos_map[i])
            indices[pos] = put_row32
            pos_map[put_row] = pos % Int32
        end
    end
    return nnz
end

# --- BTRAN R Dispatcher ---
function btran_r!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    if lu.nrows_extra == lu.nrows
        return nnz_in
    end
    if nnz_in == 0
        return 0
    end
    if nnz_in < (lu.sparse_threshold << 2) || (lu.nl == 0 && lu.has_sparse)
        nnz = btran_r_sparse!(lu, region, indices, nnz_in)
        if lu.collect_statistics
            lu.btran_count_after_r += Float64(nnz)
        end
        return nnz
    else
        btran_r_densish!(lu, region)
        if lu.collect_statistics
            lu.btran_count_after_r += Float64(min(nnz_in << 1, lu.nrows))
        end
        return lu.nrows + 1
    end
end

# --- BTRAN L: Densish by column (before go_sparse!) ---
function btran_l_densish!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
)::Int
    tol = lu.zero_tol
    nr = lu.nrows
    nnz = 0

    first = nr
    @inbounds while first >= 1 && region[first] == 0.0
        first -= 1
    end
    if first >= 1
        base = lu.base_l + 1          # 1-based first L column
        last = lu.base_l + lu.nl      # 1-based last L column
        start_col = lu.start_col_l
        idx_row = lu.index_row_l
        el_l = lu.element_l

        if first > last
            first = last
        end
        @inbounds for i in first:-1:base
            pval = region[i]
            js = Int(start_col[i])
            je = Int(start_col[i + 1]) - 1
            for j in js:je
                pval = _lu_fmsub(el_l[j], region[Int(idx_row[j])], pval)
            end
            if abs(pval) > tol
                region[i] = pval
                nnz += 1
                indices[nnz] = i % Int32
            else
                region[i] = 0.0
            end
        end
        top = first < base ? first : (base - 1)
        @inbounds for i in top:-1:1
            pval = region[i]
            if abs(pval) > tol
                nnz += 1
                indices[nnz] = i % Int32
            else
                region[i] = 0.0
            end
        end
    end
    return nnz
end

# --- BTRAN L: By Row (dense scan using row copy of L) ---
function btran_l_by_row!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
)::Int
    tol = lu.zero_tol
    el_l = lu.element_row_l
    start_row = lu.start_row_l
    idx_col = lu.index_col_l

    first = lu.nrows
    @inbounds while first >= 1 && region[first] == 0.0
        first -= 1
    end
    nnz = 0
    @inbounds for i in first:-1:1
        pval = region[i]
        if pval != 0.0
            if abs(pval) > tol
                nnz += 1
                indices[nnz] = i % Int32
                js = Int(start_row[i])
                je = Int(start_row[i + 1]) - 1
                for j in je:-1:js
                    ir = Int(idx_col[j])
                    region[ir] = _lu_fmsub(pval, el_l[j], region[ir])
                end
            else
                region[i] = 0.0
            end
        end
    end
    return nnz
end

# --- BTRAN L: Sparsish by Row (8-row bitmask) ---
function btran_l_sparsish!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    nr = lu.nrows
    el_l = lu.element_row_l
    start_row = lu.start_row_l
    idx_col = lu.index_col_l
    mark = lu.sparse_mark

    @inbounds for i in 1:nnz_in
        _set_mark!(mark, Int(indices[i]))
    end

    nnz = 0
    j_last = (((nr - 1) >> 3) << 3) + 1
    @inbounds for ii in nr:-1:j_last
        pval = region[ii]
        if abs(pval) > tol
            nnz += 1
            indices[nnz] = ii % Int32
            js = Int(start_row[ii])
            je = Int(start_row[ii + 1]) - 1
            for j in je:-1:js
                ir = Int(idx_col[j])
                _set_mark!(mark, ir)
                region[ir] = _lu_fmsub(pval, el_l[j], region[ir])
            end
        else
            region[ii] = 0.0
        end
    end

    j_word = _mark_byte(nr)
    @inbounds begin
        mark[j_word] = UInt8(0)
        for k in (j_word - 1):-1:1
            if mark[k] != UInt8(0)
                hi = k << 3
                lo = hi - 7
                for ii in hi:-1:lo
                    pval = region[ii]
                    if abs(pval) > tol
                        nnz += 1
                        indices[nnz] = ii % Int32
                        js = Int(start_row[ii])
                        je = Int(start_row[ii + 1]) - 1
                        for j in je:-1:js
                            ir = Int(idx_col[j])
                            _set_mark!(mark, ir)
                            region[ir] = _lu_fmsub(pval, el_l[j], region[ir])
                        end
                    else
                        region[ii] = 0.0
                    end
                end
                mark[k] = UInt8(0)
            end
        end
    end
    return nnz
end

# --- BTRAN L: Hypersparse DFS ---
function btran_l_sparse!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    tol = lu.zero_tol
    el_l = lu.element_row_l
    start_row = lu.start_row_l
    idx_col = lu.index_col_l

    stack = lu.sparse_stack
    list = lu.sparse_list
    next = lu.sparse_aux
    mark = lu.sparse_mark

    n_list = 0
    @inbounds for k in 1:nnz_in
        kpiv32 = indices[k]
        kpiv = Int(kpiv32)
        if mark[kpiv] == UInt8(0) && region[kpiv] != 0.0
            stack[1] = kpiv32
            j = Int(start_row[kpiv + 1]) - 1
            n_stack = 1
            while n_stack >= 1
                if j >= Int(start_row[kpiv])
                    jpiv32 = idx_col[j]
                    jpiv = Int(jpiv32)
                    j -= 1
                    next[n_stack] = j % Int32
                    if mark[jpiv] == UInt8(0)
                        kpiv32 = jpiv32
                        kpiv = jpiv
                        j = Int(start_row[kpiv + 1]) - 1
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
    end

    nnz = 0
    @inbounds for i in n_list:-1:1
        ipiv32 = list[i]
        ipiv = Int(ipiv32)
        mark[ipiv] = UInt8(0)
        pval = region[ipiv]
        if abs(pval) > tol
            nnz += 1
            indices[nnz] = ipiv32
            js = Int(start_row[ipiv])
            je = Int(start_row[ipiv + 1]) - 1
            for j in js:je
                ir = Int(idx_col[j])
                region[ir] = _lu_fmsub(el_l[j], pval, region[ir])
            end
        else
            region[ipiv] = 0.0
        end
    end
    return nnz
end

# --- BTRAN L Dispatcher + Dense Trailing LAPACK Block ---
function btran_l!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    indices::AbstractVector{Int32},
    nnz_in::Int,
)::Int
    nnz = nnz_in
    nr = lu.nrows
    if lu.nl == 0 && lu.ndense == 0
        if lu.has_sparse || nnz < nr
            return nnz
        end
    end

    go_sparse = -1
    if lu.sparse_threshold > 0
        if lu.btran_avg_after_l != 0.0
            est = Float64(nnz) * lu.btran_avg_after_l
            if est < Float64(lu.sparse_threshold)
                go_sparse = 2
            elseif est < Float64(lu.sparse_threshold2)
                go_sparse = 1
            else
                go_sparse = 0
            end
        else
            go_sparse = nnz < lu.sparse_threshold ? 2 : 0
        end
    end

    if lu.ndense > 0
        last_sparse = nr - lu.ndense   # 1-based dense block starts at last_sparse + 1
        number = nnz
        do_dense = false
        if number <= nr
            i = 1
            @inbounds while i <= number
                ir = Int(indices[i])
                if ir > last_sparse
                    do_dense = true
                    indices[i] = indices[number]
                    number -= 1
                else
                    i += 1
                end
            end
        else
            @inbounds for i in nr:-1:(last_sparse + 1)
                if region[i] != 0.0
                    do_dense = true
                    number = _scan_range_tol!(region, indices, 0, 1, last_sparse, lu.zero_tol)
                    break
                end
            end
            go_sparse = lu.sparse_threshold != 0 ? 0 : -1
        end
        if do_dense
            nnz = number
            lu_lapack_dgetrs!(
                UInt8('T'), lu.dense_area, lu.ndense, lu.dense_permute,
                region, last_sparse + 1,
            )
            if go_sparse > 0 || lu.nl == 0
                nnz += _scan_range_tol!(region, indices, nnz, last_sparse + 1, nr, lu.zero_tol)
            end
        end
        if lu.nl == 0
            if nnz > nr
                nnz = _scan_range_tol!(region, indices, 0, 1, nr, lu.zero_tol)
            end
            return nnz
        end
    end

    if go_sparse > 0 && nnz > nr
        go_sparse = 0
    end

    if go_sparse == -1
        return btran_l_densish!(lu, region, indices)
    elseif go_sparse == 0
        return btran_l_by_row!(lu, region, indices)
    elseif go_sparse == 1
        return btran_l_sparsish!(lu, region, indices, nnz)
    else
        return btran_l_sparse!(lu, region, indices, nnz)
    end
end

"""
    btran!(lu::SparseLU, w_val, w_idx, w_nnz, w_packed, r_val, r_idx, r_nnz, r_packed) -> Int

Solve `B^T y = c` in place (equivalent to `CoinFactorization::updateColumnTranspose`).
`w_*` is a scratch vector cleared on entry and exit; `r_*` holds the input RHS and receives
the output solution (preserving `r_packed`). Returns the output non-zero count.
"""
function btran!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    w_nnz::Int,
    w_packed::Bool,
    vector::AbstractVector{Float64},
    index::AbstractVector{Int32},
    nnz_in::Int,
    packed::Bool,
)::Int
    _clear_vec!(region, region_idx, w_nnz, w_packed)

    piv_col = lu.pivot_col
    if packed
        @inbounds for i in 1:nnz_in
            ir_in = Int(index[i])
            val = vector[i]
            vector[i] = 0.0
            ir32 = piv_col[ir_in]
            region[Int(ir32)] = val
            region_idx[i] = ir32
        end
    else
        @inbounds for i in 1:nnz_in
            ir_in = Int(index[i])
            val = vector[ir_in]
            vector[ir_in] = 0.0
            ir32 = piv_col[ir_in]
            region[Int(ir32)] = val
            region_idx[i] = ir32
        end
    end

    nnz = nnz_in
    if lu.collect_statistics
        lu.nbtran_counts += 1
        lu.btran_count_input += Float64(nnz)
    end

    if !lu.do_forrest_tomlin
        nnz = btran_pfi!(lu, region, region_idx, nnz)
    end

    # Scale by inverse pivot diagonals and locate smallest active index
    piv_reg = lu.pivot_region
    smallest = lu.nrows_extra + 1
    @inbounds for j in 1:nnz
        ir = Int(region_idx[j])
        if ir < smallest
            smallest = ir
        end
        region[ir] *= piv_reg[ir]
    end

    # U^T solve
    nnz = btran_u!(lu, region, region_idx, nnz, smallest)
    if lu.collect_statistics
        lu.btran_count_after_u += Float64(nnz)
    end

    # R^T solve
    nnz = btran_r!(lu, region, region_idx, nnz)

    # L^T solve (+ dense LAPACK block)
    nnz = btran_l!(lu, region, region_idx, nnz)
    if lu.collect_statistics
        lu.btran_count_after_l += Float64(nnz)
    end

    # Permute back into output vector and leave scratch region zeroed
    perm_back = lu.pivot_col_back
    if packed
        @inbounds for i in 1:nnz
            ir = Int(region_idx[i])
            val = region[ir]
            region[ir] = 0.0
            vector[i] = val
            index[i] = perm_back[ir]
        end
    else
        @inbounds for i in 1:nnz
            ir = Int(region_idx[i])
            val = region[ir]
            region[ir] = 0.0
            out_row32 = perm_back[ir]
            vector[Int(out_row32)] = val
            index[i] = out_row32
        end
    end

    return nnz
end

"""
    btran!(lu::SparseLU, work::IndexedVector, rhs::IndexedVector) -> Int

Idiomatic Julia wrapper solving `B^T y = rhs` in place using `work` as scratch.
"""
function btran!(lu::SparseLU, work::IndexedVector, rhs::IndexedVector)::Int
    nnz = btran!(
        lu,
        work.values, work.indices, work.nnz, work.packed,
        rhs.values, rhs.indices, rhs.nnz, rhs.packed,
    )
    work.nnz = 0
    work.packed = false
    rhs.nnz = nnz
    return nnz
end

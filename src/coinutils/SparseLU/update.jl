# Pure idiomatic 1-based Julia Forrest-Tomlin & PFI rank-1 basis updates for `SparseLU`.
# Equivalent to CoinFactorization4.cpp (`replaceColumn`, `replaceColumnPFI`,
# `getRowSpaceIterate`, `getColumnSpaceIterateR`, `checkPivot`).
# Zero heap allocations on all hot paths (`@allocated == 0`).

@inline function _check_pivot(
    save_from_u::Float64,
    old_pivot::Float64,
    nrows::Int,
    nrows_extra::Int,
    relax_check::Float64,
)::Int
    if abs(save_from_u) > 1.0e-9
        check_tol = if nrows_extra < nrows + 2
            1.0e-5
        elseif nrows_extra < nrows + 10
            1.0e-6
        elseif nrows_extra < nrows + 50
            1.0e-8
        else
            1.0e-10
        end
        check_tol *= relax_check
        rel_err = abs(1.0 - abs(save_from_u / old_pivot))
        if rel_err < check_tol
            return 0
        elseif abs(abs(old_pivot) - abs(save_from_u)) < 1.0e-12 || rel_err < 1.0e-8
            return 1
        else
            return 2
        end
    else
        return abs(1.0 - abs(save_from_u / old_pivot)) < 1.0e-10 ? 0 : 2
    end
end

# Move row `i_row` to end of row-wise U storage, compressing if necessary.
function _get_row_space_iterate!(lu::SparseLU, i_row::Int, extra_needed::Int)::Bool
    len_row = lu.len_row_u
    start_row = lu.start_row_u
    idx_col = lu.index_col_u
    row_to_col = lu.row_to_col_u
    next_row = lu.next_row
    last_row = lu.last_row

    row_end = lu.max_rows_extra + 1
    @inbounds number = Int(len_row[i_row])
    @inbounds space = (lu.len_area_u + 1) - Int(start_row[row_end])

    if space < extra_needed + number + 2
        # Compress row-wise U storage along linked physical order
        @inbounds r = Int(next_row[row_end])
        put = 1
        @inbounds while r != row_end
            g = Int(start_row[r])
            ge = g + Int(len_row[r]) - 1
            start_row[r] = put % Int32
            for i in g:ge
                idx_col[put] = idx_col[i]
                row_to_col[put] = row_to_col[i]
                put += 1
            end
            r = Int(next_row[r])
        end
        lu.ncompressions += 1
        @inbounds start_row[row_end] = put % Int32
        space = (lu.len_area_u + 1) - put
        if space < extra_needed + number + 2
            lu.status = -99
            return false
        end
    end

    @inbounds begin
        put = Int(start_row[row_end])
        # Unlink i_row and re-insert at tail (before row_end)
        nxt = next_row[i_row]
        lst = last_row[i_row]
        next_row[Int(lst)] = nxt
        last_row[Int(nxt)] = lst

        lst2 = last_row[row_end]
        next_row[Int(lst2)] = i_row % Int32
        last_row[row_end] = i_row % Int32
        last_row[i_row] = lst2
        next_row[i_row] = row_end % Int32

        g = Int(start_row[i_row])
        start_row[i_row] = put % Int32
        for k in 0:(number - 1)
            idx_col[put + k] = idx_col[g + k]
            row_to_col[put + k] = row_to_col[g + k]
        end
        put += number
        start_row[row_end] = (put + extra_needed + 4) % Int32
    end
    return true
end

# Move row `i_col` of R-transpose to end of R-row storage, compressing if necessary.
function _get_col_space_iterate_r!(lu::SparseLU, i_col::Int, value::Float64, i_row::Int)::Bool
    el_r = lu.element_row_r
    idx_r = lu.index_col_r
    start_r = lu.start_row_r
    len_r = lu.len_row_r
    next_col = lu.next_col
    last_col = lu.last_col

    col_end = lu.max_cols_extra + 1
    lar_end = lu.len_area_r + 1
    @inbounds number = Int(len_r[i_col])

    @inbounds if lar_end - Int(start_r[col_end]) < number + 1
        jc = Int(next_col[col_end])
        put = 1
        while jc != col_end
            g = Int(start_r[jc])
            ge = g + Int(len_r[jc]) - 1
            start_r[jc] = Int32(put)
            for i in g:ge
                idx_r[put] = idx_r[i]
                el_r[put] = el_r[i]
                put += 1
            end
            jc = Int(next_col[jc])
        end
        lu.ncompressions += 1
        start_r[col_end] = Int32(put)
    end

    @inbounds if lar_end - Int(start_r[col_end]) < number + 1
        return false
    end

    @inbounds begin
        nxt = next_col[i_col]
        lst = last_col[i_col]
        next_col[Int(lst)] = nxt
        last_col[Int(nxt)] = lst

        put = Int(start_r[col_end])
        lst2 = last_col[col_end]
        next_col[Int(lst2)] = Int32(i_col)
        last_col[col_end] = Int32(i_col)
        last_col[i_col] = lst2
        next_col[i_col] = Int32(col_end)

        g = Int(start_r[i_col])
        start_r[i_col] = Int32(put)
        for k in 0:(number - 1)
            el_r[put + k] = el_r[g + k]
            idx_r[put + k] = idx_r[g + k]
        end
        put += number
        el_r[put] = value
        idx_r[put] = Int32(i_row)
        put += 1
        len_r[i_col] = Int32(number + 1)
        end_pos = put + 4
        start_r[col_end] = Int32(min(end_pos, lar_end))
    end
    return true
end

"""
    replace_column!(lu::SparseLU, region, region_idx, pivot_row, pivot_check, check_before) -> Int

Forrest-Tomlin rank-1 basis update (`CoinFactorization::replaceColumn`).
Returns `0` (OK), `1` (probably OK), `2` (singular/unstable), `3` (no room), or `5` (max pivots).
"""
function replace_column!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    pivot_row::Int,
    pivot_check_in::Float64,
    check_before_modifying::Bool = false,
)::Int
    col_end = lu.max_cols_extra + 1
    row_end = lu.max_rows_extra + 1

    if lu.ncols_extra >= lu.max_cols_extra
        return 5
    end
    @inbounds if lu.len_area_u + 1 < Int(lu.start_col_u[col_end])
        return 3
    end

    start_col_u = lu.start_col_u
    len_col_u = lu.len_col_u
    start_row_u = lu.start_row_u
    len_row_u = lu.len_row_u
    idx_col_u = lu.index_col_u
    row_to_col = lu.row_to_col_u
    el_u = lu.element_u
    idx_row_u = lu.index_row_u
    piv_reg = lu.pivot_region

    @inbounds real_piv_row = Int(lu.pivot_col[pivot_row])
    check_before = check_before_modifying || (lu.npivots == 0)

    @inbounds begin
        lu.total_elements -= Int(len_col_u[real_piv_row])
        old_pivot = piv_reg[real_piv_row]
        pivot_check = pivot_check_in / old_pivot
        piv_reg[real_piv_row] = 0.0

        col_start_old = Int(start_col_u[real_piv_row])
        save_end = col_start_old + Int(len_col_u[real_piv_row])
        len_col_u[real_piv_row] = Int32(0)
    end

    @inbounds s_row = Int(start_row_u[real_piv_row])
    @inbounds e_row = s_row + Int(len_row_u[real_piv_row]) - 1
    smallest = lu.nrows_extra + 1
    nnz = 0

    if !check_before
        @inbounds for i in s_row:e_row
            icol = Int(idx_col_u[i])
            if icol < smallest
                smallest = icol
            end
            j = Int(row_to_col[i])
            region[icol] = el_u[j]
            el_u[j] = 0.0
            nnz += 1
            region_idx[nnz] = Int32(icol)
        end
    else
        @inbounds for i in s_row:e_row
            icol = Int(idx_col_u[i])
            if icol < smallest
                smallest = icol
            end
            j = Int(row_to_col[i])
            region[icol] = el_u[j]
            nnz += 1
            region_idx[nnz] = Int32(icol)
        end
    end

    # Compute row spike via BTRAN on U
    nnz = btran_u!(lu, region, region_idx, nnz, smallest)

    new_col = lu.ncols_extra + 1
    new_row = lu.nrows_extra + 1
    @inbounds start_u = Int(start_col_u[new_col])
    @inbounds num_c = Int(len_col_u[new_col])
    u_base = start_u - 1

    if check_before
        tol = lu.zero_tol
        save_from_u = 0.0
        @inbounds for i in 1:num_c
            ir = Int(idx_row_u[u_base + i])
            ev = el_u[u_base + i]
            if abs(ev) > tol
                if ir != real_piv_row
                    save_from_u = _lu_fmsub(ev, region[ir], save_from_u)
                else
                    save_from_u += ev
                end
            end
        end
        chk = _check_pivot(save_from_u, pivot_check, lu.nrows, lu.nrows_extra, lu.relax_check)
        if chk != 0
            @inbounds begin
                piv_reg[real_piv_row] = old_pivot
                restored = save_end - col_start_old
                lu.total_elements += restored
                len_col_u[real_piv_row] = Int32(restored)
            end
            _clear_vec!(region, region_idx, nnz, false)
            return chk
        else
            @inbounds for i in s_row:e_row
                el_u[Int(row_to_col[i])] = 0.0
            end
        end
    end

    # Zero out old pivot column in U
    @inbounds for i in col_start_old:(save_end - 1)
        el_u[i] = 0.0
    end

    # Append new eta vector to R
    l_pos = lu.len_r + 1
    @inbounds lu.start_col_r[lu.nr + 1] = Int32(l_pos)
    lu.nr += 1
    lu.len_r += nnz
    @inbounds lu.start_col_r[lu.nr + 1] = Int32(lu.len_r + 1)
    lu.total_elements += nnz
    if lu.len_r >= lu.len_area_r
        _clear_vec!(region, region_idx, nnz, false)
        return 3
    end

    idx_row_r = lu.index_row_r
    el_r = lu.element_r
    @inbounds for i in 1:nnz
        ir = region_idx[i]
        idx_row_r[l_pos + i - 1] = ir
        el_r[l_pos + i - 1] = region[Int(ir)]
    end

    # Unlink old pivot row from row list
    next_row = lu.next_row
    last_row = lu.last_row
    @inbounds begin
        nxt_r = next_row[real_piv_row]
        lst_r = last_row[real_piv_row]
        next_row[Int(lst_r)] = nxt_r
        last_row[Int(nxt_r)] = lst_r
        len_row_u[real_piv_row] = Int32(0)

        # Update permutation arrays
        lu.permute[new_row] = Int32(real_piv_row)
        lu.permute_back[real_piv_row] = Int32(new_row)
        lu.permute_back[new_row] = LU_NO_LINK
        lu.permute[new_row + 1] = LU_NO_LINK

        lu.pivot_col[pivot_row] = Int32(new_row)
        lu.pivot_col_back[new_row] = Int32(pivot_row)
    end

    lu.nu += 1
    lu.total_elements += num_c
    lu.len_u += num_c
    if lu.len_u >= lu.len_area_u
        _clear_vec!(region, region_idx, nnz, false)
        return 3
    end

    # Insert new column entries into row-wise U and compute updated pivot
    save_from_u = 0.0
    @inbounds for i in 1:num_c
        ir = Int(idx_row_u[u_base + i])
        ev = el_u[u_base + i]
        if ir != real_piv_row
            nxt2 = Int(next_row[ir])
            inir = Int(len_row_u[ir])
            put = Int(start_row_u[ir]) + inir
            if Int(start_row_u[nxt2]) - put <= 0
                if !_get_row_space_iterate!(lu, ir, inir + 4)
                    _clear_vec!(region, region_idx, nnz, false)
                    return 3
                end
                put = Int(start_row_u[ir]) + inir
            end
            idx_col_u[put] = new_col % Int32
            row_to_col[put] = (u_base + i) % Int32
            len_row_u[ir] = (inir + 1) % Int32
            save_from_u = _lu_fmsub(ev, region[ir], save_from_u)
        else
            save_from_u += ev
            el_u[u_base + i] = 0.0
        end
    end

    # Insert new_row at tail of row list and new_col at head of column list
    @inbounds begin
        lst_end = last_row[row_end]
        next_row[Int(lst_end)] = Int32(new_row)
        last_row[row_end] = Int32(new_row)
        last_row[new_row] = lst_end
        next_row[new_row] = Int32(row_end)
        start_row_u[new_row] = start_row_u[row_end]
        len_row_u[new_row] = Int32(0)

        next_col = lu.next_col
        last_col = lu.last_col
        nxt_c = next_col[col_end]
        last_col[Int(nxt_c)] = Int32(new_col)
        next_col[col_end] = Int32(new_col)
        next_col[new_col] = nxt_c
        last_col[new_col] = Int32(col_end)
    end

    ret_status = check_before ? 0 : _check_pivot(save_from_u, pivot_check, lu.nrows, lu.nrows_extra, lu.relax_check)

    if ret_status != 2
        piv_val = 1.0 / save_from_u
        @inbounds begin
            piv_reg[new_row] = piv_val
            for i in 1:num_c
                el_u[u_base + i] *= piv_val
            end
        end
        cand_max_u = (start_u - 1) + num_c
        if cand_max_u > lu.max_u
            lu.max_u = cand_max_u
        end
        lu.nrows_extra += 1
        lu.ncols_extra += 1
        lu.ngood_u += 1
        lu.npivots += 1
    end

    # Fill-in refactorization heuristic (CoinFactorization4.cpp:670-692)
    nr = lu.nrows
    if lu.nrows_extra > nr + 50
        extra = lu.factor_elements >> 1
        if lu.nrows_extra > nr + 100 + div(nr, 500)
            extra = max(extra, 2 * nr)
        else
            extra = max(extra, 5 * nr)
        end
        added = lu.total_elements - lu.factor_elements
        if added > extra && added > (lu.factor_elements << 1) && ret_status == 0 &&
           3 * lu.total_elements > 2 * (lu.len_area_u + lu.len_area_l)
            ret_status = 3
        end
    end

    # Maintain row-wise transpose copy of R (for fast FTRAN R)
    if lu.has_row_r && ret_status < 2
        el_r2 = lu.element_row_r
        idx_r2 = lu.index_col_r
        start_r2 = lu.start_row_r
        len_r2 = lu.len_row_r
        next_col = lu.next_col
        lar_end = lu.len_area_r + 1
        piv_r_new = lu.nrows_extra

        @inbounds for i in 1:nnz
            ir = Int(region_idx[i])
            nxt_col = Int(next_col[ir])
            sp = nxt_col != col_end ? (Int(start_r2[nxt_col]) - Int(start_r2[ir])) : (lar_end - Int(start_r2[ir]))
            nin_r = Int(len_r2[ir])
            if sp > nin_r
                put = Int(start_r2[ir]) + nin_r
                len_r2[ir] = Int32(nin_r + 1)
                idx_r2[put] = Int32(piv_r_new)
                el_r2[put] = region[ir]
                if nxt_col == col_end
                    ep = put + 4
                    start_r2[col_end] = Int32(min(ep, lar_end))
                end
            else
                if !_get_col_space_iterate_r!(lu, ir, region[ir], piv_r_new)
                    lu.has_row_r = false
                    _clear_vec!(region, region_idx, nnz, false)
                    break
                end
            end
            region[ir] = 0.0
        end
    else
        _clear_vec!(region, region_idx, nnz, false)
    end

    return ret_status
end

"""
    replace_column_pfi!(lu::SparseLU, region, region_idx, nnz, packed, pivot_row, alpha) -> Int

Product-Form-of-Inverse rank-1 update (`CoinFactorization::replaceColumnPFI`).
"""
function replace_column_pfi!(
    lu::SparseLU,
    region::AbstractVector{Float64},
    region_idx::AbstractVector{Int32},
    nnz_in::Int,
    packed::Bool,
    pivot_row::Int,
    alpha::Float64,
)::Int
    nr = lu.nrows
    col_end = lu.max_cols_extra + 1
    p_idx = lu.npivots + 1
    if lu.npivots == 0
        @inbounds lu.start_col_u[nr + 1] = lu.start_col_u[col_end]
    end
    @inbounds start = Int(lu.start_col_u[nr + p_idx])
    if lu.npivots >= lu.max_pivots
        return 5
    end
    if (lu.len_area_u + 1) - (start + nnz_in) < 0
        return 3
    end
    if lu.npivots > 0
        if abs(alpha) < 1.0e-5
            return abs(alpha) < 1.0e-7 ? 2 : 1
        end
    elseif abs(alpha) < 1.0e-8
        return 2
    end

    piv_val = 1.0 / alpha
    tol = lu.zero_tol
    piv_col = lu.pivot_col
    idx_row = lu.index_row_u
    el_u = lu.element_u
    @inbounds lu.pivot_region[nr + p_idx] = piv_val

    put = start
    if packed
        @inbounds for i in 1:nnz_in
            ir = Int(region_idx[i])
            if ir != pivot_row && abs(region[i]) > tol
                idx_row[put] = piv_col[ir]
                el_u[put] = region[i] * piv_val
                put += 1
            end
        end
    else
        @inbounds for i in 1:nnz_in
            ir = Int(region_idx[i])
            if ir != pivot_row && abs(region[ir]) > tol
                idx_row[put] = piv_col[ir]
                el_u[put] = region[ir] * piv_val
                put += 1
            end
        end
    end

    lu.npivots += 1
    added = put - start
    @inbounds begin
        lu.start_col_u[nr + lu.npivots + 1] = Int32(put)
        lu.total_elements += added
        lu.pivot_col[nr + p_idx] = piv_col[pivot_row]
    end
    return 0
end

# --- Idiomatic IndexedVector overload ---
function replace_column!(
    lu::SparseLU,
    work::IndexedVector,
    pivot_row::Int,
    pivot_check::Float64;
    check_before::Bool = false,
)::Int
    res = replace_column!(lu, work.values, work.indices, pivot_row, pivot_check, check_before)
    work.nnz = 0
    return res
end

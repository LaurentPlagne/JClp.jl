# Pure idiomatic 1-based Julia Markowitz Sparse LU factorization for `SparseLU`.
# Equivalent to CoinFactorization1.cpp + CoinFactorization2.cpp + CoinFactorization4.cpp::goSparse.
# Zero Ptr, zero unsafe_load / unsafe_store!, 100% 1-based Vector{T}, --trim=safe compatible.

const _LU_COMPRESS_SENTINEL = Int32(typemax(Int32) - 10)
const _LU_SMALL_LIMIT       = 65530
const _LU_SMALL_SET         = Int32(65534)
const _LU_SMALL_UNSET       = Int32(65535)
const _LU_LARGE_SET         = Int32(typemax(Int32) - 1)
const _LU_LARGE_UNSET       = typemax(Int32)

# =============================================================================
# Markowitz doubly-linked count list helpers (1-based IDs: rows 1:nr, cols nr+1:nr+nc)
# =============================================================================

@inline function _add_link!(
    first_count::Vector{Int32},
    next_count::Vector{Int32},
    last_count::Vector{Int32},
    id::Int,
    count::Int,
)::Nothing
    @inbounds begin
        nxt = first_count[count + 1]
        last_count[id] = Int32(-2 - count)
        first_count[count + 1] = Int32(id)
        next_count[id] = nxt
        if nxt > LU_NO_LINK
            last_count[Int(nxt)] = Int32(id)
        end
    end
    return nothing
end

@inline function _delete_link!(
    first_count::Vector{Int32},
    next_count::Vector{Int32},
    last_count::Vector{Int32},
    id::Int,
)::Nothing
    @inbounds begin
        nxt = next_count[id]
        lst = last_count[id]
        if lst > LU_NO_LINK
            next_count[Int(lst)] = nxt
        else
            cnt = -Int(lst) - 2
            first_count[cnt + 1] = nxt
        end
        if nxt > LU_NO_LINK
            last_count[Int(nxt)] = lst
        end
        next_count[id] = LU_REMOVED
        last_count[id] = LU_REMOVED
    end
    return nothing
end

function _separate_links!(
    first_count::Vector{Int32},
    next_count::Vector{Int32},
    last_count::Vector{Int32},
    nr::Int,
    count::Int,
    rows_first::Bool,
)::Nothing
    @inbounds begin
        nxt = first_count[count + 1]
        first_row = LU_NO_LINK
        first_col = LU_NO_LINK
        last_row = LU_NO_LINK
        last_col = LU_NO_LINK
        while nxt > LU_NO_LINK
            n_i = Int(nxt)
            nxt2 = next_count[n_i]
            next_count[n_i] = LU_NO_LINK
            if n_i > nr
                if first_col > LU_NO_LINK
                    last_count[n_i] = last_col
                    next_count[Int(last_col)] = nxt
                else
                    last_count[n_i] = Int32(-2 - count)
                    first_col = nxt
                end
                last_col = nxt
            else
                if first_row > LU_NO_LINK
                    last_count[n_i] = last_row
                    next_count[Int(last_row)] = nxt
                else
                    last_count[n_i] = Int32(-2 - count)
                    first_row = nxt
                end
                last_row = nxt
            end
            nxt = nxt2
        end
        if rows_first && first_row > LU_NO_LINK
            first_count[count + 1] = first_row
            next_count[Int(last_row)] = first_col
            if first_col > LU_NO_LINK
                last_count[Int(first_col)] = last_row
            end
        elseif first_row == LU_NO_LINK
            first_count[count + 1] = first_col
        elseif first_col > LU_NO_LINK
            first_count[count + 1] = first_col
            next_count[Int(last_col)] = first_row
            last_count[Int(first_row)] = last_col
        end
    end
    return nothing
end

# =============================================================================
# Dynamic U column / row space management & compression
# =============================================================================

function _get_col_space!(lu::SparseLU, i_col::Int, extra_needed::Int)::Bool
    nic = lu.len_col_u
    nicp = lu.len_col_plus
    next_col = lu.next_col
    last_col = lu.last_col
    start_col = lu.start_col_u
    el_u = lu.element_u
    idx_row = lu.index_row_u

    col_end = lu.max_cols_extra + 1
    lau_end = lu.len_area_u + 1

    @inbounds number = Int(nicp[i_col]) + Int(nic[i_col])
    @inbounds space = lau_end - Int(start_col[col_end])

    if space < extra_needed + number + 4
        # Compress column-wise U storage along linked physical order
        @inbounds col = Int(next_col[col_end])
        put = 1
        @inbounds while col != col_end
            sc = Int(start_col[col])
            nc_val = Int(nic[col])
            if sc > 0
                ncp_val = Int(nicp[col])
                g = sc - ncp_val
                ge = sc + nc_val - 1
                start_col[col] = Int32(put + ncp_val)
            else
                g = -sc
                ge = g + nc_val - 1
                start_col[col] = Int32(-put)
            end
            for i in g:ge
                idx_row[put] = idx_row[i]
                el_u[put] = el_u[i]
                put += 1
            end
            col = Int(next_col[col])
        end
        lu.ncompressions += 1
        @inbounds start_col[col_end] = Int32(put)
        space = lau_end - put
        if extra_needed == Int(_LU_COMPRESS_SENTINEL)
            return true
        end
        if space < extra_needed + number + 2
            lu.status = -99
            return false
        end
    end

    @inbounds begin
        put = Int(start_col[col_end])
        nxt = Int(next_col[i_col])
        lst = Int(last_col[i_col])

        if extra_needed != 0 || nxt != col_end
            # Unlink i_col and re-insert at tail
            next_col[lst] = Int32(nxt)
            last_col[nxt] = Int32(lst)
            lst2 = Int(last_col[col_end])
            next_col[lst2] = Int32(i_col)
            last_col[col_end] = Int32(i_col)
            last_col[i_col] = Int32(lst2)
            next_col[i_col] = Int32(col_end)

            ncp_ic = Int(nicp[i_col])
            g = Int(start_col[i_col]) - ncp_ic
            start_col[i_col] = Int32(put + ncp_ic)
            for k in 0:(number - 1)
                idx_row[put + k] = idx_row[g + k]
                el_u[put + k] = el_u[g + k]
            end
            put += number
            new_end = put + extra_needed + 2
            start_col[col_end] = Int32(new_end)
            if new_end > lau_end
                return false
            end
        else
            start_col[col_end] = start_col[lst] + nic[lst]
        end
    end
    return true
end

function _get_row_space!(lu::SparseLU, i_row::Int, extra_needed::Int)::Bool
    nir = lu.len_row_u
    start_row = lu.start_row_u
    next_row = lu.next_row
    last_row = lu.last_row
    idx_col = lu.index_col_u

    row_end = lu.max_rows_extra + 1
    lau_end = lu.len_area_u + 1

    @inbounds number = Int(nir[i_row])
    @inbounds space = lau_end - Int(start_row[row_end])

    if space < extra_needed + number + 2
        @inbounds r = Int(next_row[row_end])
        put = 1
        @inbounds while r != row_end
            g = Int(start_row[r])
            ge = g + Int(nir[r]) - 1
            start_row[r] = Int32(put)
            for i in g:ge
                idx_col[put] = idx_col[i]
                put += 1
            end
            r = Int(next_row[r])
        end
        lu.ncompressions += 1
        @inbounds start_row[row_end] = Int32(put)
        space = lau_end - put
        if space < extra_needed + number + 2
            lu.status = -99
            return false
        end
    end

    @inbounds begin
        put = Int(start_row[row_end])
        nxt = Int(next_row[i_row])
        lst = Int(last_row[i_row])
        next_row[lst] = Int32(nxt)
        last_row[nxt] = Int32(lst)
        lst2 = Int(last_row[row_end])
        next_row[lst2] = Int32(i_row)
        last_row[row_end] = Int32(i_row)
        last_row[i_row] = Int32(lst2)
        next_row[i_row] = Int32(row_end)

        g = Int(start_row[i_row])
        start_row[i_row] = Int32(put)
        for k in 0:(number - 1)
            idx_col[put + k] = idx_col[g + k]
        end
        start_row[row_end] = Int32(put + number + extra_needed + 4)
    end
    return true
end

# =============================================================================
# preProcess (CoinFactorization1.cpp:731-1065)
# =============================================================================

"""
    preprocess!(lu::SparseLU, state::Int)

Prepare column/row copies, largest-element-first column ordering, and Markowitz
count linked lists before `factor!`. `state` matches `CoinFactorization::preProcess`:
- `0`: raw triplets in `(index_row_u, index_col_u, element_u)[1:len_u]`
- `1`/`-1`: counts known, sort triplets into columns
- `2`: columns already packed in `start_col_u`/`len_col_u`; build row copy + pivot largest
- `3`: skip row copy (used when `bias_lu < 3 && nrows == ncols`; slacks stripped first in `factor!`)
- `4`: rebuild column largest & row copy after slack stripping
"""
function preprocess!(lu::SparseLU, state::Int)::Nothing
    idx_row = lu.index_row_u
    idx_col = lu.index_col_u
    el = lu.element_u
    num_el = lu.len_u
    nir = lu.len_row_u
    nic = lu.len_col_u
    nicp = lu.len_col_plus
    start_row = lu.start_row_u
    start_col = lu.start_col_u
    nr = lu.nrows
    nc = lu.ncols
    row_end = lu.max_rows_extra + 1
    col_end = lu.max_cols_extra + 1

    if state < 4
        lu.total_elements = num_el
    end

    if state == 0
        @inbounds for i in 1:(nr + 1)
            nir[i] = Int32(0)
        end
        @inbounds for i in 1:col_end
            nic[i] = Int32(0)
        end
        @inbounds for i in 1:num_el
            r = Int(idx_row[i])
            c = Int(idx_col[i])
            nir[r] += Int32(1)
            nic[c] += Int32(1)
        end
    end

    if state == 0 || state == -1 || state == 1
        pos = 1
        @inbounds for ic in 1:nc
            pos += Int(nic[ic])
            start_col[ic] = Int32(pos)
        end
        @inbounds for k in num_el:-1:1
            col = Int(idx_col[k])
            if col > 0
                val = el[k]
                row = idx_row[k]
                idx_col[k] = LU_NO_LINK
                while true
                    look = Int(start_col[col]) - 1
                    start_col[col] = Int32(look)
                    val_save = el[look]
                    col_save = Int(idx_col[look])
                    row_save = idx_row[look]
                    el[look] = val
                    idx_row[look] = row
                    idx_col[look] = LU_NO_LINK
                    if col_save > 0
                        col = col_save
                        val = val_save
                        row = row_save
                    else
                        break
                    end
                end
            end
        end
    end

    if state == 0 || state == -1 || state == 1 || state == 2
        pos = 1
        @inbounds for ir in 1:nr
            start_row[ir] = Int32(pos)
            pos += Int(nir[ir])
            nir[ir] = Int32(0)
        end
        @inbounds for ic in 1:nc
            num = Int(nic[ic])
            if num != 0
                first = Int(start_col[ic])
                largest = first
                ir_save = idx_row[first]
                val_save = el[first]
                val_largest = abs(val_save)
                rs = Int(ir_save)
                look = Int(nir[rs])
                nir[rs] = Int32(look + 1)
                idx_col[Int(start_row[rs]) + look] = Int32(ic)
                for k in (first + 1):(first + num - 1)
                    r = Int(idx_row[k])
                    lk = Int(nir[r])
                    nir[r] = Int32(lk + 1)
                    idx_col[Int(start_row[r]) + lk] = Int32(ic)
                    va = abs(el[k])
                    if va > val_largest
                        val_largest = va
                        largest = k
                    end
                end
                idx_row[first] = idx_row[largest]
                el[first] = el[largest]
                idx_row[largest] = ir_save
                el[largest] = val_save
            end
        end
    end

    if state <= 3
        last_row = lu.last_row
        next_row = lu.next_row
        last_col = lu.last_col
        next_col = lu.next_col
        fc = lu.first_count
        ncnt = lu.next_count
        lcnt = lu.last_count

        @inbounds for i in 1:(lu.bigger_dim + 2)
            fc[i] = LU_NO_LINK
        end
        @inbounds for i in 1:nc
            lu.pivot_col[i] = LU_NO_LINK
        end
        @inbounds for i in 1:col_end
            nicp[i] = Int32(0)
        end

        @inbounds for ir in 1:nr
            last_row[ir] = Int32(ir - 1)
            next_row[ir] = Int32(ir + 1)
            _add_link!(fc, ncnt, lcnt, ir, Int(nir[ir]))
        end
        @inbounds begin
            last_row[row_end] = Int32(nr)
            next_row[row_end] = Int32(1)
            if nr > 0
                last_row[1] = Int32(row_end)
                next_row[nr] = Int32(row_end)
            end
            start_row[row_end] = Int32(num_el + 1)
        end

        @inbounds for ic in 1:nc
            last_col[ic] = Int32(ic - 1)
            next_col[ic] = Int32(ic + 1)
            _add_link!(fc, ncnt, lcnt, ic + nr, Int(nic[ic]))
        end
        @inbounds begin
            last_col[col_end] = Int32(nc)
            next_col[col_end] = Int32(1)
            if nc > 0
                last_col[1] = Int32(col_end)
                next_col[nc] = Int32(col_end)
            else
                last_col[1] = Int32(col_end)
            end
            start_col[col_end] = Int32(num_el + 1)
        end
    elseif state == 4
        piv_reg = lu.pivot_region
        @inbounds for ir in 1:nr
            if nir[ir] >= Int32(0)
                nir[ir] = Int32(0)
            end
        end
        @inbounds for ic in 1:nc
            num = Int(nic[ic])
            if num != 0
                orig_first = Int(start_col[ic])
                first = orig_first
                largest = 0
                val_largest = -1.0
                n_other = 0
                for k in first:(first + num - 1)
                    r = idx_row[k]
                    v = el[k]
                    ri = Int(r)
                    cur_nir = nir[ri]
                    if cur_nir >= Int32(0)
                        nir[ri] = cur_nir + Int32(1)
                        n_other += 1
                        va = abs(v)
                        if va > val_largest
                            val_largest = va
                            largest = n_other
                        end
                        start_row[n_other] = r
                        piv_reg[n_other] = v
                    else
                        idx_row[first] = r
                        el[first] = v
                        first += 1
                    end
                end
                nicp[ic] = Int32(first - orig_first)
                start_col[ic] = Int32(first)
                if largest > 0
                    idx_row[first] = start_row[largest]
                    el[first] = piv_reg[largest]
                    first += 1
                end
                for k in 1:n_other
                    if k != largest
                        idx_row[first] = start_row[k]
                        el[first] = piv_reg[k]
                        first += 1
                    end
                end
                nic[ic] = Int32(first - Int(start_col[ic]))
            end
        end

        pos = 1
        @inbounds for ir in 1:nr
            start_row[ir] = Int32(pos)
            n = Int(nir[ir])
            if n > 0
                nir[ir] = Int32(0)
                pos += n
            end
        end
        @inbounds for ic in 1:nc
            num = Int(nic[ic])
            if num != 0
                first = Int(start_col[ic])
                for k in first:(first + num - 1)
                    ri = Int(idx_row[k])
                    lk = Int(nir[ri])
                    nir[ri] = Int32(lk + 1)
                    idx_col[Int(start_row[ri]) + lk] = Int32(ic)
                end
            end
        end

        last_col = lu.last_col
        next_col = lu.next_col
        fc = lu.first_count
        ncnt = lu.next_count
        lcnt = lu.last_count
        slack_val = lu.slack_value

        num_good = 0
        @inbounds lu.start_col_l[1] = Int32(1)
        @inbounds for ir in 1:nr
            if nir[ir] < Int32(0)
                nir[ir] = Int32(0)
                num_good += 1
                piv_reg[num_good] = slack_val
            end
        end
        @inbounds for ic in 1:nc
            last_col[ic] = Int32(ic - 1)
            next_col[ic] = Int32(ic + 1)
            _delete_link!(fc, ncnt, lcnt, ic + nr)
            _add_link!(fc, ncnt, lcnt, ic + nr, Int(nic[ic]))
        end
        @inbounds begin
            last_col[col_end] = Int32(nc)
            next_col[col_end] = Int32(1)
            last_col[1] = Int32(col_end)
            if nc > 0
                next_col[nc] = Int32(col_end)
            end
            start_col[col_end] = Int32(num_el + 1)
        end
    end
    return nothing
end

# =============================================================================
# Specialized Pivot Kernels (row singleton, col singleton, 2-element col, general)
# =============================================================================

function _pivot_row_singleton!(lu::SparseLU, pivot_row::Int, pivot_col::Int)::Bool
    start_col_u = lu.start_col_u
    nir = lu.len_row_u
    nic = lu.len_col_u
    idx_row_u = lu.index_row_u
    start_row_u = lu.start_row_u
    idx_col_u = lu.index_col_u
    next_row = lu.next_row
    last_row = lu.last_row
    fc = lu.first_count
    ncnt = lu.next_count
    lcnt = lu.last_count
    el_u = lu.element_u

    @inbounds begin
        sc = Int(start_col_u[pivot_col])
        num_do_col = Int(nic[pivot_col]) - 1
        end_col = sc + num_do_col
        piv_pos = sc
        while Int(idx_row_u[piv_pos]) != pivot_row
            piv_pos += 1
        end
    end

    if lu.len_l + num_do_col > lu.len_area_l
        return false
    end

    l = lu.len_l + 1
    @inbounds begin
        lu.start_col_l[lu.ngood_l + 1] = Int32(l)
        lu.ngood_l += 1
        lu.len_l += num_do_col
        lu.start_col_l[lu.ngood_l + 1] = Int32(lu.len_l + 1)

        piv_mult = 1.0 / el_u[piv_pos]
        lu.pivot_region[lu.ngood_u + 1] = piv_mult

        el_l = lu.element_l
        idx_row_l = lu.index_row_l
        for i in sc:(piv_pos - 1)
            ir = Int(idx_row_u[i])
            idx_row_l[l] = Int32(ir)
            el_l[l] = el_u[i] * piv_mult
            l += 1
            s = Int(start_row_u[ir])
            inir = Int(nir[ir])
            e = s + inir - 1
            w = s
            while Int(idx_col_u[w]) != pivot_col
                w += 1
            end
            idx_col_u[w] = idx_col_u[e]
            inir -= 1
            nir[ir] = Int32(inir)
            _delete_link!(fc, ncnt, lcnt, ir)
            _add_link!(fc, ncnt, lcnt, ir, inir)
        end
        for i in (piv_pos + 1):end_col
            ir = Int(idx_row_u[i])
            idx_row_l[l] = Int32(ir)
            el_l[l] = el_u[i] * piv_mult
            l += 1
            s = Int(start_row_u[ir])
            inir = Int(nir[ir])
            e = s + inir - 1
            w = s
            while Int(idx_col_u[w]) != pivot_col
                w += 1
            end
            idx_col_u[w] = idx_col_u[e]
            inir -= 1
            nir[ir] = Int32(inir)
            _delete_link!(fc, ncnt, lcnt, ir)
            _add_link!(fc, ncnt, lcnt, ir, inir)
        end

        nic[pivot_col] = Int32(0)
        nir[pivot_row] = Int32(0)
        _delete_link!(fc, ncnt, lcnt, pivot_row)
        _delete_link!(fc, ncnt, lcnt, pivot_col + lu.nrows)

        nxt = Int(next_row[pivot_row])
        lst = Int(last_row[pivot_row])
        next_row[lst] = Int32(nxt)
        last_row[nxt] = Int32(lst)
        last_row[pivot_row] = LU_REMOVED
        next_row[pivot_row] = Int32(lu.ngood_u + 1)
    end
    return true
end

function _pivot_col_singleton!(lu::SparseLU, pivot_row::Int, pivot_col::Int)::Bool
    nir = lu.len_row_u
    nic = lu.len_col_u
    nicp = lu.len_col_plus
    start_col_u = lu.start_col_u
    start_row_u = lu.start_row_u
    idx_col_u = lu.index_col_u
    idx_row_u = lu.index_row_u
    save_col = lu.save_col
    next_row = lu.next_row
    last_row = lu.last_row
    el_u = lu.element_u
    fc = lu.first_count
    ncnt = lu.next_count
    lcnt = lu.last_count
    nr = lu.nrows

    @inbounds begin
        num_do_row = Int(nir[pivot_row]) - 1
        sc = Int(start_col_u[pivot_col])
        sr = Int(start_row_u[pivot_row])
        er = sr + num_do_row

        put = 0
        for i in sr:er
            ic = idx_col_u[i]
            if Int(ic) != pivot_col
                put += 1
                save_col[put] = ic
            end
        end

        nxt = Int(next_row[pivot_row])
        lst = Int(last_row[pivot_row])
        next_row[lst] = Int32(nxt)
        last_row[nxt] = Int32(lst)
        next_row[pivot_row] = Int32(lu.ngood_u + 1)
        last_row[pivot_row] = LU_REMOVED

        lu.pivot_region[lu.ngood_u + 1] = 1.0 / el_u[sc]
        nic[pivot_col] = Int32(0)

        for i in 1:num_do_row
            ic = Int(save_col[i])
            cur_nic = Int(nic[ic])
            if cur_nic != 0
                num = cur_nic - 1
                _delete_link!(fc, ncnt, lcnt, ic + nr)
                _add_link!(fc, ncnt, lcnt, ic + nr, num)
                if num != 0
                    s = Int(start_col_u[ic])
                    piv = s
                    while Int(idx_row_u[piv]) != pivot_row
                        piv += 1
                    end
                    if piv != s
                        val = el_u[s]
                        ir_s = idx_row_u[s]
                        el_u[s] = el_u[piv]
                        idx_row_u[s] = idx_row_u[piv]
                        el_u[piv] = el_u[s + 1]
                        idx_row_u[piv] = idx_row_u[s + 1]
                        el_u[s + 1] = val
                        idx_row_u[s + 1] = ir_s
                    else
                        ir_save = idx_row_u[s + 1]
                        val_save = el_u[s + 1]
                        val_largest = abs(val_save)
                        largest = s + 1
                        for k in (s + 2):(s + cur_nic - 1)
                            va = abs(el_u[k])
                            if va > val_largest
                                val_largest = va
                                largest = k
                            end
                        end
                        idx_row_u[s + 1] = idx_row_u[largest]
                        el_u[s + 1] = el_u[largest]
                        idx_row_u[largest] = ir_save
                        el_u[largest] = val_save
                    end
                end
                nic[ic] = Int32(num)
                nicp[ic] += Int32(1)
                start_col_u[ic] += Int32(1)
            end
        end

        _delete_link!(fc, ncnt, lcnt, pivot_row)
        _delete_link!(fc, ncnt, lcnt, pivot_col + nr)
        nir[pivot_row] = Int32(0)

        l_pos = Int32(lu.len_l + 1)
        lu.start_col_l[lu.ngood_l + 1] = l_pos
        lu.ngood_l += 1
        lu.start_col_l[lu.ngood_l + 1] = l_pos
    end
    return true
end

function _pivot_one_other_row!(lu::SparseLU, pivot_row::Int, pivot_col::Int)::Bool
    nir = lu.len_row_u
    nic = lu.len_col_u
    nicp = lu.len_col_plus
    start_row_u = lu.start_row_u
    start_col_u = lu.start_col_u
    next_row = lu.next_row
    last_row = lu.last_row
    el_u = lu.element_u
    idx_row_u = lu.index_row_u
    idx_col_u = lu.index_col_u
    save_col = lu.save_col
    next_col = lu.next_col
    fc = lu.first_count
    ncnt = lu.next_count
    lcnt = lu.last_count
    nr = lu.nrows
    tol = lu.zero_tol

    @inbounds begin
        num_in_piv_row = Int(nir[pivot_row]) - 1
        sc_piv = Int(start_col_u[pivot_col])
        sr_piv = Int(start_row_u[pivot_row])
        er_piv = sr_piv + num_in_piv_row

        nxt = Int(next_row[pivot_row])
        lst = Int(last_row[pivot_row])
        next_row[lst] = Int32(nxt)
        last_row[nxt] = Int32(lst)
        next_row[pivot_row] = Int32(lu.ngood_u + 1)
        last_row[pivot_row] = LU_REMOVED
        nir[pivot_row] = Int32(0)
    end

    if lu.len_l + 1 > lu.len_area_l
        return false
    end

    @inbounds begin
        l = lu.len_l + 1
        lu.start_col_l[lu.ngood_l + 1] = Int32(l)
        lu.ngood_l += 1
        lu.len_l += 1
        lu.start_col_l[lu.ngood_l + 1] = Int32(lu.len_l + 1)

        if Int(idx_row_u[sc_piv]) == pivot_row
            piv_el = el_u[sc_piv]
            other_mult = el_u[sc_piv + 1]
            other_row = Int(idx_row_u[sc_piv + 1])
        else
            piv_el = el_u[sc_piv + 1]
            other_mult = el_u[sc_piv]
            other_row = Int(idx_row_u[sc_piv])
        end
        num_save = Int(nir[other_row])
        piv_mult = 1.0 / piv_el
        lu.pivot_region[lu.ngood_u + 1] = piv_mult
        nic[pivot_col] = Int32(0)
        other_mult *= piv_mult
        lu.index_row_l[l] = Int32(other_row)
        lu.element_l[l] = other_mult

        s_other = Int(start_row_u[other_row])
        e_other = s_other + num_save - 1
        w = s_other
        while Int(idx_col_u[w]) != pivot_col
            w += 1
        end
        idx_col_u[w] = idx_col_u[e_other]
        e_other -= 1   # inclusive end of remaining other_row entries (s_other..e_other)

        num_added = 0
        num_deleted = 0

        for j in sr_piv:er_piv
            ic = Int(idx_col_u[j])
            ic == pivot_col && continue
            sc = Int(start_col_u[ic])
            ec = sc + Int(nic[ic]) - 1
            ir = Int(idx_row_u[sc])
            val = el_u[sc]
            found_other = false

            put = sc + 1
            pos_largest = 0
            this_piv_val = 0.0
            other_el = 0.0
            largest = 0.0

            next_val = put <= ec ? el_u[put] : 0.0
            next_ir = put <= ec ? Int(idx_row_u[put]) : 0

            if ir != pivot_row
                if ir != other_row
                    largest = abs(val)
                    el_u[put] = val
                    idx_row_u[put] = Int32(ir)
                    pos_largest = put
                    put += 1
                    for i in (sc + 1):ec
                        ir = next_ir
                        val = next_val
                        if i < ec
                            next_ir = Int(idx_row_u[i + 1])
                            next_val = el_u[i + 1]
                        end
                        if ir != pivot_row
                            if ir != other_row
                                idx_row_u[put] = Int32(ir)
                                el_u[put] = val
                                put += 1
                            else
                                other_el = val
                                found_other = true
                            end
                        else
                            this_piv_val = val
                        end
                    end
                else
                    other_el = val
                    found_other = true
                    for i in (sc + 1):ec
                        ir = next_ir
                        val = next_val
                        if i < ec
                            next_ir = Int(idx_row_u[i + 1])
                            next_val = el_u[i + 1]
                        end
                        if ir != pivot_row
                            idx_row_u[put] = Int32(ir)
                            el_u[put] = val
                            av = abs(val)
                            if av > largest
                                largest = av
                                pos_largest = put
                            end
                            put += 1
                        else
                            this_piv_val = val
                        end
                    end
                end
            else
                this_piv_val = val
                for i in (sc + 1):ec
                    ir = next_ir
                    val = next_val
                    if i < ec
                        next_ir = Int(idx_row_u[i + 1])
                        next_val = el_u[i + 1]
                    end
                    if ir != other_row
                        idx_row_u[put] = Int32(ir)
                        el_u[put] = val
                        av = abs(val)
                        if av > largest
                            largest = av
                            pos_largest = put
                        end
                        put += 1
                    else
                        other_el = val
                        found_other = true
                    end
                end
            end

            el_u[sc] = this_piv_val
            idx_row_u[sc] = Int32(pivot_row)
            sc += 1
            nic[ic] = Int32(put - sc)
            nicp[ic] += Int32(1)
            start_col_u[ic] = Int32(sc)

            other_el = _lu_fmsub(this_piv_val, other_mult, other_el)
            av = abs(other_el)
            if av > tol
                if !found_other
                    num_added += 1
                    save_col[num_added] = Int32(ic)
                    nxt_c = Int(next_col[ic])
                    sp = Int(start_col_u[nxt_c]) - put - Int(nicp[nxt_c])
                    if sp <= 0
                        num = Int(nic[ic])
                        if !_get_col_space!(lu, ic, num + 1)
                            return false
                        end
                        new_sc = Int(start_col_u[ic])
                        if pos_largest > 0
                            pos_largest += new_sc - sc
                        end
                        sc = new_sc
                        put = sc + num
                    end
                end
                el_u[put] = other_el
                idx_row_u[put] = Int32(other_row)
                if av > largest
                    largest = av
                    pos_largest = put
                end
                put += 1
            elseif found_other
                num_deleted += 1
                w2 = s_other
                while Int(idx_col_u[w2]) != ic
                    w2 += 1
                end
                idx_col_u[w2] = idx_col_u[e_other]
                e_other -= 1
            end

            new_nic = put - sc
            nic[ic] = Int32(new_nic)
            if pos_largest > 0
                v_l = el_u[pos_largest]
                r_l = idx_row_u[pos_largest]
                el_u[pos_largest] = el_u[sc]
                idx_row_u[pos_largest] = idx_row_u[sc]
                el_u[sc] = v_l
                idx_row_u[sc] = r_l
            end
            if ncnt[ic + nr] != LU_REMOVED
                _delete_link!(fc, ncnt, lcnt, ic + nr)
                _add_link!(fc, ncnt, lcnt, ic + nr, new_nic)
            end
        end

        nxt_or = Int(next_row[other_row])
        cur_len = e_other - s_other + 1
        sp = Int(start_row_u[nxt_or]) - (e_other + 1)
        lu.total_elements += num_added - num_deleted
        number = num_added + cur_len
        if sp < num_added
            nir[other_row] = Int32(cur_len)
            if !_get_row_space!(lu, other_row, number)
                return false
            end
            e_other = Int(start_row_u[other_row]) + cur_len - 1
        end
        nir[other_row] = Int32(number)
        if number != num_save
            _delete_link!(fc, ncnt, lcnt, other_row)
            _add_link!(fc, ncnt, lcnt, other_row, number)
        end
        for j in 1:num_added
            e_other += 1
            idx_col_u[e_other] = save_col[j]
        end

        _delete_link!(fc, ncnt, lcnt, pivot_row)
        _delete_link!(fc, ncnt, lcnt, pivot_col + nr)
    end
    return true
end

function _pivot_general!(
    lu::SparseLU,
    pivot_row::Int,
    pivot_col::Int,
    pivot_row_pos::Int,
    pivot_col_pos::Int,
    inc2::Int,
    mark_row::Vector{TMark},
    large_int::Int32,
)::Bool where {TMark<:Union{UInt16,Int32}}
    idx_col_u = lu.index_col_u
    start_col_u = lu.start_col_u
    nic = lu.len_col_u
    nicp = lu.len_col_plus
    el_u = lu.element_u
    idx_row_u = lu.index_row_u
    start_row_u = lu.start_row_u
    nir = lu.len_row_u
    el_l = lu.element_l
    idx_row_l = lu.index_row_l
    save_col = lu.save_col
    next_row = lu.next_row
    last_row = lu.last_row
    next_col = lu.next_col
    fc = lu.first_count
    ncnt = lu.next_count
    lcnt = lu.last_count
    work = lu.work_area
    work2 = lu.work_flags
    nr = lu.nrows
    tol = lu.zero_tol

    @inbounds begin
        num_in_piv_row = Int(nir[pivot_row]) - 1
        sc_piv = Int(start_col_u[pivot_col])
        num_in_piv_col = Int(nic[pivot_col]) - 1
        ec_piv = sc_piv + num_in_piv_col
        sr_piv = Int(start_row_u[pivot_row])
        er_piv = sr_piv + num_in_piv_row

        put = 0
        pcp = pivot_col_pos
        if pcp <= 0
            pcp = sr_piv
            while pcp <= er_piv
                ic = idx_col_u[pcp]
                if Int(ic) != pivot_col
                    put += 1
                    save_col[put] = ic
                else
                    break
                end
                pcp += 1
            end
        else
            for i in sr_piv:(pcp - 1)
                put += 1
                save_col[put] = idx_col_u[i]
            end
        end
        for i in (pcp + 1):er_piv
            put += 1
            save_col[put] = idx_col_u[i]
        end

        nxt = Int(next_row[pivot_row])
        lst = Int(last_row[pivot_row])
        next_row[lst] = Int32(nxt)
        last_row[nxt] = Int32(lst)
        next_row[pivot_row] = Int32(lu.ngood_u + 1)
        last_row[pivot_row] = LU_REMOVED
        nir[pivot_row] = Int32(0)
    end

    if lu.len_l + num_in_piv_col > lu.len_area_l
        return false
    end

    @inbounds begin
        l_save = lu.len_l + 1
        l = l_save
        lu.start_col_l[lu.ngood_l + 1] = Int32(l)
        lu.ngood_l += 1
        lu.len_l += num_in_piv_col
        lu.start_col_l[lu.ngood_l + 1] = Int32(lu.len_l + 1)

        prp = pivot_row_pos
        if prp <= 0
            prp = sc_piv
            while prp <= ec_piv
                ir = Int(idx_row_u[prp])
                if ir != pivot_row
                    idx_row_l[l] = Int32(ir)
                    el_l[l] = el_u[prp]
                    mark_row[ir] = TMark(l - l_save)
                    l += 1
                    s = Int(start_row_u[ir])
                    cur_nir = Int(nir[ir])
                    e = s + cur_nir - 1
                    w = s
                    while Int(idx_col_u[w]) != pivot_col
                        w += 1
                    end
                    idx_col_u[w] = idx_col_u[e]
                    nir[ir] = Int32(cur_nir - 1)
                else
                    break
                end
                prp += 1
            end
        else
            for i in sc_piv:(prp - 1)
                ir = Int(idx_row_u[i])
                mark_row[ir] = TMark(l - l_save)
                idx_row_l[l] = Int32(ir)
                el_l[l] = el_u[i]
                l += 1
                s = Int(start_row_u[ir])
                cur_nir = Int(nir[ir])
                e = s + cur_nir - 1
                w = s
                while Int(idx_col_u[w]) != pivot_col
                    w += 1
                end
                idx_col_u[w] = idx_col_u[e]
                nir[ir] = Int32(cur_nir - 1)
            end
        end

        piv_mult = 1.0 / el_u[prp]
        lu.pivot_region[lu.ngood_u + 1] = piv_mult
        for i in (prp + 1):ec_piv
            ir = Int(idx_row_u[i])
            mark_row[ir] = TMark(l - l_save)
            idx_row_l[l] = Int32(ir)
            el_l[l] = el_u[i]
            l += 1
            s = Int(start_row_u[ir])
            cur_nir = Int(nir[ir])
            e = s + cur_nir - 1
            w = s
            while Int(idx_col_u[w]) != pivot_col
                w += 1
            end
            idx_col_u[w] = idx_col_u[e]
            nir[ir] = Int32(cur_nir - 1)
        end

        mark_row[pivot_row] = TMark(large_int)
        nic[pivot_col] = Int32(0)

        l_base = l_save - 1
        for j in 1:num_in_piv_col
            el_l[l_base + j] *= piv_mult
        end

        erase_end = inc2 * num_in_piv_row
        for ie in 1:erase_end
            work2[ie] = UInt32(0)
        end

        added = num_in_piv_row * num_in_piv_col
        large_unset = large_int + Int32(1)
        temp2_base = 0

        for jcol in 1:num_in_piv_row
            ic = Int(save_col[jcol])
            sc = Int(start_col_u[ic])
            ec = sc + Int(nic[ic]) - 1
            ir = Int(idx_row_u[sc])
            val = el_u[sc]
            put = sc
            pos_largest = 0
            this_piv_val = 0.0
            mark = Int32(mark_row[ir])

            if mark == large_unset
                largest = abs(val)
                pos_largest = put
                put += 1
                check_largest = false
            else
                largest = 0.0
                check_largest = true
                if mark != large_int
                    work[Int(mark) + 1] = val
                    word = Int(mark >> 5) + 1
                    bit = mark & Int32(31)
                    work2[temp2_base + word] |= (UInt32(1) << bit)
                    added -= 1
                else
                    this_piv_val = val
                end
            end

            for i in (sc + 1):ec
                ir = Int(idx_row_u[i])
                val = el_u[i]
                mark = Int32(mark_row[ir])
                if mark == large_unset
                    idx_row_u[put] = Int32(ir)
                    el_u[put] = val
                    if check_largest
                        av = abs(val)
                        if av > largest
                            largest = av
                            pos_largest = put
                        end
                    end
                    put += 1
                elseif mark != large_int
                    work[Int(mark) + 1] = val
                    word = Int(mark >> 5) + 1
                    bit = mark & Int32(31)
                    work2[temp2_base + word] |= (UInt32(1) << bit)
                    added -= 1
                else
                    this_piv_val = val
                end
            end

            el_u[put] = el_u[sc]
            idx_row_u[put] = idx_row_u[sc]
            if pos_largest == sc
                pos_largest = put
            end
            put += 1
            el_u[sc] = this_piv_val
            idx_row_u[sc] = Int32(pivot_row)
            sc += 1
            nic[ic] = Int32(put - sc)
            nicp[ic] += Int32(1)
            start_col_u[ic] = Int32(sc)

            nxt_c = Int(next_col[ic])
            sp = Int(start_col_u[nxt_c]) - put - Int(nicp[nxt_c])
            if num_in_piv_col > sp
                if !_get_col_space!(lu, ic, num_in_piv_col)
                    return false
                end
                new_sc = Int(start_col_u[ic])
                if pos_largest > 0
                    pos_largest += new_sc - sc
                end
                sc = new_sc
                put = sc + Int(nic[ic])
            end

            for j in 1:num_in_piv_col
                v = _lu_fmsub(this_piv_val, el_l[l_base + j], work[j])
                work[j] = 0.0
                av = abs(v)
                if av > tol
                    el_u[put] = v
                    idx_row_u[put] = idx_row_l[l_base + j]
                    if av > largest
                        largest = av
                        pos_largest = put
                    end
                    put += 1
                else
                    added -= 1
                    jm1 = j - 1
                    word = (jm1 >> 5) + 1
                    bit = jm1 & 31
                    wval = work2[temp2_base + word]
                    mask = UInt32(1) << bit
                    if (wval & mask) != UInt32(0)
                        ir_j = Int(idx_row_l[l_base + j])
                        s = Int(start_row_u[ir_j])
                        cur_nir = Int(nir[ir_j])
                        e = s + cur_nir - 1
                        w = s
                        while Int(idx_col_u[w]) != ic
                            w += 1
                        end
                        idx_col_u[w] = idx_col_u[e]
                        nir[ir_j] = Int32(cur_nir - 1)
                    else
                        work2[temp2_base + word] = wval | mask
                    end
                end
            end

            new_nic = put - sc
            nic[ic] = Int32(new_nic)
            if pos_largest > 0
                v_l = el_u[pos_largest]
                r_l = idx_row_u[pos_largest]
                el_u[pos_largest] = el_u[sc]
                idx_row_u[pos_largest] = idx_row_u[sc]
                el_u[sc] = v_l
                idx_row_u[sc] = r_l
            end
            if ncnt[ic + nr] != LU_REMOVED
                _delete_link!(fc, ncnt, lcnt, ic + nr)
                _add_link!(fc, ncnt, lcnt, ic + nr, new_nic)
            end
            temp2_base += inc2
        end

        # Update row lists and Markowitz row counts from work2 bitmask
        unset_mark = TMark(large_unset)
        for i in 1:num_in_piv_col
            im1 = i - 1
            word_off = (im1 >> 5) + 1
            bit = im1 & 31
            ir = Int(idx_row_l[l_base + i])
            number = 0
            pos_w = word_off
            for _ in 1:num_in_piv_row
                number += 1 - Int((work2[pos_w] >> bit) & UInt32(1))
                pos_w += inc2
            end
            nxt_r = Int(next_row[ir])
            sp = Int(start_row_u[nxt_r]) - Int(start_row_u[ir])
            number += Int(nir[ir])
            if sp < number
                if !_get_row_space!(lu, ir, number)
                    return false
                end
            end
            nxt_r = Int(next_row[ir])
            sr_i = Int(start_row_u[ir])
            e = sr_i + Int(nir[ir])
            nxt_sr = Int(start_row_u[nxt_r])
            save_idx = idx_col_u[nxt_sr]
            pos_w = word_off
            for jcol in 1:num_in_piv_row
                inc = 1 - Int((work2[pos_w] >> bit) & UInt32(1))
                pos_w += inc2
                idx_col_u[e] = save_col[jcol]
                e += inc
            end
            idx_col_u[nxt_sr] = save_idx
            mark_row[ir] = unset_mark
            new_nir = e - sr_i
            nir[ir] = Int32(new_nir)
            _delete_link!(fc, ncnt, lcnt, ir)
            _add_link!(fc, ncnt, lcnt, ir, new_nir)
        end

        mark_row[pivot_row] = unset_mark
        _delete_link!(fc, ncnt, lcnt, pivot_row)
        _delete_link!(fc, ncnt, lcnt, pivot_col + nr)
        lu.total_elements += added
    end
    return true
end

# =============================================================================
# Markowitz Sparse Factorization Loop
# =============================================================================

function _factor_sparse_impl!(
    lu::SparseLU,
    mark_row::Vector{TMark},
    mark_set::Int32,
    mark_unset::Int32,
)::Int where {TMark<:Union{UInt16,Int32}}
    idx_row = lu.index_row_u
    idx_col = lu.index_col_u
    el = lu.element_u
    nr = lu.nrows
    nc = lu.ncols
    bigger_dim = lu.bigger_dim
    dense_thresh = abs(lu.dense_threshold)
    piv_tol = lu.pivot_tol
    num_trials = lu.num_trials
    bias_lu = lu.bias_lu

    work = lu.work_area
    @inbounds for i in 1:nr
        work[i] = 0.0
    end
    unset_val = TMark(mark_unset)
    @inbounds for i in 1:nr
        mark_row[i] = unset_val
    end

    nir = lu.len_row_u
    nic = lu.len_col_u
    nicp = lu.len_col_plus
    start_col_u = lu.start_col_u
    start_col_l = lu.start_col_l
    start_row = lu.start_row_u
    piv_col = lu.pivot_col
    next_row = lu.next_row
    last_row = lu.last_row
    fc = lu.first_count
    ncnt = lu.next_count
    lcnt = lu.last_count

    status = 0

    # Strip slack singletons first
    if bias_lu < 3 && nc == nr
        slack_val = lu.slack_value
        @inbounds for ipc in 1:nc
            if nic[ipc] == Int32(1)
                s = Int(start_col_u[ipc])
                if el[s] == slack_val && nicp[ipc] == Int32(0)
                    ir = Int(idx_row[s])
                    cur_nir = Int(nir[ir])
                    if cur_nir > 0
                        lu.total_elements -= cur_nir
                        nxt = Int(next_row[ir])
                        lst = Int(last_row[ir])
                        next_row[lst] = Int32(nxt)
                        last_row[nxt] = Int32(lst)
                        next_row[ir] = Int32(lu.ngood_u + 1)
                        last_row[ir] = LU_REMOVED
                        _delete_link!(fc, ncnt, lcnt, ir)
                        nir[ir] = Int32(-1)
                        nic[ipc] = Int32(0)
                        lu.ngood_l += 1
                        start_col_l[lu.ngood_l + 1] = Int32(1)
                        lu.ngood_u += 1
                        piv_col[lu.ngood_u] = Int32(ipc)
                    end
                end
            end
        end
        preprocess!(lu, 4)
        @inbounds for i in 1:nr
            mark_row[i] = unset_val
        end
    end

    lu.nslacks = lu.ngood_u
    _separate_links!(fc, ncnt, lcnt, nr, 1, bias_lu > 1)

    count = 1
    while count <= bigger_dim
        min_count = typemax(Int)
        min_cost = floatmax(Float64)
        i_piv_row = 0
        i_piv_col = 0
        piv_row_pos = 0
        piv_col_pos = 0
        @inbounds look = Int(fc[count + 1])
        trials = 0

        if count == 1 && look > 0 && bias_lu == 0
            @inbounds while look > 0
                if look <= nr
                    look = Int(ncnt[look])
                else
                    ic = look - nr
                    s = Int(start_col_u[ic])
                    i_piv_row = Int(idx_row[s])
                    piv_row_pos = s
                    i_piv_col = ic
                    piv_col_pos = 0
                    look = 0
                    break
                end
            end
            if i_piv_row == 0
                @inbounds look = Int(fc[2])
            end
        end

        @inbounds while look > 0
            if look <= nr
                ir = look
                look = Int(ncnt[look])
                rejected = false
                s = Int(start_row[ir])
                e = s + count - 1
                cm1 = count - 1
                for i in s:e
                    ic = Int(idx_col[i])
                    n_ic = Int(nic[ic])
                    cost = Float64(cm1 * n_ic)
                    if cost < min_cost
                        w = Int(start_col_u[ic])
                        min_val = abs(el[w]) * piv_tol
                        while Int(idx_row[w]) != ir
                            w += 1
                        end
                        if abs(el[w]) >= min_val
                            min_cost = cost
                            min_count = n_ic
                            i_piv_row = ir
                            piv_row_pos = 0
                            i_piv_col = ic
                            piv_col_pos = i
                            rejected = false
                            if min_count < count
                                look = 0
                                break
                            end
                        elseif i_piv_row == 0
                            rejected = true
                        end
                    end
                end
                trials += 1
                if trials >= num_trials && i_piv_row > 0
                    look = 0
                    break
                end
                if rejected
                    _delete_link!(fc, ncnt, lcnt, ir)
                    _add_link!(fc, ncnt, lcnt, ir, bigger_dim + 1)
                end
            else
                ic = look - nr
                look = Int(ncnt[look])
                s = Int(start_col_u[ic])
                e = s + Int(nic[ic]) - 1
                min_val = abs(el[s]) * piv_tol
                cm1 = count - 1
                for i in s:e
                    if abs(el[i]) >= min_val
                        ir = Int(idx_row[i])
                        n_ir = Int(nir[ir])
                        cost = Float64(cm1 * n_ir)
                        if cost < min_cost
                            min_cost = cost
                            min_count = n_ir
                            i_piv_row = ir
                            piv_row_pos = i
                            i_piv_col = ic
                            piv_col_pos = 0
                            if min_count <= count + 1
                                look = 0
                                break
                            end
                        end
                    end
                end
                trials += 1
                if trials >= num_trials && i_piv_row > 0
                    look = 0
                    break
                end
            end
        end

        if i_piv_row > 0
            @inbounds num_do_row = Int(nir[i_piv_row]) - 1
            @inbounds num_do_col = Int(nic[i_piv_col]) - 1
            lu.total_elements -= (num_do_row + num_do_col + 1)

            good_piv = if num_do_col > 0
                if num_do_row > 0
                    if num_do_col > 1
                        inc = num_do_col + 5
                        if (inc & 15) != 0
                            inc = (inc & ~15) + 16
                        end
                        inc2 = (inc + 31) >> 5
                        sz = inc2 * num_do_row
                        if sz > length(lu.work_flags)
                            resize!(lu.work_flags, sz)
                        end
                        _pivot_general!(lu, i_piv_row, i_piv_col, piv_row_pos, piv_col_pos, inc2, mark_row, mark_set)
                    else
                        _pivot_one_other_row!(lu, i_piv_row, i_piv_col)
                    end
                else
                    _pivot_row_singleton!(lu, i_piv_row, i_piv_col)
                end
            else
                _pivot_col_singleton!(lu, i_piv_row, i_piv_col)
            end

            if !good_piv
                status = -99
                break
            end

            lu.ngood_u += 1
            @inbounds piv_col[lu.ngood_u] = Int32(i_piv_col)
            if lu.ngood_u == nr
                break
            end

            if dense_thresh != 0
                left_rows = nr - lu.ngood_u
                full = Float64(left_rows) * Float64(left_rows)
                left_el = Float64(lu.total_elements)
                ratio = if left_rows > 2000
                    3.5
                elseif left_rows > 800
                    2.75
                elseif left_rows > 256
                    2.0
                else
                    1.5
                end
                if ratio * left_el > full && left_rows > dense_thresh && (left_rows & 7) == 0
                    if status != 0
                        break
                    end
                    chk = 0
                    @inbounds for ic in 1:nc
                        if nic[ic] != Int32(0)
                            chk += 1
                        end
                    end
                    if chk != left_rows
                        dense_thresh = 0
                    else
                        status = 2
                        break
                    end
                end
            end
            count = 1
        else
            count += 1
        end
    end

    lu.status = status
    return status
end

# =============================================================================
# Dense Trailing Block Factorization (LAPACK Accelerate dgetrf_ + naive fallback)
# =============================================================================

function _factor_dense!(lu::SparseLU)::Int
    nr = lu.nrows
    nc = lu.ncols
    nd = nr - lu.ngood_u
    full = nd * nd
    lu.total_elements = full

    _ensure_zero!(lu.dense_area, full)
    _ensure_len!(lu.dense_permute, nd)

    dense_area = lu.dense_area
    dense_perm = lu.dense_permute
    next_row = lu.next_row
    last_row = lu.last_row
    nic = lu.len_col_u
    nicp = lu.len_col_plus
    idx_row = lu.index_row_u
    el_u = lu.element_u
    start_col_l = lu.start_col_l
    el_l = lu.element_l
    idx_row_l = lu.index_row_l
    piv_col = lu.pivot_col
    piv_reg = lu.pivot_region
    start_col_u = lu.start_col_u

    @inbounds for i in 1:nr
        if last_row[i] > LU_NO_LINK
            last_row[i] = Int32(0)
        end
    end
    which = 0
    @inbounds for i in 1:nr
        if last_row[i] == Int32(0)
            which += 1
            last_row[i] = Int32(which)
            next_row[i] = Int32(lu.ngood_u + which)
            dense_perm[which] = Int32(i)
        end
    end

    @inbounds end_l = start_col_l[lu.ngood_l + 1]
    col_off = 0
    rows_done = 1
    @inbounds for ic in 1:nc
        num = Int(nic[ic])
        if num != 0
            s = Int(start_col_u[ic])
            for k in s:(s + num - 1)
                ir = Int(last_row[Int(idx_row[k])])
                dense_area[col_off + ir] = el_u[k]
            end
            col_off += nd
            while last_row[rows_done] < Int32(0)
                rows_done += 1
            end
            lu.ngood_u += 1
            next_row[rows_done] = Int32(lu.ngood_u)
            rows_done += 1
            start_col_l[lu.ngood_u + 1] = end_l
            nic[ic] = Int32(0)
            piv_col[lu.ngood_u] = Int32(ic)
            piv_reg[lu.ngood_u] = 1.0
        end
    end

    if lu.dense_threshold != 0
        lu.ngood_l = nr
        info = lu_lapack_dgetrf!(dense_area, nd, dense_perm)
        lu.ndense = nd
        lu.status = info != 0 ? -1 : 0
        return lu.status
    end

    # Naive dense fallback (dense_threshold == 0)
    lu.ngood_u = nr - nd
    base = lu.ngood_u
    num_to_do = -lu.dense_threshold
    tol = 1.0e-30
    next_col = lu.next_col

    @inbounds for id in 0:(num_to_do - 1)
        ic_d = Int(piv_col[base + id + 1])
        nxt_c = Int(next_col[ic_d])
        sp = Int(start_col_u[nxt_c]) - Int(start_col_u[ic_d]) - Int(nicp[nxt_c])
        if id > sp
            if !_get_col_space!(lu, ic_d, id)
                lu.ndense = nd
                lu.dense_threshold = 0
                lu.status = -99
                return -99
            end
        end
        nic[ic_d] = Int32(id)
    end

    @inbounds for ic_f in (base + num_to_do + 1):nr
        next_row[ic_f] = Int32(ic_f)
        start_col_l[ic_f + 1] = end_l
        piv_reg[ic_f] = 1.0
    end

    if Float64(lu.len_l) + Float64(full) * 0.5 > Float64(lu.len_area_l)
        lu.ndense = nd
        lu.dense_threshold = 0
        lu.status = -99
        return -99
    end

    @inbounds for id in 1:num_to_do
        piv_r = 0
        cbase = (id - 1) * nd
        largest = 1.0e-12
        for ir in id:nd
            av = abs(dense_area[cbase + ir])
            if av > largest
                largest = av
                piv_r = ir
            end
        end
        if piv_r == 0
            lu.ndense = nd
            lu.dense_threshold = 0
            lu.status = -1
            return -1
        end
        ic_d = Int(piv_col[base + id])
        piv_el = dense_area[cbase + piv_r]
        orig_row = Int(dense_perm[piv_r])
        lu.ngood_u += 1
        next_row[orig_row] = Int32(lu.ngood_u)
        last_row[orig_row] = LU_REMOVED
        dense_perm[piv_r] = dense_perm[id]
        dense_perm[id] = Int32(orig_row)
        for jd in id:nd
            jb = (jd - 1) * nd
            v = dense_area[jb + id]
            dense_area[jb + id] = dense_area[jb + piv_r]
            dense_area[jb + piv_r] = v
        end
        piv_mult = 1.0 / piv_el
        piv_reg[lu.ngood_u] = piv_mult
        l = lu.len_l + 1
        start_col_l[lu.ngood_l + 1] = Int32(l)
        for ir in (id + 1):nd
            v = dense_area[cbase + ir] * piv_mult
            dense_area[cbase + ir] = v
            if abs(v) > tol
                idx_row_l[l] = dense_perm[ir]
                el_l[l] = v
                l += 1
            end
        end
        lu.ngood_l += 1
        lu.len_l = l - 1
        start_col_l[lu.ngood_l + 1] = Int32(l)

        s = Int(start_col_u[ic_d])
        orig_s = s
        for ir in 1:(id - 1)
            v = dense_area[cbase + ir]
            if abs(v) > tol
                idx_row[s] = dense_perm[ir]
                el_u[s] = v
                s += 1
            end
        end
        nic[ic_d] = Int32(0)
        nicp[ic_d] += Int32(s - orig_s)
        start_col_u[ic_d] = Int32(s)

        for jd in (id + 1):num_to_do
            jb = (jd - 1) * nd
            v = dense_area[jb + id]
            for ir in (id + 1):nd
                dense_area[jb + ir] = _lu_fmsub(v, dense_area[cbase + ir], dense_area[jb + ir])
            end
        end
    end

    lu.ndense = 0
    lu.dense_threshold = 0
    lu.status = 0
    return 0
end

# =============================================================================
# Singular Basis Handling (CoinFactorization1.cpp:1180-1239)
# =============================================================================

function _factor_singular!(lu::SparseLU)::Int
    nr = lu.nrows
    nc = lu.ncols
    row_end = lu.max_rows_extra + 1
    next_row = lu.next_row
    last_row = lu.last_row
    last_col = lu.last_col
    piv_col = lu.pivot_col
    permute_a = lu.permute

    @inbounds begin
        k = Int(next_row[row_end])
        while k != row_end && k > 0
            ir = k
            k = Int(next_row[k])
            next_row[ir] = LU_NO_LINK
        end
        for i in 1:nr
            igood = Int(next_row[i])
            if igood > 0
                permute_a[igood] = Int32(i)
            end
        end
        for i in 1:nr
            last_row[i] = LU_NO_LINK
        end
        for i in 1:nc
            last_col[i] = LU_NO_LINK
        end
        for i in 1:lu.ngood_u
            good_row = Int(permute_a[i])
            good_col = Int(piv_col[i])
            last_row[good_row] = Int32(good_col)
            last_col[good_col] = Int32(good_row)
        end
        for i in 1:nr
            next_row[i] = last_row[i]
        end
        for i in 1:nc
            piv_col[i] = last_col[i]
        end
    end
    lu.permute, lu.next_row = lu.next_row, lu.permute
    lu.total_elements = 0
    lu.status = -1
    return -1
end

# =============================================================================
# Post-factor Cleanup (CoinFactorization1.cpp:1705-2105)
# =============================================================================

function _cleanup!(lu::SparseLU)::Nothing
    nr = lu.nrows
    nc = lu.ncols
    row_end = lu.max_rows_extra + 1
    col_end = lu.max_cols_extra + 1
    num_slacks = lu.nslacks

    # Compress U columns
    _get_col_space!(lu, 1, Int(_LU_COMPRESS_SENTINEL))

    # O(1) vector swaps matching C++ numberInColumn_.swap(numberInColumnPlus_) & permute_.swap(nextRow_)
    lu.len_col_u, lu.len_col_plus = lu.len_col_plus, lu.len_col_u
    lu.permute, lu.next_row = lu.next_row, lu.permute

    nic = lu.len_col_u
    nicp = lu.len_col_plus
    permute = lu.permute
    next_row = lu.next_row
    start_col_u = lu.start_col_u
    @inbounds last_u = Int(start_col_u[col_end]) - 1

    # permute & permute_back
    @inbounds permute[nr + 1] = Int32(1)
    permute_back = lu.permute_back
    @inbounds for i in 1:nr
        ir = Int(permute[i])
        permute_back[ir] = Int32(i)
    end

    # Shift start_col_u back to column start
    total_elements = 0
    @inbounds for i in 1:nc
        num = Int(nic[i])
        total_elements += num
        start_col_u[i] -= Int32(num)
    end

    piv_col = lu.pivot_col
    piv_col_b = lu.pivot_col_back
    idx_col_u = lu.index_col_u
    idx_row_u = lu.index_row_u
    el_u = lu.element_u

    num_u = 0
    @inbounds for i in 1:nc
        ic = Int(piv_col[i])
        piv_col_b[ic] = Int32(i)
        if ic > 0
            num_u += 1
            nicp[ic] = num_u != ic ? Int32(num_u) : LU_NO_LINK
        end
    end

    # Cycle-permute len_col_u & start_col_u into pivot order
    @inbounds for i in 1:nc
        num = nic[i]
        w = Int(nicp[i])
        nicp[i] = LU_NO_LINK
        s = start_col_u[i]
        while w > 0
            num_next = nic[w]
            w_next = Int(nicp[w])
            s_next = start_col_u[w]
            nic[w] = num
            nicp[w] = LU_NO_LINK
            start_col_u[w] = s
            num = num_next
            w = w_next
            s = s_next
        end
    end

    # Sort index_row_u & element_u into pivot column order
    @inbounds for i in 1:last_u
        idx_col_u[i] = LU_NO_LINK
    end
    k = 1
    @inbounds for i in (num_slacks + 1):nr
        s = Int(start_col_u[i])
        e = s + Int(nic[i]) - 1
        for j in s:e
            idx_col_u[j] = Int32(k)
            k += 1
        end
    end
    @inbounds for i in (num_slacks + 1):nr
        s = Int(start_col_u[i])
        e = s + Int(nic[i]) - 1
        for j in s:e
            kk = Int(idx_col_u[j])
            ir = idx_row_u[j]
            ev = el_u[j]
            while kk > 0
                kk_next = Int(idx_col_u[kk])
                ir_next = idx_row_u[kk]
                ev_next = el_u[kk]
                idx_col_u[kk] = LU_NO_LINK
                idx_row_u[kk] = ir
                el_u[kk] = ev
                kk = kk_next
                ir = ir_next
                ev = ev_next
            end
        end
    end

    @inbounds for i in 1:num_slacks
        start_col_u[i] = Int32(1)
    end
    k = 1
    @inbounds for i in (num_slacks + 1):nr
        start_col_u[i] = Int32(k)
        k += Int(nic[i])
    end
    max_u = k - 1
    @inbounds start_col_u[col_end] = Int32(k)

    total_elements += lu.ndense * lu.ndense + lu.len_l
    next_col = lu.next_col
    last_col = lu.last_col

    use_extra_r = !(max_u > 10 * nr || nr < 200)
    if use_extra_r
        @inbounds for i in 1:nc
            last_col[i] = Int32(i - 1)
            next_col[i] = Int32(i + 1)
        end
        @inbounds begin
            if nc > 0
                next_col[nc] = Int32(col_end)
            end
            last_col[col_end] = Int32(nc)
            next_col[col_end] = Int32(1)
            last_col[1] = Int32(col_end)
        end
    end

    lu.nu = num_u
    lu.ngood_u = num_u
    nir = lu.len_row_u
    piv_reg = lu.pivot_region

    @inbounds for i in 1:nr
        nir[i] = Int32(0)
    end

    @inbounds for i in (num_slacks + 1):num_u
        s = Int(start_col_u[i])
        n_c = Int(nic[i])
        total_elements += n_c
        for j in s:(s + n_c - 1)
            ir = permute[Int(idx_row_u[j])]
            idx_row_u[j] = ir
            nir[Int(ir)] += Int32(1)
        end
    end

    # Cross-referenced row copy of U
    conv = lu.row_to_col_u
    start_row = lu.start_row_u
    pos = 1
    @inbounds for ir in 1:nr
        start_row[ir] = Int32(pos)
        pos += Int(nir[ir])
        nir[ir] = Int32(0)
    end

    @inbounds for i in (num_slacks + 1):nr
        s = Int(start_col_u[i])
        e = s + Int(nic[i]) - 1
        pv = piv_reg[i]
        for j in s:e
            ir = Int(idx_row_u[j])
            lk = Int(nir[ir])
            nir[ir] = Int32(lk + 1)
            kk = Int(start_row[ir]) + lk
            idx_col_u[kk] = Int32(i)
            conv[kk] = Int32(j)
            el_u[j] *= pv
        end
    end

    last_row = lu.last_row
    @inbounds for j in 1:nr
        last_row[j] = Int32(j - 1)
        next_row[j] = Int32(j + 1)
    end
    @inbounds begin
        if nr > 0
            next_row[nr] = Int32(row_end)
            last_row[1] = Int32(row_end)
        end
        last_row[row_end] = Int32(nr)
        next_row[row_end] = Int32(1)
        start_row[row_end] = Int32(pos)
    end

    # Permute L row indices & find base_l (0-based count of empty leading L cols)
    first_real = nr
    start_col_l = lu.start_col_l
    idx_row_l = lu.index_row_l
    @inbounds for i in nr:-1:1
        s = Int(start_col_l[i])
        e = Int(start_col_l[i + 1]) - 1
        total_elements += (e - s + 1)
        if e >= s
            first_real = i - 1
            for j in s:e
                idx_row_l[j] = permute[Int(idx_row_l[j])]
            end
        end
    end
    lu.base_l = first_real
    lu.nl = lu.ngood_l - first_real
    lu.max_u = max_u
    lu.total_elements = total_elements
    lu.factor_elements = total_elements

    # Size R storage & adjust area_factor (matching CoinFactorization1.cpp:2063-2090)
    space = lu.len_area_l - lu.len_l
    space_used = lu.len_l + lu.len_u
    needed = div(space_used + nr - 1, nr) * 2 * lu.max_pivots
    needed = max(needed, 2 * nr)
    if use_extra_r
        space = div(space, 2)
    end
    if space < needed
        lu.area_factor = lu.area_factor != 0.0 ? lu.area_factor * 1.1 : 1.1
    end

    lu.nr = 0
    lu.len_r = 0
    lu.len_area_r = max(space, 0)
    _ensure_len!(lu.element_r, lu.len_area_r + 2)
    _ensure_len!(lu.index_row_r, lu.len_area_r + 2)
    _ensure_len!(lu.start_col_r, lu.max_pivots + 2)
    @inbounds lu.start_col_r[1] = Int32(1)

    lu.has_row_r = use_extra_r
    if use_extra_r
        _ensure_len!(lu.element_row_r, lu.len_area_r + 2)
        _ensure_len!(lu.index_col_r, lu.len_area_r + 2)
        _ensure_len!(lu.start_row_r, col_end + 1)
        _ensure_zero!(lu.len_row_r, col_end + 1)
        @inbounds for i in 1:col_end
            lu.start_row_r[i] = Int32(1)
        end
    end
    return nothing
end

# =============================================================================
# Top-level factor!, finalize_basis_permutation!, go_sparse!, check_sparse!, get_weights!
# =============================================================================

"""
    factor!(lu::SparseLU) -> Int

Perform Markowitz sparse LU factorization (with dense LAPACK Accelerate fallback)
and post-factor cleanup. Returns `0` (OK), `-1` (singular), or `-99` (insufficient memory).
"""
function factor!(lu::SparseLU)::Int
    larger = max(lu.nrows, lu.ncols)
    st = if larger < _LU_SMALL_LIMIT
        _factor_sparse_impl!(lu, lu.mark_row16, _LU_SMALL_SET, _LU_SMALL_UNSET)
    else
        _factor_sparse_impl!(lu, lu.mark_row, _LU_LARGE_SET, _LU_LARGE_UNSET)
    end

    if st == 0
        lu.total_elements = 0
        if lu.ngood_u < lu.nrows
            return _factor_singular!(lu)
        end
    elseif st == 2
        st = _factor_dense!(lu)
        if st != 0
            return st
        end
    else
        return st
    end

    if lu.ncompressions > 10
        lu.area_factor *= 1.1
    end
    lu.ncompressions = 0
    _cleanup!(lu)
    return 0
end

"""
    finalize_basis_permutation!(lu::SparseLU)

Initialize `pivot_col = permute` and `pivot_col_back = permute_back` after a successful
factorization (matching `ClpFactorization::factorize` lines 2335-2336).
"""
function finalize_basis_permutation!(lu::SparseLU)::Nothing
    nr = lu.nrows
    @inbounds for i in 1:nr
        lu.pivot_col[i] = lu.permute[i]
        lu.pivot_col_back[i] = lu.permute_back[i]
    end
    return nothing
end

"""
    go_sparse!(lu::SparseLU)

Enable hypersparse BTRAN/FTRAN and build the row-wise copy of `L` (`CoinFactorization::goSparse`).
"""
function go_sparse!(lu::SparseLU)::Nothing
    nr = lu.nrows
    if lu.sparse_threshold == 0
        if nr > 300
            if nr < 10000
                lu.sparse_threshold = min(div(nr, 6), 500)
                lu.sparse_threshold2 = div(nr, 2)
            else
                lu.sparse_threshold = 500
                lu.sparse_threshold2 = max(500, nr >> 3)
            end
        else
            lu.sparse_threshold = 0
            lu.sparse_threshold2 = 0
        end
    else
        lu.sparse_threshold2 = lu.sparse_threshold
    end
    if lu.sparse_threshold == 0
        lu.has_sparse = false
        lu.has_row_l = false
        return nothing
    end

    mre = lu.max_rows_extra
    _ensure_len!(lu.sparse_stack, mre + 2)
    _ensure_len!(lu.sparse_list, mre + 2)
    _ensure_len!(lu.sparse_aux, mre + 2)
    _ensure_zero!(lu.sparse_mark, mre + 16)
    lu.has_sparse = true

    _ensure_zero!(lu.start_row_l, nr + 1)
    if lu.len_area_l > 0
        _ensure_len!(lu.element_row_l, lu.len_area_l + 2)
        _ensure_len!(lu.index_col_l, lu.len_area_l + 2)
    end
    lu.has_row_l = true

    start_row_l = lu.start_row_l
    start_col_l = lu.start_col_l
    el_l = lu.element_l
    idx_row_l = lu.index_row_l
    base = lu.base_l + 1
    end_col = lu.base_l + lu.nl

    @inbounds for i in base:end_col
        s = Int(start_col_l[i])
        e = Int(start_col_l[i + 1]) - 1
        for j in s:e
            start_row_l[Int(idx_row_l[j])] += Int32(1)
        end
    end

    cnt = 1
    @inbounds for i in 1:nr
        n_r = Int(start_row_l[i])
        cnt += n_r
        start_row_l[i] = Int32(cnt)
    end
    @inbounds start_row_l[nr + 1] = Int32(cnt)

    el_row_l = lu.element_row_l
    idx_col_l = lu.index_col_l
    @inbounds for i in end_col:-1:base
        s = Int(start_col_l[i])
        e = Int(start_col_l[i + 1]) - 1
        for j in s:e
            ir = Int(idx_row_l[j])
            pos = Int(start_row_l[ir]) - 1
            start_row_l[ir] = Int32(pos)
            el_row_l[pos] = el_l[j]
            idx_col_l[pos] = Int32(i)
        end
    end
    return nothing
end

"""
    check_sparse!(lu::SparseLU)

Update running hypersparse density statistics between factorizations (`CoinFactorization::checkSparse`).
"""
function check_sparse!(lu::SparseLU)::Nothing
    if lu.nftran_counts > 100
        lu.ftran_count_input = max(lu.ftran_count_input, 1.0)
        lu.ftran_avg_after_l = max(lu.ftran_count_after_l / lu.ftran_count_input, 1.0)
        lu.ftran_avg_after_r = max(lu.ftran_count_after_r / lu.ftran_count_after_l, 1.0)
        lu.ftran_avg_after_u = max(lu.ftran_count_after_u / lu.ftran_count_after_r, 1.0)
        if lu.btran_count_input != 0.0 && lu.btran_count_after_u != 0.0 && lu.btran_count_after_r != 0.0
            lu.btran_avg_after_u = max(lu.btran_count_after_u / lu.btran_count_input, 1.0)
            lu.btran_avg_after_r = max(lu.btran_count_after_r / lu.btran_count_after_u, 1.0)
            lu.btran_avg_after_l = max(lu.btran_count_after_l / lu.btran_count_after_r, 1.0)
        else
            lu.btran_avg_after_u = 1.0
            lu.btran_avg_after_r = 1.0
            lu.btran_avg_after_l = 1.0
        end
    end
    lu.ftran_count_input *= 0.8
    lu.ftran_count_after_l *= 0.8
    lu.ftran_count_after_r *= 0.8
    lu.ftran_count_after_u *= 0.8
    lu.btran_count_input *= 0.8
    lu.btran_count_after_u *= 0.8
    lu.btran_count_after_r *= 0.8
    lu.btran_count_after_l *= 0.8
    return nothing
end

"""
    reset_statistics!(lu::SparseLU)

Reset all hypersparse running counters and averages (`CoinFactorization::resetStatistics`).
"""
function reset_statistics!(lu::SparseLU)::Nothing
    lu.ftran_count_input = 0.0
    lu.ftran_count_after_l = 0.0
    lu.ftran_count_after_r = 0.0
    lu.ftran_count_after_u = 0.0
    lu.btran_count_input = 0.0
    lu.btran_count_after_u = 0.0
    lu.btran_count_after_r = 0.0
    lu.btran_count_after_l = 0.0
    lu.nftran_counts = 0
    lu.nbtran_counts = 0
    lu.ftran_avg_after_l = 0.0
    lu.ftran_avg_after_r = 0.0
    lu.ftran_avg_after_u = 0.0
    lu.btran_avg_after_u = 0.0
    lu.btran_avg_after_r = 0.0
    lu.btran_avg_after_l = 0.0
    return nothing
end

"""
    get_weights!(lu::SparseLU, weights::AbstractVector{Int32})

Compute initial steepest-edge row-density weights (`ClpFactorization::getWeights`).
"""
function get_weights!(lu::SparseLU, weights::AbstractVector{Int32})::Nothing
    nr = lu.nrows
    perm_back = lu.pivot_col_back
    if !lu.has_row_l
        temp = lu.sparse_aux
        _ensure_zero!(temp, nr)
        start_col_u = lu.start_col_u
        len_col_u = lu.len_col_u
        idx_row_u = lu.index_row_u
        @inbounds for i in 1:nr
            temp[i] += Int32(1)
            s = Int(start_col_u[i])
            e = s + Int(len_col_u[i]) - 1
            for j in s:e
                temp[Int(idx_row_u[j])] += Int32(1)
            end
        end
        start_col_l = lu.start_col_l
        idx_row_l = lu.index_row_l
        base = lu.base_l + 1
        @inbounds for i in base:(lu.base_l + lu.nl)
            for j in Int(start_col_l[i]):(Int(start_col_l[i + 1]) - 1)
                temp[Int(idx_row_l[j])] += Int32(1)
            end
        end
        @inbounds for i in 1:nr
            weights[Int(perm_back[i])] = temp[i]
        end
    else
        start_row_l = lu.start_row_l
        len_row_u = lu.len_row_u
        @inbounds for i in 1:nr
            num = (start_row_l[i + 1] - start_row_l[i]) + len_row_u[i] + Int32(1)
            weights[Int(perm_back[i])] = num
        end
    end
    return nothing
end

"""
    load_csc!(lu::SparseLU, nrows, ncols, col_starts, col_lens, row_indices, values;
              zero_based=false, row_counts=nothing)

Load a basis matrix in CSC format into `lu` (shifting indices 0-based -> 1-based if
`zero_based=true`). Used both by pure Julia callers and by the thin C++ bridge.
"""
function load_csc!(
    lu::SparseLU,
    nrows::Int,
    ncols::Int,
    nnz::Int,
    col_starts::AbstractVector{Int32},
    col_lens::AbstractVector{Int32},
    row_indices::AbstractVector{Int32},
    values::AbstractVector{Float64},
    row_counts::Union{Nothing, AbstractVector{Int32}} = nothing;
    zero_based::Bool = false,
)::Nothing
    el_u = lu.element_u
    idx_row = lu.index_row_u
    start_col = lu.start_col_u
    len_col = lu.len_col_u
    len_row = lu.len_row_u

    lu.len_u = nnz
    if zero_based
        @inbounds for i in 1:nnz
            el_u[i] = values[i]
            idx_row[i] = row_indices[i] + Int32(1)
        end
        @inbounds for j in 1:ncols
            start_col[j] = col_starts[j] + Int32(1)
            len_col[j] = col_lens[j]
        end
    else
        @inbounds for i in 1:nnz
            el_u[i] = values[i]
            idx_row[i] = row_indices[i]
        end
        @inbounds for j in 1:ncols
            start_col[j] = col_starts[j]
            len_col[j] = col_lens[j]
        end
    end
    if row_counts !== nothing
        @inbounds for i in 1:nrows
            len_row[i] = row_counts[i]
        end
    end
    return nothing
end

"""
    load_triplets!(lu::SparseLU, nrows, ncols, nnz, rows, cols, values; zero_based=false)

Load a basis matrix in COO triplet format into `lu` before `preprocess!(lu, 0)`.
"""
function load_triplets!(
    lu::SparseLU,
    nrows::Int,
    ncols::Int,
    nnz::Int,
    rows::AbstractVector{Int32},
    cols::AbstractVector{Int32},
    values::AbstractVector{Float64};
    zero_based::Bool = false,
)::Nothing
    el_u = lu.element_u
    idx_row = lu.index_row_u
    idx_col = lu.index_col_u
    lu.len_u = nnz
    if zero_based
        @inbounds for i in 1:nnz
            el_u[i] = values[i]
            idx_row[i] = rows[i] + Int32(1)
            idx_col[i] = cols[i] + Int32(1)
        end
    else
        @inbounds for i in 1:nnz
            el_u[i] = values[i]
            idx_row[i] = rows[i]
            idx_col[i] = cols[i]
        end
    end
    return nothing
end

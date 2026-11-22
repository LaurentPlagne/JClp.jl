# Faithful translation of CoinUtils/src/CoinFactorization4.cpp — B5b Forrest-Tomlin `replaceColumn`
# (`getRowSpaceIterate`, `getColumnSpaceIterateR`, `checkPivot`, `replaceColumn`).
# Zero-allocation, hardware arm64 FMA (`_fmsub`), `--trim=safe` compatible.

# Mirror of CoinFactorizationReplaceState as immutable isbits struct for zero-alloc C pointer load/store
struct CoinFactorizationReplaceStatePod
    nextRow::Ptr{Int32}
    lastRow::Ptr{Int32}
    relaxCheck::Float64
    numberRowsExtra::Int32
    numberColumnsExtra::Int32
    numberU::Int32
    numberGoodU::Int32
    numberPivots::Int32
    numberR::Int32
    lengthR::Int32
    lengthU::Int32
    maximumU::Int32
    totalElements::Int32
    factorElements::Int32
    lengthAreaL::Int32
    numberCompressions::Int32
    status::Int32
    dropNumberInColumnPlus::Int32
    _pad::Int32
end

# --- checkPivot (CoinFactorization4.cpp:205-256) ---
@inline function cf_check_pivot(
    save_from_u::Float64,
    old_pivot::Float64,
    number_rows::Int32,
    number_rows_extra::Int32,
    relax_check::Float64,
)::Int32
    if abs(save_from_u) > 1.0e-9
        check_tol = if number_rows_extra < number_rows + Int32(2)
            1.0e-5
        elseif number_rows_extra < number_rows + Int32(10)
            1.0e-6
        elseif number_rows_extra < number_rows + Int32(50)
            1.0e-8
        else
            1.0e-10
        end
        check_tol *= relax_check
        if abs(1.0 - abs(save_from_u / old_pivot)) < check_tol
            return Int32(0)
        elseif abs(abs(old_pivot) - abs(save_from_u)) < 1.0e-12 ||
               abs(1.0 - abs(save_from_u / old_pivot)) < 1.0e-8
            return Int32(1)
        else
            return Int32(2)
        end
    else
        if abs(1.0 - abs(save_from_u / old_pivot)) < 1.0e-10
            return Int32(0)
        else
            return Int32(2)
        end
    end
end

# --- getRowSpaceIterate (CoinFactorization4.cpp:62-130) ---
# Returns (ok::Bool, new_number_compressions::Int32, new_status::Int32)
function cf_get_row_space_iterate!(
    cf::CoinFactorizationPodView,
    next_row::Ptr{Int32},
    last_row::Ptr{Int32},
    i_row::Int32,
    extra_needed::Int32,
    n_comp::Int32,
    cur_status::Int32,
)::Tuple{Bool,Int32,Int32}
    num_in_row = cf.numberInRow
    number = unsafe_load(num_in_row, Int(i_row) + 1)
    start_row = cf.startRowU
    idx_col = cf.indexColumnU
    conv = cf.convertRowToColumnU
    mre = Int(cf.maximumRowsExtra)
    space = cf.lengthAreaU - unsafe_load(start_row, mre + 1)

    if space < extra_needed + number + Int32(2)
        # Compression
        r = Int(unsafe_load(next_row, mre + 1))
        put = Int32(0)
        while r != mre
            g = Int(unsafe_load(start_row, r + 1))
            ge = g + Int(unsafe_load(num_in_row, r + 1))
            unsafe_store!(start_row, put, r + 1)
            i = g
            while i < ge
                unsafe_store!(idx_col, unsafe_load(idx_col, i + 1), Int(put) + 1)
                unsafe_store!(conv, unsafe_load(conv, i + 1), Int(put) + 1)
                put += Int32(1)
                i += 1
            end
            r = Int(unsafe_load(next_row, r + 1))
        end
        n_comp += Int32(1)
        unsafe_store!(start_row, put, mre + 1)
        space = cf.lengthAreaU - put
        if space < extra_needed + number + Int32(2)
            return false, n_comp, Int32(-99)
        end
    end

    put = unsafe_load(start_row, mre + 1)
    ir = Int(i_row)
    nxt = unsafe_load(next_row, ir + 1)
    lst = unsafe_load(last_row, ir + 1)

    # Remove i_row from doubly linked list
    unsafe_store!(next_row, nxt, Int(lst) + 1)
    unsafe_store!(last_row, lst, Int(nxt) + 1)
    # Insert at end
    lst2 = unsafe_load(last_row, mre + 1)
    unsafe_store!(next_row, i_row, Int(lst2) + 1)
    unsafe_store!(last_row, i_row, mre + 1)
    unsafe_store!(last_row, lst2, ir + 1)
    unsafe_store!(next_row, Int32(mre), ir + 1)

    # Move existing entries
    g = Int(unsafe_load(start_row, ir + 1))
    unsafe_store!(start_row, put, ir + 1)
    rem = Int(number)
    while rem > 0
        rem -= 1
        unsafe_store!(idx_col, unsafe_load(idx_col, g + 1), Int(put) + 1)
        unsafe_store!(conv, unsafe_load(conv, g + 1), Int(put) + 1)
        put += Int32(1)
        g += 1
    end
    unsafe_store!(start_row, put + extra_needed + Int32(4), mre + 1)
    return true, n_comp, cur_status
end

# --- getColumnSpaceIterateR (CoinFactorization4.cpp:135-204) ---
# Returns (ok::Bool, new_number_compressions::Int32)
function cf_get_column_space_iterate_r!(
    cf::CoinFactorizationPodView,
    i_col::Int32,
    value::Float64,
    i_row::Int32,
    n_comp::Int32,
)::Tuple{Bool,Int32}
    el_r = cf.elementR + Int(cf.lengthAreaR) * sizeof(Float64)
    idx_r = cf.indexRowR + Int(cf.lengthAreaR) * sizeof(Int32)
    start_r = cf.startColumnR + (Int(cf.maximumPivots) + 1) * sizeof(Int32)
    nicp = cf.numberInColumnPlus
    ic = Int(i_col)
    number = unsafe_load(nicp, ic + 1)
    next_col = cf.nextColumn
    last_col = cf.lastColumn
    mce = Int(cf.maximumColumnsExtra)
    lar = cf.lengthAreaR

    if lar - unsafe_load(start_r, mce + 1) < number + Int32(1)
        # Compression
        jc = Int(unsafe_load(next_col, mce + 1))
        put = Int32(0)
        while jc != mce
            g = Int(unsafe_load(start_r, jc + 1))
            ge = g + Int(unsafe_load(nicp, jc + 1))
            unsafe_store!(start_r, put, jc + 1)
            i = g
            while i < ge
                unsafe_store!(idx_r, unsafe_load(idx_r, i + 1), Int(put) + 1)
                unsafe_store!(el_r, unsafe_load(el_r, i + 1), Int(put) + 1)
                put += Int32(1)
                i += 1
            end
            jc = Int(unsafe_load(next_col, jc + 1))
        end
        n_comp += Int32(1)
        unsafe_store!(start_r, put, mce + 1)
    end

    if lar - unsafe_load(start_r, mce + 1) < number + Int32(1)
        return false, n_comp
    end

    nxt = unsafe_load(next_col, ic + 1)
    lst = unsafe_load(last_col, ic + 1)
    unsafe_store!(next_col, nxt, Int(lst) + 1)
    unsafe_store!(last_col, lst, Int(nxt) + 1)

    put = unsafe_load(start_r, mce + 1)
    lst2 = unsafe_load(last_col, mce + 1)
    unsafe_store!(next_col, i_col, Int(lst2) + 1)
    unsafe_store!(last_col, i_col, mce + 1)
    unsafe_store!(last_col, lst2, ic + 1)
    unsafe_store!(next_col, Int32(mce), ic + 1)

    g = Int(unsafe_load(start_r, ic + 1))
    unsafe_store!(start_r, put, ic + 1)
    i = 0
    num_int = Int(number)
    while i < num_int
        unsafe_store!(el_r, unsafe_load(el_r, g + 1), Int(put) + 1)
        unsafe_store!(idx_r, unsafe_load(idx_r, g + 1), Int(put) + 1)
        put += Int32(1)
        g += 1
        i += 1
    end
    unsafe_store!(el_r, value, Int(put) + 1)
    unsafe_store!(idx_r, i_row, Int(put) + 1)
    put += Int32(1)
    unsafe_store!(nicp, number + Int32(1), ic + 1)
    end_pos = put + Int32(4)
    unsafe_store!(start_r, end_pos < lar ? end_pos : lar, mce + 1)
    return true, n_comp
end

# Helper to write back CoinFactorizationReplaceStatePod
@inline function _store_replace_state!(
    st_ptr::Ptr{CoinFactorizationReplaceStatePod},
    st::CoinFactorizationReplaceStatePod,
    nre::Int32, nce::Int32, nu::Int32, ngu::Int32, np::Int32,
    nr_cnt::Int32, lr::Int32, lu::Int32, max_u::Int32,
    tot_el::Int32, n_comp::Int32, cf_status::Int32, drop_nicp::Int32,
)::Nothing
    unsafe_store!(st_ptr, CoinFactorizationReplaceStatePod(
        st.nextRow, st.lastRow, st.relaxCheck,
        nre, nce, nu, ngu, np, nr_cnt, lr, lu, max_u,
        tot_el, st.factorElements, st.lengthAreaL, n_comp, cf_status, drop_nicp, Int32(0),
    ))
    return nothing
end

# --- Top-level B5b: CoinFactorization::replaceColumn (CoinFactorization4.cpp:290-741) ---
function cf_replace_column!(
    cf::CoinFactorizationPodView,
    st_ptr::Ptr{CoinFactorizationReplaceStatePod},
    reg_ptr::Ptr{CoinIndexedVecPod},
    pivot_row::Int32,
    pivot_check_in::Float64,
    check_before_modifying::Bool,
)::Int32
    st = unsafe_load(st_ptr)
    r = unsafe_load(reg_ptr)
    start_col_u = cf.startColumnU

    nre = st.numberRowsExtra
    nce = st.numberColumnsExtra
    nu = st.numberU
    ngu = st.numberGoodU
    np = st.numberPivots
    nr_cnt = st.numberR
    lr = st.lengthR
    lu = st.lengthU
    max_u = st.maximumU
    tot_el = st.totalElements
    n_comp = st.numberCompressions
    cf_status = st.status
    drop_nicp = Int32(0)

    if nce >= cf.maximumColumnsExtra
        return Int32(5)
    end
    if cf.lengthAreaU < unsafe_load(start_col_u, Int(cf.maximumColumnsExtra) + 1)
        return Int32(3)
    end

    num_in_row = cf.numberInRow
    num_in_col = cf.numberInColumn
    nicp = cf.numberInColumnPlus
    real_piv_row = unsafe_load(cf.pivotColumn, Int(pivot_row) + 1)
    rpr = Int(real_piv_row)

    region = r.elements
    region_index = r.indices
    el_u_all = cf.elementU

    check_before = check_before_modifying || (np == Int32(0))

    tot_el -= unsafe_load(num_in_col, rpr + 1)
    piv_reg = cf.pivotRegion
    old_pivot = unsafe_load(piv_reg, rpr + 1)
    pivot_check = pivot_check_in / old_pivot
    unsafe_store!(piv_reg, 0.0, rpr + 1)

    save_end = unsafe_load(start_col_u, rpr + 1) + unsafe_load(num_in_col, rpr + 1)
    unsafe_store!(num_in_col, Int32(0), rpr + 1)

    idx_col = cf.indexColumnU
    conv = cf.convertRowToColumnU
    start_row = cf.startRowU

    s_row = Int(unsafe_load(start_row, rpr + 1))
    e_row = s_row + Int(unsafe_load(num_in_row, rpr + 1))
    smallest_idx = nre
    nnz = Int32(0)

    if !check_before
        i = s_row
        while i < e_row
            icol = unsafe_load(idx_col, i + 1)
            if icol < smallest_idx
                smallest_idx = icol
            end
            j = Int(unsafe_load(conv, i + 1))
            unsafe_store!(region, unsafe_load(el_u_all, j + 1), Int(icol) + 1)
            unsafe_store!(el_u_all, 0.0, j + 1)
            nnz += Int32(1)
            unsafe_store!(region_index, icol, Int(nnz))
            i += 1
        end
    else
        i = s_row
        while i < e_row
            icol = unsafe_load(idx_col, i + 1)
            if icol < smallest_idx
                smallest_idx = icol
            end
            j = Int(unsafe_load(conv, i + 1))
            unsafe_store!(region, unsafe_load(el_u_all, j + 1), Int(icol) + 1)
            nnz += Int32(1)
            unsafe_store!(region_index, icol, Int(nnz))
            i += 1
        end
    end

    # BTRAN U directly in Julia!
    nnz = cf_update_column_transpose_u!(cf, region, region_index, nnz, smallest_idx)

    save_from_u = 0.0
    start_u = Int(unsafe_load(start_col_u, Int(nce) + 1))
    idx_u = cf.indexRowU + start_u * sizeof(Int32)
    el_u = cf.elementU + start_u * sizeof(Float64)

    if check_before
        tol = cf.zeroTolerance
        num_c = Int(unsafe_load(num_in_col, Int(nce) + 1))
        i = 1
        while i <= num_c
            ir = unsafe_load(idx_u, i)
            ev = unsafe_load(el_u, i)
            if abs(ev) > tol
                if ir != real_piv_row
                    save_from_u = _fmsub(ev, unsafe_load(region, Int(ir) + 1), save_from_u)
                else
                    save_from_u += ev
                end
            end
            i += 1
        end
        chk_st = cf_check_pivot(save_from_u, pivot_check, cf.numberRows, nre, st.relaxCheck)
        if chk_st != Int32(0)
            # Rollback
            unsafe_store!(piv_reg, old_pivot, rpr + 1)
            restored = save_end - unsafe_load(start_col_u, rpr + 1)
            tot_el += restored
            unsafe_store!(num_in_col, restored, rpr + 1)
            civ_clear_ptr!(region, region_index, nnz, r.capacity, false)
            _store_replace_state!(st_ptr, st, nre, nce, nu, ngu, np, nr_cnt, lr, lu, max_u, tot_el, n_comp, cf_status, drop_nicp)
            return chk_st
        else
            i = s_row
            while i < e_row
                j = Int(unsafe_load(conv, i + 1))
                unsafe_store!(el_u_all, 0.0, j + 1)
                i += 1
            end
        end
    end

    # Zero out old pivot column in U
    i = Int(unsafe_load(start_col_u, rpr + 1))
    se = Int(save_end)
    while i < se
        unsafe_store!(el_u_all, 0.0, i + 1)
        i += 1
    end

    # Add to R
    start_col_r = cf.startColumnR
    idx_row_r = cf.indexRowR
    el_r = cf.elementR
    l_pos = lr
    unsafe_store!(start_col_r, l_pos, Int(nr_cnt) + 1)
    nr_cnt += Int32(1)
    unsafe_store!(start_col_r, l_pos + nnz, Int(nr_cnt) + 1)
    lr = l_pos + nnz
    tot_el += nnz
    if lr >= cf.lengthAreaR
        civ_clear_ptr!(region, region_index, nnz, r.capacity, false)
        _store_replace_state!(st_ptr, st, nre, nce, nu, ngu, np, nr_cnt, lr, lu, max_u, tot_el, n_comp, cf_status, drop_nicp)
        return Int32(3)
    end

    n_nz = Int(nnz)
    l_idx = Int(l_pos)
    i = 1
    while i <= n_nz
        ir = unsafe_load(region_index, i)
        unsafe_store!(idx_row_r, ir, l_idx + 1)
        unsafe_store!(el_r, unsafe_load(region, Int(ir) + 1), l_idx + 1)
        l_idx += 1
        i += 1
    end

    # Remove old pivot row from linked list
    next_row = st.nextRow
    last_row = st.lastRow
    nxt_r = unsafe_load(next_row, rpr + 1)
    lst_r = unsafe_load(last_row, rpr + 1)
    unsafe_store!(next_row, nxt_r, Int(lst_r) + 1)
    unsafe_store!(last_row, lst_r, Int(nxt_r) + 1)
    unsafe_store!(num_in_row, Int32(0), rpr + 1)

    # Permute updates
    unsafe_store!(cf.permute, real_piv_row, Int(nre) + 1)
    unsafe_store!(cf.permuteBack, nre, rpr + 1)
    unsafe_store!(cf.permuteBack, Int32(-1), Int(nre) + 1)
    unsafe_store!(cf.permute, Int32(0), Int(nre) + 2)

    unsafe_store!(cf.pivotColumn, nre, Int(pivot_row) + 1)
    unsafe_store!(cf.pivotColumnBack, pivot_row, Int(nre) + 1)

    nu += Int32(1)
    num_c = unsafe_load(num_in_col, Int(nce) + 1)
    tot_el += num_c
    lu += num_c
    if lu >= cf.lengthAreaU
        civ_clear_ptr!(region, region_index, nnz, r.capacity, false)
        _store_replace_state!(st_ptr, st, nre, nce, nu, ngu, np, nr_cnt, lr, lu, max_u, tot_el, n_comp, cf_status, drop_nicp)
        return Int32(3)
    end

    save_from_u = 0.0
    nc_int = Int(num_c)
    i = 0
    while i < nc_int
        ir = unsafe_load(idx_u, i + 1)
        ev = unsafe_load(el_u, i + 1)
        if ir != real_piv_row
            ir_int = Int(ir)
            nxt2 = unsafe_load(next_row, ir_int + 1)
            inir = unsafe_load(num_in_row, ir_int + 1)
            put = unsafe_load(start_row, ir_int + 1) + inir
            sp = unsafe_load(start_row, Int(nxt2) + 1) - put
            if sp <= Int32(0)
                _, n_comp, cf_status = cf_get_row_space_iterate!(cf, next_row, last_row, ir, inir + Int32(4), n_comp, cf_status)
                put = unsafe_load(start_row, ir_int + 1) + inir
            end
            unsafe_store!(idx_col, nce, Int(put) + 1)
            unsafe_store!(conv, Int32(i + start_u), Int(put) + 1)
            unsafe_store!(num_in_row, inir + Int32(1), ir_int + 1)
            save_from_u = _fmsub(ev, unsafe_load(region, ir_int + 1), save_from_u)
        else
            save_from_u += ev
            unsafe_store!(el_u, 0.0, i + 1)
        end
        i += 1
    end

    # Insert new row nre at end of row list
    mre = Int(cf.maximumRowsExtra)
    lst_end = unsafe_load(last_row, mre + 1)
    unsafe_store!(next_row, nre, Int(lst_end) + 1)
    unsafe_store!(last_row, nre, mre + 1)
    unsafe_store!(last_row, lst_end, Int(nre) + 1)
    unsafe_store!(next_row, Int32(mre), Int(nre) + 1)
    unsafe_store!(start_row, unsafe_load(start_row, mre + 1), Int(nre) + 1)
    unsafe_store!(num_in_row, Int32(0), Int(nre) + 1)

    # Insert new column nce at beginning of column list
    next_col = cf.nextColumn
    last_col = cf.lastColumn
    mce = Int(cf.maximumColumnsExtra)
    nxt_c = unsafe_load(next_col, mce + 1)
    unsafe_store!(last_col, nce, Int(nxt_c) + 1)
    unsafe_store!(next_col, nce, mce + 1)
    unsafe_store!(next_col, nxt_c, Int(nce) + 1)
    unsafe_store!(last_col, Int32(mce), Int(nce) + 1)

    ret_status = check_before ? Int32(0) : cf_check_pivot(save_from_u, pivot_check, cf.numberRows, nre, st.relaxCheck)

    if ret_status != Int32(2)
        piv_val = 1.0 / save_from_u
        unsafe_store!(piv_reg, piv_val, Int(nre) + 1)
        i = 1
        while i <= nc_int
            unsafe_store!(el_u, unsafe_load(el_u, i) * piv_val, i)
            i += 1
        end
        cand_max_u = Int32(start_u) + num_c
        if cand_max_u > max_u
            max_u = cand_max_u
        end
        nre += Int32(1)
        nce += Int32(1)
        ngu += Int32(1)
        np += Int32(1)
    end

    # Fill-in refactorization heuristic (CoinFactorization4.cpp:670-692)
    nr = cf.numberRows
    if nre > nr + Int32(50)
        extra = st.factorElements >> 1
        if nre > nr + Int32(100) + div(nr, Int32(500))
            if extra < Int32(2) * nr
                extra = Int32(2) * nr
            end
        else
            if extra < Int32(5) * nr
                extra = Int32(5) * nr
            end
        end
        added = tot_el - st.factorElements
        if added > extra && added > (st.factorElements << 1) && ret_status == Int32(0) &&
           Int32(3) * tot_el > Int32(2) * (cf.lengthAreaU + st.lengthAreaL)
            ret_status = Int32(3)
        end
    end

    # Maintain transpose copy of R in second half of R area
    if nicp != C_NULL && ret_status < Int32(2)
        el_r2 = cf.elementR + Int(cf.lengthAreaR) * sizeof(Float64)
        idx_r2 = cf.indexRowR + Int(cf.lengthAreaR) * sizeof(Int32)
        start_r2 = cf.startColumnR + (Int(cf.maximumPivots) + 1) * sizeof(Int32)
        piv_r_new = nre - Int32(1)
        lar = cf.lengthAreaR
        cleared_all = false
        i = 1
        while i <= n_nz
            ir = unsafe_load(region_index, i)
            ir_int = Int(ir)
            nxt_col = unsafe_load(next_col, ir_int + 1)
            sp = if Int(nxt_col) != mce
                unsafe_load(start_r2, Int(nxt_col) + 1) - unsafe_load(start_r2, ir_int + 1)
            else
                lar - unsafe_load(start_r2, ir_int + 1)
            end
            nin_r = unsafe_load(nicp, ir_int + 1)
            if sp > nin_r
                put = unsafe_load(start_r2, ir_int + 1) + nin_r
                unsafe_store!(nicp, nin_r + Int32(1), ir_int + 1)
                unsafe_store!(idx_r2, piv_r_new, Int(put) + 1)
                unsafe_store!(el_r2, unsafe_load(region, ir_int + 1), Int(put) + 1)
                if Int(nxt_col) == mce
                    ep = put + Int32(4)
                    unsafe_store!(start_r2, ep < lar ? ep : lar, mce + 1)
                end
            else
                ok, n_comp = cf_get_column_space_iterate_r!(cf, ir, unsafe_load(region, ir_int + 1), piv_r_new, n_comp)
                if !ok
                    drop_nicp = Int32(1)
                    civ_clear_ptr!(region, region_index, nnz, r.capacity, false)
                    cleared_all = true
                    break
                end
            end
            unsafe_store!(region, 0.0, ir_int + 1)
            i += 1
        end
        _ = cleared_all
    else
        civ_clear_ptr!(region, region_index, nnz, r.capacity, false)
    end

    _store_replace_state!(st_ptr, st, nre, nce, nu, ngu, np, nr_cnt, lr, lu, max_u, tot_el, n_comp, cf_status, drop_nicp)
    return ret_status
end

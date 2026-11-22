# Faithful translation of CoinUtils Presolve B7a:
#   - CoinPresolveHelperFunctions.cpp
#   - CoinPresolveMatrix.{hpp,cpp} core helpers
#   - CoinPresolveZeros.cpp
#   - CoinPresolveEmpty.cpp
#   - CoinPresolveIsolated.cpp
#   - CoinPresolveUseless.cpp
#   - CoinPresolveFixed.cpp
# Zero-allocation, bit-identical arm64 FMA (`_fmadd`/`_fmsub`), `--trim=safe` compatible.

# =============================================================================
# 1. Core Linked-List, Status, ToDo, and Matrix Helpers
# =============================================================================

@inline function ps_remove_link!(lnk::Ptr{PresolveHLinkPod}, i::Int32)::Nothing
    i1 = Int(i) + 1
    cur = unsafe_load(lnk, i1)
    ipre = cur.pre
    isuc = cur.suc
    if ipre >= Int32(0)
        p1 = Int(ipre) + 1
        unsafe_store!(lnk, PresolveHLinkPod(unsafe_load(lnk, p1).pre, isuc), p1)
    end
    if isuc >= Int32(0)
        s1 = Int(isuc) + 1
        unsafe_store!(lnk, PresolveHLinkPod(ipre, unsafe_load(lnk, s1).suc), s1)
    end
    unsafe_store!(lnk, PresolveHLinkPod(PRESOLVE_NO_LINK, PRESOLVE_NO_LINK), i1)
    return nothing
end

@inline function ps_insert_link!(lnk::Ptr{PresolveHLinkPod}, i::Int32, j::Int32)::Nothing
    j1 = Int(j) + 1
    lj = unsafe_load(lnk, j1)
    isuc = lj.suc
    unsafe_store!(lnk, PresolveHLinkPod(lj.pre, i), j1)
    unsafe_store!(lnk, PresolveHLinkPod(j, isuc), Int(i) + 1)
    if isuc >= Int32(0)
        s1 = Int(isuc) + 1
        unsafe_store!(lnk, PresolveHLinkPod(i, unsafe_load(lnk, s1).suc), s1)
    end
    return nothing
end

@inline function ps_move_link!(lnk::Ptr{PresolveHLinkPod}, i::Int32, j::Int32)::Nothing
    i1 = Int(i) + 1
    cur = unsafe_load(lnk, i1)
    ipre = cur.pre
    isuc = cur.suc
    if ipre >= Int32(0)
        p1 = Int(ipre) + 1
        unsafe_store!(lnk, PresolveHLinkPod(unsafe_load(lnk, p1).pre, j), p1)
    end
    if isuc >= Int32(0)
        s1 = Int(isuc) + 1
        unsafe_store!(lnk, PresolveHLinkPod(j, unsafe_load(lnk, s1).suc), s1)
    end
    unsafe_store!(lnk, PresolveHLinkPod(PRESOLVE_NO_LINK, PRESOLVE_NO_LINK), i1)
    return nothing
end

@inline function ps_set_row_status!(rowstat::Ptr{UInt8}, seq::Int32, st::UInt8)::Nothing
    s1 = Int(seq) + 1
    b = (unsafe_load(rowstat, s1) & ~UInt8(7)) | st
    unsafe_store!(rowstat, b, s1)
    return nothing
end

@inline function ps_set_col_status!(colstat::Ptr{UInt8}, seq::Int32, st::UInt8)::Nothing
    s1 = Int(seq) + 1
    b = (unsafe_load(colstat, s1) & ~UInt8(7)) | st
    unsafe_store!(colstat, b, s1)
    return nothing
end

@inline function ps_set_row_status_using_value!(
    rowstat::Ptr{UInt8},
    acts::Ptr{Float64},
    rlo::Ptr{Float64},
    rup::Ptr{Float64},
    ztolzb::Float64,
    i_row::Int32,
)::Nothing
    i1 = Int(i_row) + 1
    val = unsafe_load(acts, i1)
    lo = unsafe_load(rlo, i1)
    hi = unsafe_load(rup, i1)
    st = if lo < -1.0e20 && hi > 1.0e20
        PS_STATUS_IS_FREE
    elseif abs(lo - val) <= ztolzb
        PS_STATUS_AT_LOWER_BOUND
    elseif abs(hi - val) <= ztolzb
        PS_STATUS_AT_UPPER_BOUND
    else
        PS_STATUS_SUPER_BASIC
    end
    ps_set_row_status!(rowstat, i_row, st)
    return nothing
end

@inline function ps_set_col_status_using_value!(
    colstat::Ptr{UInt8},
    sol::Ptr{Float64},
    clo::Ptr{Float64},
    cup::Ptr{Float64},
    ztolzb::Float64,
    i_col::Int32,
)::Nothing
    i1 = Int(i_col) + 1
    val = unsafe_load(sol, i1)
    lo = unsafe_load(clo, i1)
    hi = unsafe_load(cup, i1)
    st = if lo < -1.0e20 && hi > 1.0e20
        PS_STATUS_IS_FREE
    elseif abs(lo - val) <= ztolzb
        PS_STATUS_AT_LOWER_BOUND
    elseif abs(hi - val) <= ztolzb
        PS_STATUS_AT_UPPER_BOUND
    else
        PS_STATUS_SUPER_BASIC
    end
    ps_set_col_status!(colstat, i_col, st)
    return nothing
end

@inline function ps_col_prohibited2(any_prohib::Int32, col_changed::Ptr{UInt8}, j::Int32)::Bool
    any_prohib == Int32(0) && return false
    return (unsafe_load(col_changed, Int(j) + 1) & UInt8(2)) != UInt8(0)
end

@inline function ps_row_changed(row_changed::Ptr{UInt8}, i::Int32)::Bool
    return (unsafe_load(row_changed, Int(i) + 1) & UInt8(1)) != UInt8(0)
end

@inline function ps_add_col!(
    col_changed::Ptr{UInt8},
    next_cols::Ptr{Int32},
    n_next_cols::Int32,
    j::Int32,
)::Int32
    j1 = Int(j) + 1
    b = unsafe_load(col_changed, j1)
    if (b & UInt8(1)) == UInt8(0)
        unsafe_store!(col_changed, b | UInt8(1), j1)
        unsafe_store!(next_cols, j, Int(n_next_cols) + 1)
        n_next_cols += Int32(1)
    end
    return n_next_cols
end

@inline function ps_add_row!(
    row_changed::Ptr{UInt8},
    next_rows::Ptr{Int32},
    n_next_rows::Int32,
    i::Int32,
)::Int32
    i1 = Int(i) + 1
    b = unsafe_load(row_changed, i1)
    if (b & UInt8(1)) == UInt8(0)
        unsafe_store!(row_changed, b | UInt8(1), i1)
        unsafe_store!(next_rows, i, Int(n_next_rows) + 1)
        n_next_rows += Int32(1)
    end
    return n_next_rows
end

# --- CoinPresolveHelperFunctions.cpp ---

function ps_compact_rep!(
    elems::Ptr{Float64},
    indices::Ptr{Int32},
    starts::Ptr{Int32},
    lengths::Ptr{Int32},
    n::Int32,
    lnk::Ptr{PresolveHLinkPod},
)::Nothing
    i = n
    while true
        pre = unsafe_load(lnk, Int(i) + 1).pre
        pre == PRESOLVE_NO_LINK && break
        i = pre
    end
    j = Int32(0)
    while i != n
        i1 = Int(i) + 1
        s = Int(unsafe_load(starts, i1))
        e = s + Int(unsafe_load(lengths, i1))
        unsafe_store!(starts, j, i1)
        k = s
        while k < e
            j1 = Int(j) + 1
            unsafe_store!(elems, unsafe_load(elems, k + 1), j1)
            unsafe_store!(indices, unsafe_load(indices, k + 1), j1)
            j += Int32(1)
            k += 1
        end
        i = unsafe_load(lnk, i1).suc
    end
    return nothing
end

function ps_make_memlists!(
    lengths::Ptr{Int32},
    lnk::Ptr{PresolveHLinkPod},
    n::Int32,
)::Nothing
    pre = PRESOLVE_NO_LINK
    ni = Int(n)
    i = 0
    while i < ni
        i1 = i + 1
        if unsafe_load(lengths, i1) != Int32(0)
            unsafe_store!(lnk, PresolveHLinkPod(pre, unsafe_load(lnk, i1).suc), i1)
            if pre != PRESOLVE_NO_LINK
                p1 = Int(pre) + 1
                unsafe_store!(lnk, PresolveHLinkPod(unsafe_load(lnk, p1).pre, Int32(i)), p1)
            end
            pre = Int32(i)
        else
            unsafe_store!(lnk, PresolveHLinkPod(PRESOLVE_NO_LINK, PRESOLVE_NO_LINK), i1)
        end
        i += 1
    end
    if pre != PRESOLVE_NO_LINK
        p1 = Int(pre) + 1
        unsafe_store!(lnk, PresolveHLinkPod(unsafe_load(lnk, p1).pre, n), p1)
    end
    unsafe_store!(lnk, PresolveHLinkPod(pre, PRESOLVE_NO_LINK), ni + 1)
    return nothing
end

function ps_expand_major!(
    majstrts::Ptr{Int32},
    els::Ptr{Float64},
    minndxs::Ptr{Int32},
    majlens::Ptr{Int32},
    majlinks::Ptr{PresolveHLinkPod},
    nmaj::Int32,
    k::Int32,
)::Bool
    nm1 = Int(nmaj) + 1
    k1 = Int(k) + 1
    bulk_cap = unsafe_load(majstrts, nm1)
    kcsx = unsafe_load(majstrts, k1)
    klen = unsafe_load(majlens, k1)
    kcex = kcsx + klen
    nextcol = unsafe_load(majlinks, k1).suc

    if kcex + Int32(1) < unsafe_load(majstrts, Int(nextcol) + 1)
        return false
    elseif nextcol == nmaj
        ps_compact_rep!(els, minndxs, majstrts, majlens, nmaj, majlinks)
        kcsx = unsafe_load(majstrts, k1)
        kcex = kcsx + klen
        return (kcex + Int32(1) >= bulk_cap)
    else
        lastcol = unsafe_load(majlinks, nm1).pre
        lc1 = Int(lastcol) + 1
        newkcsx = unsafe_load(majstrts, lc1) + unsafe_load(majlens, lc1)
        newkcex = newkcsx + klen
        if newkcex + Int32(1) >= bulk_cap
            ps_compact_rep!(els, minndxs, majstrts, majlens, nmaj, majlinks)
            kcsx = unsafe_load(majstrts, k1)
            newkcsx = unsafe_load(majstrts, lc1) + unsafe_load(majlens, lc1)
            newkcex = newkcsx + klen
        end
        kl = Int(klen)
        coin_memcpy_n!(minndxs + Int(kcsx) * sizeof(Int32), kl, minndxs + Int(newkcsx) * sizeof(Int32))
        coin_memcpy_n!(els + Int(kcsx) * sizeof(Float64), kl, els + Int(newkcsx) * sizeof(Float64))
        unsafe_store!(majstrts, newkcsx, k1)
        ps_remove_link!(majlinks, k)
        ps_insert_link!(majlinks, k, lastcol)
        if newkcex + Int32(1) >= bulk_cap
            ps_compact_rep!(els, minndxs, majstrts, majlens, nmaj, majlinks)
            kcsx = unsafe_load(majstrts, k1)
            kcex = kcsx + klen
            if kcex > bulk_cap
                return true
            end
        end
        return false
    end
end

@inline function ps_find_minor1(tgt::Int32, ks::Int32, ke::Int32, minndxs::Ptr{Int32})::Int32
    k = ks
    while k < ke
        if unsafe_load(minndxs, Int(k) + 1) == tgt
            return k
        end
        k += Int32(1)
    end
    return k
end

@inline function ps_find_minor2(
    tgt::Int32,
    ks::Int32,
    majlen::Int32,
    minndxs::Ptr{Int32},
    majlinks::Ptr{Int32},
)::Int32
    i = Int32(0)
    while i < majlen
        if unsafe_load(minndxs, Int(ks) + 1) == tgt
            return ks
        end
        ks = unsafe_load(majlinks, Int(ks) + 1)
        i += Int32(1)
    end
    return Int32(-1)
end

@inline function ps_find_minor3(
    tgt::Int32,
    ks::Int32,
    majlen::Int32,
    minndxs::Ptr{Int32},
    majlinks::Ptr{Int32},
)::Int32
    i = Int32(0)
    while i < majlen
        if unsafe_load(minndxs, Int(ks) + 1) == tgt
            return ks
        end
        ks = unsafe_load(majlinks, Int(ks) + 1)
        i += Int32(1)
    end
    return Int32(-1)
end

@inline function ps_delete_from_major!(
    majndx::Int32,
    minndx::Int32,
    majstrts::Ptr{Int32},
    majlens::Ptr{Int32},
    minndxs::Ptr{Int32},
    els::Ptr{Float64},
)::Nothing
    m1 = Int(majndx) + 1
    ks = unsafe_load(majstrts, m1)
    len = unsafe_load(majlens, m1)
    ke = ks + len
    kmi = ps_find_minor1(minndx, ks, ke, minndxs)
    unsafe_store!(minndxs, unsafe_load(minndxs, Int(ke)), Int(kmi) + 1)
    unsafe_store!(els, unsafe_load(els, Int(ke)), Int(kmi) + 1)
    unsafe_store!(majlens, len - Int32(1), m1)
    return nothing
end

@inline function ps_delete_many_from_major!(
    majndx::Int32,
    marked::Ptr{UInt8},
    majstrts::Ptr{Int32},
    majlens::Ptr{Int32},
    minndxs::Ptr{Int32},
    els::Ptr{Float64},
)::Nothing
    m1 = Int(majndx) + 1
    ks = Int(unsafe_load(majstrts, m1))
    ke = ks + Int(unsafe_load(majlens, m1))
    put = ks
    k = ks
    while k < ke
        imin = unsafe_load(minndxs, k + 1)
        im1 = Int(imin) + 1
        if unsafe_load(marked, im1) == UInt8(0)
            unsafe_store!(minndxs, imin, put + 1)
            unsafe_store!(els, unsafe_load(els, k + 1), put + 1)
            put += 1
        else
            unsafe_store!(marked, UInt8(0), im1)
        end
        k += 1
    end
    unsafe_store!(majlens, Int32(put - ks), m1)
    return nothing
end

function ps_delete_from_major2!(
    majndx::Int32,
    minndx::Int32,
    majstrts::Ptr{Int32},
    majlens::Ptr{Int32},
    minndxs::Ptr{Int32},
    majlinks::Ptr{Int32},
    free_list_ptr::Ptr{Int32},
)::Nothing
    m1 = Int(majndx) + 1
    k = unsafe_load(majstrts, m1)
    k1 = Int(k) + 1
    fl = unsafe_load(free_list_ptr)
    if unsafe_load(minndxs, k1) == minndx
        unsafe_store!(majstrts, unsafe_load(majlinks, k1), m1)
        unsafe_store!(majlinks, fl, k1)
        unsafe_store!(free_list_ptr, k)
        unsafe_store!(majlens, unsafe_load(majlens, m1) - Int32(1), m1)
        return nothing
    else
        len = unsafe_load(majlens, m1)
        kpre = k
        k = unsafe_load(majlinks, k1)
        i = Int32(1)
        while i < len
            ki1 = Int(k) + 1
            if unsafe_load(minndxs, ki1) == minndx
                unsafe_store!(majlinks, unsafe_load(majlinks, ki1), Int(kpre) + 1)
                unsafe_store!(majlinks, fl, ki1)
                unsafe_store!(free_list_ptr, k)
                unsafe_store!(majlens, len - Int32(1), m1)
                return nothing
            end
            kpre = k
            k = unsafe_load(majlinks, ki1)
            i += Int32(1)
        end
    end
    return nothing
end

# --- CoinPresolveMatrix helpers ---

function ps_init_cols_todo!(p_ptr::Ptr{CoinPresolvePodView})::Nothing
    p = unsafe_load(p_ptr)
    nc = Int(p.ncols)
    cols_todo = p.colsToDo
    col_changed = p.colChanged
    _ps_store_i32!(p_ptr, _PS_OFF_numberNextColsToDo, Int32(0))
    if p.anyProhibited == Int32(0)
        j = 0
        while j < nc
            unsafe_store!(cols_todo, Int32(j), j + 1)
            j += 1
        end
        _ps_store_i32!(p_ptr, _PS_OFF_numberColsToDo, p.ncols)
    else
        cnt = Int32(0)
        j = 0
        while j < nc
            if (unsafe_load(col_changed, j + 1) & UInt8(2)) == UInt8(0)
                cnt += Int32(1)
                unsafe_store!(cols_todo, Int32(j), Int(cnt))
            end
            j += 1
        end
        _ps_store_i32!(p_ptr, _PS_OFF_numberColsToDo, cnt)
    end
    return nothing
end

function ps_init_rows_todo!(p_ptr::Ptr{CoinPresolvePodView})::Nothing
    p = unsafe_load(p_ptr)
    nr = Int(p.nrows)
    rows_todo = p.rowsToDo
    row_changed = p.rowChanged
    _ps_store_i32!(p_ptr, _PS_OFF_numberNextRowsToDo, Int32(0))
    if p.anyProhibited == Int32(0)
        i = 0
        while i < nr
            unsafe_store!(rows_todo, Int32(i), i + 1)
            i += 1
        end
        _ps_store_i32!(p_ptr, _PS_OFF_numberRowsToDo, p.nrows)
    else
        cnt = Int32(0)
        i = 0
        while i < nr
            if (unsafe_load(row_changed, i + 1) & UInt8(2)) == UInt8(0)
                cnt += Int32(1)
                unsafe_store!(rows_todo, Int32(i), Int(cnt))
            end
            i += 1
        end
        _ps_store_i32!(p_ptr, _PS_OFF_numberRowsToDo, cnt)
    end
    return nothing
end

function ps_step_cols_todo!(p_ptr::Ptr{CoinPresolvePodView})::Int32
    p = unsafe_load(p_ptr)
    n = Int(p.numberNextColsToDo)
    next_c = p.nextColsToDo
    cols = p.colsToDo
    cc = p.colChanged
    k = 1
    while k <= n
        j = unsafe_load(next_c, k)
        j1 = Int(j) + 1
        unsafe_store!(cc, unsafe_load(cc, j1) & ~UInt8(1), j1)
        unsafe_store!(cols, j, k)
        k += 1
    end
    _ps_store_i32!(p_ptr, _PS_OFF_numberColsToDo, p.numberNextColsToDo)
    _ps_store_i32!(p_ptr, _PS_OFF_numberNextColsToDo, Int32(0))
    return p.numberNextColsToDo
end

function ps_step_rows_todo!(p_ptr::Ptr{CoinPresolvePodView})::Int32
    p = unsafe_load(p_ptr)
    n = Int(p.numberNextRowsToDo)
    next_r = p.nextRowsToDo
    rows = p.rowsToDo
    rc = p.rowChanged
    k = 1
    while k <= n
        i = unsafe_load(next_r, k)
        i1 = Int(i) + 1
        unsafe_store!(rc, unsafe_load(rc, i1) & ~UInt8(1), i1)
        unsafe_store!(rows, i, k)
        k += 1
    end
    _ps_store_i32!(p_ptr, _PS_OFF_numberRowsToDo, p.numberNextRowsToDo)
    _ps_store_i32!(p_ptr, _PS_OFF_numberNextRowsToDo, Int32(0))
    return p.numberNextRowsToDo
end

function ps_recompute_sums!(p_ptr::Ptr{CoinPresolvePodView}, one_row::Int32)::Int32
    p = unsafe_load(p_ptr)
    nr = p.nrows
    nc = p.ncols
    clo = p.clo; cup = p.cup
    rlo = p.rlo; rup = p.rup
    rowels = p.rowels; hcol = p.hcol
    mrstrt = p.mrstrt; hinrow = p.hinrow
    inf_up = p.infiniteUp; sum_up = p.sumUp
    inf_dn = p.infiniteDown; sum_dn = p.sumDown
    large = PRESOLVE_SMALL_INF
    tol = p.feasibilityTolerance

    i_first = one_row >= Int32(0) ? Int(one_row) : 0
    i_last = one_row >= Int32(0) ? Int(one_row) : Int(nr)
    infeas = Int32(0)
    ir = i_first
    while ir < i_last
        ir1 = ir + 1
        unsafe_store!(inf_up, Int32(0), ir1)
        unsafe_store!(sum_up, 0.0, ir1)
        unsafe_store!(inf_dn, Int32(0), ir1)
        unsafe_store!(sum_dn, 0.0, ir1)
        rl = unsafe_load(rlo, ir1)
        ru = unsafe_load(rup, ir1)
        rlen = unsafe_load(hinrow, ir1)
        if (rl > -large || ru < large) && rlen > Int32(0)
            iup = Int32(0); idn = Int32(0)
            mup = 0.0; mdn = 0.0
            rs = Int(unsafe_load(mrstrt, ir1))
            re = rs + Int(rlen)
            j = rs
            while j < re
                v = unsafe_load(rowels, j + 1)
                ic1 = Int(unsafe_load(hcol, j + 1)) + 1
                lj = unsafe_load(clo, ic1)
                uj = unsafe_load(cup, ic1)
                if v > 0.0
                    if uj < large
                        mup = _fmadd(uj, v, mup)
                    else
                        iup += Int32(1)
                    end
                    if lj > -large
                        mdn = _fmadd(lj, v, mdn)
                    else
                        idn += Int32(1)
                    end
                elseif v < 0.0
                    if uj < large
                        mdn = _fmadd(uj, v, mdn)
                    else
                        idn += Int32(1)
                    end
                    if lj > -large
                        mup = _fmadd(lj, v, mup)
                    else
                        iup += Int32(1)
                    end
                end
                j += 1
            end
            unsafe_store!(inf_up, iup, ir1)
            unsafe_store!(sum_up, mup, ir1)
            unsafe_store!(inf_dn, idn, ir1)
            unsafe_store!(sum_dn, mdn, ir1)
            max_up = _fmadd(Float64(iup), large, mup)
            max_dn = _fmsub(Float64(idn), large, mdn)
            if max_up <= ru + tol && max_dn >= rl - tol
                unsafe_store!(inf_up, nc + Int32(1), ir1)
                unsafe_store!(inf_dn, nc + Int32(1), ir1)
            elseif max_up < rl - tol
                infeas += Int32(1)
            elseif max_dn > ru + tol
                infeas += Int32(1)
            end
        elseif rlen > Int32(0)
            unsafe_store!(inf_up, nc + Int32(1), ir1)
            unsafe_store!(inf_dn, nc + Int32(1), ir1)
        else
            if rl > 0.0 || ru < 0.0
                tol2 = 10.0 * tol
                if rl > 0.0 && rl < tol2
                    unsafe_store!(rlo, 0.0, ir1)
                else
                    infeas += Int32(1)
                end
                if ru < 0.0 && ru > -tol2
                    unsafe_store!(rup, 0.0, ir1)
                else
                    infeas += Int32(1)
                end
            end
        end
        ir += 1
    end
    return infeas
end

# =============================================================================
# 2. CoinPresolveZeros (CoinPresolveZeros.cpp)
# =============================================================================

# Simple iterative heapsort for ascending Int32 sort (zero-alloc, --trim=safe)
function _ps_sort_i32!(a::Ptr{Int32}, n::Int)::Nothing
    n <= 1 && return nothing
    # Build max-heap
    start = n >> 1
    while start >= 1
        root = start
        while (root << 1) <= n
            child = root << 1
            sw = root
            if unsafe_load(a, sw) < unsafe_load(a, child)
                sw = child
            end
            if child + 1 <= n && unsafe_load(a, sw) < unsafe_load(a, child + 1)
                sw = child + 1
            end
            sw == root && break
            tmp = unsafe_load(a, root)
            unsafe_store!(a, unsafe_load(a, sw), root)
            unsafe_store!(a, tmp, sw)
            root = sw
        end
        start -= 1
    end
    en = n
    while en > 1
        tmp = unsafe_load(a, en)
        unsafe_store!(a, unsafe_load(a, 1), en)
        unsafe_store!(a, tmp, 1)
        en -= 1
        root = 1
        while (root << 1) <= en
            child = root << 1
            sw = root
            if unsafe_load(a, sw) < unsafe_load(a, child)
                sw = child
            end
            if child + 1 <= en && unsafe_load(a, sw) < unsafe_load(a, child + 1)
                sw = child + 1
            end
            sw == root && break
            t2 = unsafe_load(a, root)
            unsafe_store!(a, unsafe_load(a, sw), root)
            unsafe_store!(a, t2, sw)
            root = sw
        end
    end
    return nothing
end

function ps_zeros_count!(
    p_ptr::Ptr{CoinPresolvePodView},
    checkcols::Ptr{Int32},
    ncheck_ptr::Ptr{Int32},
)::Int32
    p = unsafe_load(p_ptr)
    mcstrt = p.mcstrt; colels = p.colels; hincol = p.hincol
    ncheck = unsafe_load(ncheck_ptr)

    if ncheck != Int32(0) && ncheck != p.ncols
        _ps_sort_i32!(checkcols, Int(ncheck))
        n = Int(ncheck)
        out_n = 1
        ichk = unsafe_load(checkcols, 1)
        i = 2
        while i <= n
            v = unsafe_load(checkcols, i)
            if v != ichk
                ichk = v
                out_n += 1
                unsafe_store!(checkcols, ichk, out_n)
            end
            i += 1
        end
        ncheck = Int32(out_n)
    end

    nzeros = Int32(0)
    zero_cols = Int32(0)
    if ncheck == p.ncols
        nc = Int(ncheck)
        j = 0
        while j < nc
            kcs = Int(unsafe_load(mcstrt, j + 1))
            kce = kcs + Int(unsafe_load(hincol, j + 1))
            zj = Int32(0)
            k = kcs
            while k < kce
                if abs(unsafe_load(colels, k + 1)) < PRESOLVE_ZTOLDP
                    zj += Int32(1)
                end
                k += 1
            end
            if zj != Int32(0)
                zero_cols += Int32(1)
                unsafe_store!(checkcols, Int32(j), Int(zero_cols))
                nzeros += zj
            end
            j += 1
        end
    else
        nchk = Int(ncheck)
        ndx = 1
        while ndx <= nchk
            j = Int(unsafe_load(checkcols, ndx))
            kcs = Int(unsafe_load(mcstrt, j + 1))
            kce = kcs + Int(unsafe_load(hincol, j + 1))
            zj = Int32(0)
            k = kcs
            while k < kce
                if abs(unsafe_load(colels, k + 1)) < PRESOLVE_ZTOLDP
                    zj += Int32(1)
                end
                k += 1
            end
            if zj != Int32(0)
                zero_cols += Int32(1)
                unsafe_store!(checkcols, Int32(j), Int(zero_cols))
                nzeros += zj
            end
            ndx += 1
        end
    end
    unsafe_store!(ncheck_ptr, zero_cols)
    return nzeros
end

function ps_zeros_drop!(
    p_ptr::Ptr{CoinPresolvePodView},
    checkcols::Ptr{Int32},
    ncheckcols::Int32,
    actions::Ptr{DroppedZeroPod},
)::Int32
    p = unsafe_load(p_ptr)
    mcstrt = p.mcstrt; colels = p.colels; hrow = p.hrow; hincol = p.hincol; clink = p.clink
    mrstrt = p.mrstrt; rowels = p.rowels; hcol = p.hcol; hinrow = p.hinrow; rlink = p.rlink

    nactions = Int32(0)
    nchk = Int(ncheckcols)
    i = 1
    while i <= nchk
        col = unsafe_load(checkcols, i)
        c1 = Int(col) + 1
        kcs = Int(unsafe_load(mcstrt, c1))
        kce = kcs + Int(unsafe_load(hincol, c1))
        k = kcs
        while k < kce
            if abs(unsafe_load(colels, k + 1)) < PRESOLVE_ZTOLDP
                nactions += Int32(1)
                unsafe_store!(actions, DroppedZeroPod(unsafe_load(hrow, k + 1), col), Int(nactions))
                kce -= 1
                unsafe_store!(colels, unsafe_load(colels, kce + 1), k + 1)
                unsafe_store!(hrow, unsafe_load(hrow, kce + 1), k + 1)
                unsafe_store!(hincol, unsafe_load(hincol, c1) - Int32(1), c1)
                continue
            end
            k += 1
        end
        if unsafe_load(hincol, c1) == Int32(0)
            ps_remove_link!(clink, col)
        end
        i += 1
    end

    # Drop row zeros
    na = Int(nactions)
    i = 1
    while i <= na
        row = unsafe_load(actions, i).row
        r1 = Int(row) + 1
        krs = Int(unsafe_load(mrstrt, r1))
        kre = krs + Int(unsafe_load(hinrow, r1))
        k = krs
        while k < kre
            if abs(unsafe_load(rowels, k + 1)) < PRESOLVE_ZTOLDP
                kre -= 1
                unsafe_store!(rowels, unsafe_load(rowels, kre + 1), k + 1)
                unsafe_store!(hcol, unsafe_load(hcol, kre + 1), k + 1)
                unsafe_store!(hinrow, unsafe_load(hinrow, r1) - Int32(1), r1)
                continue
            end
            k += 1
        end
        if unsafe_load(hinrow, r1) == Int32(0)
            ps_remove_link!(rlink, row)
        end
        i += 1
    end
    return nactions
end

function ps_zeros_postsolve!(
    p_ptr::Ptr{CoinPostsolvePodView},
    zeros::Ptr{DroppedZeroPod},
    nzeros::Int32,
)::Nothing
    p = unsafe_load(p_ptr)
    colels = p.colels; hrow = p.hrow; mcstrt = p.mcstrt; hincol = p.hincol; lnk = p.link
    fl = p.free_list
    idx = Int(nzeros)
    while idx >= 1
        z = unsafe_load(zeros, idx)
        j1 = Int(z.col) + 1
        k = fl
        k1 = Int(k) + 1
        fl = unsafe_load(lnk, k1)
        unsafe_store!(hrow, z.row, k1)
        unsafe_store!(colels, 0.0, k1)
        unsafe_store!(lnk, unsafe_load(mcstrt, j1), k1)
        unsafe_store!(mcstrt, k, j1)
        unsafe_store!(hincol, unsafe_load(hincol, j1) + Int32(1), j1)
        idx -= 1
    end
    _post_store_i32!(p_ptr, _POST_OFF_free_list, fl)
    return nothing
end

# =============================================================================
# 3. CoinPresolveEmpty (CoinPresolveEmpty.cpp)
# =============================================================================

# Returns (status_code, bad_col) where status_code: 0=ok, 1=col_infeas, 2=unbounded_lo, 3=unbounded_hi
function ps_drop_empty_cols_presolve!(
    p_ptr::Ptr{CoinPresolvePodView},
    ecols::Ptr{Int32},
    necols::Int32,
    actions::Ptr{DropEmptyColActionPod},
    colmapping::Ptr{Int32},
    newclink::Ptr{PresolveHLinkPod},
    out_bad_col::Ptr{Int32},
)::Int32
    p = unsafe_load(p_ptr)
    n_orig = Int(p.ncols)
    mcstrt = p.mcstrt; hincol = p.hincol; clink = p.clink
    clo = p.clo; cup = p.cup; cost = p.cost
    ztoldj = p.ztoldj; int_type = p.integerType; orig_col = p.originalColumn
    maxmin = p.maxmin; sol = p.sol; colstat = p.colstat
    dobias = p.dobias
    status = p.status
    fix_infeas = (p.presolveOptions & Int32(0x4000)) != Int32(0)

    coin_zero_n!(colmapping, n_orig)
    ret_code = Int32(0)
    ndx = Int(necols) - 1
    while ndx >= 0
        j = unsafe_load(ecols, ndx + 1)
        if !ps_col_prohibited2(p.anyProhibited, p.colChanged, j)
            j1 = Int(j) + 1
            unsafe_store!(colmapping, Int32(-1), j1)
            lj = unsafe_load(clo, j1)
            uj = unsafe_load(cup, j1)
            if unsafe_load(int_type, j1) != UInt8(0)
                lj = ceil(lj - 1.0e-9)
                uj = floor(uj + 1.0e-9)
                unsafe_store!(clo, lj, j1)
                unsafe_store!(cup, uj, j1)
                if lj > uj && !fix_infeas
                    status |= Int32(1)
                    unsafe_store!(out_bad_col, j)
                    ret_code = Int32(1)
                    break
                end
            end
            cj = unsafe_load(cost, j1)
            if abs(cj) < ztoldj
                cj = 0.0
                unsafe_store!(cost, 0.0, j1)
            end
            esol = 0.0
            if cj == 0.0
                if -PRESOLVE_INF_DBL < lj
                    esol = lj
                elseif uj < PRESOLVE_INF_DBL
                    esol = uj
                else
                    esol = 0.0
                end
            elseif cj * maxmin > 0.0
                if -PRESOLVE_INF_DBL < lj
                    esol = lj
                else
                    status |= Int32(2)
                    unsafe_store!(out_bad_col, j)
                    ret_code = Int32(2)
                    break
                end
            else
                if uj < PRESOLVE_INF_DBL
                    esol = uj
                else
                    status |= Int32(2)
                    unsafe_store!(out_bad_col, j)
                    ret_code = Int32(3)
                    break
                end
            end
            unsafe_store!(actions, DropEmptyColActionPod(lj, uj, cj, esol, j, Int32(0)), ndx + 1)
            dobias = _fmadd(esol, cj, dobias)
        end
        ndx -= 1
    end

    if status == Int32(0)
        n_comp = 0
        ndx = 0
        while ndx < n_orig
            n1 = ndx + 1
            if unsafe_load(colmapping, n1) == Int32(0)
                c1 = n_comp + 1
                unsafe_store!(mcstrt, unsafe_load(mcstrt, n1), c1)
                unsafe_store!(hincol, unsafe_load(hincol, n1), c1)
                unsafe_store!(clo, unsafe_load(clo, n1), c1)
                unsafe_store!(cup, unsafe_load(cup, n1), c1)
                unsafe_store!(cost, unsafe_load(cost, n1), c1)
                if sol != C_NULL
                    unsafe_store!(sol, unsafe_load(sol, n1), c1)
                    unsafe_store!(colstat, unsafe_load(colstat, n1), c1)
                end
                unsafe_store!(int_type, unsafe_load(int_type, n1), c1)
                unsafe_store!(orig_col, unsafe_load(orig_col, n1), c1)
                unsafe_store!(colmapping, Int32(n_comp), n1)
                n_comp += 1
            end
            ndx += 1
        end
        unsafe_store!(mcstrt, unsafe_load(mcstrt, n_orig + 1), n_comp + 1)
        unsafe_store!(colmapping, Int32(n_comp), n_orig + 1)

        oldj = Int32(n_orig)
        while oldj >= Int32(0)
            oldlnk = unsafe_load(clink, Int(oldj) + 1)
            newj = unsafe_load(colmapping, Int(oldj) + 1)
            nsuc = oldlnk.suc >= Int32(0) ? unsafe_load(colmapping, Int(oldlnk.suc) + 1) : PRESOLVE_NO_LINK
            npre = oldlnk.pre >= Int32(0) ? unsafe_load(colmapping, Int(oldlnk.pre) + 1) : PRESOLVE_NO_LINK
            unsafe_store!(newclink, PresolveHLinkPod(npre, nsuc), Int(newj) + 1)
            oldj = oldlnk.pre
        end
        _ps_store_i32!(p_ptr, _PS_OFF_ncols, Int32(n_comp))
    end
    _ps_store_f64!(p_ptr, _PS_OFF_dobias, dobias)
    _ps_store_i32!(p_ptr, _PS_OFF_status, status)
    return ret_code
end

function ps_drop_empty_cols_postsolve!(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{DropEmptyColActionPod},
    nactions::Int32,
    colmapping::Ptr{Int32},
)::Nothing
    p = unsafe_load(p_ptr)
    ncols = Int(p.ncols)
    na = Int(nactions)
    ncols2 = ncols + na
    mcstrt = p.mcstrt; hincol = p.hincol
    clo = p.clo; cup = p.cup; sol = p.sol; cost = p.cost; rcosts = p.rcosts; colstat = p.colstat
    maxmin = p.maxmin; ztolzb = p.ztolzb

    coin_zero_n!(colmapping, ncols2)
    ndx = 1
    while ndx <= na
        j = unsafe_load(actions, ndx).jcol
        unsafe_store!(colmapping, Int32(-1), Int(j) + 1)
        ndx += 1
    end
    j = ncols2 - 1
    while j >= 0
        j1 = j + 1
        if unsafe_load(colmapping, j1) == Int32(0)
            ncols -= 1
            n1 = ncols + 1
            unsafe_store!(mcstrt, unsafe_load(mcstrt, n1), j1)
            unsafe_store!(hincol, unsafe_load(hincol, n1), j1)
            unsafe_store!(clo, unsafe_load(clo, n1), j1)
            unsafe_store!(cup, unsafe_load(cup, n1), j1)
            unsafe_store!(cost, unsafe_load(cost, n1), j1)
            if sol != C_NULL
                unsafe_store!(sol, unsafe_load(sol, n1), j1)
            end
            if rcosts != C_NULL
                unsafe_store!(rcosts, unsafe_load(rcosts, n1), j1)
            end
            if colstat != C_NULL
                unsafe_store!(colstat, unsafe_load(colstat, n1), j1)
            end
        end
        j -= 1
    end
    ndx = 1
    while ndx <= na
        e = unsafe_load(actions, ndx)
        j = e.jcol
        j1 = Int(j) + 1
        unsafe_store!(hincol, Int32(0), j1)
        unsafe_store!(mcstrt, PRESOLVE_NO_LINK, j1)
        unsafe_store!(clo, e.clo, j1)
        unsafe_store!(cup, e.cup, j1)
        unsafe_store!(cost, e.cost, j1)
        if sol != C_NULL
            unsafe_store!(sol, e.sol, j1)
        end
        if rcosts != C_NULL
            unsafe_store!(rcosts, maxmin * e.cost, j1)
        end
        if colstat != C_NULL
            ps_set_col_status_using_value!(colstat, sol, clo, cup, ztolzb, j)
        end
        ndx += 1
    end
    _post_store_i32!(p_ptr, _POST_OFF_ncols, Int32(ncols2))
    return nothing
end

# Returns (nactions_recorded, infeas_row) — infeas_row >= 0 if infeasible
function ps_drop_empty_rows_presolve!(
    p_ptr::Ptr{CoinPresolvePodView},
    actions::Ptr{DropEmptyRowActionPod},
    rowmapping::Ptr{Int32},
    out_infeas_row::Ptr{Int32},
)::Int32
    p = unsafe_load(p_ptr)
    ncols = Int(p.ncols); nrows = Int(p.nrows)
    mcstrt = p.mcstrt; hincol = p.hincol; hrow = p.hrow; hinrow = p.hinrow
    rlo = p.rlo; rup = p.rup; rowstat = p.rowstat; acts = p.acts; orig_row = p.originalRow
    fix_infeas = (p.presolveOptions & Int32(0x4000)) != Int32(0)
    tol = 10.0 * p.feasibilityTolerance
    status = p.status

    unsafe_store!(out_infeas_row, Int32(-1))
    nactions = Int32(0)
    nrows2 = Int32(0)
    i = 0
    while i < nrows
        i1 = i + 1
        if unsafe_load(hinrow, i1) == Int32(0)
            nactions += Int32(1)
            rl = unsafe_load(rlo, i1)
            ru = unsafe_load(rup, i1)
            if rl > 0.0 || ru < 0.0
                if (rl <= tol && ru >= -tol) || fix_infeas
                    rl = 0.0; ru = 0.0
                    unsafe_store!(rlo, 0.0, i1)
                    unsafe_store!(rup, 0.0, i1)
                else
                    status |= Int32(1)
                    unsafe_store!(out_infeas_row, Int32(i))
                    break
                end
            end
            unsafe_store!(actions, DropEmptyRowActionPod(rl, ru, Int32(i), Int32(0)), Int(nactions))
            unsafe_store!(rowmapping, Int32(-1), i1)
        else
            n21 = Int(nrows2) + 1
            unsafe_store!(rlo, unsafe_load(rlo, i1), n21)
            unsafe_store!(rup, unsafe_load(rup, i1), n21)
            unsafe_store!(orig_row, Int32(i), n21)
            if acts != C_NULL
                unsafe_store!(acts, unsafe_load(acts, i1), n21)
                unsafe_store!(rowstat, unsafe_load(rowstat, i1), n21)
            end
            unsafe_store!(rowmapping, nrows2, i1)
            nrows2 += Int32(1)
        end
        i += 1
    end

    # Remap hrow
    c = 0
    while c < ncols
        s = Int(unsafe_load(mcstrt, c + 1))
        e = s + Int(unsafe_load(hincol, c + 1))
        j = s
        while j < e
            r = Int(unsafe_load(hrow, j + 1))
            unsafe_store!(hrow, unsafe_load(rowmapping, r + 1), j + 1)
            j += 1
        end
        c += 1
    end

    _ps_store_i32!(p_ptr, _PS_OFF_nrows, nrows2)
    _ps_store_i32!(p_ptr, _PS_OFF_status, status)
    return nactions
end

function ps_drop_empty_rows_postsolve!(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{DropEmptyRowActionPod},
    nactions::Int32,
    rowmapping::Ptr{Int32},
)::Nothing
    p = unsafe_load(p_ptr)
    ncols = Int(p.ncols); nrows0 = Int(p.nrows0); nrows = Int(p.nrows)
    na = Int(nactions)
    mcstrt = p.mcstrt; hincol = p.hincol; hrow = p.hrow
    rlo = p.rlo; rup = p.rup; rowstat = p.rowstat; rowduals = p.rowduals; acts = p.acts

    coin_zero_n!(rowmapping, nrows0)
    k = 1
    while k <= na
        r = unsafe_load(actions, k).row
        unsafe_store!(rowmapping, Int32(-1), Int(r) + 1)
        k += 1
    end
    i = nrows0 - 1
    while i >= 0
        i1 = i + 1
        if unsafe_load(rowmapping, i1) == Int32(0)
            nrows -= 1
            n1 = nrows + 1
            unsafe_store!(rlo, unsafe_load(rlo, n1), i1)
            unsafe_store!(rup, unsafe_load(rup, n1), i1)
            unsafe_store!(acts, unsafe_load(acts, n1), i1)
            unsafe_store!(rowduals, unsafe_load(rowduals, n1), i1)
            if rowstat != C_NULL
                unsafe_store!(rowstat, unsafe_load(rowstat, n1), i1)
            end
        end
        i -= 1
    end
    nr_map = 0
    i = 0
    while i < nrows0
        if unsafe_load(rowmapping, i + 1) == Int32(0)
            nr_map += 1
            unsafe_store!(rowmapping, Int32(i), nr_map)
        end
        i += 1
    end
    j = 0
    while j < ncols
        s = Int(unsafe_load(mcstrt, j + 1))
        e = s + Int(unsafe_load(hincol, j + 1))
        k = s
        while k < e
            r = Int(unsafe_load(hrow, k + 1))
            unsafe_store!(hrow, unsafe_load(rowmapping, r + 1), k + 1)
            k += 1
        end
        j += 1
    end
    k = 1
    while k <= na
        e = unsafe_load(actions, k)
        r = e.row
        r1 = Int(r) + 1
        unsafe_store!(rlo, e.rlo, r1)
        unsafe_store!(rup, e.rup, r1)
        unsafe_store!(acts, 0.0, r1)
        if rowstat != C_NULL
            ps_set_row_status!(rowstat, r, PS_STATUS_BASIC)
        end
        unsafe_store!(rowduals, 0.0, r1)
        k += 1
    end
    _post_store_i32!(p_ptr, _POST_OFF_nrows, p.nrows + nactions)
    return nothing
end

# =============================================================================
# 4. CoinPresolveIsolated (CoinPresolveIsolated.cpp)
# =============================================================================

function ps_isolated_check(p_ptr::Ptr{CoinPresolvePodView}, irow::Int32)::Int32
    p = unsafe_load(p_ptr)
    ir1 = Int(irow) + 1
    if unsafe_load(p.rlo, ir1) != 0.0 || unsafe_load(p.rup, ir1) != 0.0
        return Int32(0)
    end
    krs = Int(unsafe_load(p.mrstrt, ir1))
    kre = krs + Int(unsafe_load(p.hinrow, ir1))
    maxmin = p.maxmin
    k = krs
    while k < kre
        jc = unsafe_load(p.hcol, k + 1)
        jc1 = Int(jc) + 1
        lo = unsafe_load(p.clo, jc1)
        hi = unsafe_load(p.cup, jc1)
        mc = maxmin * unsafe_load(p.cost, jc1)
        if (lo != 0.0 && hi != 0.0) || (mc > 0.0 && lo != 0.0) || (mc < 0.0 && hi != 0.0) ||
           ps_col_prohibited2(p.anyProhibited, p.colChanged, jc)
            return Int32(0)
        end
        k += 1
    end
    return Int32(1)
end

function ps_isolated_apply!(
    p_ptr::Ptr{CoinPresolvePodView},
    irow::Int32,
    out_rowcols::Ptr{Int32},
    out_rowels::Ptr{Float64},
    out_costs::Ptr{Float64},
)::Nothing
    p = unsafe_load(p_ptr)
    ir1 = Int(irow) + 1
    krs = Int(unsafe_load(p.mrstrt, ir1))
    nc = Int(unsafe_load(p.hinrow, ir1))
    kre = krs + nc
    k = krs
    idx = 1
    while k < kre
        jc = unsafe_load(p.hcol, k + 1)
        jc1 = Int(jc) + 1
        unsafe_store!(out_rowcols, jc, idx)
        unsafe_store!(out_rowels, unsafe_load(p.rowels, k + 1), idx)
        unsafe_store!(out_costs, unsafe_load(p.cost, jc1), idx)
        unsafe_store!(p.cost, 0.0, jc1)
        ps_delete_from_major!(jc, irow, p.mcstrt, p.hincol, p.hrow, p.colels)
        if unsafe_load(p.hincol, jc1) == Int32(0)
            ps_remove_link!(p.clink, jc)
        end
        k += 1
        idx += 1
    end
    unsafe_store!(p.hinrow, Int32(0), ir1)
    ps_remove_link!(p.rlink, irow)
    unsafe_store!(p.rlo, 0.0, ir1)
    unsafe_store!(p.rup, 0.0, ir1)
    return nothing
end

function ps_isolated_postsolve!(
    p_ptr::Ptr{CoinPostsolvePodView},
    irow::Int32,
    ninrow::Int32,
    rlo_val::Float64,
    rup_val::Float64,
    rowcols::Ptr{Int32},
    rowels::Ptr{Float64},
)::Nothing
    p = unsafe_load(p_ptr)
    ir1 = Int(irow) + 1
    unsafe_store!(p.rup, rup_val, ir1)
    unsafe_store!(p.rlo, rlo_val, ir1)
    fl = p.free_list
    k = 1
    nr = Int(ninrow)
    while k <= nr
        jc = unsafe_load(rowcols, k)
        jc1 = Int(jc) + 1
        unsafe_store!(p.sol, 0.0, jc1)
        kk = fl
        kk1 = Int(kk) + 1
        fl = unsafe_load(p.link, kk1)
        unsafe_store!(p.mcstrt, kk, jc1)
        unsafe_store!(p.colels, unsafe_load(rowels, k), kk1)
        unsafe_store!(p.hrow, irow, kk1)
        unsafe_store!(p.link, PRESOLVE_NO_LINK, kk1)
        unsafe_store!(p.hincol, Int32(1), jc1)
        k += 1
    end
    ps_set_row_status!(p.rowstat, irow, PS_STATUS_BASIC)
    unsafe_store!(p.rowduals, 0.0, ir1)
    unsafe_store!(p.acts, 0.0, ir1)
    _post_store_i32!(p_ptr, _POST_OFF_free_list, fl)
    return nothing
end

# =============================================================================
# 5. CoinPresolveUseless (CoinPresolveUseless.cpp)
# =============================================================================

# Runs the bound-propagation + useless-row scan of testRedundant().
# Writes:
#   - useless rows into prob.usefulRowInt + m (count returned in out_nuseless)
#   - out_infeas_row (>= 0 if row infeasibility message should be emitted)
#   - out_nfixed (if presolveOptions & 0x10 fixed cols written into prob.usefulColumnInt)
# Returns numberInfeasible (0 if feasible).
function ps_test_redundant_core!(
    p_ptr::Ptr{CoinPresolvePodView},
    col_lower::Ptr{Float64},
    col_upper::Ptr{Float64},
    out_nuseless::Ptr{Int32},
    out_nfixed::Ptr{Int32},
    out_infeas_row::Ptr{Int32},
)::Int32
    p = unsafe_load(p_ptr)
    m = Int(p.nrows); n = Int(p.ncols)
    mrstrt = p.mrstrt; hinrow = p.hinrow; rowels = p.rowels; hcol = p.hcol
    mcstrt = p.mcstrt; hrow = p.hrow; hincol = p.hincol
    rlo = p.rlo; rup = p.rup
    n_next_rows = p.numberNextRowsToDo
    n_next_cols = p.numberNextColsToDo

    coin_memcpy_n!(p.clo, n, col_lower)
    coin_memcpy_n!(p.cup, n, col_upper)

    useless_rows = p.usefulRowInt + m * sizeof(Int32)
    nuseless = Int32(0)
    large = 1.0e15
    feas_tol = p.feasibilityTolerance
    relaxed_tol = 100.0 * feas_tol
    fix_infeas = (p.presolveOptions & Int32(0x4000)) != Int32(0)

    mark_ignore = Int8(1)
    mark_inplay = Int8(-1)
    mark_actok  = Int8(-2)
    mark_row = Ptr{Int8}(p.usefulRowInt)

    i = 0
    while i < m
        i1 = i + 1
        rl = unsafe_load(rlo, i1)
        ru = unsafe_load(rup, i1)
        rlen = unsafe_load(hinrow, i1)
        if (rl > -large || ru < large) && rlen > Int32(0)
            unsafe_store!(mark_row, mark_inplay, i1)
        else
            unsafe_store!(mark_row, mark_ignore, i1)
            if rlen > Int32(0)
                nuseless += Int32(1)
                unsafe_store!(useless_rows, Int32(i), Int(nuseless))
                n_next_rows = ps_add_row!(p.rowChanged, p.nextRowsToDo, n_next_rows, Int32(i))
            end
        end
        i += 1
    end

    num_infeas = Int32(0)
    num_changed = Int32(0)
    num_check = Int32(-1)
    infeas_row = Int32(-1)
    i_pass = Int32(-1)

    while i_pass < Int32(10) && num_changed > num_check
        i_pass += Int32(1)
        num_changed = Int32(0)
        i = 0
        while i < m
            i1 = i + 1
            if unsafe_load(mark_row, i1) != mark_inplay
                i += 1
                continue
            end
            inf_up = Int32(0); inf_lo = Int32(0)
            fin_up = 0.0; fin_dn = 0.0
            krs = Int(unsafe_load(mrstrt, i1))
            kre = krs + Int(unsafe_load(hinrow, i1))
            kcol = krs
            while kcol < kre
                v = unsafe_load(rowels, kcol + 1)
                j1 = Int(unsafe_load(hcol, kcol + 1)) + 1
                if v > 0.0
                    uj = unsafe_load(col_upper, j1)
                    if uj < large
                        fin_up = _fmadd(uj, v, fin_up)
                    else
                        inf_up += Int32(1)
                    end
                    lj = unsafe_load(col_lower, j1)
                    if lj > -large
                        fin_dn = _fmadd(lj, v, fin_dn)
                    else
                        inf_lo += Int32(1)
                    end
                elseif v < 0.0
                    uj = unsafe_load(col_upper, j1)
                    if uj < large
                        fin_dn = _fmadd(uj, v, fin_dn)
                    else
                        inf_lo += Int32(1)
                    end
                    lj = unsafe_load(col_lower, j1)
                    if lj > -large
                        fin_up = _fmadd(lj, v, fin_up)
                    else
                        inf_up += Int32(1)
                    end
                end
                kcol += 1
            end
            unsafe_store!(mark_row, mark_actok, i1)
            fin_up = _fmadd(1.0e-8, abs(fin_up), fin_up)
            fin_dn = _fmsub(1.0e-8, abs(fin_dn), fin_dn)
            max_up = _fmadd(Float64(inf_up), 1.0e31, fin_up)
            max_dn = _fmsub(Float64(inf_lo), 1.0e31, fin_dn)
            rloi = unsafe_load(rlo, i1)
            rupi = unsafe_load(rup, i1)

            if max_up < rloi - relaxed_tol || max_dn > rupi + relaxed_tol
                if !fix_infeas
                    num_infeas += Int32(1)
                    infeas_row = Int32(i)
                    break
                else
                    i += 1
                    continue
                end
            end
            if max_up <= rupi + feas_tol && max_dn >= rloi - feas_tol
                i += 1
                continue
            end
            if fin_up < rloi && fin_up > rloi - relaxed_tol
                fin_up = rloi
            end
            if fin_dn > rupi && fin_dn < rupi + relaxed_tol
                fin_dn = rupi
            end

            kcol = krs
            while kcol < kre
                ait = unsafe_load(rowels, kcol + 1)
                t = Int(unsafe_load(hcol, kcol + 1))
                t1 = t + 1
                lt = unsafe_load(col_lower, t1)
                ut = unsafe_load(col_upper, t1)
                if ait > 0.0
                    if rloi > -large
                        newlt = if inf_up == Int32(0)
                            v = ut + (rloi - fin_up) / ait
                            abs(fin_up) > 1.0e8 ? _fmsub(1.0e-12, abs(fin_up), v) : v
                        elseif inf_up == Int32(1) && ut >= large
                            v = (rloi - fin_up) / ait
                            abs(fin_up) > 1.0e8 ? _fmsub(1.0e-12, abs(fin_up), v) : v
                        else
                            -PRESOLVE_INF_DBL
                        end
                        if newlt > lt + 1.0e-12 && newlt > -large
                            if ut - newlt < -relaxed_tol
                                num_infeas += Int32(1)
                                break
                            end
                            unsafe_store!(col_lower, newlt, t1)
                            unsafe_store!(mark_row, mark_ignore, i1)
                            num_changed += Int32(1)
                            kcs = Int(unsafe_load(mcstrt, t1))
                            kce = kcs + Int(unsafe_load(hincol, t1))
                            kc = kcs
                            while kc < kce
                                rk1 = Int(unsafe_load(hrow, kc + 1)) + 1
                                if unsafe_load(mark_row, rk1) == mark_actok
                                    unsafe_store!(mark_row, mark_inplay, rk1)
                                end
                                kc += 1
                            end
                            if lt <= -large
                                fin_dn = _fmadd(newlt, ait, fin_dn)
                                inf_lo -= Int32(1)
                            else
                                fin_dn = _fmadd(newlt - lt, ait, fin_dn)
                            end
                            lt = newlt
                        end
                    end
                    if rupi < large
                        newut = if inf_lo == Int32(0)
                            v = lt + (rupi - fin_dn) / ait
                            abs(fin_dn) > 1.0e8 ? _fmadd(1.0e-12, abs(fin_dn), v) : v
                        elseif inf_lo == Int32(1) && lt <= -large
                            v = (rupi - fin_dn) / ait
                            abs(fin_dn) > 1.0e8 ? _fmadd(1.0e-12, abs(fin_dn), v) : v
                        else
                            PRESOLVE_INF_DBL
                        end
                        if newut < ut - 1.0e-12 && newut < large
                            unsafe_store!(col_upper, newut, t1)
                            if newut - lt < -relaxed_tol
                                num_infeas += Int32(1)
                                break
                            end
                            unsafe_store!(mark_row, mark_ignore, i1)
                            num_changed += Int32(1)
                            kcs = Int(unsafe_load(mcstrt, t1))
                            kce = kcs + Int(unsafe_load(hincol, t1))
                            kc = kcs
                            while kc < kce
                                rk1 = Int(unsafe_load(hrow, kc + 1)) + 1
                                if unsafe_load(mark_row, rk1) == mark_actok
                                    unsafe_store!(mark_row, mark_inplay, rk1)
                                end
                                kc += 1
                            end
                            if ut >= large
                                fin_up = _fmadd(newut, ait, fin_up)
                                inf_up -= Int32(1)
                            else
                                fin_up = _fmadd(newut - ut, ait, fin_up)
                            end
                            ut = newut
                        end
                    end
                else
                    if rloi > -large
                        newut = if inf_up == Int32(0)
                            v = lt + (rloi - fin_up) / ait
                            abs(fin_up) > 1.0e8 ? _fmadd(1.0e-12, abs(fin_up), v) : v
                        elseif inf_up == Int32(1) && lt <= -large
                            v = (rloi - fin_up) / ait
                            abs(fin_up) > 1.0e8 ? _fmadd(1.0e-12, abs(fin_up), v) : v
                        else
                            PRESOLVE_INF_DBL
                        end
                        if newut < ut - 1.0e-12 && newut < large
                            unsafe_store!(col_upper, newut, t1)
                            if newut - lt < -relaxed_tol
                                num_infeas += Int32(1)
                                break
                            end
                            unsafe_store!(mark_row, mark_ignore, i1)
                            num_changed += Int32(1)
                            kcs = Int(unsafe_load(mcstrt, t1))
                            kce = kcs + Int(unsafe_load(hincol, t1))
                            kc = kcs
                            while kc < kce
                                rk1 = Int(unsafe_load(hrow, kc + 1)) + 1
                                if unsafe_load(mark_row, rk1) == mark_actok
                                    unsafe_store!(mark_row, mark_inplay, rk1)
                                end
                                kc += 1
                            end
                            if ut >= large
                                fin_dn = _fmadd(newut, ait, fin_dn)
                                inf_lo -= Int32(1)
                            else
                                fin_dn = _fmadd(newut - ut, ait, fin_dn)
                            end
                            ut = newut
                        end
                    end
                    if rupi < large
                        newlt = if inf_lo == Int32(0)
                            v = ut + (rupi - fin_dn) / ait
                            abs(fin_dn) > 1.0e8 ? _fmsub(1.0e-12, abs(fin_dn), v) : v
                        elseif inf_lo == Int32(1) && ut >= large
                            v = (rupi - fin_dn) / ait
                            abs(fin_dn) > 1.0e8 ? _fmsub(1.0e-12, abs(fin_dn), v) : v
                        else
                            -PRESOLVE_INF_DBL
                        end
                        if newlt > lt + 1.0e-12 && newlt > -large
                            unsafe_store!(col_lower, newlt, t1)
                            if ut - newlt < -relaxed_tol
                                num_infeas += Int32(1)
                                break
                            end
                            unsafe_store!(mark_row, mark_ignore, i1)
                            num_changed += Int32(1)
                            kcs = Int(unsafe_load(mcstrt, t1))
                            kce = kcs + Int(unsafe_load(hincol, t1))
                            kc = kcs
                            while kc < kce
                                rk1 = Int(unsafe_load(hrow, kc + 1)) + 1
                                if unsafe_load(mark_row, rk1) == mark_actok
                                    unsafe_store!(mark_row, mark_inplay, rk1)
                                end
                                kc += 1
                            end
                            if lt <= -large
                                fin_up = _fmadd(newlt, ait, fin_up)
                                inf_up -= Int32(1)
                            else
                                fin_up = _fmadd(newlt - lt, ait, fin_up)
                            end
                            lt = newlt
                        end
                    end
                end
                kcol += 1
            end
            i += 1
        end
        if i_pass == Int32(1)
            num_check = max(Int32(10), num_changed >> 5)
        end
        if num_infeas != Int32(0)
            _ps_store_i32!(p_ptr, _PS_OFF_status, Int32(1))
            break
        end
    end

    nfixed = Int32(0)
    if num_infeas == Int32(0)
        i = 0
        while i < m
            i1 = i + 1
            if unsafe_load(mark_row, i1) == mark_ignore
                i += 1
                continue
            end
            inf_up = Int32(0); inf_lo = Int32(0)
            fin_up = 0.0; fin_dn = 0.0
            krs = Int(unsafe_load(mrstrt, i1))
            kre = krs + Int(unsafe_load(hinrow, i1))
            krow = krs
            while krow < kre
                v = unsafe_load(rowels, krow + 1)
                j1 = Int(unsafe_load(hcol, krow + 1)) + 1
                if v > 0.0
                    uj = unsafe_load(col_upper, j1)
                    if uj < large
                        fin_up = _fmadd(uj, v, fin_up)
                    else
                        inf_up += Int32(1)
                    end
                    lj = unsafe_load(col_lower, j1)
                    if lj > -large
                        fin_dn = _fmadd(lj, v, fin_dn)
                    else
                        inf_lo += Int32(1)
                    end
                elseif v < 0.0
                    uj = unsafe_load(col_upper, j1)
                    if uj < large
                        fin_dn = _fmadd(uj, v, fin_dn)
                    else
                        inf_lo += Int32(1)
                    end
                    lj = unsafe_load(col_lower, j1)
                    if lj > -large
                        fin_up = _fmadd(lj, v, fin_up)
                    else
                        inf_up += Int32(1)
                    end
                end
                krow += 1
            end
            fin_up = _fmadd(1.0e-8, abs(fin_up), fin_up)
            fin_dn = _fmsub(1.0e-8, abs(fin_dn), fin_dn)
            max_up = _fmadd(Float64(inf_up), 1.0e31, fin_up)
            max_dn = _fmsub(Float64(inf_lo), 1.0e31, fin_dn)
            if max_up <= unsafe_load(rup, i1) + feas_tol && max_dn >= unsafe_load(rlo, i1) - feas_tol
                nuseless += Int32(1)
                unsafe_store!(useless_rows, Int32(i), Int(nuseless))
            end
            i += 1
        end

        if (p.presolveOptions & Int32(0x10)) != Int32(0)
            int_type = p.integerType
            csol = p.sol; clo = p.clo; cup = p.cup
            fixed = p.usefulColumnInt
            j = 0
            while j < n
                j1 = j + 1
                if unsafe_load(clo, j1) != unsafe_load(cup, j1)
                    lo = unsafe_load(col_lower, j1)
                    hi = unsafe_load(col_upper, j1)
                    is_int = unsafe_load(int_type, j1) != UInt8(0)
                    if is_int
                        hi = floor(hi + 1.0e-4)
                        lo = ceil(lo - 1.0e-4)
                    end
                    if hi - lo < 1.0e-8
                        if hi - lo < -feas_tol
                            num_infeas += Int32(1)
                        end
                        if min(abs(hi), abs(lo)) <= 1.0e-7
                            hi = 0.0
                        end
                        nfixed += Int32(1)
                        unsafe_store!(fixed, Int32(j), Int(nfixed))
                        n_next_cols = ps_add_col!(p.colChanged, p.nextColsToDo, n_next_cols, Int32(j))
                        unsafe_store!(cup, hi, j1)
                        unsafe_store!(clo, hi, j1)
                        if csol != C_NULL
                            unsafe_store!(csol, hi, j1)
                        end
                    elseif is_int
                        if hi < unsafe_load(cup, j1)
                            unsafe_store!(cup, hi, j1)
                            n_next_cols = ps_add_col!(p.colChanged, p.nextColsToDo, n_next_cols, Int32(j))
                        end
                        if lo > unsafe_load(clo, j1)
                            unsafe_store!(clo, lo, j1)
                            n_next_cols = ps_add_col!(p.colChanged, p.nextColsToDo, n_next_cols, Int32(j))
                        end
                    end
                end
                j += 1
            end
        end
    end

    _ps_store_i32!(p_ptr, _PS_OFF_numberNextRowsToDo, n_next_rows)
    _ps_store_i32!(p_ptr, _PS_OFF_numberNextColsToDo, n_next_cols)
    unsafe_store!(out_nuseless, nuseless)
    unsafe_store!(out_nfixed, nfixed)
    unsafe_store!(out_infeas_row, infeas_row)
    return num_infeas
end

function ps_useless_remove_rows!(
    p_ptr::Ptr{CoinPresolvePodView},
    useless_rows::Ptr{Int32},
    nuseless::Int32,
)::Nothing
    p = unsafe_load(p_ptr)
    mcstrt = p.mcstrt; hincol = p.hincol; hrow = p.hrow; colels = p.colels
    mrstrt = p.mrstrt; hinrow = p.hinrow; hcol = p.hcol
    rlo = p.rlo; rup = p.rup; clink = p.clink; rlink = p.rlink
    nu = Int(nuseless)
    i = 1
    while i <= nu
        irow = unsafe_load(useless_rows, i)
        ir1 = Int(irow) + 1
        krs = Int(unsafe_load(mrstrt, ir1))
        kre = krs + Int(unsafe_load(hinrow, ir1))
        k = krs
        while k < kre
            jc = unsafe_load(hcol, k + 1)
            ps_delete_from_major!(jc, irow, mcstrt, hincol, hrow, colels)
            if unsafe_load(hincol, Int(jc) + 1) == Int32(0)
                ps_remove_link!(clink, jc)
            end
            k += 1
        end
        unsafe_store!(hinrow, Int32(0), ir1)
        ps_remove_link!(rlink, irow)
        unsafe_store!(rlo, 0.0, ir1)
        unsafe_store!(rup, 0.0, ir1)
        i += 1
    end
    return nothing
end

function ps_useless_postsolve!(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{UselessActionPod},
    nactions::Int32,
)::Nothing
    p = unsafe_load(p_ptr)
    colels = p.colels; hrow = p.hrow; mcstrt = p.mcstrt; lnk = p.link; hincol = p.hincol
    rowacts = p.acts; sol = p.sol; rlo = p.rlo; rup = p.rup
    fl = p.free_list
    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        irow = f.row
        ir1 = Int(irow) + 1
        nin = Int(f.ninrow)
        rc = f.rowcols
        re = f.rowels
        rowact = 0.0
        unsafe_store!(rup, f.rup, ir1)
        unsafe_store!(rlo, f.rlo, ir1)
        k = 1
        while k <= nin
            jc = unsafe_load(rc, k)
            jc1 = Int(jc) + 1
            kk = fl
            kk1 = Int(kk) + 1
            fl = unsafe_load(lnk, kk1)
            unsafe_store!(hrow, irow, kk1)
            val = unsafe_load(re, k)
            unsafe_store!(colels, val, kk1)
            unsafe_store!(lnk, unsafe_load(mcstrt, jc1), kk1)
            unsafe_store!(mcstrt, kk, jc1)
            rowact = _fmadd(val, unsafe_load(sol, jc1), rowact)
            unsafe_store!(hincol, unsafe_load(hincol, jc1) + Int32(1), jc1)
            k += 1
        end
        unsafe_store!(rowacts, rowact, ir1)
        idx -= 1
    end
    _post_store_i32!(p_ptr, _POST_OFF_free_list, fl)
    return nothing
end

# =============================================================================
# 6. CoinPresolveFixed (CoinPresolveFixed.cpp)
# =============================================================================

function ps_remove_fixed_filter!(
    p_ptr::Ptr{CoinPresolvePodView},
    fcols::Ptr{Int32},
    nfcols_ptr::Ptr{Int32},
)::Int32
    p = unsafe_load(p_ptr)
    nf = Int(unsafe_load(nfcols_ptr))
    est = Int32(0)
    n = Int32(0)
    k = 1
    while k <= nf
        j = unsafe_load(fcols, k)
        if !ps_col_prohibited2(p.anyProhibited, p.colChanged, j)
            est += unsafe_load(p.hincol, Int(j) + 1)
            n += Int32(1)
            unsafe_store!(fcols, j, Int(n))
        end
        k += 1
    end
    unsafe_store!(nfcols_ptr, n)
    return est
end

function ps_remove_fixed_presolve!(
    p_ptr::Ptr{CoinPresolvePodView},
    fcols::Ptr{Int32},
    nfcols::Int32,
    actions::Ptr{RemoveFixedActionPod},
    els_action::Ptr{Float64},
    rows_action::Ptr{Int32},
    rstrt::Ptr{Int32},
    column_buf::Ptr{Int32},
    mark_buf::Ptr{UInt8},
)::Int32
    p = unsafe_load(p_ptr)
    colels = p.colels; hrow = p.hrow; mcstrt = p.mcstrt; hincol = p.hincol
    rowels = p.rowels; hcol = p.hcol; mrstrt = p.mrstrt; hinrow = p.hinrow
    clo = p.clo; rlo = p.rlo; rup = p.rup; sol = p.sol; acts = p.acts
    clink = p.clink; rlink = p.rlink
    nrows = Int(p.nrows); ncols = Int(p.ncols)
    n_next_rows = p.numberNextRowsToDo
    n_next_cols = p.numberNextColsToDo

    coin_zero_n!(rstrt, nrows)
    actsize = Int32(0)
    nf = Int(nfcols)
    ckc = 0
    while ckc < nf
        j = unsafe_load(fcols, ckc + 1)
        j1 = Int(j) + 1
        solj = unsafe_load(clo, j1)
        kcs = Int(unsafe_load(mcstrt, j1))
        kce = kcs + Int(unsafe_load(hincol, j1))
        unsafe_store!(actions, RemoveFixedActionPod(j, actsize, solj), ckc + 1)
        k = kcs
        while k < kce
            row = unsafe_load(hrow, k + 1)
            r1 = Int(row) + 1
            coeff = unsafe_load(colels, k + 1)
            actsize += Int32(1)
            unsafe_store!(els_action, coeff, Int(actsize))
            unsafe_store!(rows_action, row, Int(actsize))
            unsafe_store!(rstrt, unsafe_load(rstrt, r1) + Int32(1), r1)
            rl = unsafe_load(rlo, r1)
            if -PRESOLVE_INF_DBL < rl
                unsafe_store!(rlo, _fmsub(solj, coeff, rl), r1)
            end
            ru = unsafe_load(rup, r1)
            if ru < PRESOLVE_INF_DBL
                unsafe_store!(rup, _fmsub(solj, coeff, ru), r1)
            end
            if sol != C_NULL
                unsafe_store!(acts, _fmsub(solj, coeff, unsafe_load(acts, r1)), r1)
            end
            k += 1
        end
        ps_remove_link!(clink, j)
        unsafe_store!(hincol, Int32(0), j1)
        ckc += 1
    end
    unsafe_store!(actions, RemoveFixedActionPod(Int32(0), actsize, 0.0), nf + 1)

    # Bucket columns by row
    nel = Int32(0)
    ir = 0
    while ir < nrows
        n = unsafe_load(rstrt, ir + 1)
        unsafe_store!(rstrt, nel, ir + 1)
        nel += n
        ir += 1
    end
    unsafe_store!(rstrt, nel, nrows + 1)

    ckc = 0
    while ckc < nf
        a = unsafe_load(actions, ckc + 1)
        kcs = Int(a.start)
        j = a.col
        kce = ckc < nf - 1 ? Int(unsafe_load(actions, ckc + 2).start) : Int(actsize)
        k = kcs
        while k < kce
            r1 = Int(unsafe_load(rows_action, k + 1)) + 1
            put = unsafe_load(rstrt, r1)
            unsafe_store!(rstrt, put + Int32(1), r1)
            unsafe_store!(column_buf, j, Int(put) + 1)
            k += 1
        end
        ckc += 1
    end

    # Zero mark_buf
    i_m = 1
    while i_m <= ncols
        unsafe_store!(mark_buf, UInt8(0), i_m)
        i_m += 1
    end

    nel_i = 0
    ir = 0
    while ir < nrows
        ir1 = ir + 1
        end_r = Int(unsafe_load(rstrt, ir1))
        k = nel_i
        while k < end_r
            unsafe_store!(mark_buf, UInt8(1), Int(unsafe_load(column_buf, k + 1)) + 1)
            k += 1
        end
        ps_delete_many_from_major!(Int32(ir), mark_buf, mrstrt, hinrow, hcol, rowels)
        if unsafe_load(hinrow, ir1) == Int32(0)
            ps_remove_link!(rlink, Int32(ir))
        end
        if !ps_row_changed(p.rowChanged, Int32(ir))
            n_next_rows = ps_add_row!(p.rowChanged, p.nextRowsToDo, n_next_rows, Int32(ir))
            krs = Int(unsafe_load(mrstrt, ir1))
            kre = krs + Int(unsafe_load(hinrow, ir1))
            kr = krs
            while kr < kre
                jc = unsafe_load(hcol, kr + 1)
                n_next_cols = ps_add_col!(p.colChanged, p.nextColsToDo, n_next_cols, jc)
                kr += 1
            end
        end
        nel_i = end_r
        ir += 1
    end

    _ps_store_i32!(p_ptr, _PS_OFF_numberNextRowsToDo, n_next_rows)
    _ps_store_i32!(p_ptr, _PS_OFF_numberNextColsToDo, n_next_cols)
    return actsize
end

function ps_remove_fixed_postsolve!(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{RemoveFixedActionPod},
    nactions::Int32,
    els_action::Ptr{Float64},
    rows_action::Ptr{Int32},
)::Nothing
    p = unsafe_load(p_ptr)
    colels = p.colels; hrow = p.hrow; mcstrt = p.mcstrt; hincol = p.hincol; lnk = p.link
    clo = p.clo; cup = p.cup; rlo = p.rlo; rup = p.rup
    sol = p.sol; dcost = p.cost; rcosts = p.rcosts; acts = p.acts; rowduals = p.rowduals
    colstat = p.colstat; maxmin = p.maxmin
    fl = p.free_list

    na = Int(nactions)
    en = Int(unsafe_load(actions, na + 1).start)
    idx = na
    while idx >= 1
        f = unsafe_load(actions, idx)
        icol = f.col
        ic1 = Int(icol) + 1
        thesol = f.sol
        unsafe_store!(sol, thesol, ic1)
        unsafe_store!(clo, thesol, ic1)
        unsafe_store!(cup, thesol, ic1)
        cs = PRESOLVE_NO_LINK
        st_i = Int(f.start)
        dj = maxmin * unsafe_load(dcost, ic1)
        i = st_i
        while i < en
            row = unsafe_load(rows_action, i + 1)
            r1 = Int(row) + 1
            coeff = unsafe_load(els_action, i + 1)
            k = fl
            k1 = Int(k) + 1
            fl = unsafe_load(lnk, k1)
            unsafe_store!(hrow, row, k1)
            unsafe_store!(colels, coeff, k1)
            unsafe_store!(lnk, cs, k1)
            cs = k
            rl = unsafe_load(rlo, r1)
            if -PRESOLVE_INF_DBL < rl
                unsafe_store!(rlo, _fmadd(coeff, thesol, rl), r1)
            end
            ru = unsafe_load(rup, r1)
            if ru < PRESOLVE_INF_DBL
                unsafe_store!(rup, _fmadd(coeff, thesol, ru), r1)
            end
            unsafe_store!(acts, _fmadd(coeff, thesol, unsafe_load(acts, r1)), r1)
            dj = _fmsub(unsafe_load(rowduals, r1), coeff, dj)
            i += 1
        end
        unsafe_store!(mcstrt, cs, ic1)
        unsafe_store!(rcosts, dj, ic1)
        unsafe_store!(hincol, Int32(en - st_i), ic1)
        en = st_i
        if colstat != C_NULL
            ps_set_col_status!(colstat, icol, dj < 0.0 ? PS_STATUS_AT_UPPER_BOUND : PS_STATUS_AT_LOWER_BOUND)
        end
        idx -= 1
    end
    _post_store_i32!(p_ptr, _POST_OFF_free_list, fl)
    return nothing
end

function ps_make_fixed_presolve!(
    p_ptr::Ptr{CoinPresolvePodView},
    fcols::Ptr{Int32},
    nfcols::Int32,
    fix_to_lower::Bool,
    actions::Ptr{MakeFixedActionPod},
)::Nothing
    p = unsafe_load(p_ptr)
    clo = p.clo; cup = p.cup; csol = p.sol
    colels = p.colels; hrow = p.hrow; mcstrt = p.mcstrt; hincol = p.hincol; acts = p.acts
    nf = Int(nfcols)
    ckc = 1
    while ckc <= nf
        j = unsafe_load(fcols, ckc)
        j1 = Int(j) + 1
        movement = 0.0
        if fix_to_lower
            bnd = unsafe_load(cup, j1)
            lj = unsafe_load(clo, j1)
            unsafe_store!(cup, lj, j1)
            if csol != C_NULL
                movement = lj - unsafe_load(csol, j1)
                unsafe_store!(csol, lj, j1)
            end
            unsafe_store!(actions, MakeFixedActionPod(bnd, j, Int32(0)), ckc)
        else
            bnd = unsafe_load(clo, j1)
            uj = unsafe_load(cup, j1)
            unsafe_store!(clo, uj, j1)
            if csol != C_NULL
                movement = uj - unsafe_load(csol, j1)
                unsafe_store!(csol, uj, j1)
            end
            unsafe_store!(actions, MakeFixedActionPod(bnd, j, Int32(0)), ckc)
        end
        if movement != 0.0
            ks = Int(unsafe_load(mcstrt, j1))
            ke = ks + Int(unsafe_load(hincol, j1))
            k = ks
            while k < ke
                r1 = Int(unsafe_load(hrow, k + 1)) + 1
                unsafe_store!(acts, _fmadd(movement, unsafe_load(colels, k + 1), unsafe_load(acts, r1)), r1)
                k += 1
            end
        end
        ckc += 1
    end
    return nothing
end

function ps_make_fixed_postsolve!(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{MakeFixedActionPod},
    nactions::Int32,
    fix_to_lower::Bool,
)::Nothing
    p = unsafe_load(p_ptr)
    clo = p.clo; cup = p.cup; sol = p.sol; colstat = p.colstat
    cnt = Int(nactions)
    while cnt >= 1
        f = unsafe_load(actions, cnt)
        icol = f.col
        ic1 = Int(icol) + 1
        xj = unsafe_load(sol, ic1)
        if fix_to_lower
            ub = f.bound
            unsafe_store!(cup, ub, ic1)
            if colstat != C_NULL
                if ub >= PRESOLVE_INF_DBL || xj != ub
                    ps_set_col_status!(colstat, icol, PS_STATUS_AT_LOWER_BOUND)
                end
            end
        else
            lb = f.bound
            unsafe_store!(clo, lb, ic1)
            if colstat != C_NULL
                if lb <= -PRESOLVE_INF_DBL || xj != lb
                    ps_set_col_status!(colstat, icol, PS_STATUS_AT_UPPER_BOUND)
                end
            end
        end
        cnt -= 1
    end
    return nothing
end

function ps_transfer_costs!(p_ptr::Ptr{CoinPresolvePodView})::Nothing
    p = unsafe_load(p_ptr)
    colels = p.colels; hrow = p.hrow; mcstrt = p.mcstrt; hincol = p.hincol
    rowels = p.rowels; hcol = p.hcol; mrstrt = p.mrstrt; hinrow = p.hinrow
    rlo = p.rlo; rup = p.rup; clo = p.clo; cup = p.cup
    ncols = Int(p.ncols); cost = p.cost; int_type = p.integerType
    bias = p.dobias

    js = 0
    while js < ncols
        js1 = js + 1
        if unsafe_load(hincol, js1) == Int32(1) && unsafe_load(cost, js1) != 0.0 &&
           unsafe_load(cup, js1) > unsafe_load(clo, js1) && unsafe_load(int_type, js1) == UInt8(0)
            jsstrt = Int(unsafe_load(mcstrt, js1))
            i = Int(unsafe_load(hrow, jsstrt + 1))
            i1 = i + 1
            rl = unsafe_load(rlo, i1)
            if rl == unsafe_load(rup, i1)
                ratio = unsafe_load(cost, js1) / unsafe_load(colels, jsstrt + 1)
                bias = _fmadd(rl, ratio, bias)
                istrt = Int(unsafe_load(mrstrt, i1))
                iend = istrt + Int(unsafe_load(hinrow, i1))
                jj = istrt
                while jj < iend
                    j1 = Int(unsafe_load(hcol, jj + 1)) + 1
                    aij = unsafe_load(rowels, jj + 1)
                    unsafe_store!(cost, _fmsub(ratio, aij, unsafe_load(cost, j1)), j1)
                    jj += 1
                end
                unsafe_store!(cost, 0.0, js1)
            end
        end
        js += 1
    end
    _ps_store_f64!(p_ptr, _PS_OFF_dobias, bias)
    return nothing
end

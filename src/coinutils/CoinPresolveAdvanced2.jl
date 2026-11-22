# Faithful translation of CoinUtils Presolve B7c2:
#   - CoinPresolveSubst.cpp   (subst_constraint_action)
#   - CoinPresolveDupcol.cpp  (coin_init_random_vec, dupcol_action, duprow_action,
#                              gubrow_action, twoxtwo_action)
# Zero-allocation, bit-identical arm64 FMA (`_fmadd`/`_fmsub`), `--trim=safe` compatible.

@inline function ps_row_prohibited2(any_prohib::Int32, row_changed::Ptr{UInt8}, i::Int32)::Bool
    any_prohib == Int32(0) && return false
    return (unsafe_load(row_changed, Int(i) + 1) & UInt8(2)) != UInt8(0)
end

# =============================================================================
# 1. CoinPresolveSubst.cpp (subst_constraint_action)
# =============================================================================

function ps_subst_add_row!(
    mrstrt::Ptr{Int32},
    rlo::Ptr{Float64},
    acts::Ptr{Float64},
    rup::Ptr{Float64},
    rowels::Ptr{Float64},
    hcol::Ptr{Int32},
    hinrow::Ptr{Int32},
    rlink::Ptr{PresolveHLinkPod},
    nrows::Int32,
    coeff_factor::Float64,
    kill_ratio::Float64,
    irowx::Int32,
    irowy::Int32,
    x_to_y::Ptr{Int32},
)::Bool
    rx1 = Int(irowx) + 1
    ry1 = Int(irowy) + 1
    krsy = unsafe_load(mrstrt, ry1)
    krey = krsy + unsafe_load(hinrow, ry1)
    krsx = unsafe_load(mrstrt, rx1)
    krex = krsx + unsafe_load(hinrow, rx1)

    rhsy = unsafe_load(rlo, ry1)
    rhscorr = rhsy * coeff_factor
    tolerance = kill_ratio * coeff_factor

    rlo_x = unsafe_load(rlo, rx1)
    if -PRESOLVE_INF_DBL < rlo_x
        unsafe_store!(rlo, rlo_x + rhscorr, rx1)
    end
    rup_x = unsafe_load(rup, rx1)
    if rup_x < PRESOLVE_INF_DBL
        unsafe_store!(rup, rup_x + rhscorr, rx1)
    end
    if acts != C_NULL
        unsafe_store!(acts, unsafe_load(acts, rx1) + rhscorr, rx1)
    end

    krowx = krsx
    krex0 = krex
    x_to_y_i = 0

    krowy = krsy
    while krowy < krey
        ky1 = Int(krowy) + 1
        j = unsafe_load(hcol, ky1)

        while krowx < krex0 && unsafe_load(hcol, Int(krowx) + 1) < j
            krowx += Int32(1)
        end

        if krowx < krex0 && unsafe_load(hcol, Int(krowx) + 1) == j
            kx1 = Int(krowx) + 1
            newcoeff = _fmadd(unsafe_load(rowels, ky1), coeff_factor, unsafe_load(rowels, kx1))
            if abs(newcoeff) < tolerance
                newcoeff = 0.0
            end
            unsafe_store!(rowels, newcoeff, kx1)
            x_to_y_i += 1
            unsafe_store!(x_to_y, krowx - krsx, x_to_y_i)
            krowx += Int32(1)
        else
            newValue = unsafe_load(rowels, ky1) * coeff_factor
            if ps_expand_major!(mrstrt, rowels, hcol, hinrow, rlink, nrows, irowx)
                return true
            end
            krowy = unsafe_load(mrstrt, ry1) + (krowy - krsy)
            krsy = unsafe_load(mrstrt, ry1)
            krey = krsy + unsafe_load(hinrow, ry1)

            krowx = unsafe_load(mrstrt, rx1) + (krowx - krsx)
            krex0 = unsafe_load(mrstrt, rx1) + (krex0 - krsx)
            krsx = unsafe_load(mrstrt, rx1)
            lenx = unsafe_load(hinrow, rx1)
            krex = krsx + lenx

            krex1 = Int(krex) + 1
            unsafe_store!(hcol, j, krex1)
            unsafe_store!(rowels, newValue, krex1)
            x_to_y_i += 1
            unsafe_store!(x_to_y, krex - krsx, x_to_y_i)
            unsafe_store!(hinrow, lenx + Int32(1), rx1)
            krex += Int32(1)
        end
        krowy += Int32(1)
    end
    return false
end

function ps_subst_presolve!(
    pv::Ptr{CoinPresolvePodView},
    implied_free::Ptr{Int32},
    whichFree::Ptr{Int32},
    numberFree::Int32,
    maxLook::Int32,
    actions_out::Ptr{SubstActionPod},
    zerocols::Ptr{Int32},
    x_to_y::Ptr{Int32},
    alloc_subst_cb::Ptr{Cvoid},
    out_nactions::Ptr{Int32},
    out_nzerocols::Ptr{Int32},
    out_nomem::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    ncols = p.ncols
    nrows = p.nrows

    rowStarts = p.mrstrt
    rowLengths = p.hinrow
    rowCoeffs = p.rowels
    colIndices = p.hcol
    rlink = p.rlink

    colStarts = p.mcstrt
    colLengths = p.hincol
    colCoeffs = p.colels
    rowIndices = p.hrow
    clink = p.clink

    rlo = p.rlo
    rup = p.rup
    acts = p.acts
    cost = p.cost
    tol = p.feasibilityTolerance

    col_changed = p.colChanged
    row_changed = p.rowChanged
    next_cols = p.nextColsToDo
    next_rows = p.nextRowsToDo
    n_next_cols = p.numberNextColsToDo
    n_next_rows = p.numberNextRowsToDo
    dobias = p.dobias

    rowsUsed = p.usefulRowInt
    nRowsUsed = Int32(0)
    nactions = Int32(0)
    nzerocols = Int32(0)
    nomem = Int32(0)

    relax = (p.presolveOptions & Int32(0x60000)) >> 17
    tolerance = 1.0e-12
    ri = Int32(0)
    while ri < relax
        tolerance *= 10.0
        ri += Int32(1)
    end

    nf = Int(numberFree)
    iLook = 1
    while iLook <= nf
        tgtcol = unsafe_load(whichFree, iLook)
        tc1 = Int(tgtcol) + 1
        tgtcol_len = unsafe_load(colLengths, tc1)
        tgtrow = unsafe_load(implied_free, iLook)
        tr1 = Int(tgtrow) + 1
        tgtrow_len = unsafe_load(rowLengths, tr1)

        if tgtcol_len < Int32(2) || tgtcol_len > maxLook
            iLook += 1
            continue
        end

        tgtcs = Int(unsafe_load(colStarts, tc1))
        tgtce = tgtcs + Int(tgtcol_len)

        tgtcoeff = 0.0
        dealBreaker = false
        kcol = tgtcs
        while kcol < tgtce
            kc1 = kcol + 1
            ir = unsafe_load(rowIndices, kc1)
            if unsafe_load(rowLengths, Int(ir) + 1) < Int32(2) || ps_row_used(row_changed, ir)
                dealBreaker = true
                break
            end
            aij = unsafe_load(colCoeffs, kc1)
            if abs(aij) <= PRESOLVE_ZTOLDP2
                dealBreaker = true
                break
            end
            if ir == tgtrow
                tgtcoeff = aij
            end
            kcol += 1
        end
        if dealBreaker
            iLook += 1
            continue
        end

        kcol = tgtcs
        while kcol < tgtce
            if abs(unsafe_load(colCoeffs, kcol + 1) / tgtcoeff) > 10.0
                dealBreaker = true
            end
            kcol += 1
        end
        if !dealBreaker && (nzerocols + tgtrow_len >= ncols)
            dealBreaker = true
        end
        if dealBreaker
            iLook += 1
            continue
        end

        nonzero_cost = abs(unsafe_load(cost, tc1)) > tol
        costs_len = nonzero_cost ? tgtrow_len : Int32(0)

        ntotels = Int32(0)
        kcol = tgtcs
        while kcol < tgtce
            ir = unsafe_load(rowIndices, kcol + 1)
            ntotels += unsafe_load(rowLengths, Int(ir) + 1)
            ps_set_row_used!(row_changed, ir)
            nRowsUsed += Int32(1)
            unsafe_store!(rowsUsed, ir, Int(nRowsUsed))
            kcol += 1
        end

        slot_ptr = actions_out + Int(nactions) * sizeof(SubstActionPod)
        nactions += Int32(1)
        ccall(alloc_subst_cb, Cvoid,
              (Ptr{SubstActionPod}, Int32, Int32, Int32),
              slot_ptr, tgtcol_len, ntotels, costs_len)
        ap = unsafe_load(slot_ptr)
        ap_rows = ap.rows
        ap_rlos = ap.rlos
        ap_rups = ap.rups
        ap_coeffxs = ap.coeffxs
        ap_ninrowxs = ap.ninrowxs
        ap_rowcolsxs = ap.rowcolsxs
        ap_rowelsxs = ap.rowelsxs
        ap_costsx = ap.costsx

        unsafe_store!(slot_ptr,
            SubstActionPod(ap_rlos, ap_rups, ap_coeffxs, ap_rows, ap_ninrowxs,
                           ap_rowcolsxs, ap_rowelsxs, ap_costsx,
                           tgtcol, tgtrow, tgtcol_len, Int32(0)))

        off = 0
        kcol = tgtcs
        while kcol < tgtce
            ndx1 = (kcol - tgtcs) + 1
            ir = unsafe_load(rowIndices, kcol + 1)
            ir1 = Int(ir) + 1
            krs = Int(unsafe_load(rowStarts, ir1))
            rlen = unsafe_load(rowLengths, ir1)
            n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, ir)
            unsafe_store!(ap_rows, ir, ndx1)
            unsafe_store!(ap_ninrowxs, rlen, ndx1)
            unsafe_store!(ap_rlos, unsafe_load(rlo, ir1), ndx1)
            unsafe_store!(ap_rups, unsafe_load(rup, ir1), ndx1)
            unsafe_store!(ap_coeffxs, unsafe_load(colCoeffs, kcol + 1), ndx1)
            rleni = Int(rlen)
            kk = 1
            while kk <= rleni
                unsafe_store!(ap_rowcolsxs, unsafe_load(colIndices, krs + kk), off + kk)
                unsafe_store!(ap_rowelsxs, unsafe_load(rowCoeffs, krs + kk), off + kk)
                kk += 1
            end
            off += rleni
            kcol += 1
        end

        tgtrs = Int(unsafe_load(rowStarts, tr1))
        tgtre = tgtrs + Int(tgtrow_len)

        if nonzero_cost
            tgtcost = unsafe_load(cost, tc1)
            krow = tgtrs
            while krow < tgtre
                kr1 = krow + 1
                j = unsafe_load(colIndices, kr1)
                j1 = Int(j) + 1
                n_next_cols = ps_add_col!(col_changed, next_cols, n_next_cols, j)
                cj = unsafe_load(cost, j1)
                unsafe_store!(ap_costsx, cj, (krow - tgtrs) + 1)
                unsafe_store!(cost, cj - (tgtcost * unsafe_load(rowCoeffs, kr1)) / tgtcoeff, j1)
                krow += 1
            end
            dobias += (tgtcost * unsafe_load(rlo, tr1)) / tgtcoeff
            unsafe_store!(cost, 0.0, tc1)
        end

        coin_sort_2_std!(colIndices + tgtrs * sizeof(Int32),
                         rowCoeffs + tgtrs * sizeof(Float64), Int(tgtrow_len))

        tlen_i = Int(tgtcol_len)
        colndx = 1
        while colndx <= tlen_i
            ir = unsafe_load(ap_rows, colndx)
            if ir == tgtrow
                colndx += 1
                continue
            end
            ait = unsafe_load(ap_coeffxs, colndx)
            coeff_factor = -ait / tgtcoeff
            ir1 = Int(ir) + 1
            krs = Int(unsafe_load(rowStarts, ir1))
            rlen_i = Int(unsafe_load(rowLengths, ir1))

            coin_sort_2_std!(colIndices + krs * sizeof(Int32),
                             rowCoeffs + krs * sizeof(Float64), rlen_i)

            if ps_subst_add_row!(rowStarts, rlo, acts, rup, rowCoeffs, colIndices,
                                 rowLengths, rlink, nrows, coeff_factor, tolerance,
                                 ir, tgtrow, x_to_y)
                nomem = Int32(1)
                break
            end

            krs = Int(unsafe_load(rowStarts, ir1))
            tgtrs = Int(unsafe_load(rowStarts, tr1))
            tgtre = tgtrs + Int(unsafe_load(rowLengths, tr1))

            trlen_i = Int(tgtrow_len)
            rowndx = 0
            while rowndx < trlen_i
                j = unsafe_load(colIndices, tgtrs + rowndx + 1)
                j1 = Int(j) + 1
                kcs = unsafe_load(colStarts, j1)
                clen_j = unsafe_load(colLengths, j1)
                kce = kcs + clen_j
                coeff = unsafe_load(rowCoeffs, krs + Int(unsafe_load(x_to_y, rowndx + 1)) + 1)
                kcol_pos = ps_find_minor1(ir, kcs, kce, rowIndices)
                if kcol_pos < kce
                    unsafe_store!(colCoeffs, coeff, Int(kcol_pos) + 1)
                else
                    if ps_expand_major!(colStarts, colCoeffs, rowIndices, colLengths, clink, ncols, j)
                        nomem = Int32(1)
                        break
                    end
                    kcs = unsafe_load(colStarts, j1)
                    clen_j = unsafe_load(colLengths, j1)
                    kce = kcs + clen_j
                    kce1 = Int(kce) + 1
                    unsafe_store!(rowIndices, ir, kce1)
                    unsafe_store!(colCoeffs, coeff, kce1)
                    unsafe_store!(colLengths, clen_j + Int32(1), j1)
                end
                rowndx += 1
            end
            nomem != Int32(0) && break

            ps_delete_from_major!(ir, tgtcol, rowStarts, rowLengths, colIndices, rowCoeffs)
            colndx += 1
        end
        nomem != Int32(0) && break

        tgtrs = Int(unsafe_load(rowStarts, tr1))
        trlen_now = Int(unsafe_load(rowLengths, tr1))
        kk = 1
        while kk <= trlen_now
            nzerocols += Int32(1)
            unsafe_store!(zerocols, unsafe_load(colIndices, tgtrs + kk), Int(nzerocols))
            kk += 1
        end

        tgtre = tgtrs + trlen_now
        krow = tgtrs
        while krow < tgtre
            j = unsafe_load(colIndices, krow + 1)
            ps_delete_from_major!(j, tgtrow, colStarts, colLengths, rowIndices, colCoeffs)
            if unsafe_load(colLengths, Int(j) + 1) == Int32(0)
                ps_remove_link!(clink, j)
            end
            krow += 1
        end

        ps_remove_link!(clink, tgtcol)
        unsafe_store!(colLengths, Int32(0), tc1)

        ps_remove_link!(rlink, tgtrow)
        unsafe_store!(rowLengths, Int32(0), tr1)
        unsafe_store!(rlo, 0.0, tr1)
        unsafe_store!(rup, 0.0, tr1)

        iLook += 1
    end

    nru = Int(nRowsUsed)
    k = 1
    while k <= nru
        ps_unset_row_used!(row_changed, unsafe_load(rowsUsed, k))
        k += 1
    end

    _ps_store_f64!(pv, _PS_OFF_dobias, dobias)
    _ps_store_i32!(pv, _PS_OFF_numberNextColsToDo, n_next_cols)
    _ps_store_i32!(pv, _PS_OFF_numberNextRowsToDo, n_next_rows)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_nzerocols, nzerocols)
    unsafe_store!(out_nomem, nomem)
    return nothing
end

function ps_subst_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{SubstActionPod},
    nactions::Int32,
)::Nothing
    p = unsafe_load(postv)
    colStarts = p.mcstrt
    colLengths = p.hincol
    rowIndices = p.hrow
    colCoeffs = p.colels

    rlo = p.rlo
    rup = p.rup
    cost = p.cost
    sol = p.sol
    rcosts = p.rcosts
    acts = p.acts
    rowduals = p.rowduals
    colstat = p.colstat
    rowstat = p.rowstat
    link = p.link
    maxmin = p.maxmin
    fl_ptr = Ptr{Int32}(UInt(postv) + _POST_OFF_free_list)

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        tgtcol = f.col
        tc1 = Int(tgtcol) + 1
        tgtrow = f.rowy
        tr1 = Int(tgtrow) + 1
        tgtcol_len = Int(f.nincol)
        tgtcol_coeffs = f.coeffxs

        entngld_rows = f.rows
        entngld_lens = f.ninrowxs
        entngld_colndxs = f.rowcolsxs
        entngld_colcoeffs = f.rowelsxs
        entngld_rlos = f.rlos
        entngld_rups = f.rups
        costs = f.costsx

        tgtrow_len = 0
        tgtrow_colndxs = Ptr{Int32}(C_NULL)
        tgtrow_coeffs = Ptr{Float64}(C_NULL)
        tgtcoeff = 0.0
        tgtrhs = 1.0e50

        nel = 0
        cndx = 1
        while cndx <= tgtcol_len
            ir = unsafe_load(entngld_rows, cndx)
            ir1 = Int(ir) + 1
            rlo_v = unsafe_load(entngld_rlos, cndx)
            rup_v = unsafe_load(entngld_rups, cndx)
            unsafe_store!(rlo, rlo_v, ir1)
            unsafe_store!(rup, rup_v, ir1)
            leni = Int(unsafe_load(entngld_lens, cndx))
            if ir == tgtrow
                tgtrow_len = leni
                tgtrow_colndxs = entngld_colndxs + nel * sizeof(Int32)
                tgtrow_coeffs = entngld_colcoeffs + nel * sizeof(Float64)
                tgtcoeff = unsafe_load(tgtcol_coeffs, cndx)
                tgtrhs = rlo_v
            end
            nel += leni
            cndx += 1
        end

        tgtexp = tgtrhs
        unsafe_store!(sol, 0.0, tc1)
        ndx = 1
        while ndx <= tgtrow_len
            j1 = Int(unsafe_load(tgtrow_colndxs, ndx)) + 1
            tgtexp = _fmsub(unsafe_load(tgtrow_coeffs, ndx), unsafe_load(sol, j1), tgtexp)
            ndx += 1
        end
        unsafe_store!(sol, tgtexp / tgtcoeff, tc1)

        # Delete coefficients of tgtrow's cols from all other entangled rows
        rndx = 1
        while rndx <= tgtrow_len
            j = unsafe_load(tgtrow_colndxs, rndx)
            if j != tgtcol
                cndx = 1
                while cndx <= tgtcol_len
                    ir = unsafe_load(entngld_rows, cndx)
                    if ir != tgtrow
                        ps_delete_from_major2!(j, ir, colStarts, colLengths, rowIndices, link, fl_ptr)
                    end
                    cndx += 1
                end
            end
            rndx += 1
        end

        free_list = unsafe_load(fl_ptr)
        unsafe_store!(colLengths, Int32(0), tc1)
        cols_i = entngld_colndxs
        coeffs_i = entngld_colcoeffs

        cndx = 1
        while cndx <= tgtcol_len
            leni = Int(unsafe_load(entngld_lens, cndx))
            ir = unsafe_load(entngld_rows, cndx)
            ir1 = Int(ir) + 1
            if ir != tgtrow
                acti = 0.0
                rndx = 1
                while rndx <= leni
                    j = unsafe_load(cols_i, rndx)
                    j1 = Int(j) + 1
                    c_val = unsafe_load(coeffs_i, rndx)
                    kcoli = ps_find_minor3(ir, unsafe_load(colStarts, j1),
                                           unsafe_load(colLengths, j1), rowIndices, link)
                    if kcoli != Int32(-1)
                        unsafe_store!(colCoeffs, c_val, Int(kcoli) + 1)
                    else
                        kk = free_list
                        kk1 = Int(kk) + 1
                        free_list = unsafe_load(link, kk1)
                        unsafe_store!(link, unsafe_load(colStarts, j1), kk1)
                        unsafe_store!(colStarts, kk, j1)
                        unsafe_store!(colCoeffs, c_val, kk1)
                        unsafe_store!(rowIndices, ir, kk1)
                        unsafe_store!(colLengths, unsafe_load(colLengths, j1) + Int32(1), j1)
                    end
                    acti = _fmadd(c_val, unsafe_load(sol, j1), acti)
                    rndx += 1
                end
                unsafe_store!(acts, acti, ir1)
            end
            cols_i += leni * sizeof(Int32)
            coeffs_i += leni * sizeof(Float64)
            cndx += 1
        end

        rndx = 1
        while rndx <= tgtrow_len
            j = unsafe_load(tgtrow_colndxs, rndx)
            j1 = Int(j) + 1
            kk = free_list
            kk1 = Int(kk) + 1
            free_list = unsafe_load(link, kk1)
            unsafe_store!(link, unsafe_load(colStarts, j1), kk1)
            unsafe_store!(colStarts, kk, j1)
            unsafe_store!(colCoeffs, unsafe_load(tgtrow_coeffs, rndx), kk1)
            unsafe_store!(rowIndices, tgtrow, kk1)
            unsafe_store!(colLengths, unsafe_load(colLengths, j1) + Int32(1), j1)
            rndx += 1
        end
        unsafe_store!(acts, tgtrhs, tr1)
        unsafe_store!(fl_ptr, free_list)

        if costs != C_NULL
            ndx = 1
            while ndx <= tgtrow_len
                unsafe_store!(cost, unsafe_load(costs, ndx), Int(unsafe_load(tgtrow_colndxs, ndx)) + 1)
                ndx += 1
            end
        end

        dj = maxmin * unsafe_load(cost, tc1)
        unsafe_store!(rowduals, 0.0, tr1)
        cndx = 1
        while cndx <= tgtcol_len
            ir1 = Int(unsafe_load(entngld_rows, cndx)) + 1
            dj = _fmsub(unsafe_load(rowduals, ir1), unsafe_load(tgtcol_coeffs, cndx), dj)
            cndx += 1
        end
        dual_t = dj / tgtcoeff
        unsafe_store!(rowduals, dual_t, tr1)
        unsafe_store!(rcosts, 0.0, tc1)
        if dual_t > 0.0
            ps_set_row_status!(rowstat, tgtrow, PS_STATUS_AT_LOWER_BOUND)
        else
            ps_set_row_status!(rowstat, tgtrow, PS_STATUS_AT_UPPER_BOUND)
        end
        ps_set_col_status!(colstat, tgtcol, PS_STATUS_BASIC)

        idx -= 1
    end
    return nothing
end

# =============================================================================
# 2. CoinPresolveDupcol.cpp — Random Vec, Compute Sums, Dupcol, Duprow, Gubrow, Twoxtwo
# =============================================================================

function ps_coin_init_random_vec!(work::Ptr{Float64}, n::Int32)::Nothing
    deseed = 12345678.0
    dseed2 = 2147483647.0
    ni = Int(n)
    i = 1
    while i <= ni
        deseed *= 16807.0
        jseed = unsafe_trunc(Int32, deseed / dseed2)
        deseed = _fmsub(Float64(jseed), dseed2, deseed)
        unsafe_store!(work, deseed / dseed2, i)
        i += 1
    end
    return nothing
end

function ps_compute_sums!(
    majlens::Ptr{Int32},
    majstrts::Ptr{Int32},
    minndxs::Ptr{Int32},
    elems::Ptr{Float64},
    minmuls::Ptr{Float64},
    majcands::Ptr{Int32},
    majsums::Ptr{Float64},
    nlook::Int32,
)::Nothing
    nl = Int(nlook)
    cndx = 1
    while cndx <= nl
        i1 = Int(unsafe_load(majcands, cndx)) + 1
        kcs = Int(unsafe_load(majstrts, i1))
        kce = kcs + Int(unsafe_load(majlens, i1))
        val = 0.0
        k = kcs
        while k < kce
            k1 = k + 1
            ir1 = Int(unsafe_load(minndxs, k1)) + 1
            val = _fmadd(unsafe_load(minmuls, ir1), unsafe_load(elems, k1), val)
            k += 1
        end
        unsafe_store!(majsums, val, cndx)
        cndx += 1
    end
    return nothing
end

@inline function ps_create_col!(
    col::Int32,
    n::Int32,
    els::Ptr{Float64},
    mcstrt::Ptr{Int32},
    colels::Ptr{Float64},
    hrow::Ptr{Int32},
    link::Ptr{Int32},
    free_list::Int32,
)::Int32
    ni = Int(n)
    rows = Ptr{Int32}(els + ni * sizeof(Float64))
    xstart = PRESOLVE_NO_LINK
    i = 1
    while i <= ni
        k = free_list
        k1 = Int(k) + 1
        free_list = unsafe_load(link, k1)
        unsafe_store!(hrow, unsafe_load(rows, i), k1)
        unsafe_store!(colels, unsafe_load(els, i), k1)
        unsafe_store!(link, xstart, k1)
        xstart = k
        i += 1
    end
    unsafe_store!(mcstrt, xstart, Int(col) + 1)
    return free_list
end

# Exact ascending sort of distinct UInt32 elements (used for tie-breaking when CBC_PREPROCESS_EXPERIMENT swapUsed is active)
function _ps_sort_u32!(a::Ptr{UInt32}, n::Int)::Nothing
    n <= 1 && return nothing
    # Insertion sort for small ranges, heapsort for larger ranges (zero-alloc, O(n log n) worst-case)
    if n <= 32
        i = 2
        while i <= n
            key = unsafe_load(a, i)
            j = i - 1
            while j >= 1 && unsafe_load(a, j) > key
                unsafe_store!(a, unsafe_load(a, j), j + 1)
                j -= 1
            end
            unsafe_store!(a, key, j + 1)
            i += 1
        end
        return nothing
    end
    # Build max-heap (1-based)
    start = n >> 1
    while start >= 1
        root = start
        while (child = root << 1) <= n
            if child + 1 <= n && unsafe_load(a, child) < unsafe_load(a, child + 1)
                child += 1
            end
            vr = unsafe_load(a, root)
            vc = unsafe_load(a, child)
            if vr < vc
                unsafe_store!(a, vc, root)
                unsafe_store!(a, vr, child)
                root = child
            else
                break
            end
        end
        start -= 1
    end
    # Extract elements
    last_i = n
    while last_i > 1
        v1 = unsafe_load(a, 1)
        vl = unsafe_load(a, last_i)
        unsafe_store!(a, vl, 1)
        unsafe_store!(a, v1, last_i)
        last_i -= 1
        root = 1
        while (child = root << 1) <= last_i
            if child + 1 <= last_i && unsafe_load(a, child) < unsafe_load(a, child + 1)
                child += 1
            end
            vr = unsafe_load(a, root)
            vc = unsafe_load(a, child)
            if vr < vc
                unsafe_store!(a, vc, root)
                unsafe_store!(a, vr, child)
                root = child
            else
                break
            end
        end
    end
    return nothing
end

function ps_dupcol_presolve!(
    pv::Ptr{CoinPresolvePodView},
    rowmul_buf::Ptr{Float64},
    swap_used_buf::Ptr{UInt8},
    actions_out::Ptr{DupcolActionPod},
    fixed_down::Ptr{Int32},
    fixed_up::Ptr{Int32},
    alloc_dbl_cb::Ptr{Cvoid},
    out_nactions::Ptr{Int32},
    out_nfixed_down::Ptr{Int32},
    out_nfixed_up::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    maxmin = p.maxmin
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    ncols = Int(p.ncols)
    nrows = Int(p.nrows)

    clo = p.clo
    cup = p.cup
    sol = p.sol
    rlo = p.rlo
    rup = p.rup
    dcost = p.cost
    colstat = p.colstat
    integerType = p.integerType
    clink = p.clink
    rowels = p.rowels
    hcol = p.hcol
    mrstrt = p.mrstrt
    hinrow = p.hinrow

    any_prohib = p.anyProhibited
    col_changed = p.colChanged
    status = p.status

    allPositive = true
    rhs = p.usefulRowDouble
    i = 1
    while i <= nrows
        unsafe_store!(rhs, unsafe_load(rup, i), i)
        i += 1
    end

    allowIntegers = (p.presolveOptions & Int32(0x01)) != Int32(0)
    sort_arr = p.usefulColumnInt
    nlook = Int32(0)

    j = 0
    while j < ncols
        j1 = j + 1
        lenj = unsafe_load(hincol, j1)
        if lenj == Int32(0) || (unsafe_load(clo, j1) < -1.0e5 && unsafe_load(cup, j1) > 1.0e5)
            j += 1
            continue
        end
        kcs = Int(unsafe_load(mcstrt, j1))
        coin_sort_2_std!(hrow + kcs * sizeof(Int32), colels + kcs * sizeof(Float64), Int(lenj))
        if allPositive
            lower = unsafe_load(clo, j1)
            kce = kcs + Int(lenj)
            if lower < unsafe_load(cup, j1)
                k = kcs
                while k < kce
                    k1 = k + 1
                    val = unsafe_load(colels, k1)
                    if val < 0.0
                        allPositive = false
                    else
                        r1 = Int(unsafe_load(hrow, k1)) + 1
                        unsafe_store!(rhs, _fmsub(lower, val, unsafe_load(rhs, r1)), r1)
                    end
                    k += 1
                end
            else
                k = kcs
                while k < kce
                    k1 = k + 1
                    val = unsafe_load(colels, k1)
                    r1 = Int(unsafe_load(hrow, k1)) + 1
                    unsafe_store!(rhs, _fmsub(lower, val, unsafe_load(rhs, r1)), r1)
                    k += 1
                end
            end
        end
        j32 = Int32(j)
        if ps_col_prohibited2(any_prohib, col_changed, j32)
            j += 1
            continue
        end
        if integerType != C_NULL && unsafe_load(integerType, j1) != UInt8(0) && !allowIntegers
            j += 1
            continue
        end
        nlook += Int32(1)
        unsafe_store!(sort_arr, j32, Int(nlook))
        j += 1
    end

    if nlook == Int32(0)
        unsafe_store!(out_nactions, Int32(0))
        unsafe_store!(out_nfixed_down, Int32(0))
        unsafe_store!(out_nfixed_up, Int32(0))
        return nothing
    end

    colsum = p.usefulColumnDouble
    rowmul = p.randomNumber
    if rowmul == C_NULL
        rowmul = rowmul_buf
        ps_coin_init_random_vec!(rowmul, Int32(nrows))
    end
    ps_compute_sums!(hincol, mcstrt, hrow, colels, rowmul, sort_arr, colsum, nlook)

    # CBC_PREPROCESS_EXPERIMENT == 1 (default in CoinPresolveDupcol.hpp)
    use_swap = (p.presolveOptions & Int32(0x80000)) != Int32(0) && !allPositive && swap_used_buf != C_NULL
    if use_swap
        i = 1
        while i <= ncols
            unsafe_store!(swap_used_buf, UInt8(0), i)
            i += 1
        end
        nl0 = Int(nlook)
        i = 1
        while i <= nl0
            unsafe_store!(sort_arr, unsafe_load(sort_arr, i), nl0 + i)
            i += 1
        end
        i = 1
        while i <= nrows
            unsafe_store!(rowmul, -unsafe_load(rowmul, i), i)
            i += 1
        end
        ps_compute_sums!(
            hincol, mcstrt, hrow, colels, rowmul,
            sort_arr + nl0 * sizeof(Int32),
            colsum + nl0 * sizeof(Float64),
            nlook,
        )
        i = nl0 + 1
        nl2 = nl0 + nl0
        while i <= nl2
            u = reinterpret(UInt32, unsafe_load(sort_arr, i)) | UInt32(0x80000000)
            unsafe_store!(sort_arr, reinterpret(Int32, u), i)
            i += 1
        end
        nlook += nlook
    end

    coin_sort_2_std!(colsum, sort_arr, Int(nlook))

    if use_swap
        sort2 = Ptr{UInt32}(sort_arr)
        dval = unsafe_load(colsum, 1)
        first_i = 1
        nl_s = Int(nlook)
        jj_s = 2
        while jj_s <= nl_s
            while jj_s <= nl_s && unsafe_load(colsum, jj_s) == dval
                jj_s += 1
            end
            if first_i + 1 < jj_s
                _ps_sort_u32!(sort2 + (first_i - 1) * sizeof(UInt32), jj_s - first_i)
            end
            first_i = jj_s
            if jj_s <= nl_s
                dval = unsafe_load(colsum, jj_s)
            end
            jj_s += 1
        end
    end

    nactions = Int32(0)
    nfixed_down = Int32(0)
    nfixed_up = Int32(0)
    gotStuff = false
    infUp = p.infiniteUp
    infDown = p.infiniteDown
    sumUp = p.sumUp
    sumDown = p.sumDown

    isorted = Int32(-1)
    tgt = 1
    nl = Int(nlook)
    jj = 2
    while jj <= nl
        if unsafe_load(colsum, jj) != unsafe_load(colsum, jj - 1)
            tgt = jj
            jj += 1
            continue
        end
        raw_j2 = unsafe_load(sort_arr, jj)
        raw_j1 = unsafe_load(sort_arr, tgt)
        j2 = raw_j2 & Int32(0x7fffffff)
        j1 = raw_j1 & Int32(0x7fffffff)
        if use_swap
            if unsafe_load(swap_used_buf, Int(j2) + 1) != UInt8(0) ||
               unsafe_load(swap_used_buf, Int(j1) + 1) != UInt8(0) ||
               (j2 != raw_j2 && j1 != raw_j1)
                tgt = jj
                jj += 1
                continue
            end
        end
        j2_1 = Int(j2) + 1
        j1_1 = Int(j1) + 1
        len2 = unsafe_load(hincol, j2_1)
        len1 = unsafe_load(hincol, j1_1)
        if len2 != len1
            tgt = jj
            jj += 1
            continue
        end

        kcs = Int(unsafe_load(mcstrt, j2_1))
        kce = kcs + Int(len2)
        kcs1 = Int(unsafe_load(mcstrt, j1_1))
        ishift = kcs1 - kcs

        if len1 > Int32(1) && isorted < j1
            coin_sort_2_std!(hrow + kcs1 * sizeof(Int32), colels + kcs1 * sizeof(Float64), Int(len1))
            isorted = j1
        end
        if len2 > Int32(1) && isorted < j2
            coin_sort_2_std!(hrow + kcs * sizeof(Int32), colels + kcs * sizeof(Float64), Int(len2))
            isorted = j2
        end

        match = true
        k = kcs
        while k < kce
            if unsafe_load(hrow, k + 1) != unsafe_load(hrow, k + ishift + 1) ||
               unsafe_load(colels, k + 1) != unsafe_load(colels, k + ishift + 1)
                match = false
                break
            end
            k += 1
        end
        if !match
            if !use_swap
                tgt = jj
                jj += 1
                continue
            else
                match_neg = true
                k = kcs
                while k < kce
                    if unsafe_load(hrow, k + 1) != unsafe_load(hrow, k + ishift + 1) ||
                       unsafe_load(colels, k + 1) != -unsafe_load(colels, k + ishift + 1)
                        match_neg = false
                        break
                    end
                    k += 1
                end
                if !match_neg
                    tgt = jj
                    jj += 1
                    continue
                end
            end
        end

        int1 = integerType != C_NULL && unsafe_load(integerType, j1_1) != UInt8(0)
        int2 = integerType != C_NULL && unsafe_load(integerType, j2_1) != UInt8(0)
        if int1 != int2
            jj += 1
            continue
        end

        clo1 = unsafe_load(clo, j1_1)
        cup1 = unsafe_load(cup, j1_1)
        clo2 = unsafe_load(clo, j2_1)
        cup2 = unsafe_load(cup, j2_1)
        c1 = unsafe_load(dcost, j1_1) * maxmin
        c2 = unsafe_load(dcost, j2_1) * maxmin
        if use_swap
            unsafe_store!(swap_used_buf, UInt8(1), j2_1)
            unsafe_store!(swap_used_buf, UInt8(1), j1_1)
            if (reinterpret(UInt32, raw_j2) & UInt32(0x80000000)) != UInt32(0)
                c2 = -c2
                clo2 = -cup2
                cup2 = -unsafe_load(clo, j2_1)
            end
            if (reinterpret(UInt32, raw_j1) & UInt32(0x80000000)) != UInt32(0)
                c1 = -c1
                clo1 = -cup1
                cup1 = -unsafe_load(clo, j1_1)
            end
        end

        lowerBound = -floatmax(Float64)
        upperBound = floatmax(Float64)
        # USE_LBS == 1 (defined under CBC_PREPROCESS_EXPERIMENT in CoinPresolveDupcol.cpp) -> takeColumn = true
        if c1 != c2
            if !allPositive
                if !gotStuff
                    ps_recompute_sums!(pv, Int32(-1))
                    gotStuff = true
                end
                positiveInf = Int32(0)
                negativeInf = Int32(0)
                lo_s = 0.0
                up_s = 0.0
                if clo1 < -PRESOLVE_INF_DBL
                    negativeInf += Int32(1)
                else
                    lo_s += clo1
                end
                if clo2 < -PRESOLVE_INF_DBL
                    negativeInf += Int32(1)
                else
                    lo_s += clo2
                end
                if cup1 > PRESOLVE_INF_DBL
                    positiveInf += Int32(1)
                else
                    up_s += cup1
                end
                if cup2 > PRESOLVE_INF_DBL
                    positiveInf += Int32(1)
                else
                    up_s += cup2
                end
                k = kcs
                while k < kce
                    k1 = k + 1
                    ir1 = Int(unsafe_load(hrow, k1)) + 1
                    val = unsafe_load(colels, k1)
                    pInf = val > 0.0 ? positiveInf : negativeInf
                    nInf = val > 0.0 ? negativeInf : positiveInf
                    posinf = unsafe_load(infUp, ir1) - pInf
                    neginf = unsafe_load(infDown, ir1) - nInf
                    if posinf > Int32(0) && neginf > Int32(0)
                        k += 1
                        continue
                    end
                    maxup = unsafe_load(sumUp, ir1)
                    maxdown = unsafe_load(sumDown, ir1)
                    if val > 0.0
                        maxdown = _fmsub(val, lo_s, maxdown)
                        maxup = _fmsub(val, up_s, maxup)
                    else
                        maxdown = _fmsub(val, up_s, maxdown)
                        maxup = _fmsub(val, lo_s, maxup)
                    end
                    rlo_r = unsafe_load(rlo, ir1)
                    rup_r = unsafe_load(rup, ir1)
                    if val > 1.0e-5
                        if neginf == Int32(0) && rup_r < 1.0e10
                            if upperBound * val > rup_r - maxdown
                                upperBound = (rup_r - maxdown) / val
                            end
                        end
                        if posinf == Int32(0) && rlo_r > -1.0e10
                            if lowerBound * val < rlo_r - maxup
                                lowerBound = (rlo_r - maxup) / val
                            end
                        end
                    elseif val < -1.0e-5
                        if neginf == Int32(0) && rup_r < 1.0e10
                            if lowerBound * val > rup_r - maxdown
                                lowerBound = (rup_r - maxdown) / val
                            end
                        end
                        if posinf == Int32(0) && rlo_r > -1.0e10
                            if upperBound * val < rlo_r - maxup
                                upperBound = (rlo_r - maxup) / val
                            end
                        end
                    end
                    k += 1
                end
            else
                lowerBound = 0.0
                k = kcs
                while k < kce
                    k1 = k + 1
                    ir1 = Int(unsafe_load(hrow, k1)) + 1
                    val = unsafe_load(colels, k1)
                    rhs_r = unsafe_load(rhs, ir1)
                    if upperBound * val > rhs_r
                        upperBound = rhs_r / val
                    end
                    k += 1
                end
            end
        end
        upperBound -= 1.0e-9

        if c1 == c2
            if !(clo2 + cup1 <= clo1 + cup2) || use_swap
                j1, j2 = j2, j1
                j1_1, j2_1 = j2_1, j1_1
                clo1, clo2 = clo2, clo1
                cup1, cup2 = cup2, cup1
                tgt = jj
            end
            len2_now = unsafe_load(hincol, j2_1)
            s_colels = ps_dupmajor!(alloc_dbl_cb, colels, hrow, len2_now, unsafe_load(mcstrt, j2_1))
            nactions += Int32(1)
            unsafe_store!(actions_out,
                DupcolActionPod(
                    unsafe_load(clo, j2_1), unsafe_load(cup, j2_1),
                    unsafe_load(clo, j1_1), unsafe_load(cup, j1_1),
                    j2, j1, s_colels, len2_now, Int32(0),
                ),
                Int(nactions))

            clo1 += clo2
            if clo1 < -1.0e20
                clo1 = -PRESOLVE_INF_DBL
            end
            unsafe_store!(clo, clo1, j1_1)
            cup1 += cup2
            if cup1 > 1.0e20
                cup1 = PRESOLVE_INF_DBL
            end
            unsafe_store!(cup, cup1, j1_1)
            if sol != C_NULL
                unsafe_store!(sol, unsafe_load(sol, j1_1) + unsafe_load(sol, j2_1), j1_1)
            end
            if colstat != C_NULL
                if ps_col_is_basic(colstat, j1) || ps_col_is_basic(colstat, j2)
                    ps_set_col_status!(colstat, j1, PS_STATUS_BASIC)
                end
            end
            unsafe_store!(dcost, 0.0, j2_1)
            if sol != C_NULL
                unsafe_store!(sol, clo2, j2_1)
            end
            k2cs = Int(unsafe_load(mcstrt, j2_1))
            k2ce = k2cs + Int(len2_now)
            k = k2cs
            while k < k2ce
                ps_delete_from_major!(unsafe_load(hrow, k + 1), j2, mrstrt, hinrow, hcol, rowels)
                k += 1
            end
            unsafe_store!(hincol, Int32(0), j2_1)
            ps_remove_link!(clink, j2)
            jj += 1
            continue
        else
            minterm = Int32(0)
            swapped = false
            if c2 > c1
                minterm |= Int32(1)
            end
            if cup2 >= PRESOLVE_INF_DBL
                minterm |= Int32(2)
            end
            if clo2 <= -PRESOLVE_INF_DBL
                minterm |= Int32(4)
            end
            if cup1 >= PRESOLVE_INF_DBL
                minterm |= Int32(8)
            end
            if clo1 <= -PRESOLVE_INF_DBL
                minterm |= Int32(16)
            end
            if clo1 == 0.0 && clo2 == 0.0
                if c2 > c1 && cup1 >= upperBound
                    minterm |= Int32(8)
                elseif c2 < c1 && cup2 >= upperBound
                    minterm |= Int32(2)
                end
            end

            if (minterm & Int32(0x0d)) == Int32(0x01) || (minterm & Int32(0x13)) == Int32(0)
                tgt = jj
                jj += 1
                continue
            end
            if (minterm & Int32(0x13)) == Int32(0x12) || (minterm & Int32(0x0d)) == Int32(0x0d)
                status = Int32(2) # prob->setStatus(2)
                break
            end
            if (minterm & Int32(0x0d)) == Int32(0x05) || (minterm & Int32(0x13)) == Int32(0x02)
                j1, j2 = j2, j1
                j1_1, j2_1 = j2_1, j1_1
                clo1, clo2 = clo2, clo1
                cup1, cup2 = cup2, cup1
                c1, c2 = c2, c1
                tmp1 = minterm & Int32(0x18)
                tmp2 = minterm & Int32(0x06)
                tmp3 = minterm & Int32(0x01)
                minterm = (tmp1 >> 2) | (tmp2 << 2) | (tmp3 ⊻ Int32(0x01))
                swapped = true
            end
            if (minterm & Int32(0x13)) == Int32(0x10)
                nfixed_up += Int32(1)
                unsafe_store!(fixed_up, j2, Int(nfixed_up))
                if colstat != C_NULL
                    if ps_col_is_basic(colstat, j1) || ps_col_is_basic(colstat, j2)
                        ps_set_col_status!(colstat, j1, PS_STATUS_BASIC)
                    end
                    ps_set_col_status!(colstat, j2, PS_STATUS_AT_UPPER_BOUND)
                end
                if sol != C_NULL
                    delta2 = cup2 - unsafe_load(sol, j2_1)
                    unsafe_store!(sol, cup2, j2_1)
                    unsafe_store!(sol, unsafe_load(sol, j1_1) - delta2, j1_1)
                end
                if swapped
                    tgt = jj
                end
                jj += 1
                continue
            end
            if (minterm & Int32(0x0d)) == Int32(0x09)
                nfixed_down += Int32(1)
                unsafe_store!(fixed_down, j2, Int(nfixed_down))
                if colstat != C_NULL
                    if ps_col_is_basic(colstat, j1) || ps_col_is_basic(colstat, j2)
                        ps_set_col_status!(colstat, j1, PS_STATUS_BASIC)
                    end
                    ps_set_col_status!(colstat, j2, PS_STATUS_AT_LOWER_BOUND)
                end
                if sol != C_NULL
                    delta2 = clo2 - unsafe_load(sol, j2_1)
                    unsafe_store!(sol, clo2, j2_1)
                    unsafe_store!(sol, unsafe_load(sol, j1_1) - delta2, j1_1)
                end
                if swapped
                    tgt = jj
                end
                jj += 1
                continue
            end
        end
        jj += 1
    end

    _ps_store_i32!(pv, _PS_OFF_status, status)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_nfixed_down, nfixed_down)
    unsafe_store!(out_nfixed_up, nfixed_up)
    return nothing
end

function ps_dupcol_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{DupcolActionPod},
    nactions::Int32,
)::Nothing
    p = unsafe_load(postv)
    clo = p.clo
    cup = p.cup
    sol = p.sol
    dcost = p.cost
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    link = p.link
    rcosts = p.rcosts
    colstat = p.colstat
    tolerance = p.ztolzb
    free_list = p.free_list

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        icol = f.ithis
        icol2 = f.ilast
        ic1 = Int(icol) + 1
        ic2 = Int(icol2) + 1

        unsafe_store!(dcost, unsafe_load(dcost, ic2), ic1)
        l_j = f.thislo
        u_j = f.thisup
        l_k = f.lastlo
        u_k = f.lastup
        unsafe_store!(clo, l_j, ic1)
        unsafe_store!(cup, u_j, ic1)
        unsafe_store!(clo, l_k, ic2)
        unsafe_store!(cup, u_k, ic2)

        free_list = ps_create_col!(icol, f.nincol, f.colels, mcstrt, colels, hrow, link, free_list)
        unsafe_store!(hincol, f.nincol, ic1)

        x_k_sol = unsafe_load(sol, ic2)
        if l_j > -PRESOLVE_INF_DBL && x_k_sol - l_j >= l_k - tolerance && x_k_sol - l_j <= u_k + tolerance
            ps_set_col_status!(colstat, icol, PS_STATUS_AT_LOWER_BOUND)
            unsafe_store!(sol, l_j, ic1)
            unsafe_store!(sol, x_k_sol - l_j, ic2)
        elseif u_j < PRESOLVE_INF_DBL && x_k_sol - u_j >= l_k - tolerance && x_k_sol - u_j <= u_k + tolerance
            ps_set_col_status!(colstat, icol, PS_STATUS_AT_UPPER_BOUND)
            unsafe_store!(sol, u_j, ic1)
            unsafe_store!(sol, x_k_sol - u_j, ic2)
        elseif l_k > -PRESOLVE_INF_DBL && x_k_sol - l_k >= l_j - tolerance && x_k_sol - l_k <= u_j + tolerance
            ps_set_col_status!(colstat, icol, ps_get_col_status(colstat, icol2))
            unsafe_store!(sol, l_k, ic2)
            unsafe_store!(sol, x_k_sol - l_k, ic1)
            ps_set_col_status!(colstat, icol2, PS_STATUS_AT_LOWER_BOUND)
        elseif u_k < PRESOLVE_INF_DBL && x_k_sol - u_k >= l_j - tolerance && x_k_sol - u_k <= u_j + tolerance
            ps_set_col_status!(colstat, icol, ps_get_col_status(colstat, icol2))
            unsafe_store!(sol, u_k, ic2)
            unsafe_store!(sol, x_k_sol - u_k, ic1)
            ps_set_col_status!(colstat, icol2, PS_STATUS_AT_UPPER_BOUND)
        else
            unsafe_store!(sol, 0.0, ic1)
            ps_set_col_status!(colstat, icol, PS_STATUS_IS_FREE)
        end
        unsafe_store!(rcosts, unsafe_load(rcosts, ic2), ic1)
        idx -= 1
    end

    _post_store_i32!(postv, _POST_OFF_free_list, free_list)
    return nothing
end

function ps_duprow_presolve!(
    pv::Ptr{CoinPresolvePodView},
    sort_arr::Ptr{Int32},
    workrow::Ptr{Float64},
    workcol_buf::Ptr{Float64},
    out_nuseless::Ptr{Int32},
    out_infeas_row::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    rowels = p.rowels
    hcol = p.hcol
    mrstrt = p.mrstrt
    hinrow = p.hinrow
    ncols = Int(p.ncols)
    nrows = Int(p.nrows)
    rlo = p.rlo
    rup = p.rup
    acts = p.acts
    rowstat = p.rowstat

    any_prohib = p.anyProhibited
    row_changed = p.rowChanged
    status = p.status

    nlook = Int32(0)
    i = 0
    while i < nrows
        i1 = i + 1
        len_r = unsafe_load(hinrow, i1)
        if len_r == Int32(0)
            i += 1
            continue
        end
        i32 = Int32(i)
        if ps_row_prohibited2(any_prohib, row_changed, i32)
            i += 1
            continue
        end
        krs = Int(unsafe_load(mrstrt, i1))
        coin_sort_2_std!(hcol + krs * sizeof(Int32), rowels + krs * sizeof(Float64), Int(len_r))
        nlook += Int32(1)
        unsafe_store!(sort_arr, i32, Int(nlook))
        i += 1
    end

    if nlook == Int32(0)
        unsafe_store!(out_nuseless, Int32(0))
        unsafe_store!(out_infeas_row, Int32(-1))
        return nothing
    end

    workcol = p.randomNumber
    if workcol == C_NULL
        workcol = workcol_buf
        ps_coin_init_random_vec!(workcol, Int32(ncols))
    end
    ps_compute_sums!(hinrow, mrstrt, hcol, rowels, workcol, sort_arr, workrow, nlook)
    coin_sort_2_std!(workrow, sort_arr, Int(nlook))

    nuseless_rows = Int32(0)
    infeas_row = Int32(-1)
    fixInfeasibility = (p.presolveOptions & Int32(0x4000)) != Int32(0)
    allowIntersection = (p.presolveOptions & Int32(0x10)) != Int32(0)
    tolerance = p.feasibilityTolerance

    dval = unsafe_load(workrow, 1)
    nl = Int(nlook)
    jj = 2
    while jj <= nl
        wjj = unsafe_load(workrow, jj)
        if wjj == dval
            ithis = unsafe_load(sort_arr, jj)
            ilast = unsafe_load(sort_arr, jj - 1)
            it1 = Int(ithis) + 1
            il1 = Int(ilast) + 1
            len_t = unsafe_load(hinrow, it1)
            if len_t == unsafe_load(hinrow, il1)
                krs = Int(unsafe_load(mrstrt, it1))
                kre = krs + Int(len_t)
                ishift = Int(unsafe_load(mrstrt, il1)) - krs
                match = true
                k = krs
                while k < kre
                    if unsafe_load(hcol, k + 1) != unsafe_load(hcol, k + ishift + 1) ||
                       abs(unsafe_load(rowels, k + 1) - unsafe_load(rowels, k + ishift + 1)) > 1.0e-14
                        match = false
                        break
                    end
                    k += 1
                end
                if match
                    rlo1 = unsafe_load(rlo, il1)
                    rup1 = unsafe_load(rup, il1)
                    rlo2 = unsafe_load(rlo, it1)
                    rup2 = unsafe_load(rup, it1)
                    idelete = Int32(-1)
                    if rlo1 <= rlo2
                        if rup2 <= rup1
                            idelete = ilast
                        elseif abs(rlo1 - rlo2) < 1.0e-12
                            idelete = ithis
                            unsafe_store!(sort_arr, ithis, jj - 1)
                            unsafe_store!(sort_arr, ilast, jj)
                        else
                            if rup1 < rlo2 - tolerance && !fixInfeasibility
                                status |= Int32(1)
                                infeas_row = ithis
                                break
                            elseif allowIntersection
                                idelete = ilast
                                unsafe_store!(rup, rup1, it1)
                                if rowstat != C_NULL &&
                                   !ps_row_is_basic(rowstat, idelete) &&
                                   ps_row_is_basic(rowstat, ithis)
                                    act_t = unsafe_load(acts, it1)
                                    if act_t < unsafe_load(rlo, it1) + tolerance
                                        ps_set_row_status!(rowstat, ithis, PS_STATUS_AT_LOWER_BOUND)
                                    elseif act_t > rup1 - tolerance
                                        ps_set_row_status!(rowstat, ithis, PS_STATUS_AT_UPPER_BOUND)
                                    end
                                end
                            end
                        end
                    else
                        if rup1 <= rup2
                            idelete = ithis
                            unsafe_store!(sort_arr, ithis, jj - 1)
                            unsafe_store!(sort_arr, ilast, jj)
                        else
                            if rup2 < rlo1 - tolerance && !fixInfeasibility
                                status |= Int32(1)
                                infeas_row = ithis
                                break
                            elseif allowIntersection
                                idelete = ilast
                                unsafe_store!(rlo, rlo1, it1)
                            end
                        end
                    end
                    if idelete >= Int32(0)
                        nuseless_rows += Int32(1)
                        unsafe_store!(sort_arr, idelete, Int(nuseless_rows))
                    end
                end
            end
        end
        dval = wjj
        jj += 1
    end

    _ps_store_i32!(pv, _PS_OFF_status, status)
    unsafe_store!(out_nuseless, nuseless_rows)
    unsafe_store!(out_infeas_row, infeas_row)
    return nothing
end

function ps_gubrow_presolve!(
    pv::Ptr{CoinPresolvePodView},
    actions_out::Ptr{GubrowActionPod},
    alloc_gub_cb::Ptr{Cvoid},
)::Int32
    p = unsafe_load(pv)
    rowels = p.rowels
    hcol = p.hcol
    mrstrt = p.mrstrt
    hinrow = p.hinrow
    rlink = p.rlink
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    ncols = Int(p.ncols)
    nrows = Int(p.nrows)
    rlo = p.rlo
    rup = p.rup
    any_prohib = p.anyProhibited
    row_changed = p.rowChanged

    which = p.usefulRowInt
    number = which + nrows * sizeof(Int32)
    els = p.usefulRowDouble
    markCol = Ptr{UInt8}(p.usefulColumnInt)

    j = 1
    while j <= ncols
        unsafe_store!(markCol, UInt8(0), j)
        j += 1
    end
    i = 1
    while i <= nrows
        unsafe_store!(els, 0.0, i)
        i += 1
    end

    nactions = Int32(0)
    i = 0
    while i < nrows
        i1 = i + 1
        nInRow = unsafe_load(hinrow, i1)
        i32 = Int32(i)
        if nInRow > Int32(1) && !ps_row_prohibited2(any_prohib, row_changed, i32) &&
           unsafe_load(rlo, i1) == unsafe_load(rup, i1)
            rStart = Int(unsafe_load(mrstrt, i1))
            rEnd = rStart + Int(nInRow)
            value1 = unsafe_load(rowels, rStart + 1)
            all_same = true
            k = rStart + 1
            while k < rEnd
                if unsafe_load(rowels, k + 1) != value1
                    all_same = false
                    break
                end
                k += 1
            end
            if all_same
                nLook = 0
                k = rStart
                while k < rEnd
                    iCol = unsafe_load(hcol, k + 1)
                    ic1 = Int(iCol) + 1
                    unsafe_store!(markCol, UInt8(1), ic1)
                    kk = Int(unsafe_load(mcstrt, ic1))
                    cEnd = kk + Int(unsafe_load(hincol, ic1))
                    while kk < cEnd
                        kk1 = kk + 1
                        iRow = unsafe_load(hrow, kk1)
                        if iRow != i32
                            ir1 = Int(iRow) + 1
                            val = unsafe_load(colels, kk1)
                            val2 = unsafe_load(els, ir1)
                            if val2 != 0.0
                                if val == val2
                                    unsafe_store!(number, unsafe_load(number, ir1) + Int32(1), ir1)
                                end
                            else
                                unsafe_store!(els, val, ir1)
                                unsafe_store!(number, Int32(1), ir1)
                                nLook += 1
                                unsafe_store!(which, iRow, nLook)
                            end
                        end
                        kk += 1
                    end
                    k += 1
                end

                nDrop = Int32(0)
                rlo_i = unsafe_load(rlo, i1)
                j = 1
                while j <= nLook
                    iRow = unsafe_load(which, j)
                    ir1 = Int(iRow) + 1
                    if unsafe_load(number, ir1) == nInRow
                        kk = rStart
                        while kk < rEnd
                            ps_delete_from_major!(unsafe_load(hcol, kk + 1), iRow, mcstrt, hincol, hrow, colels)
                            kk += 1
                        end
                        nInRow2 = unsafe_load(hinrow, ir1)
                        rStart2 = Int(unsafe_load(mrstrt, ir1))
                        rEnd2 = rStart2 + Int(nInRow2)
                        dst = rStart2
                        kk = rStart2
                        while kk < rEnd2
                            kk1 = kk + 1
                            iCol = unsafe_load(hcol, kk1)
                            if unsafe_load(markCol, Int(iCol) + 1) == UInt8(0)
                                dst1 = dst + 1
                                unsafe_store!(hcol, iCol, dst1)
                                unsafe_store!(rowels, unsafe_load(rowels, kk1), dst1)
                                dst += 1
                            end
                            kk += 1
                        end
                        new_len = nInRow2 - nInRow
                        unsafe_store!(hinrow, new_len, ir1)
                        nDrop += Int32(1)
                        if new_len == Int32(0)
                            ps_remove_link!(rlink, iRow)
                        end
                        val_shift = (rlo_i / value1) * unsafe_load(els, ir1)
                        rlo_r = unsafe_load(rlo, ir1)
                        if rlo_r > -1.0e20
                            unsafe_store!(rlo, rlo_r - val_shift, ir1)
                        end
                        rup_r = unsafe_load(rup, ir1)
                        if rup_r < 1.0e20
                            unsafe_store!(rup, rup_r - val_shift, ir1)
                        end
                    else
                        unsafe_store!(number, Int32(0), ir1)
                    end
                    j += 1
                end

                if nDrop != Int32(0)
                    slot_ptr = actions_out + Int(nactions) * sizeof(GubrowActionPod)
                    ccall(alloc_gub_cb, Cvoid, (Ptr{GubrowActionPod}, Int32, Int32),
                          slot_ptr, nDrop, nInRow)
                    ap = unsafe_load(slot_ptr)
                    delRow = ap.deletedRow
                    a_rowels = ap.rowels
                    indices = ap.indices
                    nInRow_i = Int(nInRow)
                    kk = 1
                    while kk <= nInRow_i
                        unsafe_store!(indices, unsafe_load(hcol, rStart + kk), kk)
                        kk += 1
                    end
                    nd1 = Int(nDrop) + 1
                    unsafe_store!(delRow, i32, nd1)
                    unsafe_store!(a_rowels, value1, nd1)
                    cnt = Int32(0)
                    j = 1
                    while j <= nLook
                        iRow = unsafe_load(which, j)
                        ir1 = Int(iRow) + 1
                        if unsafe_load(number, ir1) != Int32(0)
                            cnt += Int32(1)
                            unsafe_store!(delRow, iRow, Int(cnt))
                            unsafe_store!(a_rowels, unsafe_load(els, ir1), Int(cnt))
                        end
                        j += 1
                    end
                    unsafe_store!(slot_ptr, GubrowActionPod(rlo_i, delRow, a_rowels, indices, cnt, nInRow))
                    nactions += Int32(1)
                end

                j = 1
                while j <= nLook
                    unsafe_store!(els, 0.0, Int(unsafe_load(which, j)) + 1)
                    j += 1
                end
                k = rStart
                while k < rEnd
                    unsafe_store!(markCol, UInt8(0), Int(unsafe_load(hcol, k + 1)) + 1)
                    k += 1
                end
            end
        end
        i += 1
    end
    return nactions
end

function ps_gubrow_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{GubrowActionPod},
    nactions::Int32,
)::Nothing
    p = unsafe_load(postv)
    colStarts = p.mcstrt
    colLengths = p.hincol
    rowIndices = p.hrow
    colCoeffs = p.colels
    rlo = p.rlo
    rup = p.rup
    acts = p.acts
    rowduals = p.rowduals
    link = p.link
    free_list = p.free_list

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        deletedRow = f.deletedRow
        els = f.rowels
        indices = f.indices
        nDrop = Int(f.nDrop)
        ninrow = Int(f.ninrow)
        rhs = f.rhs
        coeff = unsafe_load(els, nDrop + 1)
        iRow1 = Int(unsafe_load(deletedRow, nDrop + 1)) + 1

        i = 1
        while i <= nDrop
            tgtrow = unsafe_load(deletedRow, i)
            tr1 = Int(tgtrow) + 1
            coeffy = unsafe_load(els, i)
            dualValue = unsafe_load(rowduals, tr1)
            unsafe_store!(rowduals, unsafe_load(rowduals, iRow1) - (coeffy * dualValue) / coeff, iRow1)

            rndx = 1
            while rndx <= ninrow
                j1 = Int(unsafe_load(indices, rndx)) + 1
                kk = free_list
                kk1 = Int(kk) + 1
                free_list = unsafe_load(link, kk1)
                unsafe_store!(link, unsafe_load(colStarts, j1), kk1)
                unsafe_store!(colStarts, kk, j1)
                unsafe_store!(colCoeffs, coeffy, kk1)
                unsafe_store!(rowIndices, tgtrow, kk1)
                unsafe_store!(colLengths, unsafe_load(colLengths, j1) + Int32(1), j1)
                rndx += 1
            end

            val = (rhs / coeff) * coeffy
            unsafe_store!(acts, unsafe_load(acts, tr1) + val, tr1)
            rlo_t = unsafe_load(rlo, tr1)
            if rlo_t > -1.0e20
                unsafe_store!(rlo, rlo_t + val, tr1)
            end
            rup_t = unsafe_load(rup, tr1)
            if rup_t < 1.0e20
                unsafe_store!(rup, rup_t + val, tr1)
            end
            i += 1
        end
        idx -= 1
    end

    _post_store_i32!(postv, _POST_OFF_free_list, free_list)
    return nothing
end

function ps_twoxtwo_presolve!(
    pv::Ptr{CoinPresolvePodView},
    boundRecords::Ptr{TwoxtwoActionPod},
    scratch200::Ptr{Float64},
    out_offset::Ptr{Float64},
)::Int32
    p = unsafe_load(pv)
    ncols = Int(p.ncols)
    mcstrt = p.mcstrt
    hincol = p.hincol
    hrow = p.hrow
    colels = p.colels
    cost = p.cost
    integerType = p.integerType
    clo = p.clo
    cup = p.cup
    mrstrt = p.mrstrt
    hinrow = p.hinrow
    hcol = p.hcol
    rowels = p.rowels
    rlo = p.rlo
    rup = p.rup

    costPair = scratch200
    solAt = scratch200 + 100 * sizeof(Float64)

    nactions = Int32(0)
    offset = 0.0
    alpha0 = 0.0
    alpha1 = 0.0

    icol = 0
    while icol < ncols
        ic1 = icol + 1
        if unsafe_load(hincol, ic1) == Int32(2)
            start_c = Int(unsafe_load(mcstrt, ic1))
            row0 = unsafe_load(hrow, start_c + 1)
            r0_1 = Int(row0) + 1
            if unsafe_load(hinrow, r0_1) != Int32(2)
                icol += 1
                continue
            end
            row1 = unsafe_load(hrow, start_c + 2)
            r1_1 = Int(row1) + 1
            if unsafe_load(hinrow, r1_1) != Int32(2)
                icol += 1
                continue
            end

            element0 = unsafe_load(colels, start_c + 1)
            rowUpper0 = unsafe_load(rup, r0_1)
            swapSigns0 = false
            rlo0 = unsafe_load(rlo, r0_1)
            if rlo0 > -1.0e30
                if rowUpper0 > 1.0e30
                    swapSigns0 = true
                    rowUpper0 = -rlo0
                    element0 = -element0
                else
                    icol += 1
                    continue
                end
            elseif rowUpper0 > 1.0e30
                icol += 1
                continue
            end

            element1 = unsafe_load(colels, start_c + 2)
            rowUpper1 = unsafe_load(rup, r1_1)
            swapSigns1 = false
            rlo1 = unsafe_load(rlo, r1_1)
            if rlo1 > -1.0e30
                if rowUpper1 > 1.0e30
                    swapSigns1 = true
                    rowUpper1 = -rlo1
                    element1 = -element1
                else
                    icol += 1
                    continue
                end
            elseif rowUpper1 > 1.0e30
                icol += 1
                continue
            end

            lowerX = unsafe_load(clo, ic1)
            upperX = unsafe_load(cup, ic1)
            otherCol = Int32(-1)
            sr0 = Int(unsafe_load(mrstrt, r0_1))
            jj = sr0
            while jj < sr0 + 2
                jc = unsafe_load(hcol, jj + 1)
                if jc != Int32(icol)
                    v = unsafe_load(rowels, jj + 1)
                    alpha0 = swapSigns0 ? -v : v
                    otherCol = jc
                end
                jj += 1
            end
            sr1 = Int(unsafe_load(mrstrt, r1_1))
            possible = true
            jj = sr1
            while jj < sr1 + 2
                jc = unsafe_load(hcol, jj + 1)
                if jc != Int32(icol)
                    if jc == otherCol
                        v = unsafe_load(rowels, jj + 1)
                        alpha1 = swapSigns1 ? -v : v
                    else
                        possible = false
                    end
                end
                jj += 1
            end

            if possible
                oc1 = Int(otherCol) + 1
                lb0 = lowerX; ub0 = upperX; obj0 = unsafe_load(cost, ic1)
                lb1 = unsafe_load(clo, oc1); ub1 = unsafe_load(cup, oc1); obj1 = unsafe_load(cost, oc1)
                isint0 = (integerType != C_NULL && unsafe_load(integerType, ic1) != UInt8(0)) ? Int32(1) : Int32(0)
                isint1 = (integerType != C_NULL && unsafe_load(integerType, oc1) != UInt8(0)) ? Int32(1) : Int32(0)

                rhs0 = swapSigns0 ? -rlo0 : unsafe_load(rup, r0_1)
                rhs1 = swapSigns1 ? -rlo1 : unsafe_load(rup, r1_1)
                els00 = 0.0; els01 = 0.0
                jj = sr0
                while jj < sr0 + 2
                    jc = unsafe_load(hcol, jj + 1)
                    v = unsafe_load(rowels, jj + 1)
                    if swapSigns0; v = -v; end
                    if jc == Int32(icol); els00 = v; else; els01 = v; end
                    jj += 1
                end
                els10 = 0.0; els11 = 0.0
                jj = sr1
                while jj < sr1 + 2
                    jc = unsafe_load(hcol, jj + 1)
                    v = unsafe_load(rowels, jj + 1)
                    if swapSigns1; v = -v; end
                    if jc == Int32(icol); els10 = v; else; els11 = v; end
                    jj += 1
                end

                swapCol0 = false; swapCol1 = false
                if abs(lb0) > abs(ub0)
                    swapCol0 = true
                    lb0, ub0 = -ub0, -lb0
                    els00 = -els00; els10 = -els10; obj0 = -obj0
                end
                if abs(lb1) > abs(ub1)
                    swapCol1 = true
                    lb1, ub1 = -ub1, -lb1
                    els01 = -els01; els11 = -els11; obj1 = -obj1
                end
                adjBound0 = 0.0; adjBound1 = 0.0
                if lb0 != 0.0
                    adjBound0 = lb0
                    ub0 -= lb0
                    rhs0 = _fmsub(lb0, els00, rhs0)
                    rhs1 = _fmsub(lb0, els10, rhs1)
                end
                if lb1 != 0.0
                    adjBound1 = lb1
                    ub1 -= lb1
                    rhs0 = _fmsub(lb1, els01, rhs0)
                    rhs1 = _fmsub(lb1, els11, rhs1)
                end

                bound0 = unsafe_load(clo, oc1)
                bound1 = unsafe_load(cup, oc1)
                if unsafe_load(cost, ic1) == 0.0
                    icol += 1
                    continue
                end
                if isint1 != Int32(0) && bound1 - bound0 == 1.0
                    if rhs0 == rhs1 && els00 == els10 && els01 == els11
                        unsafe_store!(rlo, -PRESOLVE_INF_DBL, r0_1)
                        unsafe_store!(rup, PRESOLVE_INF_DBL, r0_1)
                    end
                    icol += 1
                    continue
                end

                lowestLowest = PRESOLVE_INF_DBL
                highestLowest = -PRESOLVE_INF_DBL
                lowestHighest = PRESOLVE_INF_DBL
                highestHighest = -PRESOLVE_INF_DBL
                binding0 = Int32(0)
                binding1 = Int32(0)
                k = 0
                while k < 2
                    infLow0 = false; infLow1 = false
                    sum0 = 0.0; sum1 = 0.0
                    val = k == 0 ? bound0 : bound1
                    if abs(val) < 1.0e30
                        sum0 = alpha0 * val
                        sum1 = alpha1 * val
                    else
                        if alpha0 > 0.0
                            if val < 0.0; infLow0 = true; end
                        elseif alpha0 < 0.0
                            if val > 0.0; infLow0 = true; end
                        end
                        if alpha1 > 0.0
                            if val < 0.0; infLow1 = true; end
                        elseif alpha1 < 0.0
                            if val > 0.0; infLow1 = true; end
                        end
                    end
                    thisLowest0 = -PRESOLVE_INF_DBL
                    thisHighest0 = PRESOLVE_INF_DBL
                    if element0 > 0.0
                        if !infLow0; thisHighest0 = (rowUpper0 - sum0) / element0; end
                    else
                        if !infLow0; thisLowest0 = (rowUpper0 - sum0) / element0; end
                    end
                    thisLowest1 = -PRESOLVE_INF_DBL
                    thisHighest1 = PRESOLVE_INF_DBL
                    if element1 > 0.0
                        if !infLow1; thisHighest1 = (rowUpper1 - sum1) / element1; end
                    else
                        if !infLow1; thisLowest1 = (rowUpper1 - sum1) / element1; end
                    end
                    bit = Int32(1) << k
                    if thisLowest0 > thisLowest1 + 1.0e-12
                        if thisLowest0 > lowerX + 1.0e-12; binding0 |= bit; end
                    elseif thisLowest1 > thisLowest0 + 1.0e-12
                        if thisLowest1 > lowerX + 1.0e-12; binding1 |= bit; end
                        thisLowest0 = thisLowest1
                    end
                    if thisHighest0 < thisHighest1 - 1.0e-12
                        if thisHighest0 < upperX - 1.0e-12; binding0 |= bit; end
                    elseif thisHighest1 < thisHighest0 - 1.0e-12
                        if thisHighest1 < upperX - 1.0e-12; binding1 |= bit; end
                        thisHighest0 = thisHighest1
                    end
                    if thisLowest0 < lowestLowest; lowestLowest = thisLowest0; end
                    if thisHighest0 > highestHighest; highestHighest = thisHighest0; end
                    if thisHighest0 < lowestHighest; lowestHighest = thisHighest0; end
                    if thisLowest0 > highestLowest; highestLowest = thisLowest0; end
                    k += 1
                end

                if binding0 != Int32(0) && binding1 != Int32(0)
                    if binding0 + binding1 != Int32(3)
                        if isint0 == Int32(0) && isint1 != Int32(0) && ub1 < 100.0
                            nBound = unsafe_trunc(Int32, ub1)
                            upInt = Int32(-1)
                            loInt = nBound + Int32(1)
                            iB = Int32(0)
                            while iB <= nBound
                                lo0 = -PRESOLVE_INF_DBL
                                hi0 = PRESOLVE_INF_DBL
                                fb = Float64(iB)
                                r_a = _fmsub(fb, els01, rhs0)
                                if els00 > 0.0
                                    q = r_a / els00
                                    if q < hi0; hi0 = q; end
                                else
                                    q = r_a / els00
                                    if q > lo0; lo0 = q; end
                                end
                                r_b = _fmsub(fb, els11, rhs1)
                                if els10 > 0.0
                                    q = r_b / els10
                                    if q < hi0; hi0 = q; end
                                else
                                    q = r_b / els10
                                    if q > lo0; lo0 = q; end
                                end
                                ib1 = Int(iB) + 1
                                if lo0 > hi0
                                    unsafe_store!(costPair, PRESOLVE_INF_DBL, ib1)
                                else
                                    unsafe_store!(costPair, _fmadd(lo0, obj0, fb * obj1), ib1)
                                    unsafe_store!(solAt, lo0, ib1)
                                    if iB > upInt; upInt = iB; end
                                    if iB < loInt; loInt = iB; end
                                end
                                iB += Int32(1)
                            end
                            if loInt > upInt
                                unsafe_store!(costPair, PRESOLVE_INF_DBL, 1)
                            end
                            if !swapCol1
                                unsafe_store!(clo, adjBound1 + Float64(loInt), oc1)
                                unsafe_store!(cup, adjBound1 + Float64(upInt), oc1)
                            else
                                unsafe_store!(clo, -(adjBound1 + Float64(upInt)), oc1)
                                unsafe_store!(cup, -(adjBound1 + Float64(loInt)), oc1)
                            end
                            linear = true
                            gap = 0.0
                            if loInt < upInt
                                gap = unsafe_load(costPair, Int(loInt) + 2) - unsafe_load(costPair, Int(loInt) + 1)
                                ii = Int(loInt) + 1
                                lim = Int(upInt) - 1
                                while ii < lim
                                    if abs(gap - (unsafe_load(costPair, ii + 2) - unsafe_load(costPair, ii + 1))) > 1.0e-10
                                        linear = false
                                        break
                                    end
                                    ii += 1
                                end
                            end
                            if linear
                                fixVal = unsafe_load(solAt, Int(loInt) + 1) + adjBound0
                                if swapCol0; fixVal = -fixVal; end
                                unsafe_store!(clo, fixVal, ic1)
                                unsafe_store!(cup, fixVal, ic1)
                            end
                            if linear && loInt < upInt
                                unsafe_store!(rlo, -PRESOLVE_INF_DBL, r0_1)
                                unsafe_store!(rup, PRESOLVE_INF_DBL, r0_1)
                                unsafe_store!(rlo, -PRESOLVE_INF_DBL, r1_1)
                                unsafe_store!(rup, PRESOLVE_INF_DBL, r1_1)
                                clo_ic = unsafe_load(clo, ic1)
                                cost_ic = unsafe_load(cost, ic1)
                                oldCost = _fmadd(Float64(loInt) + adjBound1, unsafe_load(cost, oc1), clo_ic * cost_ic)
                                newCost = swapCol1 ? -gap : gap
                                unsafe_store!(cost, newCost, oc1)
                                newCost2 = _fmadd(Float64(loInt) + adjBound1, newCost, clo_ic * cost_ic)
                                offset -= newCost2 - oldCost
                            end
                        end
                        binding0 = Int32(0)
                        binding1 = Int32(0)
                    end
                end

                cost_ic = unsafe_load(cost, ic1)
                if cost_ic < 0.0
                    icol += 1
                    continue
                end
                if binding0 != Int32(0) && binding1 != Int32(0)
                    if isint0 != Int32(0)
                        lowestLowest = ceil(lowestLowest - 1.0e-5)
                        highestLowest = ceil(highestLowest - 1.0e-5)
                        lowestHighest = floor(lowestHighest + 1.0e-5)
                        highestHighest = floor(highestHighest + 1.0e-5)
                    end
                    if cost_ic >= 0.0
                        if highestLowest < upperX && highestLowest >= lowerX && highestHighest < 1.0e30
                            if highestLowest < highestHighest; highestHighest = highestLowest; end
                        end
                    end
                    if cost_ic <= 0.0
                        if lowestHighest > lowerX && lowestHighest <= upperX && lowestHighest > -1.0e30
                            if lowestHighest > lowestLowest; lowestLowest = lowestHighest; end
                        end
                    end
                    if lowestLowest > lowerX + 1.0e-8
                        lowerX = lowestLowest
                    end
                    if highestHighest < upperX - 1.0e-8
                        upperX = highestHighest
                    end

                    if binding0 == Int32(3) && binding1 == Int32(3)
                        icol += 1
                        continue
                    elseif binding0 == Int32(1) && binding1 == Int32(3)
                        binding0 = Int32(1); binding1 = Int32(2)
                    elseif binding0 == Int32(3) && binding1 == Int32(1)
                        binding0 = Int32(2); binding1 = Int32(1)
                    end

                    divisor = _fmsub(alpha1, element0, alpha0 * element1)
                    if abs(divisor) <= 1.0e-6
                        icol += 1
                        continue
                    end
                    xValue = _fmsub(rowUpper1, element0, rowUpper0 * element1) / divisor
                    yValue0 = _fmsub(xValue, alpha0, rowUpper0) / element0
                    yValue1 = _fmsub(xValue, alpha1, rowUpper1) / element1
                    ym = yValue0 > yValue1 ? yValue0 : yValue1
                    newLower = ym
                    newUpper = ym
                    xValueEqual = xValue
                    yValueEqual = yValue0
                    cost_oc = unsafe_load(cost, oc1)
                    costEqual = _fmadd(yValueEqual, cost_ic, xValue * cost_oc)
                    slope0 = 0.0; slope1 = 0.0
                    if binding0 == Int32(1)
                        x = xValue - 1.0
                        y = _fmsub(x, alpha0, rowUpper0) / element0
                        slope0 = costEqual - _fmadd(y, cost_ic, x * cost_oc)
                        x = xValue + 1.0
                        y = _fmsub(x, alpha1, rowUpper1) / element0
                        slope1 = _fmadd(y, cost_ic, x * cost_oc) - costEqual
                    else
                        x = xValue - 1.0
                        y = _fmsub(x, alpha1, rowUpper1) / element0
                        slope1 = costEqual - _fmadd(y, cost_ic, x * cost_oc)
                        x = xValue + 1.0
                        y = _fmsub(x, alpha0, rowUpper0) / element0
                        slope0 = _fmadd(y, cost_ic, x * cost_oc) - costEqual
                    end
                    yValue0 = _fmsub(bound0, alpha0, rowUpper0) / element0
                    yValue1 = _fmsub(bound0, alpha1, rowUpper1) / element1
                    ym = yValue0 > yValue1 ? yValue0 : yValue1
                    if ym < newLower; newLower = ym; end
                    if ym > newUpper; newUpper = ym; end

                    yValue0 = _fmsub(bound1, alpha0, rowUpper0) / element0
                    yValue1 = _fmsub(bound1, alpha1, rowUpper1) / element1
                    ym = yValue0 > yValue1 ? yValue0 : yValue1
                    if ym < newLower; newLower = ym; end
                    if ym > newUpper; newUpper = ym; end

                    nl_adj = _fmsub(1.0e-12, abs(newLower), newLower)
                    nu_adj = _fmadd(1.0e-12, abs(newUpper), newUpper)
                    if nl_adj > lowerX; lowerX = nl_adj; end
                    if nu_adj < upperX; upperX = nu_adj; end

                    costOther = cost_oc + slope1
                    costThis = _fmadd(slope1, element0 / alpha0, cost_ic)
                    xValue = xValueEqual
                    yt = _fmsub(xValue, alpha0, rowUpper0) / element0
                    yValue0 = yt > lowerX ? yt : lowerX
                    thisOffset = costEqual - _fmadd(costThis, yValue0, costOther * xValue)
                    offset += thisOffset

                    xValue = xValueEqual - 1.0
                    yt = _fmsub(xValue, alpha0, rowUpper0) / element0
                    yValue0 = yt > lowerX ? yt : lowerX
                    if abs((_fmadd(costThis, yValue0, costOther * xValue) + thisOffset) - (costEqual - slope0)) > 1.0e-5
                        icol += 1
                        continue
                    end
                    xValue = xValueEqual + 1.0
                    yt = _fmsub(xValue, alpha0, rowUpper0) / element0
                    yValue0 = yt > lowerX ? yt : lowerX
                    if abs((_fmadd(costThis, yValue0, costOther * xValue) + thisOffset) - (costEqual + slope1)) > 1.0e-5
                        icol += 1
                        continue
                    end

                    nactions += Int32(1)
                    unsafe_store!(boundRecords,
                        TwoxtwoActionPod(unsafe_load(rlo, r1_1), unsafe_load(rup, r1_1),
                                         unsafe_load(clo, ic1), unsafe_load(cup, ic1),
                                         cost_ic, cost_oc, row1, Int32(icol), otherCol, Int32(0)),
                        Int(nactions))
                    unsafe_store!(cost, costOther, oc1)
                    unsafe_store!(cost, costThis, ic1)
                    unsafe_store!(clo, lowerX, ic1)
                    unsafe_store!(cup, upperX, ic1)
                    unsafe_store!(rlo, -PRESOLVE_INF_DBL, r1_1)
                    unsafe_store!(rup, PRESOLVE_INF_DBL, r1_1)
                end
            end
        end
        icol += 1
    end

    unsafe_store!(out_offset, offset)
    return nactions
end

function ps_twoxtwo_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{TwoxtwoActionPod},
    nactions::Int32,
)::Nothing
    p = unsafe_load(postv)
    mcstrt = p.mcstrt
    hincol = p.hincol
    hrow = p.hrow
    colels = p.colels
    link = p.link
    cost = p.cost
    clo = p.clo
    cup = p.cup
    rlo = p.rlo
    rup = p.rup
    sol = p.sol
    rcosts = p.rcosts
    rowacts = p.acts
    dual = p.rowduals
    colstat = p.colstat
    rowstat = p.rowstat
    tolerance = p.ztolzb
    maxmin = p.maxmin

    na = Int(nactions)
    iAction = 1
    while iAction <= na
        br = unsafe_load(actions, iAction)
        row1 = br.row
        r1_1 = Int(row1) + 1
        icol = br.col
        ic1 = Int(icol) + 1
        otherCol = br.othercol
        oc1 = Int(otherCol) + 1

        start_c = unsafe_load(mcstrt, ic1)
        nextEl = unsafe_load(link, Int(start_c) + 1)
        row0 = Int32(0)
        els0_1 = 0.0
        els1_1 = 0.0
        if unsafe_load(hrow, Int(start_c) + 1) == row1
            row0 = unsafe_load(hrow, Int(nextEl) + 1)
            els0_1 = unsafe_load(colels, Int(nextEl) + 1)
            els1_1 = unsafe_load(colels, Int(start_c) + 1)
        else
            row0 = unsafe_load(hrow, Int(start_c) + 1)
            els0_1 = unsafe_load(colels, Int(start_c) + 1)
            els1_1 = unsafe_load(colels, Int(nextEl) + 1)
        end
        r0_1 = Int(row0) + 1

        els0_0 = 0.0
        els1_0 = 0.0
        nextEl = unsafe_load(mcstrt, oc1)
        len_oc = Int(unsafe_load(hincol, oc1))
        j = 0
        while j < len_oc
            n1 = Int(nextEl) + 1
            hr = unsafe_load(hrow, n1)
            if hr == row0
                els0_0 = unsafe_load(colels, n1)
            elseif hr == row1
                els1_0 = unsafe_load(colels, n1)
            end
            nextEl = unsafe_load(link, n1)
            j += 1
        end

        ps_set_row_status!(rowstat, row1, PS_STATUS_BASIC)
        unsafe_store!(rlo, br.lbound_row, r1_1)
        unsafe_store!(rup, br.ubound_row, r1_1)
        unsafe_store!(clo, br.lbound_col, ic1)
        unsafe_store!(cup, br.ubound_col, ic1)
        oldCost = unsafe_load(cost, ic1)
        unsafe_store!(cost, br.cost_col, ic1)
        unsafe_store!(cost, br.cost_othercol, oc1)

        els0real_1 = els0_1
        els1real_1 = els1_1

        rowUpper0 = unsafe_load(rup, r0_1)
        if unsafe_load(rlo, r0_1) > -1.0e30
            rowUpper0 = -unsafe_load(rlo, r0_1)
            els0_0 = -els0_0
            els0_1 = -els0_1
        end
        rowUpper1 = br.ubound_row
        swapSigns1 = false
        if br.lbound_row > -1.0e30
            swapSigns1 = true
            rowUpper1 = -br.lbound_row
            els1_0 = -els1_0
            els1_1 = -els1_1
        end

        valueOther = unsafe_load(sol, oc1)
        lowerBoundPossible = br.lbound_col > -1.0e30
        if lowerBoundPossible
            val = br.lbound_col
            if _fmadd(els0_1, val, els0_0 * valueOther) > rowUpper0 + tolerance
                lowerBoundPossible = false
            end
            if _fmadd(els1_1, val, els1_0 * valueOther) > rowUpper1 + tolerance
                lowerBoundPossible = false
            end
        end
        upperBoundPossible = br.ubound_col < 1.0e30
        if upperBoundPossible
            val = br.ubound_col
            if _fmadd(els0_1, val, els0_0 * valueOther) > rowUpper0 + tolerance
                upperBoundPossible = false
            end
            if _fmadd(els1_1, val, els1_0 * valueOther) > rowUpper1 + tolerance
                upperBoundPossible = false
            end
        end

        cost_ic = br.cost_col
        if lowerBoundPossible && cost_ic >= 0.0
            ps_set_col_status!(colstat, icol, PS_STATUS_AT_LOWER_BOUND)
            unsafe_store!(sol, br.lbound_col, ic1)
            unsafe_store!(rcosts, _fmsub(unsafe_load(dual, r0_1), els0real_1, maxmin * cost_ic), ic1)
        elseif upperBoundPossible && cost_ic <= 0.0
            ps_set_col_status!(colstat, icol, PS_STATUS_AT_UPPER_BOUND)
            unsafe_store!(sol, br.ubound_col, ic1)
            unsafe_store!(rcosts, _fmsub(unsafe_load(dual, r0_1), els0real_1, maxmin * cost_ic), ic1)
        else
            val0 = _fmsub(els0_0, valueOther, rowUpper0) / els0_1
            val1 = _fmsub(els1_0, valueOther, rowUpper1) / els1_1
            val = cost_ic > 0.0 ? (val0 > val1 ? val0 : val1) : (val0 < val1 ? val0 : val1)
            unsafe_store!(sol, val, ic1)
            if ps_col_is_basic(colstat, icol)
                if !ps_row_is_basic(rowstat, row0)
                    unsafe_store!(dual, maxmin * ((cost_ic - oldCost) / els0real_1), r0_1)
                end
                iAction += 1
                continue
            end
            ps_set_col_status!(colstat, icol, PS_STATUS_BASIC)
            unsafe_store!(rcosts, 0.0, ic1)
            if !swapSigns1
                ps_set_row_status!(rowstat, row1, PS_STATUS_AT_UPPER_BOUND)
                unsafe_store!(rowacts, br.ubound_row, r1_1)
            else
                ps_set_row_status!(rowstat, row1, PS_STATUS_AT_LOWER_BOUND)
                unsafe_store!(rowacts, br.lbound_row, r1_1)
            end
            unsafe_store!(dual, maxmin * ((cost_ic - oldCost) / els1real_1), r1_1)
        end
        iAction += 1
    end
    return nothing
end

# Faithful translation of CoinUtils Presolve B7c1:
#   - CoinPresolveDoubleton.cpp    (doubleton_action)
#   - CoinPresolveTripleton.cpp    (tripleton_action)
#   - CoinPresolveImpliedFree.cpp  (implied_free_action, check_row)
# Zero-allocation, bit-identical arm64 FMA (`_fmadd`/`_fmsub`/`_fnmsub`), `--trim=safe` compatible.

@inline function ps_col_can_touch(col_changed::Ptr{UInt8}, j::Int32)::Bool
    return (unsafe_load(col_changed, Int(j) + 1) & UInt8(16)) == UInt8(0)
end

@inline function ps_row_used(row_changed::Ptr{UInt8}, i::Int32)::Bool
    return (unsafe_load(row_changed, Int(i) + 1) & UInt8(4)) != UInt8(0)
end

@inline function ps_set_row_used!(row_changed::Ptr{UInt8}, i::Int32)::Nothing
    i1 = Int(i) + 1
    unsafe_store!(row_changed, unsafe_load(row_changed, i1) | UInt8(4), i1)
    return nothing
end

@inline function ps_unset_row_used!(row_changed::Ptr{UInt8}, i::Int32)::Nothing
    i1 = Int(i) + 1
    unsafe_store!(row_changed, unsafe_load(row_changed, i1) & ~UInt8(4), i1)
    return nothing
end

@inline function ps_dupmajor!(
    alloc_dbl_cb::Ptr{Cvoid},
    elems::Ptr{Float64},
    indices::Ptr{Int32},
    length_in::Int32,
    offset::Int32,
    tgt::Int32 = Int32(-1),
)::Ptr{Float64}
    len = tgt >= Int32(0) ? (length_in - Int32(1)) : length_in
    n = Int32((Int(3) * Int(len) + 1) >> 1)
    dArray = ccall(alloc_dbl_cb, Ptr{Float64}, (Int32,), n)
    iArray = Ptr{Int32}(dArray + Int(len) * sizeof(Float64))
    off = Int(offset)
    leni = Int(len)
    if tgt < Int32(0)
        k = 1
        while k <= leni
            unsafe_store!(dArray, unsafe_load(elems, off + k), k)
            unsafe_store!(iArray, unsafe_load(indices, off + k), k)
            k += 1
        end
    else
        kcopy = 0
        korig = 0
        while korig <= leni
            idx = unsafe_load(indices, off + korig + 1)
            if idx != tgt
                kcopy += 1
                unsafe_store!(dArray, unsafe_load(elems, off + korig + 1), kcopy)
                unsafe_store!(iArray, idx, kcopy)
            end
            korig += 1
        end
    end
    return dArray
end

# =============================================================================
# 1. CoinPresolveDoubleton.cpp
# =============================================================================

function ps_elim_doubleton!(
    mcstrt::Ptr{Int32},
    rlo::Ptr{Float64},
    rup::Ptr{Float64},
    colels::Ptr{Float64},
    hrow::Ptr{Int32},
    hcol::Ptr{Int32},
    hinrow::Ptr{Int32},
    hincol::Ptr{Int32},
    clink::Ptr{PresolveHLinkPod},
    ncols::Int32,
    mrstrt::Ptr{Int32},
    rowels::Ptr{Float64},
    coeff_factor::Float64,
    bounds_factor::Float64,
    icolx::Int32,
    icoly::Int32,
)::Bool
    ix1 = Int(icolx) + 1
    iy1 = Int(icoly) + 1
    kcsx = unsafe_load(mcstrt, ix1)
    kcex = kcsx + unsafe_load(hincol, ix1)
    base = unsafe_load(mcstrt, iy1)
    numberInY = Int(unsafe_load(hincol, iy1))

    kwhere = 0
    while kwhere < numberInY
        kcoly1 = Int(base) + kwhere + 1
        row = unsafe_load(hrow, kcoly1)
        row1 = Int(row) + 1
        coeffy = unsafe_load(colels, kcoly1)
        delta = coeffy * coeff_factor

        kcolx = ps_find_minor1(row, kcsx, kcex, hrow)
        if kcolx < kcex
            kx1 = Int(kcolx) + 1
            new_v = unsafe_load(colels, kx1) + delta
            unsafe_store!(colels, new_v, kx1)

            mrs = unsafe_load(mrstrt, row1)
            mre = mrs + unsafe_load(hinrow, row1)
            kmi = ps_find_minor1(icolx, mrs, mre, hcol)
            unsafe_store!(rowels, new_v, Int(kmi) + 1)
            ps_delete_from_major!(row, icoly, mrstrt, hinrow, hcol, rowels)
        else
            no_mem = ps_expand_major!(mcstrt, colels, hrow, hincol, clink, ncols, icolx)
            no_mem && return true
            kcsx = unsafe_load(mcstrt, ix1)
            lenx = unsafe_load(hincol, ix1)
            kcex = kcsx + lenx
            base = unsafe_load(mcstrt, iy1)

            kcex1 = Int(kcex) + 1
            unsafe_store!(hrow, row, kcex1)
            unsafe_store!(colels, delta, kcex1)
            unsafe_store!(hincol, lenx + Int32(1), ix1)
            kcex += Int32(1)

            mrs = unsafe_load(mrstrt, row1)
            mre = mrs + unsafe_load(hinrow, row1)
            k2 = ps_find_minor1(icoly, mrs, mre, hcol)
            k21 = Int(k2) + 1
            unsafe_store!(hcol, icolx, k21)
            unsafe_store!(rowels, delta, k21)
        end

        if bounds_factor != 0.0
            delta_b = coeffy * bounds_factor
            rlo_r = unsafe_load(rlo, row1)
            if -PRESOLVE_INF_DBL < rlo_r
                unsafe_store!(rlo, rlo_r - delta_b, row1)
            end
            rup_r = unsafe_load(rup, row1)
            if rup_r < PRESOLVE_INF_DBL
                unsafe_store!(rup, rup_r - delta_b, row1)
            end
        end
        kwhere += 1
    end
    return false
end

function ps_doubleton_presolve!(
    pv::Ptr{CoinPresolvePodView},
    actions_out::Ptr{DoubletonActionPod},
    alloc_dbl_cb::Ptr{Cvoid},
    out_nactions::Ptr{Int32},
    out_nzeros::Ptr{Int32},
    out_nfixed::Ptr{Int32},
    out_nomem::Ptr{Int32},
    out_infeas_col::Ptr{Int32},
    out_infeas_lo::Ptr{Float64},
    out_infeas_up::Ptr{Float64},
)::Nothing
    p = unsafe_load(pv)
    n = p.ncols
    m = p.nrows
    ni = Int(n)
    mi = Int(m)

    colStarts = p.mcstrt
    colLengths = p.hincol
    colCoeffs = p.colels
    rowIndices = p.hrow
    clink = p.clink

    clo = p.clo
    cup = p.cup

    rowStarts = p.mrstrt
    rowLengths = p.hinrow
    rowCoeffs = p.rowels
    colIndices = p.hcol
    rlink = p.rlink

    rlo = p.rlo
    rup = p.rup
    integerType = p.integerType
    cost = p.cost

    numberLook = Int(p.numberRowsToDo)
    look = p.rowsToDo
    ztolzb = p.ztolzb
    feas_tol = p.feasibilityTolerance
    ztolzero = 1.0e-12

    zeros_arr = p.usefulColumnInt
    nzeros = Int32(0)
    fixed_arr = zeros_arr + ni * sizeof(Int32)
    nfixed = Int32(0)

    rowstat = p.rowstat
    colstat = p.colstat
    acts = p.acts
    sol = p.sol

    any_prohib = p.anyProhibited
    col_changed = p.colChanged
    row_changed = p.rowChanged
    next_cols = p.nextColsToDo
    next_rows = p.nextRowsToDo
    n_next_cols = p.numberNextColsToDo
    n_next_rows = p.numberNextRowsToDo
    dobias = p.dobias
    status = p.status

    has_careful = (p.presolveOptions & Int32(0x100000)) != Int32(0)
    maxdown = Ptr{Float64}(C_NULL)
    maxup = Ptr{Float64}(C_NULL)
    flagged = Ptr{Int32}(C_NULL)
    if has_careful
        maxdown = p.usefulRowDouble
        maxup = maxdown + mi * sizeof(Float64)
        flagged = p.usefulRowInt
        i = 0
        while i < mi
            i1 = i + 1
            unsafe_store!(flagged, Int32(0), i1)
            unsafe_store!(maxdown, -1.0e50, i1)
            unsafe_store!(maxup, 1.0e50, i1)
            if unsafe_load(rlo, i1) > -1.0e50 || unsafe_load(rup, i1) < 1.0e50
                iflagu = Int32(0)
                iflagl = Int32(0)
                dmaxup = 0.0
                dmaxdown = 0.0
                krs = Int(unsafe_load(rowStarts, i1))
                kre = krs + Int(unsafe_load(rowLengths, i1))
                k = krs
                while k < kre
                    k1 = k + 1
                    val = unsafe_load(rowCoeffs, k1)
                    j1 = Int(unsafe_load(colIndices, k1)) + 1
                    u = unsafe_load(cup, j1)
                    l = unsafe_load(clo, j1)
                    if val < 0.0
                        if u < 1.0e30
                            dmaxdown = _fmadd(u, val, dmaxdown)
                        else
                            iflagl += Int32(1)
                        end
                        if l > -1.0e30
                            dmaxup = _fmadd(l, val, dmaxup)
                        else
                            iflagu += Int32(1)
                        end
                    else
                        if u < 1.0e30
                            dmaxup = _fmadd(u, val, dmaxup)
                        else
                            iflagu += Int32(1)
                        end
                        if l > -1.0e30
                            dmaxdown = _fmadd(l, val, dmaxdown)
                        else
                            iflagl += Int32(1)
                        end
                    end
                    k += 1
                end
                if iflagu != Int32(0)
                    dmaxup = 1.0e50
                end
                if iflagl != Int32(0)
                    dmaxdown = -1.0e50
                end
                unsafe_store!(maxdown, dmaxdown, i1)
                unsafe_store!(maxup, dmaxup, i1)
            end
            i += 1
        end
    end

    fixInfeasibility = (p.presolveOptions & Int32(0x4000)) != Int32(0)
    nactions = Int32(0)
    nomem = Int32(0)
    infeas_col = Int32(-1)
    infeas_lo = 0.0
    infeas_up = 0.0

    iLook = 0
    while iLook < numberLook
        tgtrow = unsafe_load(look, iLook + 1)
        tr1 = Int(tgtrow) + 1
        if unsafe_load(rowLengths, tr1) == Int32(2)
            rlo_t = unsafe_load(rlo, tr1)
            rup_t = unsafe_load(rup, tr1)
            if abs(rup_t - rlo_t) > PRESOLVE_ZTOLDP
                if !has_careful
                    iLook += 1
                    continue
                end
                if unsafe_load(flagged, tr1) != Int32(0)
                    iLook += 1
                    continue
                end
                krs = Int(unsafe_load(rowStarts, tr1))
                lc0 = unsafe_load(colIndices, krs + 1)
                lc1 = unsafe_load(colIndices, krs + 2)
                lc0_1 = Int(lc0) + 1
                lc1_1 = Int(lc1) + 1
                way0 = Int32(3)
                way1 = Int32(3)
                redundant = Int32(3)
                it0 = unsafe_load(integerType, lc0_1) != UInt8(0)
                it1 = unsafe_load(integerType, lc1_1) != UInt8(0)
                if it0 || it1
                    if !it0
                        way1 = Int32(0)
                    elseif !it1
                        way0 = Int32(0)
                    else
                        coeff0 = abs(unsafe_load(rowCoeffs, krs + 1))
                        coeff1 = abs(unsafe_load(rowCoeffs, krs + 2))
                        iKeep = Int32(3)
                        if coeff0 < coeff1
                            ratio = coeff1 / coeff0
                            iKeep = Int32(1)
                            if abs(floor(ratio + 0.5) - ratio) > 1.0e-12
                                iLook += 1
                                continue
                            end
                            coeff1 = coeff0
                        elseif coeff0 > coeff1
                            ratio = coeff0 / coeff1
                            iKeep = Int32(2)
                            if abs(floor(ratio + 0.5) - ratio) > 1.0e-12
                                iLook += 1
                                continue
                            end
                        end
                        if rlo_t > -1.0e30
                            ratio = rlo_t / coeff1
                            if abs(floor(ratio + 0.5) - ratio) > 1.0e-12
                                iLook += 1
                                continue
                            end
                        end
                        if rup_t < 1.0e30
                            ratio = rup_t / coeff1
                            if abs(floor(ratio + 0.5) - ratio) > 1.0e-12
                                iLook += 1
                                continue
                            end
                        end
                        if (iKeep & Int32(1)) == Int32(0)
                            way0 = Int32(0)
                        end
                        if (iKeep & Int32(2)) == Int32(0)
                            way1 = Int32(0)
                        end
                    end
                end
                if way0 != Int32(0) || way1 != Int32(0)
                    side = 0
                    while side < 2
                        icol = side == 0 ? lc0 : lc1
                        ic1 = Int(icol) + 1
                        thisCost = unsafe_load(cost, ic1)
                        intWay = side == 0 ? way0 : way1
                        if thisCost > 0.0
                            intWay &= Int32(1)
                        elseif thisCost < 0.0
                            intWay &= Int32(2)
                        end
                        if intWay != Int32(0)
                            cs = Int(unsafe_load(colStarts, ic1))
                            ce = cs + Int(unsafe_load(colLengths, ic1))
                            kk = cs
                            while kk < ce
                                kk1 = kk + 1
                                iRow = unsafe_load(rowIndices, kk1)
                                ir1 = Int(iRow) + 1
                                val = unsafe_load(colCoeffs, kk1)
                                if unsafe_load(flagged, ir1) != Int32(0)
                                    way0 = Int32(0)
                                    way1 = Int32(0)
                                    intWay = Int32(0)
                                    break
                                end
                                rlo_r = unsafe_load(rlo, ir1)
                                rup_r = unsafe_load(rup, ir1)
                                md_r = unsafe_load(maxdown, ir1)
                                mu_r = unsafe_load(maxup, ir1)
                                if iRow != tgtrow
                                    if val > 0.0
                                        if (intWay & Int32(1)) != Int32(0) && rlo_r > -1.0e30 && md_r < rlo_r
                                            intWay &= Int32(2)
                                        end
                                        if (intWay & Int32(2)) != Int32(0) && rup_r < 1.0e30 && mu_r > rup_r
                                            intWay &= Int32(1)
                                        end
                                    else
                                        if (intWay & Int32(2)) != Int32(0) && rlo_r > -1.0e30 && md_r < rlo_r
                                            intWay &= Int32(1)
                                        end
                                        if (intWay & Int32(1)) != Int32(0) && rup_r < 1.0e30 && mu_r > rup_r
                                            intWay &= Int32(2)
                                        end
                                    end
                                end
                                # tryRedundant == 3
                                if rlo_r > -1.0e30 && md_r < rlo_r
                                    redundant &= Int32(2)
                                end
                                if rup_r < 1.0e30 && mu_r > rup_r
                                    redundant &= Int32(1)
                                end
                                kk += 1
                            end
                        end
                        if side == 0
                            way0 = intWay
                        else
                            way1 = intWay
                        end
                        side += 1
                    end
                end
                if way0 != Int32(0) || way1 != Int32(0)
                    if redundant == Int32(0)
                        way0 = Int32(0); way1 = Int32(0)
                    elseif redundant == Int32(3)
                        unsafe_store!(rlo, -PRESOLVE_INF_DBL, tr1)
                        unsafe_store!(rup, PRESOLVE_INF_DBL, tr1)
                        way0 = Int32(0); way1 = Int32(0)
                    elseif (redundant == Int32(1) && rup_t != 0.0) ||
                           (redundant == Int32(2) && rlo_t != 0.0)
                        way0 = Int32(0); way1 = Int32(0)
                    end
                end
                if way0 == Int32(0) && way1 == Int32(0)
                    iLook += 1
                    continue
                end
                if (way0 != Int32(0) && way1 != Int32(0)) || way0 == Int32(3) || way1 == Int32(3)
                    iLook += 1
                    continue
                else
                    side = 0
                    while side < 2
                        ic1 = Int(side == 0 ? lc0 : lc1) + 1
                        cs = Int(unsafe_load(colStarts, ic1))
                        ce = cs + Int(unsafe_load(colLengths, ic1))
                        kk = cs
                        while kk < ce
                            unsafe_store!(flagged, Int32(1), Int(unsafe_load(rowIndices, kk + 1)) + 1)
                            kk += 1
                        end
                        side += 1
                    end
                    if redundant == Int32(1)
                        unsafe_store!(rlo, rup_t, tr1)
                    else
                        unsafe_store!(rup, rlo_t, tr1)
                    end
                end
            end
        else
            iLook += 1
            continue
        end

        krs = Int(unsafe_load(rowStarts, tr1))
        tgtcolx = unsafe_load(colIndices, krs + 1)
        tgtcoly = unsafe_load(colIndices, krs + 2)
        tx1 = Int(tgtcolx) + 1
        ty1 = Int(tgtcoly) + 1

        if unsafe_load(colLengths, tx1) == Int32(1) && unsafe_load(colLengths, ty1) == Int32(1)
            iLook += 1
            continue
        end
        if ps_col_prohibited(col_changed, tgtcolx) && ps_col_prohibited(col_changed, tgtcoly)
            iLook += 1
            continue
        end
        if !ps_col_can_touch(col_changed, tgtcolx) || !ps_col_can_touch(col_changed, tgtcoly)
            iLook += 1
            continue
        end
        if abs(unsafe_load(rowCoeffs, krs + 1)) < PRESOLVE_ZTOLDP2 ||
           abs(unsafe_load(rowCoeffs, krs + 2)) < PRESOLVE_ZTOLDP2
            iLook += 1
            continue
        end
        if abs(unsafe_load(cup, tx1) - unsafe_load(clo, tx1)) < PRESOLVE_ZTOLDP ||
           abs(unsafe_load(cup, ty1) - unsafe_load(clo, ty1)) < PRESOLVE_ZTOLDP
            iLook += 1
            continue
        end

        csx = unsafe_load(colStarts, tx1)
        krowx = ps_find_minor1(tgtrow, csx, csx + unsafe_load(colLengths, tx1), rowIndices)
        coeffx = unsafe_load(colCoeffs, Int(krowx) + 1)
        csy = unsafe_load(colStarts, ty1)
        krowy = ps_find_minor1(tgtrow, csy, csy + unsafe_load(colLengths, ty1), rowIndices)
        coeffy = unsafe_load(colCoeffs, Int(krowy) + 1)
        rhs = unsafe_load(rlo, tr1)

        integerX = Int32(unsafe_load(integerType, tx1))
        integerY = Int32(unsafe_load(integerType, ty1))
        if ps_col_prohibited2(any_prohib, col_changed, tgtcolx)
            integerX = Int32(1)
        end
        if ps_col_prohibited2(any_prohib, col_changed, tgtcoly)
            integerY = Int32(1)
        end
        integerStatus = (integerY << 1) | integerX

        if integerStatus == Int32(3)
            good = Int32(0)
            rhs2 = rhs
            cx_t = coeffx
            cy_t = coeffy
            if cx_t < 0.0
                cx_t = -cx_t
                rhs2 += 1.0
            end
            if unsafe_load(cup, tx1) == 1.0 && unsafe_load(clo, tx1) == 0.0 &&
               abs(cx_t - 1.0) < 1.0e-7 && !ps_col_prohibited2(any_prohib, col_changed, tgtcoly)
                good = Int32(1)
            end
            if cy_t < 0.0
                cy_t = -cy_t
                rhs2 += 1.0
            end
            if unsafe_load(cup, ty1) == 1.0 && unsafe_load(clo, ty1) == 0.0 &&
               abs(cy_t - 1.0) < 1.0e-7 && !ps_col_prohibited2(any_prohib, col_changed, tgtcolx)
                good |= Int32(2)
            end
            if !(good == Int32(3) && abs(rhs2 - 1.0) < 1.0e-7)
                integerStatus = Int32(-1)
            end
            if integerStatus < Int32(0) && rhs == 0.0
                cx_t = unsafe_load(colCoeffs, Int(krowx) + 1)
                cy_t = unsafe_load(colCoeffs, Int(krowy) + 1)
                swap = false
                if abs(cx_t) > abs(cy_t)
                    ratio = abs(cx_t / cy_t)
                else
                    ratio = abs(cy_t / cx_t)
                    swap = true
                end
                if abs(ratio - floor(ratio + 0.5)) < ztolzero
                    integerStatus = swap ? Int32(2) : Int32(1)
                end
            end
            if integerStatus < Int32(0)
                canDo = false
                cx_t = unsafe_load(colCoeffs, Int(krowx) + 1)
                cy_t = unsafe_load(colCoeffs, Int(krowy) + 1)
                swap = false
                if abs(cx_t) > abs(cy_t)
                    ratio = abs(cx_t / cy_t)
                    rhsRatio = rhs / cx_t
                else
                    ratio = abs(cy_t / cx_t)
                    rhsRatio = rhs / cy_t
                    swap = true
                end
                if abs(ratio - floor(ratio + 0.5)) < ztolzero
                    integerStatus = swap ? Int32(2) : Int32(1)
                    if rhsRatio == floor(rhsRatio + 0.5)
                        canDo = true
                    end
                end
                if !canDo
                    iLook += 1
                    continue
                end
            end
        end

        if integerStatus == Int32(2)
            tgtcolx, tgtcoly = tgtcoly, tgtcolx
            krowx, krowy = krowy, krowx
        elseif integerStatus == Int32(0)
            if abs(unsafe_load(colCoeffs, Int(krowy) + 1)) < abs(unsafe_load(colCoeffs, Int(krowx) + 1))
                tgtcolx, tgtcoly = tgtcoly, tgtcolx
                krowx, krowy = krowy, krowx
            end
        end
        tx1 = Int(tgtcolx) + 1
        ty1 = Int(tgtcoly) + 1

        kcsy = Int(unsafe_load(colStarts, ty1))
        kcey = kcsy + Int(unsafe_load(colLengths, ty1))
        singletonRow = false
        kcol = kcsy
        while kcol < kcey
            if unsafe_load(rowLengths, Int(unsafe_load(rowIndices, kcol + 1)) + 1) == Int32(1)
                singletonRow = true
                break
            end
            kcol += 1
        end
        if singletonRow || ps_col_prohibited2(any_prohib, col_changed, tgtcoly)
            iLook += 1
            continue
        end

        coeffx = unsafe_load(colCoeffs, Int(krowx) + 1)
        coeffy = unsafe_load(colCoeffs, Int(krowy) + 1)

        lenx = unsafe_load(colLengths, tx1)
        leny = unsafe_load(colLengths, ty1)
        s_ncolx = lenx
        s_ncoly = leny
        if leny < lenx
            s_colel = ps_dupmajor!(alloc_dbl_cb, colCoeffs, rowIndices, leny, unsafe_load(colStarts, ty1), tgtrow)
            s_ncolx = Int32(0)
        else
            s_colel = ps_dupmajor!(alloc_dbl_cb, colCoeffs, rowIndices, lenx, unsafe_load(colStarts, tx1), tgtrow)
            s_ncoly = Int32(0)
        end

        clox_v = unsafe_load(clo, tx1)
        cupx_v = unsafe_load(cup, tx1)
        costx_v = unsafe_load(cost, tx1)
        costy_v = unsafe_load(cost, ty1)

        nactions += Int32(1)
        unsafe_store!(actions_out,
            DoubletonActionPod(clox_v, cupx_v, costx_v, costy_v, rhs, coeffx, coeffy,
                               s_colel, tgtcolx, tgtcoly, tgtrow, s_ncolx, s_ncoly, Int32(0)),
            Int(nactions))

        # Transfer bounds from y to x
        lo1 = -PRESOLVE_INF_DBL
        up1 = PRESOLVE_INF_DBL
        cloy_v = unsafe_load(clo, ty1)
        cupy_v = unsafe_load(cup, ty1)
        neg_cx = -coeffx
        prod_xy = coeffx * coeffy
        if -PRESOLVE_INF_DBL < cloy_v
            v = _fnmsub(coeffy, cloy_v, rhs) / neg_cx
            if prod_xy < 0.0
                lo1 = v
            else
                up1 = v
            end
        end
        if cupy_v < PRESOLVE_INF_DBL
            v = _fnmsub(coeffy, cupy_v, rhs) / neg_cx
            if prod_xy < 0.0
                up1 = v
            else
                lo1 = v
            end
        end

        unsafe_store!(cost, costx_v + (costy_v * neg_cx) / coeffy, tx1)
        dobias += (costy_v * rhs) / coeffy

        lo2 = clox_v > lo1 ? clox_v : lo1
        up2 = cupx_v < up1 ? cupx_v : up1
        if lo2 > up2
            if lo2 <= up2 + feas_tol || fixInfeasibility
                nearest = floor(lo2 + 0.5)
                if abs(nearest - lo2) < 2.0 * feas_tol
                    lo2 = nearest
                    up2 = nearest
                else
                    lo2 = up2
                end
            else
                status |= Int32(1)
                infeas_col = tgtcolx
                infeas_lo = lo2
                infeas_up = up2
                break
            end
        end
        if unsafe_load(integerType, tx1) != UInt8(0)
            lo2 = ceil(lo2 - 1.0e-7)
            up2 = floor(up2 + 1.0e-7)
        end
        unsafe_store!(clo, lo2, tx1)
        unsafe_store!(cup, up2, tx1)

        if rowstat != C_NULL && sol != C_NULL
            numberBasic = Int32(0)
            movement = 0.0
            if ps_col_is_basic(colstat, tgtcolx)
                numberBasic += Int32(1)
            end
            if ps_col_is_basic(colstat, tgtcoly)
                numberBasic += Int32(1)
            end
            if ps_row_is_basic(rowstat, tgtrow)
                numberBasic += Int32(1)
            end
            sol_x = unsafe_load(sol, tx1)
            if sol_x <= lo2 + ztolzb
                movement = lo2 - sol_x
                sol_x = lo2
                unsafe_store!(sol, lo2, tx1)
                ps_set_col_status!(colstat, tgtcolx, PS_STATUS_AT_LOWER_BOUND)
            elseif sol_x >= up2 - ztolzb
                movement = up2 - sol_x
                sol_x = up2
                unsafe_store!(sol, up2, tx1)
                ps_set_col_status!(colstat, tgtcolx, PS_STATUS_AT_UPPER_BOUND)
            end
            if numberBasic > Int32(1)
                ps_set_col_status!(colstat, tgtcolx, PS_STATUS_BASIC)
            end
            if movement != 0.0
                kkcsx = Int(unsafe_load(colStarts, tx1))
                kkcex = kkcsx + Int(unsafe_load(colLengths, tx1))
                kc = kkcsx
                while kc < kkcex
                    kc1 = kc + 1
                    r1 = Int(unsafe_load(rowIndices, kc1)) + 1
                    if unsafe_load(rowLengths, r1) != Int32(0)
                        unsafe_store!(acts, _fmadd(movement, unsafe_load(colCoeffs, kc1), unsafe_load(acts, r1)), r1)
                    end
                    kc += 1
                end
            end
            movement = ((neg_cx * sol_x) / coeffy) - unsafe_load(sol, ty1)
            if movement != 0.0
                kkcsy = Int(unsafe_load(colStarts, ty1))
                kkcey = kkcsy + Int(unsafe_load(colLengths, ty1))
                kc = kkcsy
                while kc < kkcey
                    kc1 = kc + 1
                    r1 = Int(unsafe_load(rowIndices, kc1)) + 1
                    if unsafe_load(rowLengths, r1) != Int32(0)
                        unsafe_store!(acts, _fmadd(movement, unsafe_load(colCoeffs, kc1), unsafe_load(acts, r1)), r1)
                    end
                    kc += 1
                end
            end
        end
        if lo2 == up2
            nfixed += Int32(1)
            unsafe_store!(fixed_arr, tgtcolx, Int(nfixed))
        end

        n_next_cols = ps_add_col!(col_changed, next_cols, n_next_cols, tgtcolx)
        kkcsy = Int(unsafe_load(colStarts, ty1))
        kkcey = kkcsy + Int(unsafe_load(colLengths, ty1))
        kc = kkcsy
        while kc < kkcey
            n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, unsafe_load(rowIndices, kc + 1))
            kc += 1
        end
        kkcsx = Int(unsafe_load(colStarts, tx1))
        kkcex = kkcsx + Int(unsafe_load(colLengths, tx1))
        kc = kkcsx
        while kc < kkcex
            n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, unsafe_load(rowIndices, kc + 1))
            kc += 1
        end

        ps_delete_from_major!(tgtcolx, tgtrow, colStarts, colLengths, rowIndices, colCoeffs)
        ps_delete_from_major!(tgtcoly, tgtrow, colStarts, colLengths, rowIndices, colCoeffs)

        unsafe_store!(rowLengths, Int32(0), tr1)
        ps_remove_link!(rlink, tgtrow)

        if ps_elim_doubleton!(colStarts, rlo, rup, colCoeffs, rowIndices, colIndices,
                              rowLengths, colLengths, clink, n, rowStarts, rowCoeffs,
                              neg_cx / coeffy, rhs / coeffy, tgtcolx, tgtcoly)
            nomem = Int32(1)
            break
        end

        unsafe_store!(colLengths, Int32(0), ty1)
        ps_remove_link!(clink, tgtcoly)
        unsafe_store!(cost, 0.0, ty1)

        unsafe_store!(rlo, 0.0, tr1)
        unsafe_store!(rup, 0.0, tr1)
        if rowstat != C_NULL
            ps_set_row_status!(rowstat, tgtrow, PS_STATUS_AT_LOWER_BOUND)
        end

        nzeros += Int32(1)
        unsafe_store!(zeros_arr, tgtcolx, Int(nzeros))

        iLook += 1
    end

    _ps_store_f64!(pv, _PS_OFF_dobias, dobias)
    _ps_store_i32!(pv, _PS_OFF_status, status)
    _ps_store_i32!(pv, _PS_OFF_numberNextColsToDo, n_next_cols)
    _ps_store_i32!(pv, _PS_OFF_numberNextRowsToDo, n_next_rows)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_nzeros, nzeros)
    unsafe_store!(out_nfixed, nfixed)
    unsafe_store!(out_nomem, nomem)
    unsafe_store!(out_infeas_col, infeas_col)
    unsafe_store!(out_infeas_lo, infeas_lo)
    unsafe_store!(out_infeas_up, infeas_up)
    return nothing
end

function ps_doubleton_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{DoubletonActionPod},
    nactions::Int32,
    index1::Ptr{Int32},
    element1::Ptr{Float64},
)::Nothing
    p = unsafe_load(postv)
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    link = p.link

    clo = p.clo
    cup = p.cup
    rlo = p.rlo
    rup = p.rup
    dcost = p.cost
    sol = p.sol
    acts = p.acts
    rowduals = p.rowduals
    rcosts = p.rcosts
    colstat = p.colstat
    rowstat = p.rowstat
    origRlo = p.originalRowLower
    origRup = p.originalRowUpper

    maxmin = p.maxmin
    free_list = p.free_list
    ztolzb = p.ztolzb
    ztoldj = p.ztoldj
    ztolzero = 1.0e-12

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        irow = f.row
        ir1 = Int(irow) + 1
        lo0 = f.clox
        up0 = f.cupx
        coeffx = f.coeffx
        coeffy = f.coeffy
        jcolx = f.icolx
        jcoly = f.icoly
        jx1 = Int(jcolx) + 1
        jy1 = Int(jcoly) + 1
        rhs = f.rlo

        unsafe_store!(rlo, rhs, ir1)
        unsafe_store!(rup, rhs, ir1)
        unsafe_store!(clo, lo0, jx1)
        unsafe_store!(cup, up0, jx1)
        unsafe_store!(dcost, f.costx, jx1)
        unsafe_store!(dcost, f.costy, jy1)

        sol_x = unsafe_load(sol, jx1)
        diffy = _fmsub(coeffx, sol_x, rhs)
        sol_y = abs(diffy) < ztolzero ? 0.0 : (diffy / coeffy)
        unsafe_store!(sol, sol_y, jy1)
        unsafe_store!(acts, rhs, ir1)
        if rowstat != C_NULL
            ps_set_row_status!(rowstat, irow, PS_STATUS_AT_LOWER_BOUND)
        end

        djy = maxmin * f.costy
        djx = maxmin * f.costx

        if f.ncoly != Int32(0)
            ncoly = Int(f.ncoly) - 1
            f_colel = f.colel
            indy = Ptr{Int32}(f_colel + ncoly * sizeof(Float64))
            ystart = PRESOLVE_NO_LINK
            nX = 0
            kcol = 1
            while kcol <= ncoly
                i = unsafe_load(indy, kcol)
                i1 = Int(i) + 1
                yValue = unsafe_load(f_colel, kcol)
                shift = (yValue * rhs) / coeffy
                rlo_i = unsafe_load(rlo, i1)
                if -PRESOLVE_INF_DBL < rlo_i
                    unsafe_store!(rlo, rlo_i + shift, i1)
                end
                rup_i = unsafe_load(rup, i1)
                if rup_i < PRESOLVE_INF_DBL
                    unsafe_store!(rup, rup_i + shift, i1)
                end
                unsafe_store!(acts, unsafe_load(acts, i1) + shift, i1)
                djy = _fmsub(unsafe_load(rowduals, i1), yValue, djy)

                kfree = free_list
                kf1 = Int(kfree) + 1
                free_list = unsafe_load(link, kf1)
                unsafe_store!(hrow, i, kf1)
                unsafe_store!(colels, yValue, kf1)
                unsafe_store!(link, ystart, kf1)
                ystart = kfree

                yCorr = (yValue * coeffx) / coeffy
                unsafe_store!(element1, yCorr, i1)
                nX += 1
                unsafe_store!(index1, i, nX)
                kcol += 1
            end

            kfree = free_list
            kf1 = Int(kfree) + 1
            free_list = unsafe_load(link, kf1)
            unsafe_store!(hrow, irow, kf1)
            unsafe_store!(colels, coeffy, kf1)
            unsafe_store!(link, ystart, kf1)
            ystart = kfree

            unsafe_store!(element1, coeffx, ir1)
            nX += 1
            unsafe_store!(index1, irow, nX)

            unsafe_store!(mcstrt, ystart, jy1)
            unsafe_store!(hincol, f.ncoly, jy1)

            kcs = unsafe_load(mcstrt, jx1)
            last_nonzero = PRESOLVE_NO_LINK
            numberInColumn = unsafe_load(hincol, jx1)
            numberToDo = Int(numberInColumn)
            kcol = 0
            while kcol < numberToDo
                kcs1 = Int(kcs) + 1
                i = unsafe_load(hrow, kcs1)
                i1 = Int(i) + 1
                val = unsafe_load(colels, kcs1) + unsafe_load(element1, i1)
                unsafe_store!(element1, 0.0, i1)
                if abs(val) >= 1.0e-15
                    unsafe_store!(colels, val, kcs1)
                    last_nonzero = kcs
                    kcs = unsafe_load(link, kcs1)
                    djx = _fmsub(unsafe_load(rowduals, i1), val, djx)
                else
                    numberInColumn -= Int32(1)
                    nextk = unsafe_load(link, kcs1)
                    unsafe_store!(link, free_list, kcs1)
                    free_list = kcs
                    kcs = nextk
                    if last_nonzero != PRESOLVE_NO_LINK
                        unsafe_store!(link, kcs, Int(last_nonzero) + 1)
                    else
                        unsafe_store!(mcstrt, kcs, jx1)
                    end
                end
                kcol += 1
            end
            if last_nonzero != PRESOLVE_NO_LINK
                unsafe_store!(link, PRESOLVE_NO_LINK, Int(last_nonzero) + 1)
            end

            kcol = 1
            while kcol <= nX
                i = unsafe_load(index1, kcol)
                i1 = Int(i) + 1
                xValue = unsafe_load(element1, i1)
                unsafe_store!(element1, 0.0, i1)
                if abs(xValue) >= 1.0e-15
                    if i != irow
                        djx = _fmsub(unsafe_load(rowduals, i1), xValue, djx)
                    end
                    numberInColumn += Int32(1)
                    kf = free_list
                    kf1 = Int(kf) + 1
                    free_list = unsafe_load(link, kf1)
                    unsafe_store!(hrow, i, kf1)
                    unsafe_store!(colels, xValue, kf1)
                    unsafe_store!(link, unsafe_load(mcstrt, jx1), kf1)
                    unsafe_store!(mcstrt, kf, jx1)
                end
                kcol += 1
            end
            unsafe_store!(hincol, numberInColumn, jx1)
        else
            ncolx = Int(f.ncolx) - 1
            f_colel = f.colel
            indx = Ptr{Int32}(f_colel + ncolx * sizeof(Float64))

            kcs = unsafe_load(mcstrt, jx1)
            nX = 0
            hx_m1 = Int(unsafe_load(hincol, jx1)) - 1
            kcol = 0
            while kcol < hx_m1
                kcs1 = Int(kcs) + 1
                cv = unsafe_load(colels, kcs1)
                if cv != 0.0
                    i = unsafe_load(hrow, kcs1)
                    nX += 1
                    unsafe_store!(index1, i, nX)
                    unsafe_store!(element1, -(cv * coeffy) / coeffx, Int(i) + 1)
                end
                kcs = unsafe_load(link, kcs1)
                kcol += 1
            end
            kcs1 = Int(kcs) + 1
            cv = unsafe_load(colels, kcs1)
            if cv != 0.0
                i = unsafe_load(hrow, kcs1)
                nX += 1
                unsafe_store!(index1, i, nX)
                unsafe_store!(element1, -(cv * coeffy) / coeffx, Int(i) + 1)
            end

            unsafe_store!(link, free_list, kcs1)
            free_list = unsafe_load(mcstrt, jx1)
            xstart = PRESOLVE_NO_LINK
            kcol = 1
            while kcol <= ncolx
                i = unsafe_load(indx, kcol)
                i1 = Int(i) + 1
                xValue = unsafe_load(f_colel, kcol)
                kf = free_list
                kf1 = Int(kf) + 1
                free_list = unsafe_load(link, kf1)
                unsafe_store!(hrow, i, kf1)
                unsafe_store!(colels, xValue, kf1)
                unsafe_store!(link, xstart, kf1)
                xstart = kf

                djx = _fmsub(unsafe_load(rowduals, i1), xValue, djx)
                xScaled = (xValue * coeffy) / coeffx
                cur_e = unsafe_load(element1, i1)
                if cur_e == 0.0
                    unsafe_store!(element1, xScaled, i1)
                    nX += 1
                    unsafe_store!(index1, i, nX)
                else
                    unsafe_store!(element1, cur_e + xScaled, i1)
                end
                kcol += 1
            end

            kf = free_list
            kf1 = Int(kf) + 1
            free_list = unsafe_load(link, kf1)
            unsafe_store!(hrow, irow, kf1)
            unsafe_store!(colels, coeffx, kf1)
            unsafe_store!(link, xstart, kf1)
            xstart = kf
            unsafe_store!(element1, coeffy, ir1)
            nX += 1
            unsafe_store!(index1, irow, nX)

            unsafe_store!(mcstrt, xstart, jx1)
            unsafe_store!(hincol, f.ncolx, jx1)

            ystart = PRESOLVE_NO_LINK
            leny = Int32(0)
            kcol = 1
            while kcol <= nX
                i = unsafe_load(index1, kcol)
                i1 = Int(i) + 1
                yValue = unsafe_load(element1, i1)
                unsafe_store!(element1, 0.0, i1)
                if abs(yValue) >= ztolzero
                    leny += Int32(1)
                    ky = free_list
                    ky1 = Int(ky) + 1
                    free_list = unsafe_load(link, ky1)
                    unsafe_store!(hrow, i, ky1)
                    unsafe_store!(colels, yValue, ky1)
                    unsafe_store!(link, ystart, ky1)
                    ystart = ky
                end
                kcol += 1
            end

            unsafe_store!(mcstrt, ystart, jy1)
            unsafe_store!(hincol, leny, jy1)

            kcs = ystart
            ny = Int(leny)
            kcol = 0
            while kcol < ny
                kcs1 = Int(kcs) + 1
                row = unsafe_load(hrow, kcs1)
                coeff = unsafe_load(colels, kcs1)
                kcs = unsafe_load(link, kcs1)
                if row != irow
                    r1 = Int(row) + 1
                    shift = (coeff * rhs) / coeffy
                    rlo_r = unsafe_load(rlo, r1)
                    if -PRESOLVE_INF_DBL < rlo_r
                        unsafe_store!(rlo, rlo_r + shift, r1)
                    end
                    rup_r = unsafe_load(rup, r1)
                    if rup_r < PRESOLVE_INF_DBL
                        unsafe_store!(rup, rup_r + shift, r1)
                    end
                    unsafe_store!(acts, unsafe_load(acts, r1) + shift, r1)
                    djy = _fmsub(unsafe_load(rowduals, r1), coeff, djy)
                end
                kcol += 1
            end
        end

        if colstat != C_NULL
            basicx = ps_col_is_basic(colstat, jcolx)
            rcx = unsafe_load(rcosts, jx1)
            nblbxok = (abs(lo0 - sol_x) < ztolzb) && (rcx >= -ztoldj)
            nbubxok = (abs(up0 - sol_x) < ztolzb) && (rcx <= ztoldj)
            basicColumn = Int32(-1)
            if basicx || nblbxok || nbubxok
                if !basicx
                    if nblbxok
                        ps_set_col_status!(colstat, jcolx, PS_STATUS_AT_LOWER_BOUND)
                    elseif nbubxok
                        ps_set_col_status!(colstat, jcolx, PS_STATUS_AT_UPPER_BOUND)
                    end
                end
                ps_set_col_status!(colstat, jcoly, PS_STATUS_BASIC)
                basicColumn = jcoly
                dual_i = djy / coeffy
                unsafe_store!(rowduals, dual_i, ir1)
                unsafe_store!(rcosts, _fmsub(dual_i, coeffx, djx), jx1)
                unsafe_store!(rcosts, 0.0, jy1)
            else
                ps_set_col_status!(colstat, jcolx, PS_STATUS_BASIC)
                basicColumn = jcolx
                ps_set_col_status_using_value!(colstat, sol, clo, cup, ztolzb, jcoly)
                dual_i = djx / coeffx
                unsafe_store!(rowduals, dual_i, ir1)
                unsafe_store!(rcosts, _fmsub(dual_i, coeffy, djy), jy1)
                unsafe_store!(rcosts, 0.0, jx1)
            end

            if origRlo != C_NULL &&
               unsafe_load(origRlo, ir1) != unsafe_load(origRup, ir1) &&
               !basicx
                statusx = if sol_x - lo0 < ztolzb
                    Int32(0)
                elseif up0 - sol_x < ztolzb
                    Int32(1)
                else
                    Int32(2)
                end
                statusy = if sol_y - unsafe_load(clo, jy1) < ztolzb
                    Int32(0)
                elseif unsafe_load(cup, jy1) - sol_y < ztolzb
                    Int32(1)
                else
                    Int32(2)
                end
                if statusx != Int32(2) && statusy != Int32(2)
                    activity = unsafe_load(acts, ir1)
                    lower = unsafe_load(origRlo, ir1)
                    upper = unsafe_load(origRup, ir1)
                    statusRow = Int32(2)
                    if activity - lower < 10.0 * ztolzb
                        statusRow = Int32(0)
                    elseif upper - activity < 10.0 * ztolzb
                        statusRow = Int32(1)
                    end
                    mult_r = statusRow == Int32(0) ? 1.0 : (statusRow == Int32(1) ? -1.0 : 0.0)
                    mult_x = statusx == Int32(0) ? 1.0 : -1.0
                    mult_y = statusy == Int32(0) ? 1.0 : -1.0
                    if unsafe_load(rowduals, ir1) * mult_r < -ztoldj
                        if djx * mult_x > -ztoldj && djy * mult_y > -ztoldj
                            unsafe_store!(rowduals, 0.0, ir1)
                            ps_set_row_status!(rowstat, irow, PS_STATUS_BASIC)
                            unsafe_store!(rcosts, djx, jx1)
                            ps_set_col_status!(colstat, jcolx, statusx != Int32(0) ? PS_STATUS_AT_UPPER_BOUND : PS_STATUS_AT_LOWER_BOUND)
                            ps_set_col_status!(colstat, jcoly, statusy != Int32(0) ? PS_STATUS_AT_UPPER_BOUND : PS_STATUS_AT_LOWER_BOUND)
                            unsafe_store!(rcosts, djy, jy1)
                            basicColumn = Int32(-1)
                        else
                            if basicColumn == jcoly
                                newDual = djx / coeffx
                                newDj = _fmsub(newDual, coeffy, djy)
                                if newDj * mult_y > -ztoldj && newDual * mult_r >= -ztoldj
                                    unsafe_store!(rowduals, newDual, ir1)
                                    unsafe_store!(rcosts, newDj, jy1)
                                    unsafe_store!(rcosts, 0.0, jx1)
                                    basicColumn = jcolx
                                end
                            else
                                newDual = djy / coeffy
                                newDj = _fmsub(newDual, coeffx, djx)
                                if newDj * mult_y > -ztoldj && newDual >= -ztoldj
                                    unsafe_store!(rowduals, newDual, ir1)
                                    unsafe_store!(rcosts, newDj, jx1)
                                    unsafe_store!(rcosts, 0.0, jy1)
                                    basicColumn = jcoly
                                end
                            end
                        end
                        if basicColumn == jcolx
                            ps_set_col_status!(colstat, jcolx, PS_STATUS_BASIC)
                            ps_set_col_status!(colstat, jcoly, statusy != Int32(0) ? PS_STATUS_AT_UPPER_BOUND : PS_STATUS_AT_LOWER_BOUND)
                        elseif basicColumn == jcoly
                            ps_set_col_status!(colstat, jcoly, PS_STATUS_BASIC)
                            ps_set_col_status!(colstat, jcolx, statusx != Int32(0) ? PS_STATUS_AT_UPPER_BOUND : PS_STATUS_AT_LOWER_BOUND)
                        end
                    end
                end
            end
        else
            unsafe_store!(rowduals, djy / coeffy, ir1)
            unsafe_store!(rcosts, 0.0, jy1)
        end

        idx -= 1
    end

    _post_store_i32!(postv, _POST_OFF_free_list, free_list)
    return nothing
end

# =============================================================================
# 2. CoinPresolveTripleton.cpp
# =============================================================================

function ps_elim_tripleton!(
    mcstrt::Ptr{Int32},
    rlo::Ptr{Float64},
    acts::Ptr{Float64},
    rup::Ptr{Float64},
    colels::Ptr{Float64},
    hrow::Ptr{Int32},
    hcol::Ptr{Int32},
    hinrow::Ptr{Int32},
    hincol::Ptr{Int32},
    clink::Ptr{PresolveHLinkPod},
    ncols::Int32,
    rlink::Ptr{PresolveHLinkPod},
    nrows::Int32,
    mrstrt::Ptr{Int32},
    rowels::Ptr{Float64},
    coeff_factorx::Float64,
    coeff_factorz::Float64,
    bounds_factor::Float64,
    row0::Int32,
    icolx::Int32,
    icoly::Int32,
    icolz::Int32,
)::Bool
    iy1 = Int(icoly) + 1
    kcs = unsafe_load(mcstrt, iy1)
    kce = kcs + unsafe_load(hincol, iy1)
    kcsx = unsafe_load(mcstrt, Int(icolx) + 1)
    kcex = kcsx + unsafe_load(hincol, Int(icolx) + 1)
    kcsz = unsafe_load(mcstrt, Int(icolz) + 1)
    kcez = kcsz + unsafe_load(hincol, Int(icolz) + 1)

    kcoly = kcs
    while kcoly < kce
        ky1 = Int(kcoly) + 1
        row = unsafe_load(hrow, ky1)
        if row != row0
            r1 = Int(row) + 1
            cy = unsafe_load(colels, ky1)
            if bounds_factor != 0.0
                rlo_r = unsafe_load(rlo, r1)
                if -PRESOLVE_INF_DBL < rlo_r
                    unsafe_store!(rlo, _fmsub(cy, bounds_factor, rlo_r), r1)
                end
                rup_r = unsafe_load(rup, r1)
                if rup_r < PRESOLVE_INF_DBL
                    unsafe_store!(rup, _fmsub(cy, bounds_factor, rup_r), r1)
                end
                if acts != C_NULL
                    unsafe_store!(acts, _fmsub(cy, bounds_factor, unsafe_load(acts, r1)), r1)
                end
            end

            kcolx = ps_find_minor1(row, kcsx, kcex, hrow)
            kcolz = ps_find_minor1(row, kcsz, kcez, hrow)

            if kcolx >= kcex && kcolz < kcez
                kcolx, kcolz = kcolz, kcolx
                kcsx, kcsz = kcsz, kcsx
                kcex, kcez = kcez, kcex
                icolx, icolz = icolz, icolx
                coeff_factorx, coeff_factorz = coeff_factorz, coeff_factorx
            end

            ix1 = Int(icolx) + 1
            iz1 = Int(icolz) + 1
            if kcolx < kcex
                kx1 = Int(kcolx) + 1
                new_x = _fmadd(unsafe_load(colels, Int(kcoly) + 1), coeff_factorx, unsafe_load(colels, kx1))
                unsafe_store!(colels, new_x, kx1)
                mrs = unsafe_load(mrstrt, r1)
                k2 = ps_find_minor1(icolx, mrs, mrs + unsafe_load(hinrow, r1), hcol)
                unsafe_store!(rowels, new_x, Int(k2) + 1)

                if kcolz < kcez
                    kz1 = Int(kcolz) + 1
                    new_z = _fmadd(unsafe_load(colels, Int(kcoly) + 1), coeff_factorz, unsafe_load(colels, kz1))
                    unsafe_store!(colels, new_z, kz1)
                    mrs = unsafe_load(mrstrt, r1)
                    k2z = ps_find_minor1(icolz, mrs, mrs + unsafe_load(hinrow, r1), hcol)
                    unsafe_store!(rowels, new_z, Int(k2z) + 1)
                    ps_delete_from_major!(row, icoly, mrstrt, hinrow, hcol, rowels)
                else
                    mrs = unsafe_load(mrstrt, r1)
                    k2y = ps_find_minor1(icoly, mrs, mrs + unsafe_load(hinrow, r1), hcol)
                    k2y1 = Int(k2y) + 1
                    unsafe_store!(hcol, icolz, k2y1)
                    unsafe_store!(rowels, unsafe_load(colels, Int(kcoly) + 1) * coeff_factorz, k2y1)

                    if ps_expand_major!(mcstrt, colels, hrow, hincol, clink, ncols, icolz)
                        return true
                    end
                    kcsx = unsafe_load(mcstrt, ix1)
                    kcex = kcsx + unsafe_load(hincol, ix1)
                    kcoly = unsafe_load(mcstrt, iy1) + (kcoly - kcs)
                    kcs = unsafe_load(mcstrt, iy1)
                    kce = kcs + unsafe_load(hincol, iy1)
                    kcsz = unsafe_load(mcstrt, iz1)
                    lenz = unsafe_load(hincol, iz1)
                    kcez = kcsz + lenz

                    kcez1 = Int(kcez) + 1
                    unsafe_store!(hrow, row, kcez1)
                    unsafe_store!(colels, unsafe_load(colels, Int(kcoly) + 1) * coeff_factorz, kcez1)
                    unsafe_store!(hincol, lenz + Int32(1), iz1)
                    kcez += Int32(1)
                end
            else
                mrs = unsafe_load(mrstrt, r1)
                k2y = ps_find_minor1(icoly, mrs, mrs + unsafe_load(hinrow, r1), hcol)
                k2y1 = Int(k2y) + 1
                cy_now = unsafe_load(colels, Int(kcoly) + 1)
                unsafe_store!(hcol, icolx, k2y1)
                unsafe_store!(rowels, cy_now * coeff_factorx, k2y1)

                ps_expand_major!(mrstrt, rowels, hcol, hinrow, rlink, nrows, row)
                lenr = unsafe_load(hinrow, r1)
                krez1 = Int(unsafe_load(mrstrt, r1) + lenr) + 1
                unsafe_store!(hcol, icolz, krez1)
                unsafe_store!(rowels, cy_now * coeff_factorz, krez1)
                unsafe_store!(hinrow, lenr + Int32(1), r1)

                if ps_expand_major!(mcstrt, colels, hrow, hincol, clink, ncols, icolx)
                    return true
                end
                kcoly = unsafe_load(mcstrt, iy1) + (kcoly - kcs)
                kcs = unsafe_load(mcstrt, iy1)
                kce = kcs + unsafe_load(hincol, iy1)
                kcsx = unsafe_load(mcstrt, ix1)
                lenx = unsafe_load(hincol, ix1)
                kcex = kcsx + lenx
                kcsz = unsafe_load(mcstrt, iz1)
                kcez = kcsz + unsafe_load(hincol, iz1)

                kcex1 = Int(kcex) + 1
                unsafe_store!(hrow, row, kcex1)
                unsafe_store!(colels, unsafe_load(colels, Int(kcoly) + 1) * coeff_factorx, kcex1)
                unsafe_store!(hincol, lenx + Int32(1), ix1)
                kcex += Int32(1)

                if ps_expand_major!(mcstrt, colels, hrow, hincol, clink, ncols, icolz)
                    return true
                end
                kcoly = unsafe_load(mcstrt, iy1) + (kcoly - kcs)
                kcs = unsafe_load(mcstrt, iy1)
                kce = kcs + unsafe_load(hincol, iy1)
                kcsx = unsafe_load(mcstrt, ix1)
                kcex = kcsx + unsafe_load(hincol, ix1)
                kcsz = unsafe_load(mcstrt, iz1)
                lenz = unsafe_load(hincol, iz1)
                kcez = kcsz + lenz

                kcez1 = Int(kcez) + 1
                unsafe_store!(hrow, row, kcez1)
                unsafe_store!(colels, unsafe_load(colels, Int(kcoly) + 1) * coeff_factorz, kcez1)
                unsafe_store!(hincol, lenz + Int32(1), iz1)
                kcez += Int32(1)
            end
        end
        kcoly += Int32(1)
    end
    unsafe_store!(hincol, Int32(0), iy1)
    return false
end

function ps_tripleton_presolve!(
    pv::Ptr{CoinPresolvePodView},
    actions_out::Ptr{TripletonActionPod},
    alloc_dbl_cb::Ptr{Cvoid},
    out_nactions::Ptr{Int32},
    out_nzeros::Ptr{Int32},
    out_nomem::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    ncols = p.ncols
    ni = Int(ncols)

    clo = p.clo
    cup = p.cup
    rowels = p.rowels
    hcol = p.hcol
    mrstrt = p.mrstrt
    hinrow = p.hinrow
    nrows = p.nrows
    rlo = p.rlo
    rup = p.rup
    clink = p.clink
    rlink = p.rlink
    integerType = p.integerType
    cost = p.cost

    numberLook = Int(p.numberRowsToDo)
    look = p.rowsToDo
    ztolzb = p.ztolzb
    rowstat = p.rowstat
    colstat = p.colstat
    acts = p.acts

    any_prohib = p.anyProhibited != Int32(0)
    col_changed = p.colChanged
    row_changed = p.rowChanged
    next_cols = p.nextColsToDo
    next_rows = p.nextRowsToDo
    n_next_cols = p.numberNextColsToDo
    n_next_rows = p.numberNextRowsToDo
    dobias = p.dobias

    zeros_arr = p.usefulColumnInt
    mark = Ptr{UInt8}(zeros_arr + ni * sizeof(Int32))
    j = 1
    while j <= ni
        unsafe_store!(mark, UInt8(0), j)
        j += 1
    end
    nzeros = Int32(0)
    nactions = Int32(0)
    nomem = Int32(0)

    iLook = 0
    while iLook < numberLook
        irow = unsafe_load(look, iLook + 1)
        ir1 = Int(irow) + 1
        if unsafe_load(hinrow, ir1) == Int32(3) &&
           abs(unsafe_load(rup, ir1) - unsafe_load(rlo, ir1)) <= PRESOLVE_ZTOLDP
            rhs = unsafe_load(rlo, ir1)
            krs = Int(unsafe_load(mrstrt, ir1))
            kre = krs + 3

            k = krs
            while k < kre
                unsafe_load(hincol, Int(unsafe_load(hcol, k + 1)) + 1) > Int32(0) && break
                k += 1
            end
            coeffx = unsafe_load(rowels, k + 1)
            if abs(coeffx) < PRESOLVE_ZTOLDP2
                iLook += 1
                continue
            end
            icolx = unsafe_load(hcol, k + 1)

            k += 1
            while k < kre
                unsafe_load(hincol, Int(unsafe_load(hcol, k + 1)) + 1) > Int32(0) && break
                k += 1
            end
            coeffy = unsafe_load(rowels, k + 1)
            if abs(coeffy) < PRESOLVE_ZTOLDP2
                iLook += 1
                continue
            end
            icoly = unsafe_load(hcol, k + 1)

            k += 1
            while k < kre
                unsafe_load(hincol, Int(unsafe_load(hcol, k + 1)) + 1) > Int32(0) && break
                k += 1
            end
            coeffz = unsafe_load(rowels, k + 1)
            if abs(coeffz) < PRESOLVE_ZTOLDP2
                iLook += 1
                continue
            end
            icolz = unsafe_load(hcol, k + 1)

            if coeffx * coeffz > 0.0
                if coeffx * coeffy > 0.0
                    iLook += 1
                    continue
                end
            elseif coeffx * coeffy > 0.0
                icoly, icolz = icolz, icoly
                coeffy, coeffz = coeffz, coeffy
            else
                icoly, icolx = icolx, icoly
                coeffy, coeffx = coeffx, coeffy
            end

            if any_prohib
                if !ps_col_can_touch(col_changed, icolx) ||
                   !ps_col_can_touch(col_changed, icoly) ||
                   !ps_col_can_touch(col_changed, icolz)
                    iLook += 1
                    continue
                end
            end

            ix1 = Int(icolx) + 1
            iy1 = Int(icoly) + 1
            iz1 = Int(icolz) + 1
            # Matches CoinPresolveTripleton.cpp line 440 verbatim
            if !(abs(unsafe_load(cup, ix1) - unsafe_load(clo, ix1)) < PRESOLVE_ZTOLDP) &&
               !(abs(unsafe_load(cup, iy1) - unsafe_load(clo, ix1)) < PRESOLVE_ZTOLDP) &&
               !(abs(unsafe_load(cup, iz1) - unsafe_load(clo, iy1)) < PRESOLVE_ZTOLDP)
                cx = -coeffx / coeffy
                cz = -coeffz / coeffy
                if unsafe_load(integerType, iy1) != UInt8(0)
                    if unsafe_load(integerType, ix1) == UInt8(0) || unsafe_load(integerType, iz1) == UInt8(0)
                        iLook += 1
                        continue
                    end
                    if cx != floor(cx + 0.5) || cz != floor(cz + 0.5)
                        iLook += 1
                        continue
                    end
                end
                rhsRatio = rhs / coeffy
                cloy_v = unsafe_load(clo, iy1)
                if cloy_v > -1.0e30
                    clox_v = unsafe_load(clo, ix1)
                    cloz_v = unsafe_load(clo, iz1)
                    if clox_v < -1.0e30 || cloz_v < -1.0e30
                        iLook += 1
                        continue
                    end
                    if _fmadd(cx, clox_v, cz * cloz_v) + rhsRatio < cloy_v - ztolzb
                        iLook += 1
                        continue
                    end
                end
                cupy_v = unsafe_load(cup, iy1)
                if cupy_v < 1.0e30
                    cupx_v = unsafe_load(cup, ix1)
                    cupz_v = unsafe_load(cup, iz1)
                    if cupx_v > 1.0e30 || cupz_v > 1.0e30
                        iLook += 1
                        continue
                    end
                    if _fmadd(cx, cupx_v, cz * cupz_v) + rhsRatio > cupy_v + ztolzb
                        iLook += 1
                        continue
                    end
                end

                singleton = false
                csy = Int(unsafe_load(mcstrt, iy1))
                leny = unsafe_load(hincol, iy1)
                cey = csy + Int(leny)
                kc = csy
                while kc < cey
                    jr = unsafe_load(hrow, kc + 1)
                    if unsafe_load(hinrow, Int(jr) + 1) == Int32(1)
                        singleton = true
                    end
                    if jr != irow
                        ps_set_row_used!(row_changed, jr)
                    end
                    kc += 1
                end
                nDuplicate = Int32(0)
                csx = Int(unsafe_load(mcstrt, ix1))
                lenx = unsafe_load(hincol, ix1)
                cex = csx + Int(lenx)
                kc = csx
                while kc < cex
                    jr = unsafe_load(hrow, kc + 1)
                    if jr != irow && ps_row_used(row_changed, jr)
                        nDuplicate += Int32(1)
                    end
                    kc += 1
                end
                csz = Int(unsafe_load(mcstrt, iz1))
                lenz = unsafe_load(hincol, iz1)
                cez = csz + Int(lenz)
                kc = csz
                while kc < cez
                    jr = unsafe_load(hrow, kc + 1)
                    if jr != irow && ps_row_used(row_changed, jr)
                        nDuplicate += Int32(1)
                    end
                    kc += 1
                end
                nAdded = leny - Int32(3) - nDuplicate
                kc = csy
                while kc < cey
                    ps_unset_row_used!(row_changed, unsafe_load(hrow, kc + 1))
                    kc += 1
                end

                if singleton || nAdded > Int32(2)
                    iLook += 1
                    continue
                end
                if (lenx == Int32(1) || lenz == Int32(1)) && leny == Int32(1)
                    iLook += 1
                    continue
                end

                costy_v = unsafe_load(cost, iy1)
                s_colel = ps_dupmajor!(alloc_dbl_cb, colels, hrow, leny, unsafe_load(mcstrt, iy1))
                nactions += Int32(1)
                unsafe_store!(actions_out,
                    TripletonActionPod(icolx, icolz, irow, icoly,
                                       cloy_v, cupy_v, costy_v,
                                       0.0, 0.0, 0.0,
                                       unsafe_load(rlo, ir1), unsafe_load(rup, ir1),
                                       coeffx, coeffy, coeffz,
                                       s_colel, Int32(0), leny),
                    Int(nactions))

                unsafe_store!(cost, _fmadd(costy_v, cx, unsafe_load(cost, ix1)), ix1)
                unsafe_store!(cost, _fmadd(costy_v, cz, unsafe_load(cost, iz1)), iz1)
                dobias += (costy_v * rhs) / coeffy

                if rowstat != C_NULL
                    numberBasic = Int32(0)
                    if ps_col_is_basic(colstat, icoly)
                        numberBasic += Int32(1)
                    end
                    if ps_row_is_basic(rowstat, irow)
                        numberBasic += Int32(1)
                    end
                    if numberBasic > Int32(1)
                        if !ps_col_is_basic(colstat, icolx)
                            ps_set_col_status!(colstat, icolx, PS_STATUS_BASIC)
                        else
                            ps_set_col_status!(colstat, icolz, PS_STATUS_BASIC)
                        end
                    end
                end

                n_next_cols = ps_add_col!(col_changed, next_cols, n_next_cols, icolx)
                kc = csy
                while kc < cey
                    n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, unsafe_load(hrow, kc + 1))
                    kc += 1
                end
                kc = csx
                while kc < cex
                    n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, unsafe_load(hrow, kc + 1))
                    kc += 1
                end
                n_next_cols = ps_add_col!(col_changed, next_cols, n_next_cols, icolz)
                kc = csz
                while kc < cez
                    n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, unsafe_load(hrow, kc + 1))
                    kc += 1
                end

                if ps_elim_tripleton!(mcstrt, rlo, acts, rup, colels, hrow, hcol, hinrow, hincol,
                                      clink, ncols, rlink, nrows, mrstrt, rowels,
                                      cx, cz, rhs / coeffy, irow, icolx, icoly, icolz)
                    nomem = Int32(1)
                    break
                end

                ps_delete_from_major!(icolx, irow, mcstrt, hincol, hrow, colels)
                ps_delete_from_major!(icolz, irow, mcstrt, hincol, hrow, colels)

                unsafe_store!(hinrow, Int32(0), ir1)
                ps_remove_link!(rlink, irow)
                ps_remove_link!(clink, icoly)
                unsafe_store!(cost, 0.0, iy1)
                unsafe_store!(rlo, 0.0, ir1)
                unsafe_store!(rup, 0.0, ir1)

                if unsafe_load(mark, ix1) == UInt8(0)
                    unsafe_store!(mark, UInt8(1), ix1)
                    nzeros += Int32(1)
                    unsafe_store!(zeros_arr, icolx, Int(nzeros))
                end
                if unsafe_load(mark, iz1) == UInt8(0)
                    unsafe_store!(mark, UInt8(1), iz1)
                    nzeros += Int32(1)
                    unsafe_store!(zeros_arr, icolz, Int(nzeros))
                end
            end
        end
        iLook += 1
    end

    _ps_store_f64!(pv, _PS_OFF_dobias, dobias)
    _ps_store_i32!(pv, _PS_OFF_numberNextColsToDo, n_next_cols)
    _ps_store_i32!(pv, _PS_OFF_numberNextRowsToDo, n_next_rows)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_nzeros, nzeros)
    unsafe_store!(out_nomem, nomem)
    return nothing
end

function ps_tripleton_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{TripletonActionPod},
    nactions::Int32,
    index1::Ptr{Int32},
    element1::Ptr{Float64},
    index2::Ptr{Int32},
    element2::Ptr{Float64},
)::Nothing
    p = unsafe_load(postv)
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    link = p.link

    clo = p.clo
    cup = p.cup
    rlo = p.rlo
    rup = p.rup
    dcost = p.cost
    sol = p.sol
    rcosts = p.rcosts
    acts = p.acts
    rowduals = p.rowduals
    colstat = p.colstat
    rowstat = p.rowstat

    maxmin = p.maxmin
    free_list = p.free_list
    ztolzb = p.ztolzb
    ztoldj = p.ztoldj

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        irow = f.row
        ir1 = Int(irow) + 1
        coeffx = f.coeffx
        coeffy = f.coeffy
        coeffz = f.coeffz
        jcolx = f.icolx
        jcoly = f.icoly
        jcolz = f.icolz
        jx1 = Int(jcolx) + 1
        jy1 = Int(jcoly) + 1
        jz1 = Int(jcolz) + 1
        rhs = f.rlo

        unsafe_store!(rlo, f.rlo, ir1)
        unsafe_store!(rup, f.rup, ir1)
        unsafe_store!(clo, f.cloy, jy1)
        unsafe_store!(cup, f.cupy, jy1)

        unsafe_store!(dcost, f.costy, jy1)
        cost_x = unsafe_load(dcost, jx1) + (f.costy * coeffx) / coeffy
        cost_z = unsafe_load(dcost, jz1) + (f.costy * coeffz) / coeffy
        unsafe_store!(dcost, cost_x, jx1)
        unsafe_store!(dcost, cost_z, jz1)

        sol_x = unsafe_load(sol, jx1)
        sol_z = unsafe_load(sol, jz1)
        sol_y = _fmsub(coeffz, sol_z, _fmsub(coeffx, sol_x, rhs)) / coeffy
        unsafe_store!(sol, sol_y, jy1)
        unsafe_store!(acts, rhs, ir1)
        if rowstat != C_NULL
            ps_set_row_status!(rowstat, irow, PS_STATUS_AT_LOWER_BOUND)
        end

        djy = maxmin * f.costy
        djx = maxmin * cost_x
        djz = maxmin * cost_z
        bounds_factor = rhs / coeffy
        multiplier1 = coeffx / coeffy
        multiplier2 = coeffz / coeffy

        ny = Int(f.ncoly)
        f_colel = f.colel
        indy = Ptr{Int32}(f_colel + ny * sizeof(Float64))
        ystart = PRESOLVE_NO_LINK
        nX = 0
        nZ = 0
        i = 1
        while i <= ny
            iRow = unsafe_load(indy, i)
            iR1 = Int(iRow) + 1
            yValue = unsafe_load(f_colel, i)
            k = free_list
            k1 = Int(k) + 1
            free_list = unsafe_load(link, k1)
            if iRow != irow
                rlo_r = unsafe_load(rlo, iR1)
                if -PRESOLVE_INF_DBL < rlo_r
                    unsafe_store!(rlo, _fmadd(yValue, bounds_factor, rlo_r), iR1)
                end
                rup_r = unsafe_load(rup, iR1)
                if rup_r < PRESOLVE_INF_DBL
                    unsafe_store!(rup, _fmadd(yValue, bounds_factor, rup_r), iR1)
                end
                unsafe_store!(acts, _fmadd(yValue, bounds_factor, unsafe_load(acts, iR1)), iR1)
                djy = _fmsub(unsafe_load(rowduals, iR1), yValue, djy)
            end
            unsafe_store!(hrow, iRow, k1)
            unsafe_store!(colels, yValue, k1)
            unsafe_store!(link, ystart, k1)
            ystart = k
            unsafe_store!(element1, yValue * multiplier1, iR1)
            nX += 1
            unsafe_store!(index1, iRow, nX)
            unsafe_store!(element2, yValue * multiplier2, iR1)
            nZ += 1
            unsafe_store!(index2, iRow, nZ)
            i += 1
        end
        unsafe_store!(mcstrt, ystart, jy1)
        unsafe_store!(hincol, f.ncoly, jy1)

        # Reconstruct column x
        k = unsafe_load(mcstrt, jx1)
        last = PRESOLVE_NO_LINK
        numberInColumn = unsafe_load(hincol, jx1)
        numberToDo = Int(numberInColumn)
        i = 0
        while i < numberToDo
            k1 = Int(k) + 1
            iRow = unsafe_load(hrow, k1)
            iR1 = Int(iRow) + 1
            val = unsafe_load(colels, k1) + unsafe_load(element1, iR1)
            unsafe_store!(element1, 0.0, iR1)
            if abs(val) >= 1.0e-15
                unsafe_store!(colels, val, k1)
                last = k
                k = unsafe_load(link, k1)
                if iRow != irow
                    djx = _fmsub(unsafe_load(rowduals, iR1), val, djx)
                end
            else
                numberInColumn -= Int32(1)
                nextk = unsafe_load(link, k1)
                unsafe_store!(link, free_list, k1)
                free_list = k
                k = nextk
                if last != PRESOLVE_NO_LINK
                    unsafe_store!(link, k, Int(last) + 1)
                else
                    unsafe_store!(mcstrt, k, jx1)
                end
            end
            i += 1
        end
        i = 1
        while i <= nX
            iRow = unsafe_load(index1, i)
            iR1 = Int(iRow) + 1
            xValue = unsafe_load(element1, iR1)
            unsafe_store!(element1, 0.0, iR1)
            if abs(xValue) >= 1.0e-15
                if iRow != irow
                    djx = _fmsub(unsafe_load(rowduals, iR1), xValue, djx)
                end
                numberInColumn += Int32(1)
                kf = free_list
                kf1 = Int(kf) + 1
                free_list = unsafe_load(link, kf1)
                unsafe_store!(hrow, iRow, kf1)
                unsafe_store!(colels, xValue, kf1)
                if last != PRESOLVE_NO_LINK
                    unsafe_store!(link, kf, Int(last) + 1)
                else
                    unsafe_store!(mcstrt, kf, jx1)
                end
                last = kf
            end
            i += 1
        end
        unsafe_store!(link, PRESOLVE_NO_LINK, Int(last) + 1)
        unsafe_store!(hincol, numberInColumn, jx1)

        # Reconstruct column z
        k = unsafe_load(mcstrt, jz1)
        last = PRESOLVE_NO_LINK
        numberInColumn = unsafe_load(hincol, jz1)
        numberToDo = Int(numberInColumn)
        i = 0
        while i < numberToDo
            k1 = Int(k) + 1
            iRow = unsafe_load(hrow, k1)
            iR1 = Int(iRow) + 1
            val = unsafe_load(colels, k1) + unsafe_load(element2, iR1)
            unsafe_store!(element2, 0.0, iR1)
            if abs(val) >= 1.0e-15
                unsafe_store!(colels, val, k1)
                last = k
                k = unsafe_load(link, k1)
                if iRow != irow
                    djz = _fmsub(unsafe_load(rowduals, iR1), val, djz)
                end
            else
                numberInColumn -= Int32(1)
                nextk = unsafe_load(link, k1)
                unsafe_store!(link, free_list, k1)
                free_list = k
                k = nextk
                if last != PRESOLVE_NO_LINK
                    unsafe_store!(link, k, Int(last) + 1)
                else
                    unsafe_store!(mcstrt, k, jz1)
                end
            end
            i += 1
        end
        i = 1
        while i <= nZ
            iRow = unsafe_load(index2, i)
            iR1 = Int(iRow) + 1
            zValue = unsafe_load(element2, iR1)
            unsafe_store!(element2, 0.0, iR1)
            if abs(zValue) >= 1.0e-15
                if iRow != irow
                    djz = _fmsub(unsafe_load(rowduals, iR1), zValue, djz)
                end
                numberInColumn += Int32(1)
                kf = free_list
                kf1 = Int(kf) + 1
                free_list = unsafe_load(link, kf1)
                unsafe_store!(hrow, iRow, kf1)
                unsafe_store!(colels, zValue, kf1)
                if last != PRESOLVE_NO_LINK
                    unsafe_store!(link, kf, Int(last) + 1)
                else
                    unsafe_store!(mcstrt, kf, jz1)
                end
                last = kf
            end
            i += 1
        end
        unsafe_store!(link, PRESOLVE_NO_LINK, Int(last) + 1)
        unsafe_store!(hincol, numberInColumn, jz1)

        if colstat != C_NULL
            rcx = unsafe_load(rcosts, jx1)
            if ps_col_is_basic(colstat, jcolx) ||
               (abs(unsafe_load(clo, jx1) - sol_x) < ztolzb && rcx >= -ztoldj) ||
               (abs(unsafe_load(cup, jx1) - sol_x) < ztolzb && rcx <= ztoldj) ||
               (ps_get_col_status(colstat, jcolx) == PS_STATUS_IS_FREE && abs(rcx) <= ztoldj)
                ps_set_col_status!(colstat, jcoly, PS_STATUS_BASIC)
                dual_i = djy / coeffy
                unsafe_store!(rowduals, dual_i, ir1)
                unsafe_store!(rcosts, _fmsub(dual_i, coeffx, djx), jx1)
                unsafe_store!(rcosts, _fmsub(dual_i, coeffz, djz), jz1)
                unsafe_store!(rcosts, 0.0, jy1)
            else
                ps_set_col_status!(colstat, jcolx, PS_STATUS_BASIC)
                ps_set_col_status_using_value!(colstat, sol, clo, cup, ztolzb, jcoly)
                dual_i = djx / coeffx
                unsafe_store!(rowduals, dual_i, ir1)
                unsafe_store!(rcosts, 0.0, jx1)
                unsafe_store!(rcosts, _fmsub(dual_i, coeffz, djz), jz1)
                unsafe_store!(rcosts, _fmsub(dual_i, coeffy, djy), jy1)
            end
        else
            unsafe_store!(rowduals, djy / coeffy, ir1)
            unsafe_store!(rcosts, 0.0, jy1)
        end

        idx -= 1
    end

    _post_store_i32!(postv, _POST_OFF_free_list, free_list)
    return nothing
end

# =============================================================================
# 3. CoinPresolveImpliedFree.cpp
# =============================================================================

function ps_check_row(
    mrstrt::Ptr{Int32},
    rowels::Ptr{Float64},
    hcol::Ptr{Int32},
    hinrow::Ptr{Int32},
    coeff_factor::Float64,
    kill_ratio::Float64,
    irowx::Int32,
    irowy::Int32,
    bad_out::Ptr{Int32},
)::Int32
    tolerance = kill_ratio * coeff_factor
    ry1 = Int(irowy) + 1
    rx1 = Int(irowx) + 1
    krsy = Int(unsafe_load(mrstrt, ry1))
    krey = krsy + Int(unsafe_load(hinrow, ry1))
    krsx = Int(unsafe_load(mrstrt, rx1))
    krex = krsx + Int(unsafe_load(hinrow, rx1))
    krowx = krsx
    nFill = Int32(0)
    nbad = unsafe_load(bad_out)

    krowy = krsy
    while krowy < krey
        ky1 = krowy + 1
        j = unsafe_load(hcol, ky1)
        while krowx < krex && unsafe_load(hcol, krowx + 1) < j
            krowx += 1
        end
        wy = unsafe_load(rowels, ky1)
        newcoeff = if krowx < krex && unsafe_load(hcol, krowx + 1) == j
            _fmadd(wy, coeff_factor, unsafe_load(rowels, krowx + 1))
        else
            nFill += Int32(1)
            wy * coeff_factor
        end
        if abs(newcoeff) < tolerance
            if newcoeff > 0.1 * tolerance
                nbad += Int32(1)
            end
            nFill -= Int32(1)
        end
        krowx += 1
        krowy += 1
    end
    unsafe_store!(bad_out, nbad)
    return nFill
end

function ps_implied_free_presolve!(
    pv::Ptr{CoinPresolvePodView},
    fill_level_ptr::Ptr{Int32},
    actions_out::Ptr{ImpliedFreeActionPod},
    infiniteDown::Ptr{Int32},
    infiniteUp::Ptr{Int32},
    maxDown::Ptr{Float64},
    maxUp::Ptr{Float64},
    alloc_dbl_cb::Ptr{Cvoid},
    out_nactions::Ptr{Int32},
    out_unprocessed::Ptr{Int32},
    out_number_free::Ptr{Int32},
    out_infeas_row::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    m = Int(p.nrows)
    n = Int(p.ncols)

    rowStarts = p.mrstrt
    rowLengths = p.hinrow
    colIndices = p.hcol
    rowCoeffs = p.rowels
    rlink = p.rlink

    colStarts = p.mcstrt
    colLengths = p.hincol
    rowIndices = p.hrow
    colCoeffs = p.colels
    clink = p.clink

    clo = p.clo
    cup = p.cup
    rlo = p.rlo
    rup = p.rup
    cost = p.cost
    integerType = p.integerType

    stopSomeStuff = (p.presolveOptions & Int32(0x04)) != Int32(0)
    fixInfeasibility = (p.presolveOptions & Int32(0x4000)) != Int32(0)
    feasTol = p.feasibilityTolerance
    any_prohib = p.anyProhibited != Int32(0)
    col_changed = p.colChanged
    row_changed = p.rowChanged
    next_cols = p.nextColsToDo
    n_next_cols = p.numberNextColsToDo
    dobias = p.dobias
    status = p.status

    fill_level = unsafe_load(fill_level_ptr)

    nactions = Int32(0)
    implied_free = p.usefulColumnInt
    whichFree = implied_free + n * sizeof(Int32)
    numberFree = Int32(0)
    infeas_row = Int32(-1)

    i = 1
    while i <= m
        unsafe_store!(infiniteUp, unsafe_load(rowLengths, i) > Int32(1) ? Int32(-1) : Int32(-2), i)
        i += 1
    end

    if any_prohib
        i = 1
        while i <= m
            rStart = Int(unsafe_load(rowStarts, i))
            rEnd = rStart + Int(unsafe_load(rowLengths, i))
            badRow = false
            j = rStart
            while j < rEnd
                if ps_col_prohibited(col_changed, unsafe_load(colIndices, j + 1))
                    badRow = true
                    break
                end
                j += 1
            end
            if badRow
                unsafe_store!(infiniteUp, Int32(-2), i)
            end
            i += 1
        end
    end

    large = 1.0e20

    numberLook = Int(p.numberColsToDo)
    look = p.colsToDo
    if fill_level < Int32(0)
        look = whichFree
        if !any_prohib
            k = 0
            while k < n
                unsafe_store!(look, Int32(k), k + 1)
                k += 1
            end
            numberLook = n
        else
            numberLook = 0
            k = 0
            while k < n
                if !ps_col_prohibited(col_changed, Int32(k))
                    numberLook += 1
                    unsafe_store!(look, Int32(k), numberLook)
                end
                k += 1
            end
        end
    end

    infeas = false
    maxLook = Int32(abs(fill_level))
    iLook = 0
    while iLook < numberLook
        tgtcol = unsafe_load(look, iLook + 1)
        tc1 = Int(tgtcol) + 1
        tgtcol_len = unsafe_load(colLengths, tc1)
        if tgtcol_len <= Int32(0) || tgtcol_len > maxLook
            iLook += 1
            continue
        end

        kcs = Int(unsafe_load(colStarts, tc1))
        kce = kcs + Int(tgtcol_len)
        singletonCol = tgtcol_len == Int32(1)
        possibleRow = false
        singletonRow = false
        ait_max = 20.0 * PRESOLVE_ZTOLDP2

        if singletonCol
            ir1 = Int(unsafe_load(rowIndices, kcs + 1)) + 1
            singletonRow = unsafe_load(rowLengths, ir1) == Int32(1)
            possibleRow = abs(unsafe_load(colCoeffs, kcs + 1)) > PRESOLVE_ZTOLDP2
        else
            kcol = kcs
            while kcol < kce
                kc1 = kcol + 1
                ir1 = Int(unsafe_load(rowIndices, kc1)) + 1
                if unsafe_load(rowLengths, ir1) == Int32(1)
                    singletonRow = true
                    break
                end
                abs_ait = abs(unsafe_load(colCoeffs, kc1))
                if abs_ait > ait_max
                    ait_max = abs_ait
                end
                if abs(unsafe_load(rlo, ir1) - unsafe_load(rup, ir1)) < feasTol && abs_ait > 0.1 * ait_max
                    possibleRow = true
                end
                kcol += 1
            end
        end

        if singletonRow || !possibleRow
            iLook += 1
            continue
        end

        lt = unsafe_load(clo, tc1)
        ut = unsafe_load(cup, tc1)
        impliedLow = -PRESOLVE_INF_DBL
        impliedHigh = PRESOLVE_INF_DBL
        subst_ndx = Int32(-1)
        subst_len = Int32(n)
        is_int_t = unsafe_load(integerType, tc1) != UInt8(0)

        kcol = kcs
        while kcol < kce
            kc1 = kcol + 1
            i_r = unsafe_load(rowIndices, kc1)
            ir1 = Int(i_r) + 1
            iu_i = unsafe_load(infiniteUp, ir1)
            if iu_i <= Int32(-2)
                kcol += 1
                continue
            end

            ait = unsafe_load(colCoeffs, kc1)
            leni = unsafe_load(rowLengths, ir1)
            rloi = unsafe_load(rlo, ir1)
            rupi = unsafe_load(rup, ir1)

            rowiOK = (abs(rloi - rupi) < feasTol) && (abs(ait) > 0.1 * ait_max)
            if rowiOK && is_int_t
                q = rloi / ait
                rowiOK = abs(q - floor(q + 0.5)) < feasTol
            end

            infUi = Int32(0)
            infLi = Int32(0)
            maxUi = 0.0
            maxLi = 0.0
            krs = Int(unsafe_load(rowStarts, ir1))
            kre = krs + Int(leni)

            if iu_i == Int32(-1)
                krow = krs
                while krow < kre
                    kr1 = krow + 1
                    aik = unsafe_load(rowCoeffs, kr1)
                    k1 = Int(unsafe_load(colIndices, kr1)) + 1
                    lk = unsafe_load(clo, k1)
                    uk = unsafe_load(cup, k1)
                    if aik > 0.0
                        if uk < large
                            maxUi = _fmadd(uk, aik, maxUi)
                        else
                            infUi += Int32(1)
                        end
                        if lk > -large
                            maxLi = _fmadd(lk, aik, maxLi)
                        else
                            infLi += Int32(1)
                        end
                    elseif aik < 0.0
                        if uk < large
                            maxLi = _fmadd(uk, aik, maxLi)
                        else
                            infLi += Int32(1)
                        end
                        if lk > -large
                            maxUi = _fmadd(lk, aik, maxUi)
                        else
                            infUi += Int32(1)
                        end
                    end
                    krow += 1
                end
                maxUinf = _fmadd(Float64(infUi), 1.0e31, maxUi)
                maxLinf = _fmsub(Float64(infLi), 1.0e31, maxLi)
                if maxUinf < rloi - feasTol
                    if ps_can_be_feasible(pv, i_r) != Int32(0)
                        maxUinf = rloi - feasTol
                    end
                end
                if maxLinf > rupi + feasTol
                    if ps_can_be_feasible(pv, i_r) != Int32(0)
                        maxLinf = rupi + feasTol
                    end
                end
                if maxUinf <= rupi + feasTol && maxLinf >= rloi - feasTol
                    unsafe_store!(infiniteUp, Int32(-2), ir1)
                elseif maxUinf < rloi - feasTol && !fixInfeasibility
                    status |= Int32(1)
                    infeas = true
                    infeas_row = i_r
                    unsafe_store!(infiniteUp, Int32(-3), ir1)
                elseif maxLinf > rupi + feasTol && !fixInfeasibility
                    status |= Int32(1)
                    infeas = true
                    infeas_row = i_r
                    unsafe_store!(infiniteUp, Int32(-3), ir1)
                else
                    unsafe_store!(infiniteUp, infUi, ir1)
                    unsafe_store!(infiniteDown, infLi, ir1)
                    unsafe_store!(maxUp, maxUi, ir1)
                    unsafe_store!(maxDown, maxLi, ir1)
                end
            else
                infUi = iu_i
                infLi = unsafe_load(infiniteDown, ir1)
                maxUi = unsafe_load(maxUp, ir1)
                maxLi = unsafe_load(maxDown, ir1)
            end

            infeas && break
            if unsafe_load(infiniteUp, ir1) == Int32(-2)
                kcol += 1
                continue
            end

            if ait > 0.0
                if rloi > -large
                    ltprime = if infUi == Int32(0)
                        v = ut + (rloi - maxUi) / ait
                        (abs(maxUi) > 1.0e8 && !singletonCol) ? _fmadd(-1.0e-12, abs(maxUi), v) : v
                    elseif infUi == Int32(1) && ut > large
                        v = (rloi - maxUi) / ait
                        (abs(maxUi) > 1.0e8 && !singletonCol) ? _fmadd(-1.0e-12, abs(maxUi), v) : v
                    else
                        -PRESOLVE_INF_DBL
                    end
                    if ltprime > impliedLow
                        impliedLow = ltprime
                    end
                end
                if rupi < large
                    utprime = if infLi == Int32(0)
                        v = lt + (rupi - maxLi) / ait
                        (abs(maxLi) > 1.0e8 && !singletonCol) ? _fmadd(1.0e-12, abs(maxLi), v) : v
                    elseif infLi == Int32(1) && lt < -large
                        v = (rupi - maxLi) / ait
                        (abs(maxLi) > 1.0e8 && !singletonCol) ? _fmadd(1.0e-12, abs(maxLi), v) : v
                    else
                        PRESOLVE_INF_DBL
                    end
                    if utprime < impliedHigh
                        impliedHigh = utprime
                    end
                end
            else
                if rloi > -large
                    utprime = if infUi == Int32(0)
                        v = lt + (rloi - maxUi) / ait
                        (abs(maxUi) > 1.0e8 && !singletonCol) ? _fmadd(1.0e-12, abs(maxUi), v) : v
                    elseif infUi == Int32(1) && lt < -large
                        v = (rloi - maxUi) / ait
                        (abs(maxUi) > 1.0e8 && !singletonCol) ? _fmadd(1.0e-12, abs(maxUi), v) : v
                    else
                        PRESOLVE_INF_DBL
                    end
                    if utprime < impliedHigh
                        impliedHigh = utprime
                    end
                end
                if rupi < large
                    ltprime = if infLi == Int32(0)
                        v = ut + (rupi - maxLi) / ait
                        (abs(maxLi) > 1.0e8 && !singletonCol) ? _fmadd(-1.0e-12, abs(maxLi), v) : v
                    elseif infLi == Int32(1) && ut > large
                        v = (rupi - maxLi) / ait
                        (abs(maxLi) > 1.0e8 && !singletonCol) ? _fmadd(-1.0e-12, abs(maxLi), v) : v
                    else
                        -PRESOLVE_INF_DBL
                    end
                    if ltprime > impliedLow
                        impliedLow = ltprime
                    end
                end
            end

            if is_int_t
                possR = true
                allOnes = true
                krow = krs
                while krow < kre
                    kr1 = krow + 1
                    j1 = Int(unsafe_load(colIndices, kr1)) + 1
                    scaled_aij = unsafe_load(rowCoeffs, kr1) / ait
                    if abs(scaled_aij) != 1.0
                        allOnes = false
                    end
                    if unsafe_load(integerType, j1) == UInt8(0) ||
                       abs(scaled_aij - floor(scaled_aij + 0.5)) > feasTol
                        possR = false
                        break
                    end
                    krow += 1
                end
                if rloi == 1.0 && leni >= Int32(5) && stopSomeStuff && allOnes
                    possR = false
                end
                rowiOK = rowiOK && possR
            end

            if rowiOK
                if subst_ndx < Int32(0) || leni < subst_len
                    subst_ndx = i_r
                    subst_len = leni
                end
            end
            kcol += 1
        end

        infeas && break

        if lt <= impliedLow && impliedHigh <= ut && (subst_ndx >= Int32(0) || singletonRow)
            nf1 = Int(numberFree) + 1
            unsafe_store!(implied_free, subst_ndx, nf1)
            unsafe_store!(infiniteUp, Int32(-4), Int(subst_ndx) + 1)
            unsafe_store!(whichFree, tgtcol, nf1)
            numberFree += Int32(1)
        end
        iLook += 1
    end

    if infeas
        _ps_store_i32!(pv, _PS_OFF_status, status)
        unsafe_store!(out_nactions, Int32(0))
        unsafe_store!(out_unprocessed, Int32(0))
        unsafe_store!(out_number_free, numberFree)
        unsafe_store!(out_infeas_row, infeas_row)
        return nothing
    end

    unprocessed = Int32(0)
    nf_i = Int(numberFree)
    iLook = 1
    while iLook <= nf_i
        tgtcol = unsafe_load(whichFree, iLook)
        tc1 = Int(tgtcol) + 1
        if unsafe_load(colLengths, tc1) != Int32(1)
            unprocessed += Int32(1)
            u1 = Int(unprocessed)
            unsafe_store!(whichFree, tgtcol, u1)
            unsafe_store!(implied_free, unsafe_load(implied_free, iLook), u1)
            iLook += 1
            continue
        end

        tgtrow = unsafe_load(implied_free, iLook)
        tr1 = Int(tgtrow) + 1
        tgtrow_len = unsafe_load(rowLengths, tr1)
        kcs1 = Int(unsafe_load(colStarts, tc1)) + 1
        tgtcol_coeff = unsafe_load(colCoeffs, kcs1)
        tgtcol_cost = unsafe_load(cost, tc1)

        krs = Int(unsafe_load(rowStarts, tr1))
        kre = krs + Int(tgtrow_len)
        if tgtcol_cost != 0.0
            maxOldCost = 0.0
            maxNewCost = 0.0
            krow = krs
            while krow < kre
                kr1 = krow + 1
                j = unsafe_load(colIndices, kr1)
                if j != tgtcol
                    oldCost = unsafe_load(cost, Int(j) + 1)
                    newCost = oldCost - (tgtcol_cost * unsafe_load(rowCoeffs, kr1)) / tgtcol_coeff
                    ao = abs(oldCost)
                    an = abs(newCost)
                    if ao > maxOldCost
                        maxOldCost = ao
                    end
                    if an > maxNewCost
                        maxNewCost = an
                    end
                end
                krow += 1
            end
            if maxNewCost > 1000.0 * (maxOldCost + 1.0) && maxOldCost != 0.0
                iLook += 1
                continue
            end
        end

        s_rowels = ps_dupmajor!(alloc_dbl_cb, rowCoeffs, colIndices, tgtrow_len, unsafe_load(rowStarts, tr1))
        s_costs = Ptr{Float64}(C_NULL)

        rlo_t = unsafe_load(rlo, tr1)
        rup_t = unsafe_load(rup, tr1)
        if tgtcol_cost != 0.0
            tgtrow_rhs = rup_t
            if abs(rlo_t - rup_t) > feasTol
                if rlo_t > -PRESOLVE_INF_DBL && rup_t < PRESOLVE_INF_DBL
                    tgtrow_rhs = (tgtcol_cost * tgtcol_coeff) > 0.0 ? rlo_t : rup_t
                elseif rup_t >= PRESOLVE_INF_DBL
                    tgtrow_rhs = rlo_t
                end
            end
            save_costs = ccall(alloc_dbl_cb, Ptr{Float64}, (Int32,), tgtrow_len)
            krow = krs
            while krow < kre
                kr1 = krow + 1
                j1 = Int(unsafe_load(colIndices, kr1)) + 1
                cj = unsafe_load(cost, j1)
                unsafe_store!(save_costs, cj, (krow - krs) + 1)
                unsafe_store!(cost, cj - (tgtcol_cost * unsafe_load(rowCoeffs, kr1)) / tgtcol_coeff, j1)
                krow += 1
            end
            dobias += (tgtcol_cost * tgtrow_rhs) / tgtcol_coeff
            unsafe_store!(cost, 0.0, tc1)
            s_costs = save_costs
        end

        nactions += Int32(1)
        unsafe_store!(actions_out,
            ImpliedFreeActionPod(tgtrow, tgtcol, unsafe_load(clo, tc1), unsafe_load(cup, tc1),
                                 rlo_t, rup_t, s_rowels, s_costs, tgtrow_len, Int32(0)),
            Int(nactions))

        krow = krs
        while krow < kre
            j = unsafe_load(colIndices, krow + 1)
            ps_delete_from_major!(j, tgtrow, colStarts, colLengths, rowIndices, colCoeffs)
            if unsafe_load(colLengths, Int(j) + 1) == Int32(0)
                ps_remove_link!(clink, j)
            else
                n_next_cols = ps_add_col!(col_changed, next_cols, n_next_cols, j)
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

    # Filter non-singletons if !anyInteger()
    if unprocessed != Int32(0) && p.anyInteger == Int32(0)
        nFreeSub = Int(unprocessed)
        unprocessed = Int32(0)
        rowsUsed = p.usefulRowInt
        nRowsUsed = Int32(0)
        relax = (p.presolveOptions & Int32(0x60000)) >> 17
        tol_base = 1.0e-12
        ri = Int32(0)
        while ri < relax
            tol_base *= 10.0
            ri += Int32(1)
        end

        iLook = 1
        while iLook <= nFreeSub
            tgtcol = unsafe_load(whichFree, iLook)
            tgtrow = unsafe_load(implied_free, iLook)
            tc1 = Int(tgtcol) + 1
            tr1 = Int(tgtrow) + 1
            clen = unsafe_load(colLengths, tc1)
            if clen < Int32(2) || clen > maxLook
                iLook += 1
                continue
            end
            tgtcs = Int(unsafe_load(colStarts, tc1))
            tgtce = tgtcs + Int(clen)

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
            if dealBreaker
                iLook += 1
                continue
            end

            kcol = tgtcs
            while kcol < tgtce
                ir = unsafe_load(rowIndices, kcol + 1)
                ps_set_row_used!(row_changed, ir)
                nRowsUsed += Int32(1)
                unsafe_store!(rowsUsed, ir, Int(nRowsUsed))
                kcol += 1
            end

            tgtrs = Int(unsafe_load(rowStarts, tr1))
            tgt_rlen = unsafe_load(rowLengths, tr1)
            coin_sort_2_std!(colIndices + tgtrs * sizeof(Int32),
                             rowCoeffs + tgtrs * sizeof(Float64), Int(tgt_rlen))

            start_c = Int(unsafe_load(colStarts, tc1))
            end_c = start_c + Int(unsafe_load(colLengths, tc1))
            nbad_ref = Int32(0)
            # Track bad elements without heap ref using pointer to rowsUsed? No, pass a tiny local inline check!
            numberBadElements = Int32(0)
            numberFill = -tgt_rlen
            colndx = start_c
            while colndx < end_c
                ir = unsafe_load(rowIndices, colndx + 1)
                if ir == tgtrow
                    colndx += 1
                    continue
                end
                ait = unsafe_load(colCoeffs, colndx + 1)
                cf = -ait / tgtcoeff
                ir1 = Int(ir) + 1
                krs = Int(unsafe_load(rowStarts, ir1))
                rlen_i = Int(unsafe_load(rowLengths, ir1))
                coin_sort_2_std!(colIndices + krs * sizeof(Int32),
                                 rowCoeffs + krs * sizeof(Float64), rlen_i)

                # Inline ps_check_row to avoid Ref allocation
                tol_c = tol_base * cf
                krsy = tgtrs
                krey = tgtrs + Int(tgt_rlen)
                krowx = krs
                krex = krs + rlen_i
                krowy = krsy
                while krowy < krey
                    ky1 = krowy + 1
                    j = unsafe_load(colIndices, ky1)
                    while krowx < krex && unsafe_load(colIndices, krowx + 1) < j
                        krowx += 1
                    end
                    wy = unsafe_load(rowCoeffs, ky1)
                    newcoeff = if krowx < krex && unsafe_load(colIndices, krowx + 1) == j
                        _fmadd(wy, cf, unsafe_load(rowCoeffs, krowx + 1))
                    else
                        numberFill += Int32(1)
                        wy * cf
                    end
                    if abs(newcoeff) < tol_c
                        if newcoeff > 0.1 * tol_c
                            numberBadElements += Int32(1)
                        end
                        numberFill -= Int32(1)
                    end
                    krowx += 1
                    krowy += 1
                end
                colndx += 1
            end

            if !(numberBadElements != Int32(0) ||
                 Int32(3) * numberFill > Int32(2) * (unsafe_load(colLengths, tc1) + unsafe_load(rowLengths, tr1)))
                unprocessed += Int32(1)
                u1 = Int(unprocessed)
                unsafe_store!(whichFree, tgtcol, u1)
                unsafe_store!(implied_free, tgtrow, u1)
            end
            iLook += 1
        end

        k = 1
        nru = Int(nRowsUsed)
        while k <= nru
            ps_unset_row_used!(row_changed, unsafe_load(rowsUsed, k))
            k += 1
        end
    end

    _ps_store_f64!(pv, _PS_OFF_dobias, dobias)
    _ps_store_i32!(pv, _PS_OFF_status, status)
    _ps_store_i32!(pv, _PS_OFF_numberNextColsToDo, n_next_cols)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_unprocessed, unprocessed)
    unsafe_store!(out_number_free, numberFree)
    unsafe_store!(out_infeas_row, infeas_row)
    return nothing
end

function ps_implied_free_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{ImpliedFreeActionPod},
    nactions::Int32,
)::Nothing
    p = unsafe_load(postv)
    colStarts = p.mcstrt
    colLengths = p.hincol
    rowIndices = p.hrow
    colCoeffs = p.colels
    link = p.link
    free_list = p.free_list

    clo = p.clo
    cup = p.cup
    rlo = p.rlo
    rup = p.rup
    cost = p.cost
    sol = p.sol
    rcosts = p.rcosts
    acts = p.acts
    rowduals = p.rowduals
    colstat = p.colstat
    rowstat = p.rowstat

    large = 1.0e20

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        tgtrow = f.row
        tr1 = Int(tgtrow) + 1
        tgtcol = f.col
        tc1 = Int(tgtcol) + 1
        tgtrow_len = Int(f.ninrow)
        tgtrow_coeffs = f.rowels
        tgtrow_cols = Ptr{Int32}(tgtrow_coeffs + tgtrow_len * sizeof(Float64))
        saved_costs = f.costs

        tgt_coeff = 0.0
        tgtrow_act = 0.0
        krow = 1
        while krow <= tgtrow_len
            j = unsafe_load(tgtrow_cols, krow)
            j1 = Int(j) + 1
            atj = unsafe_load(tgtrow_coeffs, krow)

            kk = free_list
            kk1 = Int(kk) + 1
            free_list = unsafe_load(link, kk1)
            unsafe_store!(link, unsafe_load(colStarts, j1), kk1)
            unsafe_store!(colStarts, kk, j1)
            unsafe_store!(colCoeffs, atj, kk1)
            unsafe_store!(rowIndices, tgtrow, kk1)

            if saved_costs != C_NULL
                unsafe_store!(cost, unsafe_load(saved_costs, krow), j1)
            end

            if j == tgtcol
                unsafe_store!(colLengths, Int32(1), j1)
                unsafe_store!(clo, f.clo, tc1)
                unsafe_store!(cup, f.cup, tc1)
                unsafe_store!(rcosts, -unsafe_load(cost, tc1) / atj, j1)
                tgt_coeff = atj
            else
                unsafe_store!(colLengths, unsafe_load(colLengths, j1) + Int32(1), j1)
                tgtrow_act = _fmadd(atj, unsafe_load(sol, j1), tgtrow_act)
            end
            krow += 1
        end

        rlo_t = f.rlo
        rup_t = f.rup
        unsafe_store!(rlo, rlo_t, tr1)
        unsafe_store!(rup, rup_t, tr1)

        ct = unsafe_load(cost, tc1) # maxmin == 1.0 hardwired in CoinPresolveImpliedFree.cpp
        possibleDual = ct / tgt_coeff
        unsafe_store!(rowduals, possibleDual, tr1)
        if possibleDual >= 0.0 && rlo_t > -large
            unsafe_store!(sol, (rlo_t - tgtrow_act) / tgt_coeff, tc1)
            unsafe_store!(acts, rlo_t, tr1)
            ps_set_row_status!(rowstat, tgtrow, PS_STATUS_AT_LOWER_BOUND)
        elseif possibleDual <= 0.0 && rup_t < large
            unsafe_store!(sol, (rup_t - tgtrow_act) / tgt_coeff, tc1)
            unsafe_store!(acts, rup_t, tr1)
            ps_set_row_status!(rowstat, tgtrow, PS_STATUS_AT_UPPER_BOUND)
        else
            if rup_t < large
                unsafe_store!(sol, (rup_t - tgtrow_act) / tgt_coeff, tc1)
                unsafe_store!(acts, rup_t, tr1)
                ps_set_row_status!(rowstat, tgtrow, PS_STATUS_AT_UPPER_BOUND)
            else
                unsafe_store!(sol, (rlo_t - tgtrow_act) / tgt_coeff, tc1)
                unsafe_store!(acts, rlo_t, tr1)
                ps_set_row_status!(rowstat, tgtrow, PS_STATUS_AT_LOWER_BOUND)
            end
        end
        ps_set_col_status!(colstat, tgtcol, PS_STATUS_BASIC)
        unsafe_store!(rcosts, 0.0, tc1)

        idx -= 1
    end

    _post_store_i32!(postv, _POST_OFF_free_list, free_list)
    return nothing
end

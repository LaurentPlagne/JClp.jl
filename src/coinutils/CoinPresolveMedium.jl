# Faithful translation of CoinUtils Presolve B7b:
#   - CoinPresolveTighten.cpp   (do_tighten_action)
#   - CoinPresolveForcing.cpp   (forcing_constraint_action, canBeFeasible)
#   - CoinPresolveSingleton.cpp (slack_doubleton_action, slack_singleton_action)
#   - CoinPresolveDual.cpp      (remove_dual_action)
# Zero-allocation, bit-identical arm64 FMA (`_fmadd`/`_fmsub`), `--trim=safe` compatible.

@inline function ps_col_prohibited(col_changed::Ptr{UInt8}, j::Int32)::Bool
    return (unsafe_load(col_changed, Int(j) + 1) & UInt8(2)) != UInt8(0)
end

@inline function ps_get_col_status(colstat::Ptr{UInt8}, j::Int32)::UInt8
    return unsafe_load(colstat, Int(j) + 1) & UInt8(7)
end

@inline function ps_get_row_status(rowstat::Ptr{UInt8}, i::Int32)::UInt8
    return unsafe_load(rowstat, Int(i) + 1) & UInt8(7)
end

@inline function ps_col_is_basic(colstat::Ptr{UInt8}, j::Int32)::Bool
    return ps_get_col_status(colstat, j) == PS_STATUS_BASIC
end

@inline function ps_row_is_basic(rowstat::Ptr{UInt8}, i::Int32)::Bool
    return ps_get_row_status(rowstat, i) == PS_STATUS_BASIC
end

# =============================================================================
# 1. CoinPresolveTighten.cpp (do_tighten_action)
# =============================================================================

function ps_tighten_presolve!(
    pv::Ptr{CoinPresolvePodView},
    actions_out::Ptr{TightenActionPod},
    alloc_cb::Ptr{Cvoid},
    infeas_cb::Ptr{Cvoid},
    cb_ctx::Ptr{Cvoid},
    out_nactions::Ptr{Int32},
    out_nuseless::Ptr{Int32},
    out_nfixup::Ptr{Int32},
    out_nfixdown::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    ncols = p.ncols

    clo = p.clo
    cup = p.cup
    rlo = p.rlo
    rup = p.rup
    dcost = p.cost
    integerType = p.integerType

    fix_cols = p.usefulColumnInt
    nfixup_cols = Int32(0)
    nfixdown_cols = ncols

    useless_rows = p.usefulRowInt
    nuseless_rows = Int32(0)
    nactions = Int32(0)

    numberLook = Int(p.numberColsToDo)
    look = p.colsToDo
    fixInfeasibility = (p.presolveOptions & Int32(0x4000)) != Int32(0)

    any_prohib = p.anyProhibited
    col_changed = p.colChanged
    row_changed = p.rowChanged
    next_cols = p.nextColsToDo
    next_rows = p.nextRowsToDo
    n_next_cols = p.numberNextColsToDo
    n_next_rows = p.numberNextRowsToDo
    status = p.status

    iLook = 0
    while iLook < numberLook
        j = unsafe_load(look, iLook + 1)
        j1 = Int(j) + 1
        if integerType != C_NULL && unsafe_load(integerType, j1) != UInt8(0)
            cj_lo = ceil(unsafe_load(clo, j1) - 1.0e-12)
            cj_up = floor(unsafe_load(cup, j1) + 1.0e-12)
            unsafe_store!(clo, cj_lo, j1)
            unsafe_store!(cup, cj_up, j1)
            if cj_lo > cj_up && !fixInfeasibility
                status |= Int32(1)
                if infeas_cb != C_NULL
                    ccall(infeas_cb, Cvoid, (Ptr{Cvoid}, Int32, Cdouble, Cdouble),
                          cb_ctx, j, cj_lo, cj_up)
                end
            end
        end

        if unsafe_load(dcost, j1) == 0.0 && !ps_col_prohibited2(any_prohib, col_changed, j)
            iflag = Int32(0)
            nonFree = Int32(0)
            kcs = Int(unsafe_load(mcstrt, j1))
            len_j = unsafe_load(hincol, j1)
            kce = kcs + Int(len_j)

            k = kcs
            while k < kce
                k1 = k + 1
                i = unsafe_load(hrow, k1)
                i1 = Int(i) + 1
                coeff = unsafe_load(colels, k1)
                rlb = unsafe_load(rlo, i1)
                rub = unsafe_load(rup, i1)

                if -1.0e28 < rlb && rub < 1.0e28
                    iflag = Int32(0)
                    break
                elseif -1.0e28 < rlb || rub < 1.0e28
                    nonFree += Int32(1)
                end

                jflag = if coeff > 0.0
                    rub > 1.0e28 ? Int32(1) : Int32(-1)
                else
                    rlb < -1.0e28 ? Int32(1) : Int32(-1)
                end

                if iflag != Int32(0)
                    if iflag != jflag
                        iflag = Int32(0)
                        break
                    end
                else
                    iflag = jflag
                end
                k += 1
            end

            if nonFree == Int32(0)
                iflag = Int32(0)
            end

            if iflag != Int32(0)
                if iflag == Int32(1) && unsafe_load(cup, j1) < 1.0e10
                    nfixup_cols += Int32(1)
                    unsafe_store!(fix_cols, j, Int(nfixup_cols))
                elseif iflag == Int32(-1) && unsafe_load(clo, j1) > -1.0e10
                    nfixdown_cols -= Int32(1)
                    unsafe_store!(fix_cols, j, Int(nfixdown_cols) + 1)
                else
                    dir = iflag
                    if integerType != C_NULL && unsafe_load(integerType, j1) != UInt8(0)
                        dir = dir * Int32(2)
                    end
                    slot_ptr = actions_out + Int(nactions) * sizeof(TightenActionPod)
                    ccall(alloc_cb, Cvoid, (Ptr{TightenActionPod}, Int32), slot_ptr, len_j)
                    slot_init = unsafe_load(slot_ptr)
                    s_rows = slot_init.rows
                    s_lb = slot_init.lbound
                    s_ub = slot_init.ubound

                    nr = Int32(0)
                    n_next_cols = ps_add_col!(col_changed, next_cols, n_next_cols, j)
                    k = kcs
                    while k < kce
                        irow = unsafe_load(hrow, k + 1)
                        ir1 = Int(irow) + 1
                        r_lo = unsafe_load(rlo, ir1)
                        r_up = unsafe_load(rup, ir1)
                        if !(r_lo == -PRESOLVE_INF_DBL && r_up == PRESOLVE_INF_DBL)
                            n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, irow)
                            nr1 = Int(nr) + 1
                            unsafe_store!(s_rows, irow, nr1)
                            unsafe_store!(s_lb, r_lo, nr1)
                            unsafe_store!(s_ub, r_up, nr1)
                            nr += Int32(1)

                            nuseless_rows += Int32(1)
                            unsafe_store!(useless_rows, irow, Int(nuseless_rows))

                            unsafe_store!(rlo, -PRESOLVE_INF_DBL, ir1)
                            unsafe_store!(rup, PRESOLVE_INF_DBL, ir1)
                        end
                        k += 1
                    end
                    nactions += Int32(1)
                    unsafe_store!(actions_out,
                        TightenActionPod(s_rows, s_lb, s_ub, j, nr, dir, Int32(0)),
                        Int(nactions))
                end
            end
        end
        iLook += 1
    end

    _ps_store_i32!(pv, _PS_OFF_status, status)
    _ps_store_i32!(pv, _PS_OFF_numberNextColsToDo, n_next_cols)
    _ps_store_i32!(pv, _PS_OFF_numberNextRowsToDo, n_next_rows)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_nuseless, nuseless_rows)
    unsafe_store!(out_nfixup, nfixup_cols)
    unsafe_store!(out_nfixdown, nfixdown_cols)
    return nothing
end

function ps_tighten_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{TightenActionPod},
    nactions::Int32,
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
    sol = p.sol
    acts = p.acts
    colstat = p.colstat
    rowstat = p.rowstat

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        jcol = f.col
        j1 = Int(jcol) + 1
        iflag = f.direction
        nr = Int(f.nrows)
        rows = f.rows
        lbound = f.lbound
        ubound = f.ubound

        i = 1
        while i <= nr
            irow1 = Int(unsafe_load(rows, i)) + 1
            unsafe_store!(rlo, unsafe_load(lbound, i), irow1)
            unsafe_store!(rup, unsafe_load(ubound, i), irow1)
            i += 1
        end

        correction = 0.0
        last_corrected = Int32(-1)
        k = unsafe_load(mcstrt, j1)
        nk = Int(unsafe_load(hincol, j1))
        sol_j = unsafe_load(sol, j1)
        i = 0
        while i < nk
            k1 = Int(k) + 1
            irow = unsafe_load(hrow, k1)
            coeff = unsafe_load(colels, k1)
            k = unsafe_load(link, k1)
            ir1 = Int(irow) + 1
            newrlo = unsafe_load(rlo, ir1)
            newrup = unsafe_load(rup, ir1)
            activity = unsafe_load(acts, ir1)

            est = _fmadd(correction, coeff, activity)
            if est < newrlo
                last_corrected = irow
                new_correction = (newrlo - activity) / coeff
                if iflag == Int32(-2) || iflag == Int32(2)
                    new_correction += sol_j
                    if abs(floor(new_correction + 0.5) - new_correction) > 1.0e-4
                        new_correction = ceil(new_correction) - sol_j
                    end
                end
                correction = new_correction
            elseif est > newrup
                last_corrected = irow
                new_correction = (newrup - activity) / coeff
                if iflag == Int32(-2) || iflag == Int32(2)
                    new_correction += sol_j
                    if abs(floor(new_correction + 0.5) - new_correction) > 1.0e-4
                        new_correction = ceil(new_correction) - sol_j
                    end
                end
                correction = new_correction
            end
            i += 1
        end

        if last_corrected >= Int32(0)
            new_sol = sol_j + correction
            unsafe_store!(sol, new_sol, j1)
            k = unsafe_load(mcstrt, j1)
            i = 0
            while i < nk
                k1 = Int(k) + 1
                ir1 = Int(unsafe_load(hrow, k1)) + 1
                coeff = unsafe_load(colels, k1)
                k = unsafe_load(link, k1)
                unsafe_store!(acts, _fmadd(correction, coeff, unsafe_load(acts, ir1)), ir1)
                i += 1
            end

            if abs(new_sol - unsafe_load(clo, j1)) > PRESOLVE_ZTOLDP &&
               abs(new_sol - unsafe_load(cup, j1)) > PRESOLVE_ZTOLDP
                ps_set_col_status!(colstat, jcol, PS_STATUS_BASIC)
                lc1 = Int(last_corrected) + 1
                act_lc = unsafe_load(acts, lc1)
                if act_lc - unsafe_load(rlo, lc1) < unsafe_load(rup, lc1) - act_lc
                    ps_set_row_status!(rowstat, last_corrected, PS_STATUS_AT_LOWER_BOUND)
                else
                    ps_set_row_status!(rowstat, last_corrected, PS_STATUS_AT_UPPER_BOUND)
                end
            end
        end
        idx -= 1
    end
    return nothing
end

# =============================================================================
# 2. CoinPresolveForcing.cpp (forcing_constraint_action, canBeFeasible)
# =============================================================================

@inline function ps_implied_row_bounds(
    els::Ptr{Float64},
    clo::Ptr{Float64},
    cup::Ptr{Float64},
    hcol::Ptr{Int32},
    krs::Int,
    kre::Int,
)::Tuple{Float64, Float64}
    posinf = false
    neginf = false
    maxup = 0.0
    maxdown = 0.0
    kk = krs
    while kk < kre
        kk1 = kk + 1
        col1 = Int(unsafe_load(hcol, kk1)) + 1
        coeff = unsafe_load(els, kk1)
        lb = unsafe_load(clo, col1)
        ub = unsafe_load(cup, col1)
        if coeff > 0.0
            if PRESOLVE_INF_DBL <= ub
                posinf = true
                neginf && break
            else
                maxup = _fmadd(ub, coeff, maxup)
            end
            if lb <= -PRESOLVE_INF_DBL
                neginf = true
                posinf && break
            else
                maxdown = _fmadd(lb, coeff, maxdown)
            end
        else
            if PRESOLVE_INF_DBL <= ub
                neginf = true
                posinf && break
            else
                maxdown = _fmadd(ub, coeff, maxdown)
            end
            if lb <= -PRESOLVE_INF_DBL
                posinf = true
                neginf && break
            else
                maxup = _fmadd(lb, coeff, maxup)
            end
        end
        kk += 1
    end
    return (posinf ? PRESOLVE_INF_DBL : maxup, neginf ? -PRESOLVE_INF_DBL : maxdown)
end

function ps_can_be_feasible(pv::Ptr{CoinPresolvePodView}, iRow::Int32)::Int32
    p = unsafe_load(pv)
    clo = p.clo
    cup = p.cup
    mrstrt = p.mrstrt
    rowels = p.rowels
    hcol = p.hcol
    hinrow = p.hinrow
    rlo = p.rlo
    rup = p.rup
    inftol = p.feasibilityTolerance

    posinf = false
    neginf = false
    maxup = 0.0
    maxdown = 0.0

    ir1 = Int(iRow) + 1
    krs = Int(unsafe_load(mrstrt, ir1))
    kre = krs + Int(unsafe_load(hinrow, ir1))
    kk = krs
    while kk < kre
        kk1 = kk + 1
        col1 = Int(unsafe_load(hcol, kk1)) + 1
        coeff = unsafe_load(rowels, kk1)
        lb = unsafe_load(clo, col1) - inftol
        ub = unsafe_load(cup, col1) + inftol
        if coeff > 0.0
            if PRESOLVE_INF_DBL <= ub
                posinf = true
                neginf && break
            else
                maxup = _fmadd(ub, coeff, maxup)
            end
            if lb <= -PRESOLVE_INF_DBL
                neginf = true
                posinf && break
            else
                maxdown = _fmadd(lb, coeff, maxdown)
            end
        else
            if PRESOLVE_INF_DBL <= ub
                neginf = true
                posinf && break
            else
                maxdown = _fmadd(ub, coeff, maxdown)
            end
            if lb <= -PRESOLVE_INF_DBL
                posinf = true
                neginf && break
            else
                maxup = _fmadd(lb, coeff, maxup)
            end
        end
        kk += 1
    end

    feasible = true
    if maxup < unsafe_load(rlo, ir1) && !posinf
        feasible = false
    end
    if maxdown > unsafe_load(rup, ir1) && !neginf
        feasible = false
    end
    return feasible ? Int32(1) : Int32(0)
end

function ps_forcing_presolve!(
    pv::Ptr{CoinPresolvePodView},
    actions_out::Ptr{ForcingActionPod},
    fixed_cols::Ptr{Int32},
    useless_rows::Ptr{Int32},
    alloc_cb::Ptr{Cvoid},
    out_nactions::Ptr{Int32},
    out_nuseless::Ptr{Int32},
    out_nfixed::Ptr{Int32},
    out_infeas_row::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    clo = p.clo
    cup = p.cup
    csol = p.sol
    mrstrt = p.mrstrt
    rowels = p.rowels
    hcol = p.hcol
    hinrow = p.hinrow
    rlo = p.rlo
    rup = p.rup

    tol = PRESOLVE_ZTOLDP
    inftol = p.feasibilityTolerance
    inftol2 = 0.01 * inftol

    nfixed_cols = Int32(0)
    nactions = Int32(0)
    nuseless_rows = Int32(0)
    infeas_row = Int32(-1)

    numberLook = Int(p.numberRowsToDo)
    look = p.rowsToDo
    fixInfeasibility = (p.presolveOptions & Int32(0x4000)) != Int32(0)
    any_prohib = p.anyProhibited != Int32(0)
    col_changed = p.colChanged
    next_cols = p.nextColsToDo
    n_next_cols = p.numberNextColsToDo
    status = p.status

    iLook = 0
    while iLook < numberLook
        irow = unsafe_load(look, iLook + 1)
        ir1 = Int(irow) + 1
        len_r = unsafe_load(hinrow, ir1)
        if len_r <= Int32(0)
            iLook += 1
            continue
        end
        krs = Int(unsafe_load(mrstrt, ir1))
        kre = krs + Int(len_r)

        maxup, maxdown = ps_implied_row_bounds(rowels, clo, cup, hcol, krs, kre)
        rlo_i = unsafe_load(rlo, ir1)
        rup_i = unsafe_load(rup, ir1)

        if maxup < PRESOLVE_INF_DBL && maxup + inftol < rlo_i && !fixInfeasibility
            if ps_can_be_feasible(pv, irow) == Int32(0)
                status |= Int32(1)
                infeas_row = irow
                break
            end
        end
        if -PRESOLVE_INF_DBL < maxdown && rup_i < maxdown - inftol && !fixInfeasibility
            if ps_can_be_feasible(pv, irow) == Int32(0)
                status |= Int32(1)
                infeas_row = irow
                break
            end
        end

        if ((rlo_i <= -PRESOLVE_INF_DBL) || (-PRESOLVE_INF_DBL < maxdown && rlo_i <= maxdown + inftol2)) &&
           ((rup_i >= PRESOLVE_INF_DBL) || (maxup < PRESOLVE_INF_DBL && rup_i >= maxup - inftol2))
            if any_prohib
                prohib = false
                k = krs
                while k < kre
                    if ps_col_prohibited(col_changed, unsafe_load(hcol, k + 1))
                        prohib = true
                        break
                    end
                    k += 1
                end
                if prohib
                    iLook += 1
                    continue
                end
            end
            nuseless_rows += Int32(1)
            unsafe_store!(useless_rows, irow, Int(nuseless_rows))
            iLook += 1
            continue
        end

        tightAtLower = (maxup < PRESOLVE_INF_DBL) && (abs(rlo_i - maxup) < tol)
        tightAtUpper = (-PRESOLVE_INF_DBL < maxdown) && (abs(rup_i - maxdown) < tol)
        if !(tightAtLower || tightAtUpper)
            iLook += 1
            continue
        end

        if any_prohib
            prohib = false
            k = krs
            while k < kre
                if ps_col_prohibited(col_changed, unsafe_load(hcol, k + 1))
                    prohib = true
                    break
                end
                k += 1
            end
            if prohib
                iLook += 1
                continue
            end
        end

        slot_ptr = actions_out + Int(nactions) * sizeof(ForcingActionPod)
        ccall(alloc_cb, Cvoid, (Ptr{ForcingActionPod}, Int32), slot_ptr, len_r)
        slot_init = unsafe_load(slot_ptr)
        rowcols = slot_init.rowcols
        bounds = slot_init.bounds

        lk = krs
        uk = kre
        k = krs
        while k < kre
            k1 = k + 1
            j = unsafe_load(hcol, k1)
            j1 = Int(j) + 1
            lj = unsafe_load(clo, j1)
            uj = unsafe_load(cup, j1)
            coeff = unsafe_load(rowels, k1)
            if tightAtLower == (coeff > 0.0)
                uk -= 1
                pos = (uk - krs) + 1
                unsafe_store!(bounds, lj, pos)
                unsafe_store!(rowcols, j, pos)
                if csol != C_NULL
                    unsafe_store!(csol, uj, j1)
                end
                unsafe_store!(clo, uj, j1)
            else
                pos = (lk - krs) + 1
                unsafe_store!(bounds, uj, pos)
                unsafe_store!(rowcols, j, pos)
                lk += 1
                if csol != C_NULL
                    unsafe_store!(csol, lj, j1)
                end
                unsafe_store!(cup, lj, j1)
            end
            if lj != uj
                nfixed_cols += Int32(1)
                unsafe_store!(fixed_cols, j, Int(nfixed_cols))
                n_next_cols = ps_add_col!(col_changed, next_cols, n_next_cols, j)
            end
            k += 1
        end

        nactions += Int32(1)
        unsafe_store!(actions_out,
            ForcingActionPod(rowcols, bounds, irow, Int32(lk - krs), Int32(kre - uk), Int32(0)),
            Int(nactions))
        iLook += 1
    end

    _ps_store_i32!(pv, _PS_OFF_status, status)
    _ps_store_i32!(pv, _PS_OFF_numberNextColsToDo, n_next_cols)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_nuseless, nuseless_rows)
    unsafe_store!(out_nfixed, nfixed_cols)
    unsafe_store!(out_infeas_row, infeas_row)
    return nothing
end

function ps_forcing_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{ForcingActionPod},
    nactions::Int32,
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
    rcosts = p.rcosts
    acts = p.acts
    rowduals = p.rowduals
    colstat = p.colstat
    rowstat = p.rowstat
    ztoldj = p.ztoldj
    ztolzb = p.ztolzb

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        irow = f.row
        ir1 = Int(irow) + 1
        nlo = Int(f.nlo)
        nup = Int(f.nup)
        ninrow = nlo + nup
        rowcols = f.rowcols
        bounds = f.bounds

        dualfeas = true
        k = 1
        while k <= nlo
            jcol = unsafe_load(rowcols, k)
            j1 = Int(jcol) + 1
            cbarj = unsafe_load(rcosts, j1)
            olduj = unsafe_load(cup, j1)
            newuj = unsafe_load(bounds, k)
            change = abs(newuj - olduj) > ztolzb
            if change && ps_get_col_status(colstat, jcol) != PS_STATUS_BASIC
                ps_set_col_status!(colstat, jcol, PS_STATUS_AT_LOWER_BOUND)
                if cbarj < -ztoldj || unsafe_load(clo, j1) <= -PRESOLVE_INF_DBL
                    dualfeas = false
                end
            end
            unsafe_store!(cup, newuj, j1)
            k += 1
        end

        k = nlo + 1
        while k <= ninrow
            jcol = unsafe_load(rowcols, k)
            j1 = Int(jcol) + 1
            cbarj = unsafe_load(rcosts, j1)
            oldlj = unsafe_load(clo, j1)
            newlj = unsafe_load(bounds, k)
            change = abs(newlj - oldlj) > ztolzb
            if change && ps_get_col_status(colstat, jcol) != PS_STATUS_BASIC
                ps_set_col_status!(colstat, jcol, PS_STATUS_AT_UPPER_BOUND)
                if cbarj > ztoldj || unsafe_load(cup, j1) >= PRESOLVE_INF_DBL
                    dualfeas = false
                end
            end
            unsafe_store!(clo, newlj, j1)
            k += 1
        end

        if !dualfeas
            joow = Int32(-1)
            yi = 0.0
            k = 1
            while k <= ninrow
                jcol = unsafe_load(rowcols, k)
                j1 = Int(jcol) + 1
                kk = ps_find_minor2(irow, unsafe_load(mcstrt, j1), unsafe_load(hincol, j1), hrow, link)
                cbarj = unsafe_load(rcosts, j1)
                statj = ps_get_col_status(colstat, jcol)
                if (cbarj < -ztoldj && statj != PS_STATUS_AT_UPPER_BOUND) ||
                   (cbarj > ztoldj && statj != PS_STATUS_AT_LOWER_BOUND)
                    yi_j = cbarj / unsafe_load(colels, Int(kk) + 1)
                    if abs(yi_j) > abs(yi)
                        joow = jcol
                        yi = yi_j
                    end
                end
                k += 1
            end

            ps_set_col_status!(colstat, joow, PS_STATUS_BASIC)
            act_r = unsafe_load(acts, ir1)
            if act_r - unsafe_load(rlo, ir1) < unsafe_load(rup, ir1) - act_r
                ps_set_row_status!(rowstat, irow, PS_STATUS_AT_LOWER_BOUND)
            else
                ps_set_row_status!(rowstat, irow, PS_STATUS_AT_UPPER_BOUND)
            end
            unsafe_store!(rowduals, yi, ir1)

            k = 1
            while k <= ninrow
                jcol = unsafe_load(rowcols, k)
                j1 = Int(jcol) + 1
                kk = ps_find_minor2(irow, unsafe_load(mcstrt, j1), unsafe_load(hincol, j1), hrow, link)
                old_cbarj = unsafe_load(rcosts, j1)
                new_cbarj = _fmsub(yi, unsafe_load(colels, Int(kk) + 1), old_cbarj)
                unsafe_store!(rcosts, new_cbarj, j1)
                if (old_cbarj < 0.0) != (new_cbarj < 0.0)
                    if new_cbarj < -ztoldj && unsafe_load(cup, j1) < PRESOLVE_INF_DBL
                        ps_set_col_status!(colstat, jcol, PS_STATUS_AT_UPPER_BOUND)
                    elseif new_cbarj > ztoldj && unsafe_load(clo, j1) > -PRESOLVE_INF_DBL
                        ps_set_col_status!(colstat, jcol, PS_STATUS_AT_LOWER_BOUND)
                    end
                end
                k += 1
            end
        end
        idx -= 1
    end
    return nothing
end

# =============================================================================
# 3. CoinPresolveSingleton.cpp (slack_doubleton_action & slack_singleton_action)
# =============================================================================

function ps_slack_doubleton_presolve!(
    pv::Ptr{CoinPresolvePodView},
    actions_out::Ptr{SlackActionPod},
    out_nactions::Ptr{Int32},
    out_nfixed::Ptr{Int32},
    out_infeas_col::Ptr{Int32},
    out_infeas_lo::Ptr{Float64},
    out_infeas_up::Ptr{Float64},
)::Nothing
    p = unsafe_load(pv)
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    clink = p.clink

    clo = p.clo
    cup = p.cup
    rowels = p.rowels
    hcol = p.hcol
    mrstrt = p.mrstrt
    hinrow = p.hinrow
    rlink = p.rlink
    rlo = p.rlo
    rup = p.rup

    rowstat = p.rowstat
    colstat = p.colstat
    acts = p.acts
    sol = p.sol
    integerType = p.integerType

    ztolzb = p.ztolzb
    feas_tol = p.feasibilityTolerance
    numberLook = Int(p.numberRowsToDo)
    look = p.rowsToDo
    fixInfeasibility = (p.presolveOptions & Int32(0x4000)) != Int32(0)

    fixed_cols = p.usefulColumnInt
    col_changed = p.colChanged
    next_cols = p.nextColsToDo
    n_next_cols = p.numberNextColsToDo
    status = p.status

    nactions = Int32(0)
    nfixed_cols = Int32(0)
    infeas_col = Int32(-1)
    infeas_lo = 0.0
    infeas_up = 0.0

    big_inf_scaled = 1.0e-10 * PRESOLVE_INF_DBL

    iLook = 0
    while iLook < numberLook
        i = unsafe_load(look, iLook + 1)
        i1 = Int(i) + 1
        if unsafe_load(hinrow, i1) != Int32(1)
            iLook += 1
            continue
        end
        krs1 = Int(unsafe_load(mrstrt, i1)) + 1
        j = unsafe_load(hcol, krs1)
        j1 = Int(j) + 1
        aij = unsafe_load(rowels, krs1)
        lo = unsafe_load(rlo, i1)
        up = unsafe_load(rup, i1)
        abs_aij = abs(aij)

        if abs_aij < PRESOLVE_ZTOLDP2
            iLook += 1
            continue
        end
        cj_lo = unsafe_load(clo, j1)
        cj_up = unsafe_load(cup, j1)
        if abs(cj_up - cj_lo) < ztolzb
            iLook += 1
            continue
        end

        nactions += Int32(1)
        unsafe_store!(actions_out,
            SlackActionPod(cj_lo, cj_up, lo, up, aij, j, i),
            Int(nactions))

        if aij < 0.0
            tmp = lo
            lo = -up
            up = -tmp
        end

        if lo <= -big_inf_scaled
            lo = -PRESOLVE_INF_DBL
        else
            lo /= abs_aij
            if lo <= -big_inf_scaled
                lo = -PRESOLVE_INF_DBL
            else
                if lo != 0.0 && abs(lo) <= feas_tol && abs_aij > 1.0
                    nactions -= Int32(1)
                    iLook += 1
                    continue
                end
            end
        end

        if up > big_inf_scaled
            up = PRESOLVE_INF_DBL
        else
            up /= abs_aij
            if up > big_inf_scaled
                up = PRESOLVE_INF_DBL
            else
                if up != 0.0 && abs(up) <= feas_tol && abs_aij > 1.0
                    nactions -= Int32(1)
                    iLook += 1
                    continue
                end
            end
        end

        is_int = integerType != C_NULL && unsafe_load(integerType, j1) != UInt8(0)
        if cj_lo < lo && lo > -1.0e100
            if is_int
                fl = floor(lo + 0.5)
                if abs(lo - fl) < 0.000001
                    lo = fl
                end
                if cj_lo < lo
                    cj_lo = lo
                    unsafe_store!(clo, cj_lo, j1)
                end
            else
                cj_lo = lo
                unsafe_store!(clo, cj_lo, j1)
            end
        end
        if cj_up > up && up < 1.0e100
            if is_int
                fl = floor(up + 0.5)
                if abs(up - fl) < 0.000001
                    up = fl
                end
                if cj_up > up
                    cj_up = up
                    unsafe_store!(cup, cj_up, j1)
                end
            else
                cj_up = up
                unsafe_store!(cup, cj_up, j1)
            end
        end

        if abs(cj_up - cj_lo) < PRESOLVE_ZTOLDP
            nfixed_cols += Int32(1)
            unsafe_store!(fixed_cols, j, Int(nfixed_cols))
        end

        if lo > up
            if lo <= up + feas_tol || fixInfeasibility
                nearest = floor(lo + 0.5)
                if abs(nearest - lo) < 2.0 * feas_tol
                    lo = nearest
                    up = nearest
                else
                    lo = up
                end
                cj_lo = lo
                cj_up = up
                unsafe_store!(clo, cj_lo, j1)
                unsafe_store!(cup, cj_up, j1)
            else
                status |= Int32(1)
                infeas_col = j
                infeas_lo = lo
                infeas_up = up
                break
            end
        end

        unsafe_store!(hinrow, Int32(0), i1)
        ps_remove_link!(rlink, i)
        unsafe_store!(rlo, 0.0, i1)
        unsafe_store!(rup, 0.0, i1)

        ps_delete_from_major!(j, i, mcstrt, hincol, hrow, colels)
        if unsafe_load(hincol, j1) == Int32(0)
            ps_remove_link!(clink, j)
        else
            n_next_cols = ps_add_col!(col_changed, next_cols, n_next_cols, j)
        end

        if rowstat != C_NULL
            basisChoice = Int32(0)
            numberBasic = Int32(0)
            movement = 0.0
            if ps_col_is_basic(colstat, j)
                numberBasic += Int32(1)
                basisChoice = Int32(2)
            end
            if ps_row_is_basic(rowstat, i)
                numberBasic += Int32(1)
            end
            sol_j = unsafe_load(sol, j1)
            if sol_j <= cj_lo + ztolzb
                movement = cj_lo - sol_j
                unsafe_store!(sol, cj_lo, j1)
                ps_set_col_status!(colstat, j, PS_STATUS_AT_LOWER_BOUND)
            elseif sol_j >= cj_up - ztolzb
                movement = cj_up - sol_j
                unsafe_store!(sol, cj_up, j1)
                ps_set_col_status!(colstat, j, PS_STATUS_AT_UPPER_BOUND)
            else
                basisChoice = Int32(1)
            end
            if numberBasic > Int32(1) || basisChoice == Int32(1)
                ps_set_col_status!(colstat, j, PS_STATUS_BASIC)
            elseif basisChoice == Int32(2)
                ps_set_row_status!(rowstat, i, PS_STATUS_BASIC)
            end
            if movement != 0.0
                kcs = Int(unsafe_load(mcstrt, j1))
                kce = kcs + Int(unsafe_load(hincol, j1))
                kcol = kcs
                while kcol < kce
                    kc1 = kcol + 1
                    r1 = Int(unsafe_load(hrow, kc1)) + 1
                    unsafe_store!(acts, _fmadd(movement, unsafe_load(colels, kc1), unsafe_load(acts, r1)), r1)
                    kcol += 1
                end
            end
        end
        iLook += 1
    end

    _ps_store_i32!(pv, _PS_OFF_status, status)
    _ps_store_i32!(pv, _PS_OFF_numberNextColsToDo, n_next_cols)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_nfixed, nfixed_cols)
    unsafe_store!(out_infeas_col, infeas_col)
    unsafe_store!(out_infeas_lo, infeas_lo)
    unsafe_store!(out_infeas_up, infeas_up)
    return nothing
end

function ps_slack_doubleton_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{SlackActionPod},
    nactions::Int32,
)::Nothing
    p = unsafe_load(postv)
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    link = p.link
    clo = p.clo
    cup = p.cup
    sol = p.sol
    rcosts = p.rcosts
    colstat = p.colstat
    rowstat = p.rowstat
    rlo = p.rlo
    rup = p.rup
    acts = p.acts
    rowduals = p.rowduals
    free_list = p.free_list
    ztolzb = p.ztolzb

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        irow = f.row
        ir1 = Int(irow) + 1
        lo0 = f.clo
        up0 = f.cup
        coeff = f.coeff
        jcol = f.col
        j1 = Int(jcol) + 1

        unsafe_store!(rlo, f.rlo, ir1)
        unsafe_store!(rup, f.rup, ir1)
        unsafe_store!(clo, lo0, j1)
        unsafe_store!(cup, up0, j1)

        sol_j = unsafe_load(sol, j1)
        unsafe_store!(acts, coeff * sol_j, ir1)

        k = free_list
        k1 = Int(k) + 1
        free_list = unsafe_load(link, k1)
        unsafe_store!(hrow, irow, k1)
        unsafe_store!(colels, coeff, k1)
        unsafe_store!(link, unsafe_load(mcstrt, j1), k1)
        unsafe_store!(mcstrt, k, j1)
        unsafe_store!(hincol, unsafe_load(hincol, j1) + Int32(1), j1)

        if colstat == C_NULL
            unsafe_store!(rowduals, 0.0, ir1)
        else
            rc_j = unsafe_load(rcosts, j1)
            if ps_col_is_basic(colstat, jcol)
                ps_set_row_status!(rowstat, irow, PS_STATUS_BASIC)
                unsafe_store!(rowduals, 0.0, ir1)
            elseif (abs(sol_j - lo0) <= ztolzb && rc_j >= 0.0) ||
                   (abs(sol_j - up0) <= ztolzb && rc_j <= 0.0)
                ps_set_row_status!(rowstat, irow, PS_STATUS_BASIC)
                unsafe_store!(rowduals, 0.0, ir1)
            elseif !(abs(sol_j - lo0) <= ztolzb) && !(abs(sol_j - up0) <= ztolzb)
                ps_set_col_status!(colstat, jcol, PS_STATUS_BASIC)
                ps_set_row_status_using_value!(rowstat, acts, rlo, rup, ztolzb, irow)
                unsafe_store!(rowduals, rc_j / coeff, ir1)
                unsafe_store!(rcosts, 0.0, j1)
            else
                ps_set_col_status!(colstat, jcol, PS_STATUS_BASIC)
                ps_set_row_status_using_value!(rowstat, acts, rlo, rup, ztolzb, irow)
                unsafe_store!(rowduals, rc_j / coeff, ir1)
                unsafe_store!(rcosts, 0.0, j1)
            end
        end
        idx -= 1
    end

    _post_store_i32!(postv, _POST_OFF_free_list, free_list)
    return nothing
end

function ps_slack_singleton_presolve!(
    pv::Ptr{CoinPresolvePodView},
    rowObjective::Ptr{Float64},
    actions_ptr_ref::Ptr{Ptr{SlackActionPod}},
    maxActions_in::Int32,
    fixed_cols::Ptr{Int32},
    grow_cb::Ptr{Cvoid},
    out_nactions::Ptr{Int32},
    out_nfixed::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    colels = p.colels
    hrow = p.hrow
    mcstrt = p.mcstrt
    hincol = p.hincol
    clink = p.clink

    clo = p.clo
    cup = p.cup
    rowels = p.rowels
    hcol = p.hcol
    mrstrt = p.mrstrt
    hinrow = p.hinrow
    rlink = p.rlink
    rlo = p.rlo
    rup = p.rup

    rowstat = p.rowstat
    colstat = p.colstat
    acts = p.acts
    sol = p.sol
    integerType = p.integerType
    ztolzb = p.ztolzb
    dcost = p.cost
    col_changed = p.colChanged
    row_changed = p.rowChanged
    next_rows = p.nextRowsToDo
    n_next_rows = p.numberNextRowsToDo
    dobias = p.dobias

    numberLook = Int(p.numberColsToDo)
    look = p.colsToDo

    actions = unsafe_load(actions_ptr_ref)
    maxActions = maxActions_in
    nactions = Int32(0)
    nfixed_cols = Int32(0)

    iLook = 0
    while iLook < numberLook
        iCol = unsafe_load(look, iLook + 1)
        ic1 = Int(iCol) + 1
        cost_c = unsafe_load(dcost, ic1)
        if cost_c != 0.0
            iLook += 1
            continue
        end
        if unsafe_load(hincol, ic1) == Int32(1)
            kcs1 = Int(unsafe_load(mcstrt, ic1)) + 1
            iRow = unsafe_load(hrow, kcs1)
            ir1 = Int(iRow) + 1
            coeff = unsafe_load(colels, kcs1)
            acoeff = abs(coeff)
            if acoeff < PRESOLVE_ZTOLDP2
                iLook += 1
                continue
            end
            c_lo = unsafe_load(clo, ic1)
            c_up = unsafe_load(cup, ic1)
            if abs(c_up - c_lo) < ztolzb
                iLook += 1
                continue
            end
            is_int = integerType != C_NULL && unsafe_load(integerType, ic1) != UInt8(0)
            if is_int
                if acoeff != 1.0
                    iLook += 1
                    continue
                end
                allInt = true
                krs = Int(unsafe_load(mrstrt, ir1))
                kre = krs + Int(unsafe_load(hinrow, ir1))
                k = krs
                while k < kre
                    k1 = k + 1
                    col_k1 = Int(unsafe_load(hcol, k1)) + 1
                    val = abs(unsafe_load(rowels, k1))
                    if unsafe_load(integerType, col_k1) == UInt8(0) || val != 1.0
                        allInt = false
                        break
                    end
                    k += 1
                end
                if allInt
                    iLook += 1
                    continue
                end
            end

            if !ps_col_prohibited(col_changed, iCol)
                currentLower = unsafe_load(rlo, ir1)
                currentUpper = unsafe_load(rup, ir1)
                if rowObjective == C_NULL
                    if cost_c != 0.0
                        iLook += 1
                        continue
                    end
                elseif (cost_c != 0.0 && currentLower != currentUpper) || unsafe_load(rowObjective, ir1) != 0.0
                    iLook += 1
                    continue
                end

                newLower = currentLower
                newUpper = currentUpper
                if coeff < 0.0
                    if currentUpper > 1.0e20 || c_up > 1.0e20
                        newUpper = PRESOLVE_INF_DBL
                    else
                        newUpper = _fmsub(coeff, c_up, newUpper)
                        if newUpper > 1.0e20
                            newUpper = PRESOLVE_INF_DBL
                        end
                    end
                    if currentLower < -1.0e20 || c_lo < -1.0e20
                        newLower = -PRESOLVE_INF_DBL
                    else
                        newLower = _fmsub(coeff, c_lo, newLower)
                        if newLower < -1.0e20
                            newLower = -PRESOLVE_INF_DBL
                        end
                    end
                else
                    if currentUpper > 1.0e20 || c_lo < -1.0e20
                        newUpper = PRESOLVE_INF_DBL
                    else
                        newUpper = _fmsub(coeff, c_lo, newUpper)
                        if newUpper > 1.0e20
                            newUpper = PRESOLVE_INF_DBL
                        end
                    end
                    if currentLower < -1.0e20 || c_up > 1.0e20
                        newLower = -PRESOLVE_INF_DBL
                    else
                        newLower = _fmsub(coeff, c_up, newLower)
                        if newLower < -1.0e20
                            newLower = -PRESOLVE_INF_DBL
                        end
                    end
                end

                if is_int
                    if currentLower != floor(currentLower + 0.5)
                        iLook += 1
                        continue
                    end
                    if currentUpper != floor(currentUpper + 0.5)
                        iLook += 1
                        continue
                    end
                    allInt = true
                    krs = Int(unsafe_load(mrstrt, ir1))
                    kre = krs + Int(unsafe_load(hinrow, ir1))
                    k = krs
                    while k < kre
                        k1 = k + 1
                        col_k1 = Int(unsafe_load(hcol, k1)) + 1
                        val = abs(unsafe_load(rowels, k1))
                        if unsafe_load(integerType, col_k1) == UInt8(0) || val != floor(val + 0.5)
                            allInt = false
                            break
                        end
                        k += 1
                    end
                    if !allInt
                        iLook += 1
                        continue
                    end
                end

                if nactions >= maxActions
                    rem = Int32(numberLook - iLook)
                    maxActions += rem < maxActions ? rem : maxActions
                    actions = ccall(grow_cb, Ptr{SlackActionPod},
                                    (Ptr{SlackActionPod}, Int32, Int32),
                                    actions, nactions, maxActions)
                    unsafe_store!(actions_ptr_ref, actions)
                end

                nactions += Int32(1)
                unsafe_store!(actions,
                    SlackActionPod(c_lo, c_up, currentLower, currentUpper, coeff, iCol, iRow),
                    Int(nactions))

                ps_delete_from_major!(iRow, iCol, mrstrt, hinrow, hcol, rowels)
                if unsafe_load(hinrow, ir1) == Int32(0)
                    ps_remove_link!(rlink, iRow)
                end
                n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, iRow)

                unsafe_store!(rlo, newLower, ir1)
                unsafe_store!(rup, newUpper, ir1)

                if rowstat != C_NULL && sol != C_NULL
                    sol_c = unsafe_load(sol, ic1)
                    if (sol_c < c_up - ztolzb && sol_c > c_lo + ztolzb) || ps_col_is_basic(colstat, iCol)
                        ps_set_row_status!(rowstat, iRow, PS_STATUS_BASIC)
                    end
                    ps_set_col_status_using_value!(colstat, sol, clo, cup, ztolzb, iCol)
                end

                unsafe_store!(clo, 0.0, ic1)
                unsafe_store!(cup, 0.0, ic1)

                if rowObjective != C_NULL && cost_c != 0.0
                    ro_val = -cost_c / coeff
                    unsafe_store!(rowObjective, ro_val, ir1)
                    dobias = _fmsub(currentLower, ro_val, dobias)
                end

                if sol != C_NULL
                    sol_c = unsafe_load(sol, ic1)
                    # Note clo[iCol] == 0.0 and cup[iCol] == 0.0 here
                    movement = -sol_c
                    unsafe_store!(sol, 0.0, ic1)
                    if movement != 0.0
                        unsafe_store!(acts, _fmadd(movement, coeff, unsafe_load(acts, ir1)), ir1)
                    end
                end

                ps_delete_from_major!(iCol, iRow, mcstrt, hincol, hrow, colels)
                ps_remove_link!(clink, iCol)
                nfixed_cols += Int32(1)
                unsafe_store!(fixed_cols, iCol, Int(nfixed_cols))
            end
        end
        iLook += 1
    end

    _ps_store_f64!(pv, _PS_OFF_dobias, dobias)
    _ps_store_i32!(pv, _PS_OFF_numberNextRowsToDo, n_next_rows)
    unsafe_store!(out_nactions, nactions)
    unsafe_store!(out_nfixed, nfixed_cols)
    return nothing
end

function ps_slack_singleton_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    actions::Ptr{SlackActionPod},
    nactions::Int32,
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
    sol = p.sol
    rcosts = p.rcosts
    acts = p.acts
    rowduals = p.rowduals
    dcost = p.cost
    colstat = p.colstat
    rowstat = p.rowstat
    free_list = p.free_list
    ztolzb = p.ztolzb

    idx = Int(nactions)
    while idx >= 1
        f = unsafe_load(actions, idx)
        iRow = f.row
        ir1 = Int(iRow) + 1
        lo0 = f.clo
        up0 = f.cup
        coeff = f.coeff
        iCol = f.col
        ic1 = Int(iCol) + 1

        rlo_r = f.rlo
        rup_r = f.rup
        unsafe_store!(rlo, rlo_r, ir1)
        unsafe_store!(rup, rup_r, ir1)
        unsafe_store!(clo, lo0, ic1)
        unsafe_store!(cup, up0, ic1)

        movement = 0.0
        sol_c = unsafe_load(sol, ic1)
        act_r = _fmadd(coeff, sol_c, unsafe_load(acts, ir1))
        dual_r = unsafe_load(rowduals, ir1)
        rc_c = _fmsub(coeff, dual_r, unsafe_load(rcosts, ic1))
        unsafe_store!(rcosts, rc_c, ic1)

        if act_r < rlo_r - ztolzb
            movement = rlo_r - act_r
        elseif act_r > rup_r + ztolzb
            movement = rup_r - act_r
        end
        cMove = movement / coeff
        sol_c += cMove
        act_r += movement
        unsafe_store!(sol, sol_c, ic1)
        unsafe_store!(acts, act_r, ir1)

        if unsafe_load(dcost, ic1) == 0.0
            cMove = 0.0
            if sol_c > up0 + ztolzb
                cMove = up0 - sol_c
            elseif sol_c < lo0 - ztolzb
                cMove = lo0 - sol_c
            end
            sol_c += cMove
            act_r = _fmadd(cMove, coeff, act_r)
            unsafe_store!(sol, sol_c, ic1)
            unsafe_store!(acts, act_r, ir1)

            if colstat != C_NULL
                numberBasic = Int32(0)
                if ps_col_is_basic(colstat, iCol)
                    numberBasic += Int32(1)
                end
                if ps_row_is_basic(rowstat, iRow)
                    numberBasic += Int32(1)
                end
                if sol_c > lo0 + ztolzb && sol_c < up0 - ztolzb
                    ps_set_col_status!(colstat, iCol, PS_STATUS_BASIC)
                    ps_set_row_status_using_value!(rowstat, acts, rlo, rup, ztolzb, iRow)
                elseif act_r > rlo_r + ztolzb && act_r < rup_r - ztolzb
                    ps_set_row_status!(rowstat, iRow, PS_STATUS_BASIC)
                    ps_set_col_status_using_value!(colstat, sol, clo, cup, ztolzb, iCol)
                elseif numberBasic != Int32(0)
                    ps_set_row_status!(rowstat, iRow, PS_STATUS_BASIC)
                    ps_set_col_status_using_value!(colstat, sol, clo, cup, ztolzb, iCol)
                else
                    ps_set_row_status_using_value!(rowstat, acts, rlo, rup, ztolzb, iRow)
                    ps_set_col_status_using_value!(colstat, sol, clo, cup, ztolzb, iCol)
                end
            end
        else
            cost_v = _fmsub(dual_r, coeff, rc_c)
            basic = true
            if abs(sol_c - up0) < ztolzb && cost_v < -1.0e-6
                basic = false
            elseif abs(sol_c - lo0) < ztolzb && cost_v > 1.0e-6
                basic = false
            end
            row_basic = ps_row_is_basic(rowstat, iRow)
            if basic && !row_basic
                basic = false
            end
            if abs(dual_r) > 1.0e-6 && row_basic
                basic = true
            end
            if basic
                unsafe_store!(rowduals, rc_c / coeff, ir1)
                unsafe_store!(rcosts, 0.0, ic1)
            else
                unsafe_store!(rcosts, cost_v, ic1)
            end
            if colstat != C_NULL
                if basic
                    if row_basic
                        ps_set_col_status!(colstat, iCol, PS_STATUS_BASIC)
                    end
                    ps_set_row_status_using_value!(rowstat, acts, rlo, rup, ztolzb, iRow)
                else
                    ps_set_col_status_using_value!(colstat, sol, clo, cup, ztolzb, iCol)
                end
            end
        end

        k = free_list
        k1 = Int(k) + 1
        free_list = unsafe_load(link, k1)
        unsafe_store!(hrow, iRow, k1)
        unsafe_store!(colels, coeff, k1)
        unsafe_store!(link, unsafe_load(mcstrt, ic1), k1)
        unsafe_store!(mcstrt, k, ic1)
        unsafe_store!(hincol, unsafe_load(hincol, ic1) + Int32(1), ic1)

        idx -= 1
    end

    _post_store_i32!(postv, _POST_OFF_free_list, free_list)
    return nothing
end

# =============================================================================
# 4. CoinPresolveDual.cpp (remove_dual_action)
# =============================================================================

function ps_dual_presolve_phase1!(
    pv::Ptr{CoinPresolvePodView},
    cbarmin::Ptr{Float64},
    cbarmax::Ptr{Float64},
    msg_cb::Ptr{Cvoid},
    cb_ctx::Ptr{Cvoid},
    out_nfixup::Ptr{Int32},
    out_nfixdown::Ptr{Int32},
)::Nothing
    p = unsafe_load(pv)
    ncols = Int(p.ncols)
    nrows = Int(p.nrows)
    mcstrt = p.mcstrt
    hincol = p.hincol
    hrow = p.hrow
    colels = p.colels
    cost = p.cost
    integerType = p.integerType
    clo = p.clo
    cup = p.cup
    csol = p.sol
    colstat = p.colstat

    mrstrt = p.mrstrt
    hinrow = p.hinrow
    hcol = p.hcol
    rowels = p.rowels
    rlo = p.rlo
    rup = p.rup

    ekkinf2 = PRESOLVE_SMALL_INF
    ekkinf = ekkinf2 * 1.0e8
    ztolcbarj = p.ztoldj
    any_prohib = p.anyProhibited
    col_changed = p.colChanged
    status = p.status

    ymin = p.usefulRowDouble
    ymax = ymin + nrows * sizeof(Float64)

    active = Ptr{UInt8}(p.usefulColumnInt + ncols * sizeof(Int32))
    j = 1
    while j <= ncols
        unsafe_store!(active, UInt8(0), j)
        j += 1
    end

    numberLook = Int(p.numberColsToDo)
    look = p.colsToDo
    iLook = 0
    while iLook < numberLook
        jc = Int(unsafe_load(look, iLook + 1)) + 1
        cj_up = unsafe_load(cup, jc)
        cj_lo = unsafe_load(clo, jc)
        t = if cj_up >= ekkinf
            cj_lo <= -ekkinf ? UInt8(0) : UInt8(1)
        else
            if cj_lo <= -ekkinf
                UInt8(2)
            elseif cj_lo < cj_up
                UInt8(3)
            else
                UInt8(4)
            end
        end
        unsafe_store!(active, t, jc)
        iLook += 1
    end

    all_sing_opt = (p.presolveOptions & Int32(0x200000)) != Int32(0)

    i = 0
    while i < nrows
        i1 = i + 1
        krs = Int(unsafe_load(mrstrt, i1))
        len_r = Int(unsafe_load(hinrow, i1))
        kre = krs + len_r
        positive = Int32(0)
        negative = Int32(0)
        onlyPositive = -1
        onlyNegative = -1
        allSingletons = all_sing_opt

        k = krs
        while k < kre
            k1 = k + 1
            coeff = unsafe_load(rowels, k1)
            icol1 = Int(unsafe_load(hcol, k1)) + 1
            t = unsafe_load(active, icol1)
            if unsafe_load(hincol, icol1) != Int32(1)
                allSingletons = false
            end
            if t == UInt8(0) || (t & UInt8(8)) != UInt8(0)
                positive = Int32(2)
                negative = Int32(2)
                break
            elseif t == UInt8(1)
                if coeff > 0.0
                    positive += Int32(1)
                else
                    negative += Int32(1)
                end
            elseif t == UInt8(2)
                if coeff < 0.0
                    positive += Int32(1)
                else
                    negative += Int32(1)
                end
            elseif t == UInt8(3)
                if coeff > 0.0
                    onlyPositive = k
                    positive += Int32(1)
                else
                    onlyNegative = k
                    negative += Int32(1)
                end
            end
            k += 1
        end

        doneSomething = false
        if onlyPositive >= 0 && positive == Int32(1)
            coeff = unsafe_load(rowels, onlyPositive + 1)
            icol = unsafe_load(hcol, onlyPositive + 1)
            icol1 = Int(icol) + 1
            lb = unsafe_load(clo, icol1)
            ub = unsafe_load(cup, icol1)
            maxUp = 0.0
            maxDown = 0.0
            k = krs
            while k < kre
                k1 = k + 1
                icol2 = unsafe_load(hcol, k1)
                if icol2 == icol
                    k += 1
                    continue
                end
                ic2 = Int(icol2) + 1
                coeff2 = unsafe_load(rowels, k1)
                lb2 = unsafe_load(clo, ic2)
                ub2 = unsafe_load(cup, ic2)
                if coeff2 > 0.0
                    if ub2 > 1.0e30
                        maxUp = PRESOLVE_INF_DBL
                    else
                        maxUp = _fmadd(coeff2, ub2, maxUp)
                    end
                    if lb2 < -1.0e30
                        maxDown = -PRESOLVE_INF_DBL
                    else
                        maxDown = _fmadd(coeff2, lb2, maxDown)
                    end
                else
                    if lb2 < -1.0e30
                        maxUp = PRESOLVE_INF_DBL
                    else
                        maxUp = _fmadd(coeff2, lb2, maxUp)
                    end
                    if ub2 > 1.0e30
                        maxDown = -PRESOLVE_INF_DBL
                    else
                        maxDown = _fmadd(coeff2, ub2, maxDown)
                    end
                end
                k += 1
            end
            impliedLower = (unsafe_load(rlo, i1) - maxUp) / coeff
            impliedUpper = (unsafe_load(rup, i1) - maxDown) / coeff
            if impliedLower > lb - 1.0e-7
                doneSomething = true
                unsafe_store!(active, UInt8(10), icol1)
            elseif impliedUpper < ub + 1.0e-7
                doneSomething = true
                unsafe_store!(active, UInt8(9), icol1)
            end
        elseif allSingletons && onlyPositive >= 0
            dlo = unsafe_load(rlo, i1)
            possible = true
            nIntegers = Int32(0)
            k = krs
            while k < kre
                k1 = k + 1
                ic1 = Int(unsafe_load(hcol, k1)) + 1
                val = unsafe_load(rowels, k1)
                if unsafe_load(integerType, ic1) != UInt8(0)
                    nIntegers += Int32(1)
                    if val != 1.0
                        possible = false
                    end
                end
                clo_c = unsafe_load(clo, ic1)
                if clo_c != 0.0
                    dlo = _fmsub(clo_c, val, dlo)
                end
                k += 1
            end
            if nIntegers != Int32(0) && Int(nIntegers) < len_r
                possible = false
            end
            if possible
                while dlo > 1.0e-9
                    cheapest = 1.0e50
                    iCheap = -1
                    k = krs
                    while k < kre
                        k1 = k + 1
                        ic1 = Int(unsafe_load(hcol, k1)) + 1
                        if unsafe_load(cup, ic1) > unsafe_load(clo, ic1)
                            val = unsafe_load(rowels, k1)
                            thisCost = unsafe_load(cost, ic1)
                            if thisCost < val * cheapest
                                cheapest = thisCost / val
                                iCheap = k
                            end
                        end
                        k += 1
                    end
                    iCheap < 0 && break
                    val = unsafe_load(rowels, iCheap + 1)
                    ic1 = Int(unsafe_load(hcol, iCheap + 1)) + 1
                    diff2 = dlo / val
                    gap = unsafe_load(cup, ic1) - unsafe_load(clo, ic1)
                    diff = gap < diff2 ? gap : diff2
                    new_lo = unsafe_load(clo, ic1) + diff
                    unsafe_store!(clo, new_lo, ic1)
                    unsafe_store!(cup, new_lo, ic1)
                    dlo = _fmsub(diff, val, dlo)
                end
                k = krs
                while k < kre
                    ic1 = Int(unsafe_load(hcol, k + 1)) + 1
                    clo_c = unsafe_load(clo, ic1)
                    if unsafe_load(cup, ic1) > clo_c
                        unsafe_store!(cup, clo_c, ic1)
                        unsafe_store!(csol, clo_c, ic1)
                        unsafe_store!(colstat, PS_STATUS_AT_LOWER_BOUND, ic1)
                    end
                    k += 1
                end
            end
        end

        if onlyNegative >= 0 && negative == Int32(1) && !doneSomething
            coeff = unsafe_load(rowels, onlyNegative + 1)
            icol = unsafe_load(hcol, onlyNegative + 1)
            icol1 = Int(icol) + 1
            lb = unsafe_load(clo, icol1)
            ub = unsafe_load(cup, icol1)
            maxUp = 0.0
            maxDown = 0.0
            k = krs
            while k < kre
                k1 = k + 1
                icol2 = unsafe_load(hcol, k1)
                if icol2 == icol
                    k += 1
                    continue
                end
                ic2 = Int(icol2) + 1
                coeff2 = unsafe_load(rowels, k1)
                lb2 = unsafe_load(clo, ic2)
                ub2 = unsafe_load(cup, ic2)
                if coeff2 > 0.0
                    if ub2 > 1.0e30
                        maxUp = PRESOLVE_INF_DBL
                    else
                        maxUp = _fmadd(coeff2, ub2, maxUp)
                    end
                    if lb2 < -1.0e30
                        maxDown = -PRESOLVE_INF_DBL
                    else
                        maxDown = _fmadd(coeff2, lb2, maxDown)
                    end
                else
                    if lb2 < -1.0e30
                        maxUp = PRESOLVE_INF_DBL
                    else
                        maxUp = _fmadd(coeff2, lb2, maxUp)
                    end
                    if ub2 > 1.0e30
                        maxDown = -PRESOLVE_INF_DBL
                    else
                        maxDown = _fmadd(coeff2, ub2, maxDown)
                    end
                end
                k += 1
            end
            impliedLower = (unsafe_load(rup, i1) - maxDown) / coeff
            impliedUpper = (unsafe_load(rlo, i1) - maxUp) / coeff
            if impliedLower > lb - 1.0e-7
                unsafe_store!(active, UInt8(10), icol1)
            elseif impliedUpper < ub + 1.0e-7
                unsafe_store!(active, UInt8(9), icol1)
            end
        end
        i += 1
    end

    # Initialise row dual min/max from row bounds
    i = 1
    while i <= nrows
        no_lb = unsafe_load(rup, i) >= ekkinf
        no_ub = unsafe_load(rlo, i) <= -ekkinf
        unsafe_store!(ymin, (no_lb && !no_ub) ? 0.0 : -PRESOLVE_INF_DBL, i)
        unsafe_store!(ymax, (no_ub && !no_lb) ? 0.0 : PRESOLVE_INF_DBL, i)
        i += 1
    end

    # Tighten initial row dual bounds from singleton columns
    j = 1
    while j <= ncols
        if unsafe_load(integerType, j) != UInt8(0)
            j += 1
            continue
        end
        if unsafe_load(hincol, j) != Int32(1)
            j += 1
            continue
        end
        act_j = unsafe_load(active, j)
        no_ub = (act_j & UInt8(6)) == UInt8(0)
        no_lb = (act_j & UInt8(5)) == UInt8(0)
        if no_ub != no_lb
            kcs1 = Int(unsafe_load(mcstrt, j)) + 1
            ir1 = Int(unsafe_load(hrow, kcs1)) + 1
            aij = unsafe_load(colels, kcs1)
            yzero = unsafe_load(cost, j) / aij
            if (aij > 0.0) == no_ub
                if unsafe_load(ymax, ir1) > yzero
                    unsafe_store!(ymax, yzero, ir1)
                end
            else
                if unsafe_load(ymin, ir1) < yzero
                    unsafe_store!(ymin, yzero, ir1)
                end
            end
        end
        j += 1
    end

    nfixup_cols = Int32(0)
    nfixdown_cols = Int32(ncols)
    fix_cols = p.usefulColumnInt

    nPass = 0
    while nPass < 100
        nPass += 1
        tightened = Int32(0)

        j = 0
        while j < ncols
            j1 = j + 1
            len_j = unsafe_load(hincol, j1)
            if len_j <= Int32(0)
                j += 1
                continue
            end
            kcs = Int(unsafe_load(mcstrt, j1))
            kce = kcs + Int(len_j)
            nflagu = Int32(0)
            nflagl = Int32(0)
            nordl = Int32(0)
            cbarjmin = unsafe_load(cost, j1)
            cbarjmax = cbarjmin

            k = kcs
            while k < kce
                k1 = k + 1
                ir1 = Int(unsafe_load(hrow, k1)) + 1
                aij = unsafe_load(colels, k1)
                ymi = unsafe_load(ymin, ir1)
                yma = unsafe_load(ymax, ir1)
                # Note: clang++ emits separate fmul + fsub (unfused) here!
                mindelta = aij * ymi
                maxdelta = aij * yma
                if aij > 0.0
                    if ymi >= -ekkinf2
                        cbarjmax -= mindelta
                    else
                        nflagu += Int32(1)
                    end
                    if yma <= ekkinf2
                        cbarjmin -= maxdelta
                        nordl += Int32(1)
                    else
                        nflagl += Int32(1)
                    end
                else
                    if yma <= ekkinf2
                        cbarjmax -= maxdelta
                    else
                        nflagu += Int32(1)
                    end
                    if ymi >= -ekkinf2
                        cbarjmin -= mindelta
                        nordl += Int32(1)
                    else
                        nflagl += Int32(1)
                    end
                end
                k += 1
            end

            if unsafe_load(integerType, j1) == UInt8(0)
                no_cub = (unsafe_load(active, j1) & UInt8(6)) == UInt8(0)
                if no_cub
                    if nflagu == Int32(1) && cbarjmax < -ztolcbarj
                        k = kcs
                        while k < kce
                            k1 = k + 1
                            ir1 = Int(unsafe_load(hrow, k1)) + 1
                            aij = unsafe_load(colels, k1)
                            if aij > 0.0 && unsafe_load(ymin, ir1) < -ekkinf2
                                yma = unsafe_load(ymax, ir1)
                                if cbarjmax < _fmadd(yma, aij, -ztolcbarj)
                                    newValue = cbarjmax / aij
                                    if yma > ekkinf2 && newValue <= ekkinf2
                                        nflagl -= Int32(1)
                                        cbarjmin = _fmsub(aij, newValue, cbarjmin)
                                    elseif yma <= ekkinf2
                                        cbarjmin = _fmsub(aij, newValue - yma, cbarjmin)
                                    end
                                    unsafe_store!(ymax, newValue, ir1)
                                    tightened += Int32(1)
                                end
                            elseif aij < 0.0 && unsafe_load(ymax, ir1) > ekkinf2
                                ymi = unsafe_load(ymin, ir1)
                                if cbarjmax < _fmadd(ymi, aij, -ztolcbarj)
                                    newValue = cbarjmax / aij
                                    if ymi < -ekkinf2 && newValue >= -ekkinf2
                                        nflagl -= Int32(1)
                                        cbarjmin = _fmsub(aij, newValue, cbarjmin)
                                    elseif ymi >= -ekkinf2
                                        cbarjmin = _fmsub(aij, newValue - ymi, cbarjmin)
                                    end
                                    unsafe_store!(ymin, newValue, ir1)
                                    tightened += Int32(1)
                                end
                            end
                            k += 1
                        end
                    elseif nflagl == Int32(0) && nordl == Int32(1) && cbarjmin < -ztolcbarj
                        # Matches C++ line 759: if (kce != kcs + 1) abort();
                        if kce != kcs + 1
                            ccall(:abort, Cvoid, ())
                        end
                    end
                end
            end

            unsafe_store!(cbarmin, nflagl != Int32(0) ? -PRESOLVE_INF_DBL : cbarjmin, j1)
            unsafe_store!(cbarmax, nflagu != Int32(0) ? PRESOLVE_INF_DBL : cbarjmax, j1)

            j32 = Int32(j)
            if cbarjmin > ztolcbarj && nflagl == Int32(0) && !ps_col_prohibited2(any_prohib, col_changed, j32)
                clo_j = unsafe_load(clo, j1)
                if clo_j <= -ekkinf
                    if msg_cb != C_NULL
                        ccall(msg_cb, Cvoid, (Ptr{Cvoid}, Int32, Int32), cb_ctx, Int32(1), j32)
                    end
                    status |= Int32(2)
                    break
                else
                    nfixdown_cols -= Int32(1)
                    unsafe_store!(fix_cols, j32, Int(nfixdown_cols) + 1)
                    if csol != C_NULL
                        unsafe_store!(csol, clo_j, j1)
                        unsafe_store!(colstat, PS_STATUS_AT_LOWER_BOUND, j1)
                    end
                end
            elseif cbarjmax < -ztolcbarj && nflagu == Int32(0) && !ps_col_prohibited2(any_prohib, col_changed, j32)
                cup_j = unsafe_load(cup, j1)
                if cup_j >= ekkinf
                    if msg_cb != C_NULL
                        ccall(msg_cb, Cvoid, (Ptr{Cvoid}, Int32, Int32), cb_ctx, Int32(2), j32)
                    end
                    status |= Int32(2)
                    break
                else
                    nfixup_cols += Int32(1)
                    unsafe_store!(fix_cols, j32, Int(nfixup_cols))
                    if csol != C_NULL
                        unsafe_store!(csol, cup_j, j1)
                        unsafe_store!(colstat, PS_STATUS_AT_UPPER_BOUND, j1)
                    end
                end
            end
            j += 1
        end

        # PRESOLVE_TIGHTEN_DUALS > 0 (enabled in CoinPresolveDual.cpp line 30)
        i = 0
        while i < nrows
            i1 = i + 1
            no_ub = unsafe_load(rup, i1) >= ekkinf
            no_lb = unsafe_load(rlo, i1) <= -ekkinf
            if no_ub != no_lb
                krs = Int(unsafe_load(mrstrt, i1))
                kre = krs + Int(unsafe_load(hinrow, i1))
                rmax = unsafe_load(ymax, i1)
                rmin = unsafe_load(ymin, i1)
                k = krs
                while k < kre
                    k1 = k + 1
                    coeff = unsafe_load(rowels, k1)
                    ic1 = Int(unsafe_load(hcol, k1)) + 1
                    cbarmax0 = unsafe_load(cbarmax, ic1)
                    cbarmin0 = unsafe_load(cbarmin, ic1)
                    if no_ub
                        if coeff > PRESOLVE_ZTOLDP2 && cbarmax0 < PRESOLVE_INF_DBL && unsafe_load(cup, ic1) >= ekkinf
                            bnd = cbarmax0 / coeff
                            if rmax > bnd
                                rmax = bnd
                                unsafe_store!(ymax, bnd, i1)
                                tightened += Int32(1)
                            end
                        elseif coeff < -PRESOLVE_ZTOLDP2 && cbarmax0 < PRESOLVE_INF_DBL && unsafe_load(cup, ic1) >= ekkinf
                            bnd = cbarmax0 / coeff
                            if rmin < bnd
                                rmin = bnd
                                unsafe_store!(ymin, bnd, i1)
                                tightened += Int32(1)
                            end
                        end
                    else
                        if coeff > PRESOLVE_ZTOLDP2 && cbarmin0 > -PRESOLVE_INF_DBL && unsafe_load(clo, ic1) <= -ekkinf
                            bnd = cbarmin0 / coeff
                            if rmin < bnd
                                rmin = bnd
                                unsafe_store!(ymin, bnd, i1)
                                tightened += Int32(1)
                            end
                        elseif coeff < -PRESOLVE_ZTOLDP2 && cbarmin0 > -PRESOLVE_INF_DBL && unsafe_load(clo, ic1) <= -ekkinf
                            bnd = cbarmin0 / coeff
                            if rmax > bnd
                                rmax = bnd
                                unsafe_store!(ymax, bnd, i1)
                                tightened += Int32(1)
                            end
                        end
                    end
                    k += 1
                end
            end
            i += 1
        end

        if tightened < Int32(100) || Int(nfixdown_cols) < ncols || nfixup_cols != Int32(0)
            break
        end
    end

    _ps_store_i32!(pv, _PS_OFF_status, status)
    unsafe_store!(out_nfixup, nfixup_cols)
    unsafe_store!(out_nfixdown, nfixdown_cols)
    return nothing
end

function ps_dual_presolve_phase2!(pv::Ptr{CoinPresolvePodView})::Int32
    p = unsafe_load(pv)
    ncols = Int(p.ncols)
    nrows = Int(p.nrows)
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

    any_prohib = p.anyProhibited
    col_changed = p.colChanged

    ekkinf = PRESOLVE_SMALL_INF * 1.0e8
    ymin = p.usefulRowDouble
    ymax = ymin + nrows * sizeof(Float64)
    canFix = p.usefulRowInt

    i = 1
    while i <= nrows
        no_rlb = unsafe_load(rlo, i) <= -ekkinf
        no_rub = unsafe_load(rup, i) >= ekkinf
        cf = Int32(0)
        if no_rub && !no_rlb
            cf = unsafe_load(ymin, i) > 0.0 ? Int32(-1) : Int32(-2)
        elseif no_rlb && !no_rub
            cf = unsafe_load(ymax, i) < 0.0 ? Int32(1) : Int32(2)
        end
        unsafe_store!(canFix, cf, i)
        i += 1
    end

    j = 0
    while j < ncols
        j1 = j + 1
        len_j = unsafe_load(hincol, j1)
        if len_j <= Int32(1)
            j += 1
            continue
        end
        if unsafe_load(integerType, j1) != UInt8(0)
            j += 1
            continue
        end
        bindingUp = Int32(-1)
        bindingDown = Int32(-1)
        if unsafe_load(cup, j1) < ekkinf
            bindingUp = Int32(-2)
        end
        if unsafe_load(clo, j1) > -ekkinf
            bindingDown = Int32(-2)
        end
        if bindingUp == Int32(-2) && bindingDown == Int32(-2)
            j += 1
            continue
        end

        kcs = Int(unsafe_load(mcstrt, j1))
        kce = kcs + Int(len_j)
        k = kcs
        while k < kce
            k1 = k + 1
            ir = unsafe_load(hrow, k1)
            ir1 = Int(ir) + 1
            cf = unsafe_load(canFix, ir1)
            if abs(cf) != Int32(2)
                bindingUp = Int32(-2)
                bindingDown = Int32(-2)
                break
            end
            aij = unsafe_load(colels, k1)
            if aij > 0.0
                if cf == Int32(2)
                    bindingUp = bindingUp == Int32(-1) ? ir : Int32(-2)
                else
                    bindingDown = bindingDown == Int32(-1) ? ir : Int32(-2)
                end
            else
                if cf == Int32(2)
                    bindingDown = bindingDown == Int32(-1) ? ir : Int32(-2)
                else
                    bindingUp = bindingUp == Int32(-1) ? ir : Int32(-2)
                end
            end
            k += 1
        end

        if bindingUp == Int32(-2) && bindingDown == Int32(-2)
            j += 1
            continue
        end

        cj = unsafe_load(cost, j1)
        if bindingUp > Int32(-2) && cj <= 0.0
            if bindingUp >= Int32(0)
                bu1 = Int(bindingUp) + 1
                unsafe_store!(canFix, unsafe_load(canFix, bu1) ÷ Int32(2), bu1)
            end
        elseif bindingDown > Int32(-2) && cj >= 0.0
            if bindingDown >= Int32(0)
                bd1 = Int(bindingDown) + 1
                unsafe_store!(canFix, unsafe_load(canFix, bd1) ÷ Int32(2), bd1)
            end
        end
        j += 1
    end

    # Compute row activity min/max (reusing ymin/ymax and usefulRowInt + 2*nrows)
    infCount = canFix + 2 * nrows * sizeof(Int32)
    sentinel_inf = Int32(10) | (Int32(10) << 16)
    i = 0
    while i < nrows
        i1 = i + 1
        rlo_i = unsafe_load(rlo, i1)
        rup_i = unsafe_load(rup, i1)
        if rlo_i == rup_i
            unsafe_store!(infCount, sentinel_inf, i1)
        elseif rlo_i > -1.0e20 || rup_i < 1.0e20
            iflagu = Int32(0)
            iflagl = Int32(0)
            dmaxup = 0.0
            dmaxdown = 0.0
            krs = Int(unsafe_load(mrstrt, i1))
            kre = krs + Int(unsafe_load(hinrow, i1))
            k = krs
            while k < kre
                k1 = k + 1
                val = unsafe_load(rowels, k1)
                jc1 = Int(unsafe_load(hcol, k1)) + 1
                if val > 0.0
                    u = unsafe_load(cup, jc1)
                    if u < 1.0e12
                        dmaxup = _fmadd(u, val, dmaxup)
                    else
                        iflagu += Int32(1)
                    end
                    l = unsafe_load(clo, jc1)
                    if l > -1.0e12
                        dmaxdown = _fmadd(l, val, dmaxdown)
                    else
                        iflagl += Int32(1)
                    end
                elseif val < 0.0
                    u = unsafe_load(cup, jc1)
                    if u < 1.0e12
                        dmaxdown = _fmadd(u, val, dmaxdown)
                    else
                        iflagl += Int32(1)
                    end
                    l = unsafe_load(clo, jc1)
                    if l > -1.0e12
                        dmaxup = _fmadd(l, val, dmaxup)
                    else
                        iflagu += Int32(1)
                    end
                end
                k += 1
            end
            if iflagl > Int32(2)
                iflagl = Int32(2)
            end
            if iflagu > Int32(2)
                iflagu = Int32(2)
            end
            unsafe_store!(infCount, iflagl | (iflagu << 16), i1)
            unsafe_store!(ymax, dmaxup, i1)
            unsafe_store!(ymin, dmaxdown, i1)
        else
            unsafe_store!(infCount, sentinel_inf, i1)
        end
        i += 1
    end

    j = 0
    while j < ncols
        j1 = j + 1
        if unsafe_load(hincol, j1) == Int32(1) && unsafe_load(cost, j1) > -1.0e10
            kcs1 = Int(unsafe_load(mcstrt, j1)) + 1
            coeff = unsafe_load(colels, kcs1)
            irow = unsafe_load(hrow, kcs1)
            ir1 = Int(irow) + 1
            ic_val = unsafe_load(infCount, ir1)
            iflagl = ic_val & Int32(63)
            iflagu = ic_val >> 16
            if iflagu > Int32(1) && iflagl > Int32(1)
                j += 1
                continue
            end
            dmaxdown = unsafe_load(ymin, ir1)
            dmaxup = unsafe_load(ymax, ir1)
            clo_j = unsafe_load(clo, j1)
            cup_j = unsafe_load(cup, j1)

            if coeff > 0.0
                if clo_j > -1.0e12
                    dmaxdown = _fmsub(coeff, clo_j, dmaxdown)
                    if iflagl != Int32(0)
                        dmaxdown = -1.0e60
                    end
                elseif iflagl > Int32(1)
                    dmaxdown = -1.0e60
                end
                if cup_j < 1.0e12
                    dmaxup = _fmsub(coeff, cup_j, dmaxup)
                    if iflagu != Int32(0)
                        dmaxup = 1.0e60
                    end
                elseif iflagu > Int32(1)
                    dmaxup = 1.0e60
                end
            else
                if cup_j < 1.0e12
                    dmaxdown = _fmsub(coeff, cup_j, dmaxdown)
                    if iflagl != Int32(0)
                        dmaxdown = -1.0e60
                    end
                elseif iflagl > Int32(1)
                    dmaxdown = -1.0e60
                end
                if clo_j > -1.0e12
                    dmaxup = _fmsub(coeff, clo_j, dmaxup)
                    if iflagu != Int32(0)
                        dmaxup = 1.0e60
                    end
                elseif iflagu > Int32(1)
                    dmaxup = 1.0e60
                end
            end

            if unsafe_load(cost, j1) > 0.0
                if coeff > 0.0
                    rhs = unsafe_load(rlo, ir1)
                    if rhs > -1.0e12
                        if _fmadd(clo_j, coeff, dmaxdown) < rhs + 1.0e-7 &&
                           _fmadd(cup_j, coeff, dmaxup) > rhs - 1.0e-7
                            unsafe_store!(canFix, Int32(-11), ir1)
                            unsafe_store!(infCount, sentinel_inf, ir1)
                        end
                    end
                else
                    rhs = unsafe_load(rup, ir1)
                    if rhs < 1.0e12
                        if _fmadd(clo_j, coeff, dmaxdown) > rhs - 1.0e-7 &&
                           _fmadd(cup_j, coeff, dmaxup) < rhs + 1.0e-7
                            unsafe_store!(canFix, Int32(1), ir1)
                            unsafe_store!(infCount, sentinel_inf, ir1)
                        end
                    end
                end
            else
                if coeff > 0.0
                    rhs = unsafe_load(rup, ir1)
                    if rhs < 1.0e12
                        if _fmadd(cup_j, coeff, dmaxdown) > rhs - 1.0e-7 &&
                           _fmadd(clo_j, coeff, dmaxup) < rhs + 1.0e-7
                            unsafe_store!(canFix, Int32(1), ir1)
                            unsafe_store!(infCount, sentinel_inf, ir1)
                        end
                    end
                else
                    rhs = unsafe_load(rlo, ir1)
                    if rhs > -1.0e12
                        if _fmadd(cup_j, coeff, dmaxdown) < rhs + 1.0e-7 &&
                           _fmadd(clo_j, coeff, dmaxup) > rhs - 1.0e-7
                            unsafe_store!(canFix, Int32(-11), ir1)
                            unsafe_store!(infCount, sentinel_inf, ir1)
                        end
                    end
                end
            end
        end
        j += 1
    end

    makeEqCnt = nrows
    i = 0
    while i < nrows
        i1 = i + 1
        cf_i = unsafe_load(canFix, i1)
        if abs(cf_i) == Int32(1) && unsafe_load(hinrow, i1) > Int32(1)
            krs = Int(unsafe_load(mrstrt, i1))
            kre = krs + Int(unsafe_load(hinrow, i1))
            nBinary = Int32(0)
            nOther = Int32(0)
            nSameCoeff = Int32(0)
            rhs = cf_i == Int32(1) ? unsafe_load(rup, i1) : unsafe_load(rlo, i1)
            firstCoeff = 0.0
            k = krs
            while k < kre
                k1 = k + 1
                jc = unsafe_load(hcol, k1)
                jc1 = Int(jc) + 1
                coeff = unsafe_load(rowels, k1)
                clo_c = unsafe_load(clo, jc1)
                cup_c = unsafe_load(cup, jc1)
                if cup_c > clo_c
                    if firstCoeff == 0.0
                        firstCoeff = coeff
                        nSameCoeff = Int32(1)
                    else
                        if coeff == firstCoeff
                            nSameCoeff += Int32(1)
                        elseif coeff != -firstCoeff
                            nOther = Int32(1)
                        end
                    end
                    if ps_col_prohibited2(any_prohib, col_changed, jc)
                        nBinary = Int32(1)
                        nOther = Int32(1)
                        break
                    elseif unsafe_load(integerType, jc1) != UInt8(0)
                        if cup_c - clo_c == 1.0
                            nBinary += Int32(1)
                        else
                            nBinary = Int32(1)
                            nOther = Int32(1)
                            break
                        end
                    else
                        nOther += Int32(1)
                    end
                else
                    rhs = _fmsub(coeff, clo_c, rhs)
                end
                k += 1
            end

            canFixThis = true
            if nBinary != Int32(0)
                if nOther != Int32(0)
                    canFixThis = false
                else
                    if rhs == 0.0
                        if !(nSameCoeff == Int32(1) && nBinary == Int32(2))
                            canFixThis = false
                        end
                    elseif !(rhs == 1.0 && cf_i == Int32(1) && nSameCoeff == nBinary && firstCoeff == 1.0)
                        canFixThis = false
                    end
                end
            end

            if canFixThis
                makeEqCnt += 1
                unsafe_store!(canFix, Int32(i), makeEqCnt)
            else
                nInteger = Int32(0)
                nOther2 = Int32(0)
                nSameCoeff2 = Int32(0)
                nSingletons = Int32(0)
                nMixed = Int32(0)
                rhs2 = cf_i == Int32(1) ? unsafe_load(rup, i1) : unsafe_load(rlo, i1)
                firstCoeff2 = 0.0
                k = krs
                while k < kre
                    k1 = k + 1
                    jc = unsafe_load(hcol, k1)
                    jc1 = Int(jc) + 1
                    if unsafe_load(hincol, jc1) == Int32(1)
                        nSingletons += Int32(1)
                    end
                    coeff = abs(unsafe_load(rowels, k1))
                    clo_c = unsafe_load(clo, jc1)
                    cup_c = unsafe_load(cup, jc1)
                    if cup_c > clo_c
                        if firstCoeff2 == 0.0
                            firstCoeff2 = coeff
                            nSameCoeff2 = Int32(1)
                        else
                            if coeff == firstCoeff2
                                nSameCoeff2 += Int32(1)
                            elseif coeff != -firstCoeff2
                                nOther2 = Int32(1)
                            end
                        end
                        if ps_col_prohibited2(any_prohib, col_changed, jc)
                            nInteger = Int32(1)
                            nOther2 = Int32(1)
                            nMixed = Int32(3)
                            break
                        elseif unsafe_load(integerType, jc1) != UInt8(0)
                            nInteger += Int32(1)
                            if clo_c == 0.0 && cup_c == 1.0
                                if nMixed == Int32(0)
                                    nMixed = Int32(2)
                                elseif nMixed != Int32(2)
                                    nMixed = Int32(3)
                                end
                            else
                                nMixed = Int32(3)
                            end
                        else
                            nOther2 += Int32(1)
                            if nMixed == Int32(0)
                                nMixed = Int32(1)
                            elseif nMixed != Int32(1)
                                nMixed = Int32(3)
                            end
                        end
                    else
                        rhs2 = _fmsub(coeff, clo_c, rhs2)
                    end
                    k += 1
                end
                if nSingletons > Int32(1)
                    if nMixed < Int32(3)
                        nOther2 = Int32(0)
                    end
                end
                if nOther2 == Int32(0)
                    if nSameCoeff2 == nInteger
                        rhs2 /= firstCoeff2
                        if abs(rhs2 - floor(rhs2 + 0.5)) < 1.0e-9
                            makeEqCnt += 1
                            unsafe_store!(canFix, Int32(i), makeEqCnt)
                        end
                    end
                end
            end
        end
        i += 1
    end

    return Int32(makeEqCnt - nrows)
end

function ps_dual_presolve_apply!(
    pv::Ptr{CoinPresolvePodView},
    makeEqCnt::Int32,
    bndRecords::Ptr{DualActionPod},
)::Nothing
    p = unsafe_load(pv)
    nrows = Int(p.nrows)
    rlo = p.rlo
    rup = p.rup
    canFix = p.usefulRowInt
    row_changed = p.rowChanged
    next_rows = p.nextRowsToDo
    n_next_rows = p.numberNextRowsToDo

    cnt = Int(makeEqCnt)
    k = 1
    while k <= cnt
        i = unsafe_load(canFix, k + nrows)
        i1 = Int(i) + 1
        rlo_i = unsafe_load(rlo, i1)
        rup_i = unsafe_load(rup, i1)
        unsafe_store!(bndRecords, DualActionPod(rlo_i, rup_i, i, Int32(0)), k)
        cf_i = unsafe_load(canFix, i1)
        if cf_i == Int32(1)
            unsafe_store!(rlo, rup_i, i1)
            n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, i)
        elseif cf_i == Int32(-1)
            unsafe_store!(rup, rlo_i, i1)
            n_next_rows = ps_add_row!(row_changed, next_rows, n_next_rows, i)
        end
        k += 1
    end
    _ps_store_i32!(pv, _PS_OFF_numberNextRowsToDo, n_next_rows)
    return nothing
end

function ps_dual_postsolve!(
    postv::Ptr{CoinPostsolvePodView},
    bndRecords::Ptr{DualActionPod},
    numRecs::Int32,
)::Nothing
    p = unsafe_load(postv)
    rlo = p.rlo
    rup = p.rup
    rowstat = p.rowstat

    n = Int(numRecs)
    k = 1
    while k <= n
        rec = unsafe_load(bndRecords, k)
        i = rec.ndx
        i1 = Int(i) + 1
        rloi = rec.rlo
        rupi = rec.rup
        unsafe_store!(rlo, rloi, i1)
        unsafe_store!(rup, rupi, i1)
        if rowstat != C_NULL
            stati = ps_get_row_status(rowstat, i)
            if stati == PS_STATUS_AT_LOWER_BOUND
                if rloi <= -PRESOLVE_INF_DBL
                    unsafe_store!(rowstat, PS_STATUS_AT_UPPER_BOUND, i1)
                end
            elseif stati == PS_STATUS_AT_UPPER_BOUND
                if rupi >= PRESOLVE_INF_DBL
                    unsafe_store!(rowstat, PS_STATUS_AT_LOWER_BOUND, i1)
                end
            end
        end
        k += 1
    end
    return nothing
end

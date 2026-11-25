# Pure idiomatic 1-based Julia LP Presolve, Postsolve, and initial_solve! (Phase B — B13).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, --trim=safe clean.
# Implements reversible LP presolve passes (fixed/empty cols, empty rows, row singletons,
# forcing/redundant rows, bound tightening, free/implied-free col singletons,
# zero-cost slack singletons, doubleton equalities, multi-row implied-free substitution)
# with exact primal/dual postsolve and standalone ClpSimplexModel integration.

const _PS_INF = 1.0e20
const _PS_ACT_FIXED_COL       = Int8(1)  # col j fixed at val
const _PS_ACT_EMPTY_ROW       = Int8(2)  # row i removed (empty or redundant)
const _PS_ACT_DOUBLETON       = Int8(3)  # eq row i: a1*x1 + a2*x2 = rhs -> x1 = (rhs - a2*x2)/a1
const _PS_ACT_COL_SINGLETON   = Int8(4)  # eq row i, col j singleton: a*x_j + sum(row_i) = rhs
const _PS_ACT_SLACK_SINGLETON = Int8(5)  # ineq row i, zero-cost col j singleton: rlo <= a*x_j + sum(row_i) <= rhi
const _PS_ACT_ROW_SINGLETON   = Int8(6)  # row singleton: rlo <= a*x_j <= rhi
const _PS_ACT_SUBST           = Int8(7)  # multi-row implied-free col j (col_len == 2): eq r_piv -> r_other
const _PS_ACT_FORCING_ROW     = Int8(8)  # forcing row i forces cols to bounds; dual y_i fixes worst dual infeas

struct _PresolveAction
    kind::Int8
    row::Int32
    col1::Int32
    col2::Int32
    val1::Float64     # fixed_val / rhs / slack a / row_sing a / subst rhs_piv
    val2::Float64     # fixed c_min / a1 / coeff / slack col_lo / row_sing old_lo / subst a_piv
    val3::Float64     # a2 / c_orig / slack col_hi / row_sing old_hi / subst a_other
    val4::Float64     # doubleton c1_min / slack old_rlo / row_sing rlo / subst c_min
    val5::Float64     # doubleton tight_lo2 / slack old_rhi / row_sing rhi
    val6::Float64     # doubleton tight_hi2
    aux_start::Int32  # slice into aux arrays for FIXED_COL / DOUBLETON / COL_SINGLETON / SLACK_SINGLETON / SUBST
    aux_len::Int32
end

struct _PresolvedProblem
    orig_nrows::Int
    orig_ncols::Int
    red_nrows::Int
    red_ncols::Int
    obj_sense::Float64
    obj_offset_delta::Float64
    col_map::Vector{Int32}       # red_col -> orig_col
    row_map::Vector{Int32}       # red_row -> orig_row
    col_active::Vector{Bool}
    row_active::Vector{Bool}
    actions::Vector{_PresolveAction}
    aux_cols::Vector{Int32}
    aux_vals::Vector{Float64}
    status::Int                  # 0 = ok, 1 = primal infeasible, 2 = unbounded
    reduced::ClpSimplexModel
end

# Dynamic row/column incidence lists for fast O(nnz) presolve reductions
mutable struct _PresolveWork
    nrows::Int
    ncols::Int
    obj_sense::Float64
    obj_offset_delta::Float64
    col_lo::Vector{Float64}
    col_hi::Vector{Float64}
    cost::Vector{Float64}        # minimization cost = obj_sense * objective
    row_lo::Vector{Float64}
    row_hi::Vector{Float64}
    col_active::Vector{Bool}
    row_active::Vector{Bool}
    # Dynamic CSC & CSR (stored as Vector{Vector{...}} for clean O(1) swap-delete)
    col_rows::Vector{Vector{Int32}}
    col_vals::Vector{Vector{Float64}}
    row_cols::Vector{Vector{Int32}}
    row_vals::Vector{Vector{Float64}}
    actions::Vector{_PresolveAction}
    aux_cols::Vector{Int32}
    aux_vals::Vector{Float64}
    tol::Float64
end

function _pw_init(m::ClpSimplexModel):: _PresolveWork
    nrows = m.nrows
    ncols = m.ncols
    col_rows = [Int32[] for _ in 1:ncols]
    col_vals = [Float64[] for _ in 1:ncols]
    row_cols = [Int32[] for _ in 1:nrows]
    row_vals = [Float64[] for _ in 1:nrows]

    starts = m.mat_starts
    indices = m.mat_indices
    elems = m.mat_elements

    row_counts = zeros(Int, nrows)
    @inbounds for j in 1:ncols
        s = Int(starts[j])
        e = Int(starts[j + 1]) - 1
        for k in s:e
            elems[k] == 0.0 && continue
            row_counts[Int(indices[k])] += 1
        end
    end
    @inbounds for ir in 1:nrows
        cnt = row_counts[ir]
        if cnt > 0
            sizehint!(row_cols[ir], cnt)
            sizehint!(row_vals[ir], cnt)
        end
    end

    @inbounds for j in 1:ncols
        s = Int(starts[j])
        e = Int(starts[j + 1]) - 1
        len = e - s + 1
        if len > 0
            sizehint!(col_rows[j], len)
            sizehint!(col_vals[j], len)
        end
        for k in s:e
            v = elems[k]
            v == 0.0 && continue
            ir = Int(indices[k])
            push!(col_rows[j], Int32(ir))
            push!(col_vals[j], v)
            push!(row_cols[ir], Int32(j))
            push!(row_vals[ir], v)
        end
    end

    cost = Vector{Float64}(undef, ncols)
    @inbounds for j in 1:ncols
        cost[j] = m.objective[j] * m.obj_sense
    end

    return _PresolveWork(
        nrows, ncols, m.obj_sense, 0.0,
        copy(m.col_lower), copy(m.col_upper), cost,
        copy(m.row_lower), copy(m.row_upper),
        trues(ncols), trues(nrows),
        col_rows, col_vals, row_cols, row_vals,
        _PresolveAction[], Int32[], Float64[],
        1.0e-7,
    )
end

@inline function _pw_remove_from_row!(pw::_PresolveWork, i::Int, j::Int)::Nothing
    rc = pw.row_cols[i]
    rv = pw.row_vals[i]
    @inbounds for k in 1:length(rc)
        if Int(rc[k]) == j
            rc[k] = rc[end]
            rv[k] = rv[end]
            pop!(rc)
            pop!(rv)
            return nothing
        end
    end
    return nothing
end

@inline function _pw_remove_from_col!(pw::_PresolveWork, j::Int, i::Int)::Nothing
    cr = pw.col_rows[j]
    cv = pw.col_vals[j]
    @inbounds for k in 1:length(cr)
        if Int(cr[k]) == i
            cr[k] = cr[end]
            cv[k] = cv[end]
            pop!(cr)
            pop!(cv)
            return nothing
        end
    end
    return nothing
end

@inline function _pw_add_entry!(pw::_PresolveWork, i::Int, j::Int, delta::Float64)::Nothing
    delta == 0.0 && return nothing
    rc = pw.row_cols[i]
    rv = pw.row_vals[i]
    @inbounds for k in 1:length(rc)
        if Int(rc[k]) == j
            nv = rv[k] + delta
            if abs(nv) <= 1.0e-13
                rc[k] = rc[end]; rv[k] = rv[end]
                pop!(rc); pop!(rv)
                _pw_remove_from_col!(pw, j, i)
            else
                rv[k] = nv
                cr = pw.col_rows[j]
                cv = pw.col_vals[j]
                for p in 1:length(cr)
                    if Int(cr[p]) == i
                        cv[p] = nv
                        break
                    end
                end
            end
            return nothing
        end
    end
    if abs(delta) > 1.0e-13
        push!(rc, Int32(j)); push!(rv, delta)
        push!(pw.col_rows[j], Int32(i)); push!(pw.col_vals[j], delta)
    end
    return nothing
end

function _pw_fix_col!(pw::_PresolveWork, j::Int, val::Float64)::Nothing
    pw.col_active[j] || return nothing
    pw.col_active[j] = false
    pw.obj_offset_delta += pw.cost[j] * val
    cr = pw.col_rows[j]
    cv = pw.col_vals[j]
    aux_start = length(pw.aux_cols) + 1
    @inbounds for k in 1:length(cr)
        i = Int(cr[k])
        if pw.row_active[i]
            v = cv[k]
            push!(pw.aux_cols, Int32(i))
            push!(pw.aux_vals, v)
            shift = v * val
            if pw.row_lo[i] > -_PS_INF
                pw.row_lo[i] -= shift
            end
            if pw.row_hi[i] < _PS_INF
                pw.row_hi[i] -= shift
            end
            _pw_remove_from_row!(pw, i, j)
        end
    end
    aux_len = length(pw.aux_cols) - aux_start + 1
    Base.empty!(cr)
    Base.empty!(cv)
    push!(pw.actions, _PresolveAction(
        _PS_ACT_FIXED_COL, Int32(0), Int32(j), Int32(0),
        val, pw.cost[j], 0.0, 0.0, 0.0, 0.0, Int32(aux_start), Int32(aux_len),
    ))
    return nothing
end

function _pw_remove_row!(pw::_PresolveWork, i::Int; record::Bool = true)::Nothing
    pw.row_active[i] || return nothing
    pw.row_active[i] = false
    rc = pw.row_cols[i]
    rv = pw.row_vals[i]
    @inbounds for k in 1:length(rc)
        j = Int(rc[k])
        if pw.col_active[j]
            _pw_remove_from_col!(pw, j, i)
        end
    end
    Base.empty!(rc)
    Base.empty!(rv)
    if record
        push!(pw.actions, _PresolveAction(
            _PS_ACT_EMPTY_ROW, Int32(i), Int32(0), Int32(0),
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, Int32(0), Int32(0),
        ))
    end
    return nothing
end

# Pass 1: Fixed & empty columns
function _pw_pass_fixed_and_empty_cols!(pw::_PresolveWork)::Tuple{Int,Int}
    n_red = 0
    tol = pw.tol
    @inbounds for j in 1:pw.ncols
        pw.col_active[j] || continue
        lo = pw.col_lo[j]
        hi = pw.col_hi[j]
        if lo > hi + tol
            return (n_red, CLP_STATUS_PRIMAL_INFEAS)
        end
        if hi - lo <= tol
            _pw_fix_col!(pw, j, 0.5 * (lo + hi))
            n_red += 1
        elseif isempty(pw.col_rows[j])
            c = pw.cost[j]
            if c > tol
                if lo <= -_PS_INF
                    return (n_red, CLP_STATUS_DUAL_INFEAS)
                end
                _pw_fix_col!(pw, j, lo)
                n_red += 1
            elseif c < -tol
                if hi >= _PS_INF
                    return (n_red, CLP_STATUS_DUAL_INFEAS)
                end
                _pw_fix_col!(pw, j, hi)
                n_red += 1
            else
                v = lo > -_PS_INF ? lo : (hi < _PS_INF ? hi : 0.0)
                _pw_fix_col!(pw, j, v)
                n_red += 1
            end
        end
    end
    return (n_red, 0)
end

# Pass 2: Empty rows & row singletons
function _pw_pass_rows_and_singletons!(pw::_PresolveWork)::Tuple{Int,Int}
    n_red = 0
    tol = pw.tol
    @inbounds for i in 1:pw.nrows
        pw.row_active[i] || continue
        rc = pw.row_cols[i]
        rv = pw.row_vals[i]
        rlo = pw.row_lo[i]
        rhi = pw.row_hi[i]
        if rlo > rhi + tol
            return (n_red, CLP_STATUS_PRIMAL_INFEAS)
        end
        len = length(rc)
        if len == 0
            if rlo > tol || rhi < -tol
                return (n_red, CLP_STATUS_PRIMAL_INFEAS)
            end
            _pw_remove_row!(pw, i)
            n_red += 1
        elseif len == 1
            j = Int(rc[1])
            a = rv[1]
            if abs(a) <= 1.0e-12
                continue
            end
            old_lo = pw.col_lo[j]
            old_hi = pw.col_hi[j]
            if a > 0.0
                new_lo = rlo > -_PS_INF ? (rlo / a) : -floatmax(Float64)
                new_hi = rhi < _PS_INF ? (rhi / a) : floatmax(Float64)
            else
                new_lo = rhi < _PS_INF ? (rhi / a) : -floatmax(Float64)
                new_hi = rlo > -_PS_INF ? (rlo / a) : floatmax(Float64)
            end
            if new_lo > pw.col_lo[j]
                pw.col_lo[j] = new_lo
            end
            if new_hi < pw.col_hi[j]
                pw.col_hi[j] = new_hi
            end
            if pw.col_lo[j] > pw.col_hi[j] + tol * max(1.0, abs(pw.col_lo[j]))
                return (n_red, CLP_STATUS_PRIMAL_INFEAS)
            end
            _pw_remove_row!(pw, i; record = false)
            push!(pw.actions, _PresolveAction(
                _PS_ACT_ROW_SINGLETON, Int32(i), Int32(j), Int32(0),
                a, old_lo, old_hi, rlo, rhi, 0.0, Int32(0), Int32(0),
            ))
            n_red += 1
            if pw.col_hi[j] - pw.col_lo[j] <= tol
                _pw_fix_col!(pw, j, 0.5 * (pw.col_lo[j] + pw.col_hi[j]))
                n_red += 1
            end
        end
    end
    return (n_red, 0)
end

# Pass 3: Forcing & redundant rows (row activity bounds)
function _pw_pass_forcing_and_redundant!(pw::_PresolveWork)::Tuple{Int,Int}
    n_red = 0
    tol = pw.tol
    @inbounds for i in 1:pw.nrows
        pw.row_active[i] || continue
        rc = pw.row_cols[i]
        rv = pw.row_vals[i]
        isempty(rc) && continue
        rlo = pw.row_lo[i]
        rhi = pw.row_hi[i]
        if rlo <= -_PS_INF && rhi >= _PS_INF
            _pw_remove_row!(pw, i)
            n_red += 1
            continue
        end

        min_act = 0.0
        max_act = 0.0
        n_inf_min = 0
        n_inf_max = 0
        for k in 1:length(rc)
            j = Int(rc[k])
            a = rv[k]
            lo = pw.col_lo[j]
            hi = pw.col_hi[j]
            if a > 0.0
                if lo <= -_PS_INF; n_inf_min += 1; else; min_act += a * lo; end
                if hi >= _PS_INF;  n_inf_max += 1; else; max_act += a * hi; end
            else
                if hi >= _PS_INF;  n_inf_min += 1; else; min_act += a * hi; end
                if lo <= -_PS_INF; n_inf_max += 1; else; max_act += a * lo; end
            end
        end

        # Infeasibility check
        if n_inf_min == 0 && rhi < _PS_INF && min_act > rhi + tol * (1.0 + abs(rhi))
            return (n_red, CLP_STATUS_PRIMAL_INFEAS)
        end
        if n_inf_max == 0 && rlo > -_PS_INF && max_act < rlo - tol * (1.0 + abs(rlo))
            return (n_red, CLP_STATUS_PRIMAL_INFEAS)
        end

        # Redundant row check
        lo_ok = (rlo <= -_PS_INF) || (n_inf_min == 0 && min_act >= rlo - tol)
        hi_ok = (rhi >= _PS_INF)  || (n_inf_max == 0 && max_act <= rhi + tol)
        if lo_ok && hi_ok
            _pw_remove_row!(pw, i)
            n_red += 1
            continue
        end

        # Forcing row at upper bound (min_act == rhi) or lower bound (max_act == rlo)
        tight_at_rhi = (n_inf_min == 0 && rhi < _PS_INF && abs(min_act - rhi) <= 1.0e-10 * (1.0 + abs(rhi)))
        tight_at_rlo = !tight_at_rhi && (n_inf_max == 0 && rlo > -_PS_INF && abs(max_act - rlo) <= 1.0e-10 * (1.0 + abs(rlo)))
        if tight_at_rhi || tight_at_rlo
            aux_start = length(pw.aux_cols) + 1
            for k in 1:length(rc)
                j = Int(rc[k])
                pw.col_active[j] || continue
                push!(pw.aux_cols, Int32(j))
                push!(pw.aux_vals, rv[k])
            end
            aux_len = length(pw.aux_cols) - aux_start + 1
            cols_copy = copy(rc)
            vals_copy = copy(rv)
            _pw_remove_row!(pw, i; record = false)
            push!(pw.actions, _PresolveAction(
                _PS_ACT_FORCING_ROW, Int32(i), Int32(0), Int32(0),
                tight_at_rlo ? 1.0 : -1.0, rlo, rhi, 0.0, 0.0, 0.0,
                Int32(aux_start), Int32(aux_len),
            ))
            n_red += 1
            for k in 1:length(cols_copy)
                j = Int(cols_copy[k])
                pw.col_active[j] || continue
                fval = (tight_at_rlo == (vals_copy[k] > 0.0)) ? pw.col_hi[j] : pw.col_lo[j]
                _pw_fix_col!(pw, j, fval)
                n_red += 1
            end
            continue
        end
    end
    return (n_red, 0)
end

@inline function _pw_apply_implied_bound!(
    pw::_PresolveWork,
    j::Int,
    a::Float64,
    rhs::Float64,
    rest::Float64,
    abs_sum::Float64,
    is_upper_rhs::Bool,
)::Int
    new_b = (rhs - rest) / a
    abs(new_b) < 1.0e12 || return 0
    eps_c = 1.0e-13 * (abs(rhs) + abs_sum) / abs(a)
    if is_upper_rhs == (a > 0.0)
        new_hi = eps_c > 1.0e-8 ? new_b + eps_c : new_b
        lo = pw.col_lo[j]
        if new_hi < lo && lo - new_hi <= eps_c
            new_hi = lo
        end
        if pw.col_hi[j] - new_hi > max(1.0e-9, 1.0e-9 * abs(new_hi))
            pw.col_hi[j] = new_hi
            return 1
        end
    else
        new_lo = eps_c > 1.0e-8 ? new_b - eps_c : new_b
        hi = pw.col_hi[j]
        if new_lo > hi && new_lo - hi <= eps_c
            new_lo = hi
        end
        if new_lo - pw.col_lo[j] > max(1.0e-9, 1.0e-9 * abs(new_lo))
            pw.col_lo[j] = new_lo
            return 1
        end
    end
    return 0
end

# Pass 3b: Primal bound tightening from row activity bounds
function _pw_pass_bound_tightening!(pw::_PresolveWork)::Tuple{Int,Int}
    n_red = 0
    tol = pw.tol
    to_fix = Int32[]
    @inbounds for i in 1:pw.nrows
        pw.row_active[i] || continue
        rc = pw.row_cols[i]
        rv = pw.row_vals[i]
        isempty(rc) && continue
        rlo = pw.row_lo[i]
        rhi = pw.row_hi[i]
        (rlo <= -_PS_INF && rhi >= _PS_INF) && continue

        min_act = 0.0
        max_act = 0.0
        abs_min_sum = 0.0
        abs_max_sum = 0.0
        n_inf_min = 0
        n_inf_max = 0
        inf_min_k = 0
        inf_max_k = 0
        for k in 1:length(rc)
            j = Int(rc[k])
            a = rv[k]
            lo = pw.col_lo[j]
            hi = pw.col_hi[j]
            if a > 0.0
                if lo <= -_PS_INF
                    n_inf_min += 1
                    inf_min_k = k
                else
                    term = a * lo
                    min_act += term
                    abs_min_sum += abs(term)
                end
                if hi >= _PS_INF
                    n_inf_max += 1
                    inf_max_k = k
                else
                    term = a * hi
                    max_act += term
                    abs_max_sum += abs(term)
                end
            elseif a < 0.0
                if hi >= _PS_INF
                    n_inf_min += 1
                    inf_min_k = k
                else
                    term = a * hi
                    min_act += term
                    abs_min_sum += abs(term)
                end
                if lo <= -_PS_INF
                    n_inf_max += 1
                    inf_max_k = k
                else
                    term = a * lo
                    max_act += term
                    abs_max_sum += abs(term)
                end
            end
        end

        do_hi = (rhi < _PS_INF) && (n_inf_min <= 1)
        do_lo = (rlo > -_PS_INF) && (n_inf_max <= 1)
        (!do_hi && !do_lo) && continue

        Base.empty!(to_fix)
        for k in 1:length(rc)
            j = Int(rc[k])
            a = rv[k]
            abs(a) >= 1.0e-6 || continue
            lo = pw.col_lo[j]
            hi = pw.col_hi[j]

            if do_hi && (n_inf_min == 0 || k == inf_min_k)
                if n_inf_min == 0
                    contrib = a > 0.0 ? (a * lo) : (a * hi)
                    rest_min = min_act - contrib
                else
                    rest_min = min_act
                end
                n_red += _pw_apply_implied_bound!(pw, j, a, rhi, rest_min, abs_min_sum, true)
            end

            if do_lo && (n_inf_max == 0 || k == inf_max_k)
                if n_inf_max == 0
                    contrib = a > 0.0 ? (a * hi) : (a * lo)
                    rest_max = max_act - contrib
                else
                    rest_max = max_act
                end
                n_red += _pw_apply_implied_bound!(pw, j, a, rlo, rest_max, abs_max_sum, false)
            end

            cur_lo = pw.col_lo[j]
            cur_hi = pw.col_hi[j]
            if cur_lo > cur_hi + tol * max(1.0, abs(cur_lo), abs(cur_hi))
                return (n_red, CLP_STATUS_PRIMAL_INFEAS)
            end
            if cur_hi - cur_lo <= tol
                push!(to_fix, Int32(j))
            end
        end

        for kf in 1:length(to_fix)
            jf = Int(to_fix[kf])
            pw.col_active[jf] || continue
            cur_lo = pw.col_lo[jf]
            cur_hi = pw.col_hi[jf]
            if cur_hi - cur_lo <= tol
                _pw_fix_col!(pw, jf, 0.5 * (cur_lo + cur_hi))
                n_red += 1
            end
        end
    end
    return (n_red, 0)
end

@inline function _pw_is_implied_free(
    pw::_PresolveWork,
    i::Int,
    j::Int,
    a::Float64,
    rhs::Float64,
)::Bool
    tol = pw.tol
    lo_j = pw.col_lo[j]
    hi_j = pw.col_hi[j]
    (lo_j <= -_PS_INF && hi_j >= _PS_INF) && return true

    # Track lower and upper implied bounds independently (one-sided bounds suffice)
    rc = pw.row_cols[i]
    rv = pw.row_vals[i]
    sum_min = 0.0
    sum_max = 0.0
    min_valid = true
    max_valid = true
    need_min = a > 0.0 ? (hi_j < _PS_INF) : (lo_j > -_PS_INF)
    need_max = a > 0.0 ? (lo_j > -_PS_INF) : (hi_j < _PS_INF)
    @inbounds for k in 1:length(rc)
        c2 = Int(rc[k])
        c2 == j && continue
        v2 = rv[k]
        v2 == 0.0 && continue
        l2 = pw.col_lo[c2]
        u2 = pw.col_hi[c2]
        if v2 > 0.0
            if l2 <= -_PS_INF
                min_valid = false
            elseif min_valid
                sum_min += v2 * l2
            end
            if u2 >= _PS_INF
                max_valid = false
            elseif max_valid
                sum_max += v2 * u2
            end
        else
            if u2 >= _PS_INF
                min_valid = false
            elseif min_valid
                sum_min += v2 * u2
            end
            if l2 <= -_PS_INF
                max_valid = false
            elseif max_valid
                sum_max += v2 * l2
            end
        end
        ((need_min && !min_valid) || (need_max && !max_valid)) && break
    end
    if a > 0.0
        lo_valid = max_valid
        hi_valid = min_valid
        imp_lo = lo_valid ? (rhs - sum_max) / a : -_PS_INF
        imp_hi = hi_valid ? (rhs - sum_min) / a : _PS_INF
    else
        lo_valid = min_valid
        hi_valid = max_valid
        imp_lo = lo_valid ? (rhs - sum_min) / a : -_PS_INF
        imp_hi = hi_valid ? (rhs - sum_max) / a : _PS_INF
    end
    lo_ok = (lo_j <= -_PS_INF) || (lo_valid && imp_lo >= lo_j - tol)
    hi_ok = (hi_j >= _PS_INF)  || (hi_valid && imp_hi <= hi_j + tol)
    return lo_ok && hi_ok
end

# Pass 4: Free / implied-free column singletons in equality rows
function _pw_pass_col_singletons!(pw::_PresolveWork)::Tuple{Int,Int}
    n_red = 0
    tol = pw.tol
    @inbounds for j in 1:pw.ncols
        pw.col_active[j] || continue
        length(pw.col_rows[j]) == 1 || continue
        i = Int(pw.col_rows[j][1])
        pw.row_active[i] || continue
        rlo = pw.row_lo[i]
        rhi = pw.row_hi[i]
        # Only equality rows for exact substitution
        (rlo > -_PS_INF && rhi < _PS_INF && abs(rhi - rlo) <= tol) || continue
        a = pw.col_vals[j][1]
        abs(a) >= 1.0e-5 || continue

        rhs = 0.5 * (rlo + rhi)
        _pw_is_implied_free(pw, i, j, a, rhs) || continue

        # Eliminate col j and row i: x_j = (rhs - sum_{k!=j} a_ik x_k) / a
        cj = pw.cost[j]
        ratio_c = cj / a
        pw.obj_offset_delta += ratio_c * rhs

        aux_start = length(pw.aux_cols) + 1
        rc = pw.row_cols[i]
        rv = pw.row_vals[i]
        for k in 1:length(rc)
            c2 = Int(rc[k])
            c2 == j && continue
            v2 = rv[k]
            pw.cost[c2] -= ratio_c * v2
            push!(pw.aux_cols, Int32(c2))
            push!(pw.aux_vals, v2)
            _pw_remove_from_col!(pw, c2, i)
        end
        aux_len = length(pw.aux_cols) - aux_start + 1

        pw.col_active[j] = false
        pw.row_active[i] = false
        Base.empty!(pw.col_rows[j]); Base.empty!(pw.col_vals[j])
        Base.empty!(pw.row_cols[i]); Base.empty!(pw.row_vals[i])

        push!(pw.actions, _PresolveAction(
            _PS_ACT_COL_SINGLETON, Int32(i), Int32(j), Int32(0),
            rhs, a, cj, 0.0, 0.0, 0.0, Int32(aux_start), Int32(aux_len),
        ))
        n_red += 1
    end
    return (n_red, 0)
end

# Pass 4b: Zero-cost column singletons in inequality rows (bounded/free slacks)
function _pw_pass_slack_singletons!(pw::_PresolveWork)::Tuple{Int,Int}
    n_red = 0
    tol = pw.tol
    @inbounds for j in 1:pw.ncols
        pw.col_active[j] || continue
        length(pw.col_rows[j]) == 1 || continue
        i = Int(pw.col_rows[j][1])
        pw.row_active[i] || continue
        rlo = pw.row_lo[i]
        rhi = pw.row_hi[i]
        rlo < rhi - tol || continue
        (rlo <= -_PS_INF && rhi >= _PS_INF) && continue
        a = pw.col_vals[j][1]
        abs(a) >= 1.0e-6 || continue
        abs(pw.cost[j]) <= 1.0e-12 || continue

        lo = pw.col_lo[j]
        hi = pw.col_hi[j]
        lo > hi + tol && return (n_red, CLP_STATUS_PRIMAL_INFEAS)

        # Project row bounds over x_j in [lo, hi]: a*x_j in [la, ua]
        if a > 0.0
            la_fin = lo > -_PS_INF
            ua_fin = hi < _PS_INF
            la = la_fin ? a * lo : 0.0
            ua = ua_fin ? a * hi : 0.0
        else
            la_fin = hi < _PS_INF
            ua_fin = lo > -_PS_INF
            la = la_fin ? a * hi : 0.0
            ua = ua_fin ? a * lo : 0.0
        end

        if rlo <= -_PS_INF || !ua_fin
            new_rlo = -_PS_INF
        else
            if rlo != 0.0 && ua != 0.0 && 1.0e-13 * (abs(rlo) + abs(ua)) > 1.0e-6 * abs(a)
                continue
            end
            new_rlo = rlo - ua
            new_rlo <= -_PS_INF && continue
        end

        if rhi >= _PS_INF || !la_fin
            new_rhi = _PS_INF
        else
            if rhi != 0.0 && la != 0.0 && 1.0e-13 * (abs(rhi) + abs(la)) > 1.0e-6 * abs(a)
                continue
            end
            new_rhi = rhi - la
            new_rhi >= _PS_INF && continue
        end

        # Save remaining row terms before removing j (guaranteed final at postsolve time)
        aux_start = length(pw.aux_cols) + 1
        rc = pw.row_cols[i]
        rv = pw.row_vals[i]
        for k in 1:length(rc)
            c2 = Int(rc[k])
            c2 == j && continue
            push!(pw.aux_cols, Int32(c2))
            push!(pw.aux_vals, rv[k])
        end
        aux_len = length(pw.aux_cols) - aux_start + 1

        _pw_remove_from_row!(pw, i, j)
        pw.row_lo[i] = new_rlo
        pw.row_hi[i] = new_rhi
        pw.col_active[j] = false
        Base.empty!(pw.col_rows[j])
        Base.empty!(pw.col_vals[j])

        push!(pw.actions, _PresolveAction(
            _PS_ACT_SLACK_SINGLETON, Int32(i), Int32(j), Int32(0),
            a, lo, hi, rlo, rhi, 0.0, Int32(aux_start), Int32(aux_len),
        ))
        n_red += 1

        if new_rlo <= -_PS_INF && new_rhi >= _PS_INF
            _pw_remove_row!(pw, i)
            n_red += 1
        elseif isempty(pw.row_cols[i])
            if new_rlo > tol || new_rhi < -tol
                return (n_red, CLP_STATUS_PRIMAL_INFEAS)
            end
            _pw_remove_row!(pw, i)
            n_red += 1
        end
    end
    return (n_red, 0)
end

# Pass 5: Doubleton equality rows (a1*x1 + a2*x2 = rhs)
function _pw_pass_doubletons!(pw::_PresolveWork)::Tuple{Int,Int}
    n_red = 0
    tol = pw.tol
    @inbounds for i in 1:pw.nrows
        pw.row_active[i] || continue
        length(pw.row_cols[i]) == 2 || continue
        rlo = pw.row_lo[i]
        rhi = pw.row_hi[i]
        abs(rhi - rlo) <= tol || continue

        j1 = Int(pw.row_cols[i][1])
        j2 = Int(pw.row_cols[i][2])
        a1 = pw.row_vals[i][1]
        a2 = pw.row_vals[i][2]
        if abs(a1) < 1.0e-6 || abs(a2) < 1.0e-6
            continue
        end
        # Prefer eliminating the sparser column (and keep pivot well-conditioned)
        if length(pw.col_rows[j1]) > length(pw.col_rows[j2])
            j1, j2 = j2, j1
            a1, a2 = a2, a1
        end
        # Avoid fill-in and ill-conditioned pivots
        if length(pw.col_rows[j1]) > 5 || abs(a2 / a1) > 1.0e4 || abs(a1 / a2) > 1.0e4
            continue
        end

        rhs = 0.5 * (rlo + rhi)
        # x1 = (rhs - a2 * x2) / a1  =>  lo1 <= (rhs - a2 * x2)/a1 <= hi1
        lo1 = pw.col_lo[j1]
        hi1 = pw.col_hi[j1]
        old_lo2 = pw.col_lo[j2]
        old_hi2 = pw.col_hi[j2]
        ratio = a2 / a1
        # a2*x2 = rhs - a1*x1 => x2 = rhs/a2 - (a1/a2)*x1
        s_a1 = a1 / a2
        if s_a1 > 0.0
            nlo2 = hi1 < _PS_INF ? (rhs / a2 - s_a1 * hi1) : -floatmax(Float64)
            nhi2 = lo1 > -_PS_INF ? (rhs / a2 - s_a1 * lo1) : floatmax(Float64)
        else
            nlo2 = lo1 > -_PS_INF ? (rhs / a2 - s_a1 * lo1) : -floatmax(Float64)
            nhi2 = hi1 < _PS_INF ? (rhs / a2 - s_a1 * hi1) : floatmax(Float64)
        end
        tight_lo2 = (nlo2 > -_PS_INF && (old_lo2 <= -_PS_INF || nlo2 > old_lo2 + 1.0e-9 * max(1.0, abs(old_lo2)))) ? nlo2 : NaN
        tight_hi2 = (nhi2 < _PS_INF && (old_hi2 >= _PS_INF || nhi2 < old_hi2 - 1.0e-9 * max(1.0, abs(old_hi2)))) ? nhi2 : NaN
        if nlo2 > pw.col_lo[j2]; pw.col_lo[j2] = nlo2; end
        if nhi2 < pw.col_hi[j2]; pw.col_hi[j2] = nhi2; end
        if pw.col_lo[j2] > pw.col_hi[j2] + tol * max(1.0, abs(pw.col_lo[j2]))
            return (n_red, CLP_STATUS_PRIMAL_INFEAS)
        end

        # Substitute x1 = (rhs - a2 * x2) / a1 into objective and all other rows containing j1
        c1 = pw.cost[j1]
        pw.obj_offset_delta += c1 * (rhs / a1)
        pw.cost[j2] -= c1 * ratio

        # Remove row i first
        pw.row_active[i] = false
        _pw_remove_from_col!(pw, j1, i)
        _pw_remove_from_col!(pw, j2, i)
        Base.empty!(pw.row_cols[i]); Base.empty!(pw.row_vals[i])

        # Substitute j1 in remaining rows of col j1 (and save active rows for dual postsolve)
        rows_j1 = copy(pw.col_rows[j1])
        vals_j1 = copy(pw.col_vals[j1])
        pw.col_active[j1] = false
        Base.empty!(pw.col_rows[j1]); Base.empty!(pw.col_vals[j1])
        aux_start = length(pw.aux_cols) + 1
        for k in 1:length(rows_j1)
            ir = Int(rows_j1[k])
            pw.row_active[ir] || continue
            v = vals_j1[k]
            push!(pw.aux_cols, Int32(ir))
            push!(pw.aux_vals, v)
            _pw_remove_from_row!(pw, ir, j1)
            shift = v * (rhs / a1)
            if pw.row_lo[ir] > -_PS_INF; pw.row_lo[ir] -= shift; end
            if pw.row_hi[ir] < _PS_INF;  pw.row_hi[ir] -= shift; end
            _pw_add_entry!(pw, ir, j2, -v * ratio)
        end
        aux_len = length(pw.aux_cols) - aux_start + 1

        push!(pw.actions, _PresolveAction(
            _PS_ACT_DOUBLETON, Int32(i), Int32(j1), Int32(j2),
            rhs, a1, a2, c1, tight_lo2, tight_hi2, Int32(aux_start), Int32(aux_len),
        ))
        n_red += 1
    end
    return (n_red, 0)
end

function _pw_try_subst!(
    pw::_PresolveWork,
    j::Int,
    r_piv::Int,
    a_piv::Float64,
    r_other::Int,
    a_other::Float64,
)::Tuple{Int,Int}
    abs(a_piv) >= 1.0e-4 || return (0, 0)
    mult_r = a_other / a_piv
    abs(mult_r) <= 1.0e3 || return (0, 0)

    rc_piv = pw.row_cols[r_piv]
    rv_piv = pw.row_vals[r_piv]
    max_abs_other = 0.0
    @inbounds for k in 1:length(rc_piv)
        Int(rc_piv[k]) == j && continue
        av = abs(rv_piv[k])
        if av > max_abs_other
            max_abs_other = av
        end
    end
    max_abs_other * abs(mult_r) <= 1.0e8 || return (0, 0)

    rhs_piv = 0.5 * (pw.row_lo[r_piv] + pw.row_hi[r_piv])
    rshift = mult_r * rhs_piv
    abs(rshift) < 1.0e12 || return (0, 0)
    if pw.row_lo[r_other] > -_PS_INF && (pw.row_lo[r_other] - rshift <= -_PS_INF)
        return (0, 0)
    end
    if pw.row_hi[r_other] < _PS_INF && (pw.row_hi[r_other] - rshift >= _PS_INF)
        return (0, 0)
    end

    _pw_is_implied_free(pw, r_piv, j, a_piv, rhs_piv) || return (0, 0)

    # Save (c2, v2) for c2 != j in r_piv before mutating row/col incidence
    aux_start = length(pw.aux_cols) + 1
    @inbounds for k in 1:length(rc_piv)
        c2 = Int(rc_piv[k])
        c2 == j && continue
        push!(pw.aux_cols, Int32(c2))
        push!(pw.aux_vals, rv_piv[k])
    end
    aux_len = length(pw.aux_cols) - aux_start + 1

    cj = pw.cost[j]
    push!(pw.actions, _PresolveAction(
        _PS_ACT_SUBST, Int32(r_piv), Int32(j), Int32(r_other),
        rhs_piv, a_piv, a_other, cj, 0.0, 0.0, Int32(aux_start), Int32(aux_len),
    ))

    # Substitute x_j = (rhs_piv - sum v2 * x_c2) / a_piv into objective and r_other
    mult_obj = cj / a_piv
    pw.obj_offset_delta += mult_obj * rhs_piv
    @inbounds for p in aux_start:(aux_start + aux_len - 1)
        c2 = Int(pw.aux_cols[p])
        v2 = pw.aux_vals[p]
        pw.cost[c2] -= mult_obj * v2
    end

    if pw.row_lo[r_other] > -_PS_INF
        pw.row_lo[r_other] -= rshift
    end
    if pw.row_hi[r_other] < _PS_INF
        pw.row_hi[r_other] -= rshift
    end
    _pw_remove_from_row!(pw, r_other, j)
    @inbounds for p in aux_start:(aux_start + aux_len - 1)
        c2 = Int(pw.aux_cols[p])
        v2 = pw.aux_vals[p]
        _pw_add_entry!(pw, r_other, c2, -mult_r * v2)
    end

    # Remove r_piv from all c2 in r_piv and deactivate r_piv & j
    @inbounds for p in aux_start:(aux_start + aux_len - 1)
        _pw_remove_from_col!(pw, Int(pw.aux_cols[p]), r_piv)
    end
    pw.row_active[r_piv] = false
    pw.col_active[j] = false
    Base.empty!(pw.row_cols[r_piv]); Base.empty!(pw.row_vals[r_piv])
    Base.empty!(pw.col_rows[j]); Base.empty!(pw.col_vals[j])

    n_added = 1
    if isempty(pw.row_cols[r_other])
        tol = pw.tol
        if pw.row_lo[r_other] > tol || pw.row_hi[r_other] < -tol
            return (n_added, CLP_STATUS_PRIMAL_INFEAS)
        end
        _pw_remove_row!(pw, r_other)
        n_added += 1
    end
    return (n_added, 0)
end

# Pass 6: Multi-row implied-free column substitution (col_len == 2)
function _pw_pass_implied_free_subst!(pw::_PresolveWork; max_row_len::Int = 4)::Tuple{Int,Int}
    n_red = 0
    tol = pw.tol
    @inbounds for j in 1:pw.ncols
        pw.col_active[j] || continue
        length(pw.col_rows[j]) == 2 || continue
        r1 = Int(pw.col_rows[j][1])
        r2 = Int(pw.col_rows[j][2])
        (pw.row_active[r1] && pw.row_active[r2] && r1 != r2) || continue
        a1 = pw.col_vals[j][1]
        a2 = pw.col_vals[j][2]

        len1 = length(pw.row_cols[r1])
        len2 = length(pw.row_cols[r2])
        eq1 = (pw.row_lo[r1] > -_PS_INF && pw.row_hi[r1] < _PS_INF &&
               abs(pw.row_hi[r1] - pw.row_lo[r1]) <= tol &&
               2 <= len1 <= max_row_len)
        eq2 = (pw.row_lo[r2] > -_PS_INF && pw.row_hi[r2] < _PS_INF &&
               abs(pw.row_hi[r2] - pw.row_lo[r2]) <= tol &&
               2 <= len2 <= max_row_len)
        (eq1 || eq2) || continue

        first_is_1 = if eq1 && eq2
            (len1 < len2) || (len1 == len2 && abs(a1) >= abs(a2))
        else
            eq1
        end

        r_piv, a_piv, r_other, a_other = first_is_1 ? (r1, a1, r2, a2) : (r2, a2, r1, a1)
        n_sub, st = _pw_try_subst!(pw, j, r_piv, a_piv, r_other, a_other)
        if st != 0
            return (n_red + n_sub, st)
        end
        if n_sub > 0
            n_red += n_sub
            continue
        end

        if eq1 && eq2
            r_piv2, a_piv2, r_other2, a_other2 = first_is_1 ? (r2, a2, r1, a1) : (r1, a1, r2, a2)
            n_sub2, st2 = _pw_try_subst!(pw, j, r_piv2, a_piv2, r_other2, a_other2)
            if st2 != 0
                return (n_red + n_sub2, st2)
            end
            n_red += n_sub2
        end
    end
    return (n_red, 0)
end

"""
    presolve(m::ClpSimplexModel; max_passes::Int=15) -> _PresolvedProblem

Run pure-Julia LP presolve on `m` and return a `_PresolvedProblem` containing the
compacted reduced `ClpSimplexModel` and reversible postsolve stack.
"""
function presolve(m::ClpSimplexModel; max_passes::Int = 15)::_PresolvedProblem
    pw = _pw_init(m)
    status = 0
    for _ in 1:max_passes
        tot = 0
        n1, st = _pw_pass_fixed_and_empty_cols!(pw); tot += n1
        st != 0 && (status = st; break)
        n2, st = _pw_pass_rows_and_singletons!(pw); tot += n2
        st != 0 && (status = st; break)
        n3, st = _pw_pass_forcing_and_redundant!(pw); tot += n3
        st != 0 && (status = st; break)
        n5, st = _pw_pass_col_singletons!(pw); tot += n5
        st != 0 && (status = st; break)
        n6, st = _pw_pass_slack_singletons!(pw); tot += n6
        st != 0 && (status = st; break)
        n7, st = _pw_pass_doubletons!(pw); tot += n7
        st != 0 && (status = st; break)
        n8, st = _pw_pass_implied_free_subst!(pw); tot += n8
        st != 0 && (status = st; break)
        tot == 0 && break
    end

    # Build compacted reduced ClpSimplexModel
    col_map = Int32[]
    orig_to_red_col = zeros(Int32, pw.ncols)
    @inbounds for j in 1:pw.ncols
        if pw.col_active[j]
            push!(col_map, Int32(j))
            orig_to_red_col[j] = Int32(length(col_map))
        end
    end
    row_map = Int32[]
    orig_to_red_row = zeros(Int32, pw.nrows)
    @inbounds for i in 1:pw.nrows
        if pw.row_active[i]
            push!(row_map, Int32(i))
            orig_to_red_row[i] = Int32(length(row_map))
        end
    end

    rn = length(row_map)
    cn = length(col_map)
    red_starts = Vector{Int32}(undef, cn + 1)
    nnz_est = 0
    @inbounds for rc_j in 1:cn
        nnz_est += length(pw.col_rows[Int(col_map[rc_j])])
    end
    red_indices = Vector{Int32}(undef, nnz_est)
    red_elements = Vector{Float64}(undef, nnz_est)
    red_clo = Vector{Float64}(undef, cn)
    red_chi = Vector{Float64}(undef, cn)
    red_obj = Vector{Float64}(undef, cn)
    pos = 1
    @inbounds for rc_j in 1:cn
        j = Int(col_map[rc_j])
        red_starts[rc_j] = Int32(pos)
        red_clo[rc_j] = pw.col_lo[j]
        red_chi[rc_j] = pw.col_hi[j]
        red_obj[rc_j] = pw.cost[j]  # minimization cost
        cr = pw.col_rows[j]
        cv = pw.col_vals[j]
        for k in 1:length(cr)
            rr = Int(orig_to_red_row[Int(cr[k])])
            if rr > 0 && cv[k] != 0.0
                red_indices[pos] = Int32(rr)
                red_elements[pos] = cv[k]
                pos += 1
            end
        end
    end
    red_starts[cn + 1] = Int32(pos)
    resize!(red_indices, pos - 1)
    resize!(red_elements, pos - 1)

    red_rlo = Vector{Float64}(undef, rn)
    red_rhi = Vector{Float64}(undef, rn)
    @inbounds for rr_i in 1:rn
        i = Int(row_map[rr_i])
        red_rlo[rr_i] = pw.row_lo[i]
        red_rhi[rr_i] = pw.row_hi[i]
    end

    red = ClpSimplexModel()
    red.scaling_flag = m.scaling_flag
    red.primal_tol = m.primal_tol
    red.dual_tol = m.dual_tol
    load_problem!(
        red, rn, cn, red_starts, red_indices, red_elements,
        red_clo, red_chi, red_obj, red_rlo, red_rhi;
        obj_sense = 1.0, obj_offset = 0.0, name = m.name,
    )

    return _PresolvedProblem(
        m.nrows, m.ncols, rn, cn,
        m.obj_sense, pw.obj_offset_delta,
        col_map, row_map, pw.col_active, pw.row_active,
        pw.actions, pw.aux_cols, pw.aux_vals,
        status, red,
    )
end

@inline function _ps_row_nonbasic_status(ax::Float64, rlo::Float64, rhi::Float64)::UInt8
    if rlo > -_PS_INF && rhi < _PS_INF && abs(rhi - rlo) <= 1.0e-9
        return _sb_set_status(UInt8(0), CLP_STATUS_IS_FIXED)
    elseif rlo > -_PS_INF && (rhi >= _PS_INF || abs(ax - rlo) <= abs(rhi - ax))
        return _sb_set_status(UInt8(0), CLP_STATUS_AT_LOWER)
    else
        return _sb_set_status(UInt8(0), CLP_STATUS_AT_UPPER)
    end
end

"""
    postsolve!(m::ClpSimplexModel, ps::_PresolvedProblem)

Recover the full unpresolved primal solution (`col_activity`, `row_activity`),
basis `status`, duals (`dual_row_sol`, `reduced_cost`), and `objective_value`
on `m` from the solved reduced model `ps.reduced`.
"""
function postsolve!(m::ClpSimplexModel, ps::_PresolvedProblem)::Nothing
    nrows = m.nrows
    ncols = m.ncols
    red = ps.reduced
    obj_s = m.obj_sense

    fill!(m.col_activity, 0.0)
    fill!(m.row_activity, 0.0)
    fill!(m.dual_row_sol, 0.0)
    fill!(m.reduced_cost, 0.0)
    was_fixed = falses(ncols)

    # Copy reduced primal/dual solution & status back to original indices
    @inbounds for rj in 1:ps.red_ncols
        oj = Int(ps.col_map[rj])
        m.col_activity[oj] = red.col_activity[rj]
        m.status[oj] = red.status[rj]
        m.reduced_cost[oj] = red.reduced_cost[rj] * obj_s
    end
    @inbounds for ri in 1:ps.red_nrows
        oi = Int(ps.row_map[ri])
        m.row_activity[oi] = red.row_activity[ri]
        m.status[ncols + oi] = red.status[ps.red_ncols + ri]
        m.dual_row_sol[oi] = red.dual_row_sol[ri] * obj_s
    end

    # Replay presolve actions in reverse order to recover eliminated primal/dual variables & basis
    @inbounds for k in length(ps.actions):-1:1
        act = ps.actions[k]
        if act.kind == _PS_ACT_FIXED_COL
            j = Int(act.col1)
            val = act.val1
            m.col_activity[j] = val
            lo = m.col_lower[j]
            hi = m.col_upper[j]
            if abs(hi - lo) <= 1.0e-9
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FIXED)
            elseif lo > -_PS_INF && abs(val - lo) <= 1.0e-7 * max(1.0, abs(lo))
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_LOWER)
            elseif hi < _PS_INF && abs(val - hi) <= 1.0e-7 * max(1.0, abs(hi))
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_UPPER)
            elseif lo <= -_PS_INF && hi >= _PS_INF
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FREE)
            else
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FIXED)
            end
            dual_sum = 0.0
            a0 = Int(act.aux_start)
            al = Int(act.aux_len)
            for p in a0:(a0 + al - 1)
                dual_sum += ps.aux_vals[p] * m.dual_row_sol[Int(ps.aux_cols[p])]
            end
            m.reduced_cost[j] = act.val2 * obj_s - dual_sum
            was_fixed[j] = true
        elseif act.kind == _PS_ACT_EMPTY_ROW
            i = Int(act.row)
            m.status[ncols + i] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
            m.dual_row_sol[i] = 0.0
        elseif act.kind == _PS_ACT_FORCING_ROW
            i = Int(act.row)
            tight_at_rlo = (act.val1 > 0.0)
            rlo = act.val2
            rhi = act.val3
            a0 = Int(act.aux_start)
            al = Int(act.aux_len)

            j_piv = 0
            best_y_min = 0.0
            for p in a0:(a0 + al - 1)
                j = Int(ps.aux_cols[p])
                a = ps.aux_vals[p]
                abs(a) <= 1.0e-13 && continue
                forced_to_hi = (tight_at_rlo == (a > 0.0))
                if abs(m.col_upper[j] - m.col_lower[j]) <= 1.0e-9
                    m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FIXED)
                    continue
                end
                m.status[j] = _sb_set_status(UInt8(0), forced_to_hi ? CLP_STATUS_AT_UPPER : CLP_STATUS_AT_LOWER)
                cand_y_min = (obj_s * m.reduced_cost[j]) / a
                if tight_at_rlo
                    if cand_y_min > best_y_min + 1.0e-12
                        best_y_min = cand_y_min
                        j_piv = j
                    end
                else
                    if cand_y_min < best_y_min - 1.0e-12
                        best_y_min = cand_y_min
                        j_piv = j
                    end
                end
            end

            if j_piv != 0
                y_i = obj_s * best_y_min
                m.dual_row_sol[i] = y_i
                for p in a0:(a0 + al - 1)
                    j = Int(ps.aux_cols[p])
                    a = ps.aux_vals[p]
                    m.reduced_cost[j] -= a * y_i
                end
                m.reduced_cost[j_piv] = 0.0
                was_fixed[j_piv] = false
                m.status[j_piv] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
                r_act = tight_at_rlo ? rlo : rhi
                m.status[ncols + i] = _ps_row_nonbasic_status(r_act, rlo, rhi)
            else
                m.dual_row_sol[i] = 0.0
                m.status[ncols + i] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
            end
        elseif act.kind == _PS_ACT_ROW_SINGLETON
            i = Int(act.row)
            j = Int(act.col1)
            a = act.val1
            cmp_lo = min(act.val2, m.col_lower[j])
            cmp_hi = max(act.val3, m.col_upper[j])
            rlo = act.val4
            rhi = act.val5
            xj = m.col_activity[j]
            st = _sb_get_status(m.status[j])
            rc_j = m.reduced_cost[j]
            ax = a * xj
            at_rlo = rlo > -_PS_INF && abs(ax - rlo) <= 1.0e-6 * max(1.0, abs(rlo))
            at_rhi = rhi < _PS_INF && abs(ax - rhi) <= 1.0e-6 * max(1.0, abs(rhi))
            row_binding = at_rlo || at_rhi

            is_eq_row = rlo > -_PS_INF && rhi < _PS_INF && abs(rhi - rlo) <= 1.0e-9
            y_cand = rc_j / a
            dual_ok = is_eq_row ||
                      (at_rlo && y_cand * obj_s >= -1.0e-12) ||
                      (at_rhi && y_cand * obj_s <= 1.0e-12)

            strictly_inside = (cmp_lo <= -_PS_INF || xj > cmp_lo + 1.0e-7 * max(1.0, abs(cmp_lo))) &&
                              (cmp_hi >= _PS_INF  || xj < cmp_hi - 1.0e-7 * max(1.0, abs(cmp_hi)))
            tight_lo = (st == CLP_STATUS_AT_LOWER) &&
                       (cmp_lo <= -_PS_INF || xj > cmp_lo + 1.0e-7 * max(1.0, abs(cmp_lo)))
            tight_hi = (st == CLP_STATUS_AT_UPPER) &&
                       (cmp_hi >= _PS_INF  || xj < cmp_hi - 1.0e-7 * max(1.0, abs(cmp_hi)))

            degen_lo = (st != CLP_STATUS_BASIC) &&
                       (cmp_lo > -_PS_INF && abs(xj - cmp_lo) <= 1.0e-7 * max(1.0, abs(cmp_lo))) &&
                       ((a > 0.0 && at_rhi) || (a < 0.0 && at_rlo)) &&
                       (rc_j * obj_s < -1.0e-12)
            degen_hi = (st != CLP_STATUS_BASIC) &&
                       (cmp_hi < _PS_INF && abs(xj - cmp_hi) <= 1.0e-7 * max(1.0, abs(cmp_hi))) &&
                       ((a > 0.0 && at_rlo) || (a < 0.0 && at_rhi)) &&
                       (rc_j * obj_s > 1.0e-12)

            if row_binding && dual_ok && (
                (strictly_inside && (abs(rc_j) > 1.0e-12 || st == CLP_STATUS_IS_FIXED)) ||
                ((tight_lo || tight_hi) && abs(rc_j) > 1.0e-12) ||
                degen_lo || degen_hi
            )
                m.dual_row_sol[i] = y_cand
                m.reduced_cost[j] = 0.0
                was_fixed[j] = false
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
                m.status[ncols + i] = _ps_row_nonbasic_status(ax, rlo, rhi)
            elseif row_binding && (tight_lo || tight_hi)
                m.dual_row_sol[i] = 0.0
                m.reduced_cost[j] = 0.0
                was_fixed[j] = false
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
                m.status[ncols + i] = _ps_row_nonbasic_status(ax, rlo, rhi)
            else
                m.dual_row_sol[i] = 0.0
                m.status[ncols + i] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
            end
        elseif act.kind == _PS_ACT_DOUBLETON
            i = Int(act.row)
            j1 = Int(act.col1)
            j2 = Int(act.col2)
            rhs = act.val1
            a1 = act.val2
            a2 = act.val3
            c1_min = act.val4
            tight_lo2 = act.val5
            tight_hi2 = act.val6
            x2 = m.col_activity[j2]
            m.col_activity[j1] = (rhs - a2 * x2) / a1
            m.status[ncols + i] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FIXED)

            dual_sum = 0.0
            a0 = Int(act.aux_start)
            al = Int(act.aux_len)
            for p in a0:(a0 + al - 1)
                dual_sum += ps.aux_vals[p] * m.dual_row_sol[Int(ps.aux_cols[p])]
            end
            y_i = (c1_min * obj_s - dual_sum) / a1

            st2 = _sb_get_status(m.status[j2])
            rc2 = m.reduced_cost[j2]
            lo2_orig = m.col_lower[j2]
            hi2_orig = m.col_upper[j2]
            at_tight_lo = !isnan(tight_lo2) &&
                          (lo2_orig <= -_PS_INF || tight_lo2 > lo2_orig + 1.0e-9 * max(1.0, abs(lo2_orig))) &&
                          abs(x2 - tight_lo2) <= 1.0e-5 * max(1.0, abs(tight_lo2))
            at_tight_hi = !isnan(tight_hi2) &&
                          (hi2_orig >= _PS_INF || tight_hi2 < hi2_orig - 1.0e-9 * max(1.0, abs(hi2_orig))) &&
                          abs(x2 - tight_hi2) <= 1.0e-5 * max(1.0, abs(tight_hi2))

            swap_to_lo = (st2 == CLP_STATUS_AT_LOWER && at_tight_lo && abs(rc2) > 1.0e-12) ||
                         (st2 == CLP_STATUS_AT_UPPER && at_tight_lo && rc2 * obj_s > 1.0e-12) ||
                         (st2 == CLP_STATUS_IS_FIXED && at_tight_lo && (!at_tight_hi || rc2 * obj_s > 1.0e-12))
            swap_to_hi = !swap_to_lo && (
                         (st2 == CLP_STATUS_AT_UPPER && at_tight_hi && abs(rc2) > 1.0e-12) ||
                         (st2 == CLP_STATUS_AT_LOWER && at_tight_hi && rc2 * obj_s < -1.0e-12) ||
                         (st2 == CLP_STATUS_IS_FIXED && at_tight_hi && (!at_tight_lo || rc2 * obj_s < -1.0e-12)))

            if swap_to_lo || swap_to_hi
                y_i += rc2 / a2
                m.dual_row_sol[i] = y_i
                m.reduced_cost[j1] = -(a1 / a2) * rc2
                m.reduced_cost[j2] = 0.0
                was_fixed[j2] = false
                m.status[j2] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
                same_sign = (a1 / a2) > 0.0
                j1_st = if swap_to_lo
                    same_sign ? CLP_STATUS_AT_UPPER : CLP_STATUS_AT_LOWER
                else
                    same_sign ? CLP_STATUS_AT_LOWER : CLP_STATUS_AT_UPPER
                end
                m.status[j1] = _sb_set_status(UInt8(0), j1_st)
            else
                m.dual_row_sol[i] = y_i
                m.reduced_cost[j1] = 0.0
                m.status[j1] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
            end
        elseif act.kind == _PS_ACT_COL_SINGLETON
            i = Int(act.row)
            j = Int(act.col1)
            rhs = act.val1
            a = act.val2
            s = rhs
            a0 = Int(act.aux_start)
            al = Int(act.aux_len)
            for p in a0:(a0 + al - 1)
                s -= ps.aux_vals[p] * m.col_activity[Int(ps.aux_cols[p])]
            end
            m.col_activity[j] = s / a
            m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
            m.status[ncols + i] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FIXED)
            m.dual_row_sol[i] = (act.val3 * obj_s) / a
            m.reduced_cost[j] = 0.0
        elseif act.kind == _PS_ACT_SUBST
            r_piv = Int(act.row)
            j = Int(act.col1)
            r_other = Int(act.col2)
            rhs_piv = act.val1
            a_piv = act.val2
            a_other = act.val3
            c_min = act.val4
            s_val = rhs_piv
            a0 = Int(act.aux_start)
            al = Int(act.aux_len)
            for p in a0:(a0 + al - 1)
                s_val -= ps.aux_vals[p] * m.col_activity[Int(ps.aux_cols[p])]
            end
            m.col_activity[j] = s_val / a_piv
            m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
            m.status[ncols + r_piv] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FIXED)
            m.dual_row_sol[r_piv] = (c_min * obj_s - a_other * m.dual_row_sol[r_other]) / a_piv
            m.reduced_cost[j] = 0.0
        elseif act.kind == _PS_ACT_SLACK_SINGLETON
            i = Int(act.row)
            j = Int(act.col1)
            a = act.val1
            lo = act.val2
            hi = act.val3
            old_rlo = act.val4
            old_rhi = act.val5
            other_act = 0.0
            a0 = Int(act.aux_start)
            al = Int(act.aux_len)
            for p in a0:(a0 + al - 1)
                other_act += ps.aux_vals[p] * m.col_activity[Int(ps.aux_cols[p])]
            end

            if a > 0.0
                r_min = old_rlo <= -_PS_INF ? -Inf : (old_rlo - other_act) / a
                r_max = old_rhi >= _PS_INF ? Inf : (old_rhi - other_act) / a
            else
                r_min = old_rhi >= _PS_INF ? -Inf : (old_rhi - other_act) / a
                r_max = old_rlo <= -_PS_INF ? Inf : (old_rlo - other_act) / a
            end

            req_lo = max(lo, r_min)
            req_hi = min(hi, r_max)
            row_st = _sb_get_status(m.status[ncols + i])

            tol_lo = lo > -_PS_INF ? 1.0e-9 * max(1.0, abs(lo)) : 0.0
            tol_hi = hi < _PS_INF ? 1.0e-9 * max(1.0, abs(hi)) : 0.0
            if req_lo > req_hi
                if r_min > hi
                    m.col_activity[j] = hi
                    m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_UPPER)
                else
                    m.col_activity[j] = lo
                    m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_LOWER)
                end
            elseif row_st == CLP_STATUS_AT_LOWER && a > 0.0 && hi < _PS_INF && req_lo - tol_hi <= hi && hi <= r_max + tol_hi
                m.col_activity[j] = hi
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_UPPER)
            elseif row_st == CLP_STATUS_AT_LOWER && a < 0.0 && lo > -_PS_INF && r_min - tol_lo <= lo && lo <= req_hi + tol_lo
                m.col_activity[j] = lo
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_LOWER)
            elseif row_st == CLP_STATUS_AT_UPPER && a > 0.0 && lo > -_PS_INF && r_min - tol_lo <= lo && lo <= req_hi + tol_lo
                m.col_activity[j] = lo
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_LOWER)
            elseif row_st == CLP_STATUS_AT_UPPER && a < 0.0 && hi < _PS_INF && req_lo - tol_hi <= hi && hi <= r_max + tol_hi
                m.col_activity[j] = hi
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_UPPER)
            elseif lo > -_PS_INF && req_lo <= lo && lo <= req_hi
                m.col_activity[j] = lo
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_LOWER)
            elseif hi < _PS_INF && req_lo <= hi && hi <= req_hi
                m.col_activity[j] = hi
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_UPPER)
            elseif lo <= -_PS_INF && hi >= _PS_INF && 0.0 >= req_lo && 0.0 <= req_hi
                m.col_activity[j] = 0.0
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FREE)
            elseif isfinite(r_min)
                m.col_activity[j] = req_lo
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
                m.status[ncols + i] = _sb_set_status(UInt8(0), a > 0.0 ? CLP_STATUS_AT_LOWER : CLP_STATUS_AT_UPPER)
            else
                m.col_activity[j] = req_hi
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_BASIC)
                m.status[ncols + i] = _sb_set_status(UInt8(0), a > 0.0 ? CLP_STATUS_AT_UPPER : CLP_STATUS_AT_LOWER)
            end
            m.reduced_cost[j] = -a * m.dual_row_sol[i]
        end
    end

    # Recompute exact reduced_cost = objective - A' * dual_row_sol for fixed columns
    starts = m.mat_starts
    indices = m.mat_indices
    elems = m.mat_elements
    @inbounds for j in 1:ncols
        was_fixed[j] || continue
        rc = m.objective[j]
        s_idx = Int(starts[j])
        e_idx = Int(starts[j + 1]) - 1
        for k in s_idx:e_idx
            rc -= elems[k] * m.dual_row_sol[Int(indices[k])]
        end
        m.reduced_cost[j] = rc
    end

    # Recompute exact unscaled row activities A * x and objective value
    fill!(m.row_activity, 0.0)
    clp_mat_times_unscaled!(
        m.mat_starts, nothing, false, m.mat_indices, m.mat_elements,
        ncols, 1.0, m.col_activity, m.row_activity,
    )
    obj = -m.obj_offset
    @inbounds for j in 1:ncols
        obj += m.objective[j] * m.col_activity[j]
    end
    m.objective_value = obj
    m.problem_status = red.problem_status
    m.number_iterations = red.number_iterations
    return nothing
end

"""
    initial_solve!(m::ClpSimplexModel; do_presolve::Bool=true, max_iterations::Int=m.max_iterations,
                   scaling::Int=m.scaling_flag, polish::Bool=false) -> Int

Full standalone pure-Julia LP solver equivalent to `ClpSimplex::initialSolve()`:
runs pure-Julia presolve -> solves reduced LP with `dual!` -> postsolves back to
original space (with optional warm-start polish on the original problem).
"""
function initial_solve!(
    m::ClpSimplexModel;
    do_presolve::Bool = true,
    max_iterations::Int = m.max_iterations,
    scaling::Int = m.scaling_flag,
    polish::Bool = false,
)::Int
    if !do_presolve || m.nrows == 0 || m.ncols == 0
        return dual!(m; max_iterations = max_iterations, scaling = scaling)
    end

    ps = presolve(m)
    if ps.status != 0
        m.problem_status = ps.status
        m.number_iterations = 0
        return m.problem_status
    end

    st = dual!(ps.reduced; max_iterations = max_iterations, scaling = scaling)
    postsolve!(m, ps)

    # Verify relative primal feasibility of postsolved solution; polish only if needed
    if st == CLP_STATUS_OPTIMAL
        max_rel_viol = 0.0
        @inbounds for j in 1:m.ncols
            v = m.col_activity[j]
            lo = m.col_lower[j]; hi = m.col_upper[j]
            if lo > -_PS_INF && v < lo
                max_rel_viol = max(max_rel_viol, (lo - v) / (1.0 + abs(lo)))
            end
            if hi < _PS_INF && v > hi
                max_rel_viol = max(max_rel_viol, (v - hi) / (1.0 + abs(hi)))
            end
        end
        @inbounds for i in 1:m.nrows
            v = m.row_activity[i]
            lo = m.row_lower[i]; hi = m.row_upper[i]
            if lo > -_PS_INF && v < lo
                max_rel_viol = max(max_rel_viol, (lo - v) / (1.0 + abs(lo)))
            end
            if hi < _PS_INF && v > hi
                max_rel_viol = max(max_rel_viol, (v - hi) / (1.0 + abs(hi)))
            end
        end
        if polish || max_rel_viol > 1.0e-5
            prev_iters = m.number_iterations
            scale_model!(m; mode = scaling)
            m.has_row_copy = false
            st2 = dual!(m; max_iterations = max_iterations, warm_start = false, scaling = scaling)
            m.number_iterations += prev_iters
            return st2
        end
    end

    return m.problem_status
end

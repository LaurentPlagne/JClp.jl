# Pure idiomatic 1-based Julia standalone Dual & Primal Simplex drivers (Phase B — B12c-2).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, --trim=safe clean.
# Drives clp_dual_while_iterating! (B12a) and clp_primal_while_iterating! (B12b) directly.

const CLP_STATUS_OPTIMAL          = 0
const CLP_STATUS_PRIMAL_INFEAS    = 1
const CLP_STATUS_DUAL_INFEAS      = 2   # primal unbounded
const CLP_STATUS_MAX_ITERATIONS   = 3
const CLP_STATUS_ERROR            = 4

const _SM_EMPTY_F64 = Float64[]

# ==============================================================================
# Warm-start modification helpers (preserve m.status for re-optimization)
# ==============================================================================

function set_row_bounds!(m::ClpSimplexModel, i::Int, lo::Float64, hi::Float64)::Nothing
    @inbounds begin
        m.row_lower[i] = lo
        m.row_upper[i] = hi
    end
    return nothing
end

function set_col_bounds!(m::ClpSimplexModel, j::Int, lo::Float64, hi::Float64)::Nothing
    @inbounds begin
        m.col_lower[j] = lo
        m.col_upper[j] = hi
    end
    return nothing
end

function set_obj_coeff!(m::ClpSimplexModel, j::Int, c::Float64)::Nothing
    @inbounds m.objective[j] = c
    return nothing
end

"""
    set_perturbation!(m::ClpSimplexModel, value::Int) -> Int

Set the dual simplex perturbation mode (`50` = automatic on degeneracy, `100` = disabled,
`102` = eager/immediate) and return `value`.
"""
function set_perturbation!(m::ClpSimplexModel, value::Int)::Int
    m.perturbation = value
    return value
end

"""
    perturbation(m::ClpSimplexModel) -> Int

Return the current dual simplex perturbation mode for `m`.
"""
perturbation(m::ClpSimplexModel)::Int = m.perturbation

@inline function _sm_clear_flags!(status::Vector{UInt8})::Int
    n_unflagged = 0
    @inbounds for i in 1:length(status)
        if (status[i] & CLP_STATUS_FLAGGED_BIT) != UInt8(0)
            status[i] &= ~CLP_STATUS_FLAGGED_BIT
            n_unflagged += 1
        end
    end
    return n_unflagged
end

@inline function _sm_clear_workspaces!(m::ClpSimplexModel)::Nothing
    fill!(m.r0_val, 0.0)
    fill!(m.r1_val, 0.0)
    fill!(m.r2_val, 0.0)
    fill!(m.r3_val, 0.0)
    fill!(m.c0_val, 0.0)
    fill!(m.c1_val, 0.0)
    fill!(m.pcs_alt_val, 0.0)
    m.pcs_alt_nnz = 0
    return nothing
end

# ==============================================================================
# Dual Simplex helpers (Phase 1 fake bounds + Dual Row Steepest init)
# ==============================================================================

function _sm_dual_make_feasible!(m::ClpSimplexModel, dual_bound::Float64)::Tuple{Int,Int}
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols
    dtol = m.dual_tol
    n_changed = 0
    n_fake = 0
    @inbounds for seq in 1:ntotal
        st = _sb_get_status(m.status[seq])
        st == CLP_STATUS_BASIC && continue
        st == CLP_STATUS_IS_FIXED && continue
        d = m.dj[seq]
        lo = m.lower[seq]
        hi = m.upper[seq]
        if st == CLP_STATUS_AT_LOWER
            if d < -dtol
                if hi < 1.0e20
                    m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_UPPER)
                    m.solution[seq] = hi
                    n_changed += 1
                else
                    # Apply temporary upper fake bound
                    fb = max(lo + dual_bound, dual_bound)
                    m.upper[seq] = fb
                    m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_UPPER)
                    m.solution[seq] = fb
                    n_fake += 1
                    n_changed += 1
                end
            end
        elseif st == CLP_STATUS_AT_UPPER
            if d > dtol
                if lo > -1.0e20
                    m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_LOWER)
                    m.solution[seq] = lo
                    n_changed += 1
                else
                    fb = min(hi - dual_bound, -dual_bound)
                    m.lower[seq] = fb
                    m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_LOWER)
                    m.solution[seq] = fb
                    n_fake += 1
                    n_changed += 1
                end
            end
        elseif st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
            if d < -dtol
                fb = hi < 1.0e20 ? hi : dual_bound
                if hi >= 1.0e20
                    m.upper[seq] = fb
                    n_fake += 1
                end
                m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_UPPER)
                m.solution[seq] = fb
                n_changed += 1
            elseif d > dtol
                fb = lo > -1.0e20 ? lo : -dual_bound
                if lo <= -1.0e20
                    m.lower[seq] = fb
                    n_fake += 1
                end
                m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_LOWER)
                m.solution[seq] = fb
                n_changed += 1
            end
        end
    end
    return (n_changed, n_fake)
end

function _sm_drs_refresh_infeas!(m::ClpSimplexModel)::Nothing
    fill!(m.drs_infeas_val, 0.0)
    m.drs_infeas_nnz = clp_drs_recompute_infeasibilities!(
        m.drs_infeas_val, m.drs_infeas_idx, m.pivot_var,
        m.solution, m.lower, m.upper, m.nrows, m.primal_tol,
    )
    return nothing
end

function _sm_drs_init_weights!(m::ClpSimplexModel, exact::Bool)::Nothing
    nrows = m.nrows
    if exact
        fill!(m.r0_val, 0.0)
        fill!(m.r1_val, 0.0)
        @inbounds for irow in 1:nrows
            m.r0_val[1] = 1.0
            m.r0_idx[1] = Int32(irow)
            nnz = clp_fact_update_column_transpose!(
                m.cf, m.r1_val, m.r1_idx, 0, false, m.r0_val, m.r0_idx, 1, true,
            )
            w = 0.0
            for k in 1:nnz
                v = m.r0_val[k]
                w += v * v
                m.r0_val[k] = 0.0
            end
            m.drs_weights[irow] = max(w, CLP_DEVEX_TRY_NORM)
        end
    else
        fill!(m.drs_weights, 1.0)
    end
    return nothing
end

@inline function _sm_bound_pair_feasible(lo::Float64, hi::Float64, tol::Float64)::Bool
    (isnan(lo) || isnan(hi) || lo == Inf || hi == -Inf) && return false
    (lo == -Inf || hi == Inf) && return true
    return lo <= hi + tol * max(1.0, abs(lo), abs(hi))
end

function _sm_bounds_feasible(m::ClpSimplexModel)::Bool
    tol = m.primal_tol
    @inbounds for j in 1:m.ncols
        _sm_bound_pair_feasible(m.col_lower[j], m.col_upper[j], tol) || return false
    end
    @inbounds for i in 1:m.nrows
        _sm_bound_pair_feasible(m.row_lower[i], m.row_upper[i], tol) || return false
    end
    return true
end

# Zero-allocation in-place heapsort on v[1:n] for Clp C++ iteration-0 distinct-cost heuristic.
function _sm_heapsort_f64!(v::Vector{Float64}, n::Int)::Nothing
    n <= 1 && return nothing
    @inbounds begin
        for start in (n >>> 1):-1:1
            root = start
            while (child = root << 1) <= n
                if child < n && v[child] < v[child + 1]
                    child += 1
                end
                if v[root] < v[child]
                    v[root], v[child] = v[child], v[root]
                    root = child
                else
                    break
                end
            end
        end
        for last in n:-1:2
            v[1], v[last] = v[last], v[1]
            root = 1
            lim = last - 1
            while (child = root << 1) <= lim
                if child < lim && v[child] < v[child + 1]
                    child += 1
                end
                if v[root] < v[child]
                    v[root], v[child] = v[child], v[root]
                    root = child
                else
                    break
                end
            end
        end
    end
    return nothing
end

# Matches ClpSimplexDual::perturb() iteration-0 heuristic:
# perturb immediately if <= 25% of columns have distinct |objective[j]| values
# (typical of hydro/network/SDP LPs with many zero or repeated costs).
# Uses pre-allocated m.c0_val buffer -> 0 bytes allocated.
function _sm_should_perturb_init!(m::ClpSimplexModel)::Bool
    ncols = m.ncols
    ncols <= 0 && return false
    buf = m.c0_val
    @inbounds for j in 1:ncols
        buf[j] = abs(m.objective[j])
    end
    _sm_heapsort_f64!(buf, ncols)
    n_distinct = 1
    @inbounds begin
        last_v = buf[1]
        for j in 2:ncols
            v = buf[j]
            if v != last_v
                n_distinct += 1
                last_v = v
            end
        end
        for j in 1:ncols
            buf[j] = 0.0
        end
    end
    return n_distinct * 4 <= ncols
end

function _sm_dual_perturb!(m::ClpSimplexModel, pert_mode::Int)::Nothing
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols
    @inbounds for seq in 1:ntotal
        m.saved_cost[seq] = m.cost[seq]
    end
    c_floor = (pert_mode == 102 ? 100.0 : 50.0) * m.dual_tol
    state = UInt64(0x9e3779b97f4a7c15) ⊻ UInt64(m.rng_seed)
    @inbounds for seq in 1:ntotal
        st = _sb_get_status(m.status[seq])
        if st == CLP_STATUS_AT_LOWER
            pert_seq = clamp(c_floor + 1.0e-5 * abs(m.cost[seq]), 1.0e-7, 1.0e-4)
            state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            u = Float64(state >>> 11) * 0x1.0p-53
            m.cost[seq] += pert_seq * (0.25 + u)
        elseif st == CLP_STATUS_AT_UPPER
            pert_seq = clamp(c_floor + 1.0e-5 * abs(m.cost[seq]), 1.0e-7, 1.0e-4)
            state = state * UInt64(6364136223846793005) + UInt64(1442695040888963407)
            u = Float64(state >>> 11) * 0x1.0p-53
            m.cost[seq] -= pert_seq * (0.25 + u)
        end
    end
    guts_of_solution!(m; do_primals = false, do_duals = true, check_infeas = true, algorithm = -1)
    return nothing
end

function _sm_dual_unperturb!(m::ClpSimplexModel, recompute::Bool = true)::Nothing
    ntotal = m.nrows + m.ncols
    @inbounds for seq in 1:ntotal
        m.cost[seq] = m.saved_cost[seq]
    end
    if recompute
        guts_of_solution!(m; do_primals = true, do_duals = true, check_infeas = true, algorithm = -1)
    end
    return nothing
end

function _sm_restore_true_bounds!(m::ClpSimplexModel)::Nothing
    nrows = m.nrows
    ncols = m.ncols
    if m.has_scale
        clp_create_rim1!(
            m.lower, m.upper, m.col_lower, m.col_upper, m.row_lower, m.row_upper,
            m.row_scale, m.inv_col_scale, nrows, 0, ncols, nrows, ncols, m.special_options, false,
            m.rhs_scale, m.primal_tol,
        )
    else
        clp_create_rim1!(
            m.lower, m.upper, m.col_lower, m.col_upper, m.row_lower, m.row_upper,
            nothing, nothing, nrows, 0, ncols, nrows, ncols, m.special_options, false,
            m.rhs_scale, m.primal_tol,
        )
    end
    return nothing
end

function _sm_reset_fake_bounds!(m::ClpSimplexModel)::Nothing
    ntotal = m.nrows + m.ncols
    @inbounds for seq in 1:ntotal
        st = _sb_get_status(m.status[seq])
        if st == CLP_STATUS_AT_UPPER && m.upper[seq] >= 1.0e20
            if m.lower[seq] > -1.0e20
                m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_LOWER)
                m.solution[seq] = m.lower[seq]
            else
                m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_IS_FREE)
                m.solution[seq] = 0.0
            end
        elseif st == CLP_STATUS_AT_LOWER && m.lower[seq] <= -1.0e20
            if m.upper[seq] < 1.0e20
                m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_AT_UPPER)
                m.solution[seq] = m.upper[seq]
            else
                m.status[seq] = _sb_set_status(m.status[seq], CLP_STATUS_IS_FREE)
                m.solution[seq] = 0.0
            end
        end
    end
    return nothing
end

"""
    dual!(m::ClpSimplexModel; max_iterations=m.max_iterations, warm_start=false,
          scaling=m.scaling_flag, exact_weights=false, perturbation=m.perturbation) -> Int

Solve `m` using the standalone pure-Julia Dual Simplex method (`clp_dual_while_iterating!`).
Returns `m.problem_status`: `0` (optimal), `1` (primal infeasible), `2` (dual infeasible /
unbounded), `3` (max iterations), `4` (error).
"""
function dual!(
    m::ClpSimplexModel;
    max_iterations::Int = m.max_iterations,
    warm_start::Bool = false,
    scaling::Int = m.scaling_flag,
    exact_weights::Bool = false,
    perturbation::Int = m.perturbation,
)::Int
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols

    if !_sm_bounds_feasible(m)
        m.number_iterations = 0
        m.problem_status = CLP_STATUS_PRIMAL_INFEAS
        m.secondary_status = 0
        fill!(m.dual_row_sol, 0.0)
        fill!(m.reduced_cost, 0.0)
        return m.problem_status
    end

    if nrows == 0
        # Trivial unconstrained / box LP
        m.number_iterations = 0
        m.problem_status = CLP_STATUS_OPTIMAL
        m.secondary_status = 0
        obj = -m.obj_offset
        @inbounds for j in 1:ncols
            lo = m.col_lower[j]
            hi = m.col_upper[j]
            c = m.objective[j] * m.obj_sense
            m.reduced_cost[j] = m.objective[j]
            if lo > -1.0e20 && hi < 1.0e20 && abs(hi - lo) <= m.primal_tol * max(1.0, abs(lo), abs(hi))
                m.col_activity[j] = 0.5 * (lo + hi)
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FIXED)
            elseif c > 0.0
                if lo <= -1.0e20
                    m.problem_status = CLP_STATUS_DUAL_INFEAS
                    return m.problem_status
                end
                m.col_activity[j] = lo
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_LOWER)
            elseif c < 0.0
                if hi >= 1.0e20
                    m.problem_status = CLP_STATUS_DUAL_INFEAS
                    return m.problem_status
                end
                m.col_activity[j] = hi
                m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_UPPER)
            else
                if lo > -1.0e20
                    m.col_activity[j] = lo
                    m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_LOWER)
                elseif hi < 1.0e20
                    m.col_activity[j] = hi
                    m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_AT_UPPER)
                else
                    m.col_activity[j] = 0.0
                    m.status[j] = _sb_set_status(UInt8(0), CLP_STATUS_IS_FREE)
                end
            end
            obj += m.objective[j] * m.col_activity[j]
        end
        m.objective_value = obj
        return m.problem_status
    end

    m.max_iterations = max_iterations
    m.number_iterations = 0
    m.problem_status = -1
    m.secondary_status = 0
    m.force_factorization = -1
    m.last_bad_iter = -1
    m.last_good_iter = 0
    m.change_made = 0
    m.progress_flag = 0
    m.alpha_accuracy = -1.0
    m.acceptable_pivot = 1.0e-9
    fill!(m.prog_in, Int32(-1))
    fill!(m.prog_out, Int32(-1))
    fill!(m.prog_way, Int8(0))
    m.prog_bad_times = 0
    _sm_clear_flags!(m.status)

    if !warm_start
        scale_model!(m; mode = scaling)
        m.has_row_copy = false
        all_slack_basis!(m)
    elseif !m.has_row_copy
        build_row_copy!(m)
    end

    saved_weights_val = m.drs_saved_val
    saved_weights_idx = m.drs_saved_idx
    saved_back = m.drs_saved_back
    weights_initialized = warm_start && !m.drs_weights_stale &&
                          m.cf.lu.nrows == nrows && m.cf.lu.ncols == nrows && m.cf.lu.status == 0 &&
                          length(m.drs_weights) >= nrows && length(m.pivot_var) >= nrows &&
                          length(saved_weights_val) >= nrows && length(saved_weights_idx) >= nrows &&
                          length(saved_back) >= ntotal
    if weights_initialized
        n_basic_status = 0
        @inbounds for seq in 1:ntotal
            saved_back[seq] = Int32(0)
            if _sb_get_status(m.status[seq]) == CLP_STATUS_BASIC
                n_basic_status += 1
            end
        end
        if n_basic_status != nrows
            weights_initialized = false
        else
            @inbounds for i in 1:nrows
                pv = Int(m.pivot_var[i])
                w = m.drs_weights[i]
                if pv < 1 || pv > ntotal || !(isfinite(w) && w > 0.0) ||
                   _sb_get_status(m.status[pv]) != CLP_STATUS_BASIC ||
                   saved_back[pv] != Int32(0)
                    weights_initialized = false
                    break
                end
                saved_back[pv] = Int32(i)
            end
        end
    end
    if weights_initialized
        @inbounds for i in 1:nrows
            saved_weights_val[i] = m.drs_weights[i]
            saved_weights_idx[i] = m.pivot_var[i]
        end
    else
        fill!(saved_weights_val, 0.0)
        fill!(saved_weights_idx, Int32(0))
        fill!(saved_back, Int32(0))
    end
    can_reuse_cf = warm_start && m.cf.lu.nrows == nrows && m.cf.lu.ncols == nrows &&
                   m.cf.lu.status == 0 && m.cf.lu.npivots < 35 &&
                   !clp_fact_time_to_refactorize!(m.cf) &&
                   m.fact_hash != UInt64(0) && m.basis_hash != UInt64(0) &&
                   m.fact_hash == _sm_matrix_hash(m) && m.basis_hash == _sm_basis_hash(m)
    create_rim!(m)
    if can_reuse_cf && _sm_basis_hash(m) != m.basis_hash
        can_reuse_cf = false
    end

    elems = _sm_active_elements(m)
    dual_bound = 1.0e10
    number_fake = 0
    number_changed = 0
    drs_alt_val = m.pcs_alt_val
    drs_alt_idx = m.pcs_alt_idx
    drs_alt_nnz = 0

    # Local loop scalars for clp_dual_while_iterating!
    first_free = 0
    pivot_row = 0
    seq_in = 0
    seq_out = 0
    dir_in = 0
    dir_out = 0
    val_in = 0.0; lo_in = 0.0; hi_in = 0.0; dual_in = 0.0
    val_out = 0.0; lo_out = 0.0; hi_out = 0.0; dual_out = 0.0
    theta = 0.0; alpha = 0.0
    spare_int0 = 0; spare_int1 = 0
    spare_dbl0 = 0.0; spare_dbl2 = 0.0; spare_dbl3 = 0.0

    # Shared empty typed buffer for unscaled mode (zero allocation)
    rs_v = m.has_scale ? m.row_scale : _SM_EMPTY_F64
    cs_v = m.has_scale ? m.col_scale : _SM_EMPTY_F64
    ics_v = m.has_scale ? m.inv_col_scale : _SM_EMPTY_F64

    outer_iters = 0
    max_outer = min(2000, max(200, max_iterations))
    infeas_confirm = 0
    fake_bumps = 0
    force_primal_cleanup = false
    is_perturbed = false
    perturbation_done = false
    prev_iter = 0
    prev_obj = 0.0
    prev_outer_iter = -1
    stall_count = 0
    degen_trigger = false
    zero_piv_streak = 0

    while outer_iters < max_outer
        outer_iters += 1
        m.problem_status = -1
        _sm_clear_workspaces!(m)

        factorized_this_outer = false
        if outer_iters == 1 && can_reuse_cf
            guts_of_solution!(m; do_primals = true, do_duals = true, check_infeas = true, algorithm = -1)
            if !isfinite(m.largest_primal_err) || !isfinite(m.largest_dual_err) ||
               m.largest_primal_err > 1.0e-10 || m.largest_dual_err > 1.0e-10
                fstat = internal_factorize!(m)
                if fstat != 0
                    m.problem_status = CLP_STATUS_ERROR
                    break
                end
                factorized_this_outer = true
                guts_of_solution!(m; do_primals = true, do_duals = true, check_infeas = true, algorithm = -1)
            end
        else
            # Save Dual Steepest weights before refactorization reorders pivot_var
            # (for outer_iters == 1, the pre-create_rim! snapshot already has the exact pairing)
            if outer_iters > 1 && weights_initialized
                @inbounds for i in 1:nrows
                    saved_weights_val[i] = m.drs_weights[i]
                    saved_weights_idx[i] = m.pivot_var[i]
                end
            end

            fstat = internal_factorize!(m)
            if fstat != 0
                m.problem_status = CLP_STATUS_ERROR
                break
            end
            factorized_this_outer = true
            guts_of_solution!(m; do_primals = true, do_duals = true, check_infeas = true, algorithm = -1)
        end

        # Ensure dual feasibility (Phase 1 fake bounds if needed)
        if m.num_dual_infeas > 0
            nc, nf = _sm_dual_make_feasible!(m, dual_bound)
            number_changed += nc
            number_fake += nf
            if nc > 0
                guts_of_solution!(m; do_primals = true, do_duals = false, check_infeas = true, algorithm = -1)
            end
        end

        # Restore or initialize Dual Steepest weights
        if !weights_initialized
            _sm_drs_init_weights!(m, exact_weights)
            weights_initialized = true
        elseif factorized_this_outer
            clp_drs_restore_weights!(
                m.drs_weights, saved_weights_val, saved_weights_idx,
                saved_back, m.pivot_var, nrows, ntotal, 1, nrows,
            )
        end
        m.drs_weights_stale = false
        _sm_drs_refresh_infeas!(m)

        # Check termination after fresh factorization
        if m.num_primal_infeas == 0 && m.drs_infeas_nnz == 0
            if number_fake > 0
                # Restore true bounds and re-check
                _sm_restore_true_bounds!(m)
                # Reset any nonbasic variable that was sitting on a fake upper/lower bound
                _sm_reset_fake_bounds!(m)
                number_fake = 0
                if is_perturbed
                    _sm_dual_unperturb!(m, false)
                    is_perturbed = false
                    perturbation_done = true
                end
                guts_of_solution!(m; do_primals = true, do_duals = true, check_infeas = true, algorithm = -1)
                if m.num_primal_infeas == 0 && m.num_dual_infeas == 0
                    m.problem_status = CLP_STATUS_OPTIMAL
                    break
                elseif m.num_dual_infeas > 0 && m.num_primal_infeas == 0
                    # Primal feasible & dual infeasible -> finish with primal simplex!
                    force_primal_cleanup = true
                    break
                elseif fake_bumps < 2 && dual_bound < 1.0e15
                    fake_bumps += 1
                    dual_bound = min(1.0e15, dual_bound * 100.0)
                    continue
                else
                    # Fake bounds repeatedly blocked optimality / left primal infeasibility -> finish with primal!
                    force_primal_cleanup = true
                    break
                end
            elseif is_perturbed
                _sm_dual_unperturb!(m, true)
                is_perturbed = false
                perturbation_done = true
                if m.num_primal_infeas == 0 && m.num_dual_infeas == 0
                    m.problem_status = CLP_STATUS_OPTIMAL
                    break
                elseif m.num_primal_infeas == 0 && m.num_dual_infeas > 0
                    force_primal_cleanup = true
                    break
                else
                    continue
                end
            elseif m.num_dual_infeas == 0
                m.problem_status = CLP_STATUS_OPTIMAL
                break
            else
                # Primal feasible but dual infeasible -> switch to primal cleanup
                force_primal_cleanup = true
                break
            end
        end

        if m.number_iterations >= m.max_iterations
            m.problem_status = CLP_STATUS_MAX_ITERATIONS
            break
        end

        # Activate dual perturbation if configured and triggered
        if !is_perturbed && !perturbation_done && perturbation != 100
            if perturbation == 102
                if m.num_primal_infeas > 0
                    _sm_dual_perturb!(m, perturbation)
                    is_perturbed = true
                end
            elseif perturbation < 100
                if outer_iters == 1
                    prev_iter = m.number_iterations
                    prev_obj = m.objective_value
                    prev_outer_iter = m.number_iterations
                    if m.num_primal_infeas > 0 && _sm_should_perturb_init!(m)
                        _sm_dual_perturb!(m, perturbation)
                        is_perturbed = true
                    end
                else
                    obj_tol = 1.0e-9 * max(1.0, abs(m.objective_value))
                    if abs(m.objective_value - prev_obj) > obj_tol
                        prev_iter = m.number_iterations
                        prev_obj = m.objective_value
                    end
                    stalled_zero = (m.number_iterations == prev_outer_iter)
                    stall_count = stalled_zero ? (stall_count + 1) : 0
                    prev_outer_iter = m.number_iterations
                    if m.num_primal_infeas > 0 && (
                        degen_trigger ||
                        (m.number_iterations - prev_iter >= 10 && abs(m.objective_value - prev_obj) <= obj_tol) ||
                        m.force_factorization > 0 ||
                        stall_count >= 2
                    )
                        _sm_dual_perturb!(m, perturbation)
                        is_perturbed = true
                        degen_trigger = false
                    end
                end
            end
        end

        # Run pure-Julia inner dual loop!
        m.problem_status = -1
        ret_code = -1
        drs_alt_nnz = 0
        iter_before_inner = m.number_iterations
        max_zero_step = (!is_perturbed && !perturbation_done && perturbation < 100) ? 20 : 0
        dl_exit, prob_st, ret_code, n_iters, chg_made, prog_flg, mso,
        first_free, last_bad, ffac, pbad, number_fake, number_changed, n_pinf,
        drs_inf_nnz, drs_alt_nnz, _, _,
        pivot_row, seq_in, seq_out, dir_in, dir_out,
        val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
        theta, alpha, alpha_acc, obj_val,
        spare_int0, spare_int1, spare_dbl0, acc_piv, _, rng_seed = clp_dual_while_iterating!(
            m.cf,
            m.mat_starts, m.mat_lengths, m.mat_indices, m.mat_elements,
            m.mat_starts, m.mat_indices, elems,
            m.has_scale, false, rs_v, cs_v, ics_v,
            m.has_row_copy, m.rc_starts, m.rc_cols, m.rc_elements,
            m.pivot_var, m.solution, m.lower, m.upper, m.cost, m.dj, m.status,
            m.col_lower, m.col_upper, m.row_lower, m.row_upper,
            m.drs_weights, m.drs_infeas_val, m.drs_infeas_idx, m.drs_infeas_nnz,
            drs_alt_val, drs_alt_idx, drs_alt_nnz, m.drs_mode,
            m.prog_in, m.prog_out, m.prog_way,
            m.r0_val, m.r0_idx, m.r1_val, m.r1_idx, m.r2_val, m.r2_idx, m.r3_val, m.r3_idx,
            m.c0_val, m.c0_idx, m.c1_val, m.c1_idx,
            m.marked_buf, m.work_col,
            nrows, ncols,
            m.primal_tol, m.dual_tol, m.zero_tol, dual_bound, m.large_value,
            m.acceptable_pivot, m.obj_scale, m.rhs_scale,
            m.largest_primal_err, m.largest_dual_err, m.sum_dual_infeas,
            m.max_iterations, m.special_options, 1, 0,
            m.problem_status, ret_code, m.number_iterations,
            m.change_made, m.progress_flag, m.more_special_options,
            first_free, m.last_bad_iter, m.force_factorization,
            m.prog_bad_times, number_fake, number_changed, m.num_primal_infeas,
            pivot_row, seq_in, seq_out, dir_in, dir_out,
            val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
            theta, alpha, m.alpha_accuracy, m.objective_value,
            spare_int0, spare_int1, spare_dbl0, spare_dbl2, spare_dbl3, m.rng_seed,
            max_zero_step,
        )

        degen_trigger = (ret_code == -4)

        m.problem_status = prob_st
        m.number_iterations = n_iters
        m.change_made = chg_made
        m.progress_flag = prog_flg
        m.more_special_options = mso
        m.last_bad_iter = last_bad
        m.force_factorization = ffac
        m.prog_bad_times = pbad
        m.num_primal_infeas = n_pinf
        m.drs_infeas_nnz = drs_inf_nnz
        m.alpha_accuracy = alpha_acc
        m.objective_value = obj_val
        m.acceptable_pivot = abs(acc_piv)
        m.rng_seed = rng_seed

        if dl_exit == CLP_DL_EXIT_HIT_MAX || m.number_iterations >= m.max_iterations
            m.problem_status = CLP_STATUS_MAX_ITERATIONS
            break
        elseif dl_exit == CLP_DL_EXIT_SEQ_IN_NEG
            # Dual unbounded -> primal infeasible (confirm after unflagging & lowering acceptable_pivot)
            n_unf = _sm_clear_flags!(m.status)
            if m.cf.lu.npivots == 0 && n_unf == 0
                if m.acceptable_pivot > 1.0e-12
                    m.acceptable_pivot = max(1.0e-12, m.acceptable_pivot * 0.01)
                    m.cf.lu.pivot_tol = min(0.99, max(m.cf.lu.pivot_tol * 2.0, 0.5))
                else
                    infeas_confirm += 1
                    if infeas_confirm >= 2
                        if is_perturbed
                            _sm_dual_unperturb!(m, true)
                            is_perturbed = false
                            perturbation_done = true
                            infeas_confirm = 0
                            m.problem_status = -1
                            continue
                        end
                        if number_fake > 0
                            # Fake bounds may have caused primal infeasibility -> verify with primal!
                            m.problem_status = -1
                            m.num_primal_infeas = 0
                            m.num_dual_infeas = 1
                            force_primal_cleanup = true
                            break
                        end
                        m.problem_status = CLP_STATUS_PRIMAL_INFEAS
                        break
                    end
                end
            else
                infeas_confirm = 0
                if n_unf > 0 || m.cf.lu.npivots > 0
                    m.problem_status = -1
                end
            end
        elseif dl_exit == CLP_DL_EXIT_PIVOT_ROW_NEG
            infeas_confirm = 0
            _sm_clear_flags!(m.status)
            if m.drs_infeas_nnz == 0 && m.num_primal_infeas > 0 && m.cf.lu.npivots == 0
                if is_perturbed
                    _sm_dual_unperturb!(m, true)
                    is_perturbed = false
                    perturbation_done = true
                    m.problem_status = -1
                    continue
                end
                if number_fake > 0
                    force_primal_cleanup = true
                else
                    m.problem_status = CLP_STATUS_PRIMAL_INFEAS
                end
                break
            end
            # Loop back to refactorize and verify optimality / restore fake bounds
        elseif dl_exit == CLP_DL_EXIT_FALLBACK_ITER
            infeas_confirm = 0
            first_free = 0
            _sm_clear_flags!(m.status)
        else
            infeas_confirm = 0
            # CLP_DL_EXIT_DONE: refactorize and continue
        end

        if m.number_iterations == iter_before_inner
            if factorized_this_outer || m.cf.lu.npivots == 0
                zero_piv_streak += 1
                if zero_piv_streak >= 1 && m.problem_status < 0
                    if zero_piv_streak == 1
                        m.acceptable_pivot = max(1.0e-12, m.acceptable_pivot * 0.01)
                        m.cf.lu.pivot_tol = min(0.99, max(m.cf.lu.pivot_tol * 2.0, 0.5))
                        if is_perturbed
                            _sm_dual_unperturb!(m, true)
                            is_perturbed = false
                            perturbation_done = true
                            continue
                        end
                    elseif zero_piv_streak == 2
                        m.acceptable_pivot = max(1.0e-12, m.acceptable_pivot * 0.01)
                        if is_perturbed
                            _sm_dual_unperturb!(m, true)
                            is_perturbed = false
                            perturbation_done = true
                            continue
                        end
                    else
                        if is_perturbed
                            _sm_dual_unperturb!(m, true)
                            is_perturbed = false
                            perturbation_done = true
                            continue
                        elseif number_fake > 0 || m.num_primal_infeas > 0 || m.num_dual_infeas > 0 || m.drs_infeas_nnz > 0
                            force_primal_cleanup = true
                            break
                        end
                    end
                end
            end
        else
            zero_piv_streak = 0
        end
    end

    # Guarantee unperturbed costs and exact unperturbed duals before delete_rim!/primal!
    if is_perturbed
        _sm_dual_unperturb!(m, m.cf.lu.nrows > 0 && m.cf.lu.status == 0)
        is_perturbed = false
        perturbation_done = true
        if m.problem_status == CLP_STATUS_OPTIMAL && m.num_dual_infeas > 0 && m.num_primal_infeas == 0
            m.problem_status = -1
            force_primal_cleanup = true
        end
    end

    # If dual exited with primal-feasible (or fake-bounded) basis needing primal cleanup,
    # reset any nonbasic variable sitting on an infinite bound before delete_rim!/primal!
    need_primal_cleanup = force_primal_cleanup || (m.problem_status == -1 && m.num_primal_infeas == 0 && m.num_dual_infeas > 0)
    if !need_primal_cleanup && m.problem_status < 0
        m.problem_status = CLP_STATUS_MAX_ITERATIONS
    end
    if number_fake > 0
        _sm_restore_true_bounds!(m)
        _sm_reset_fake_bounds!(m)
    end
    @inbounds for seq in 1:ntotal
        m.status[seq] &= ~UInt8(24)
    end
    if need_primal_cleanup
        _sm_reset_fake_bounds!(m)
    elseif number_fake > 0 && m.cf.lu.nrows == nrows && m.cf.lu.status == 0
        guts_of_solution!(m; do_primals = true, do_duals = true, check_infeas = true, algorithm = -1)
    end
    delete_rim!(m)
    if need_primal_cleanup
        prev_iters = m.number_iterations
        rem_iters = max(0, max_iterations - prev_iters)
        primal!(m; max_iterations = rem_iters, warm_start = true)
        m.number_iterations += prev_iters
    elseif m.cf.lu.nrows == nrows && m.cf.lu.ncols == nrows && m.cf.lu.status == 0 && m.fact_hash != UInt64(0)
        m.basis_hash = _sm_basis_hash(m)
    end
    return m.problem_status
end

# ==============================================================================
# Primal Simplex helpers (NonLinearCost Phase 1/2 + Primal Column Steepest init)
# ==============================================================================

function _sm_nlc_check_infeasibilities!(m::ClpSimplexModel, tol::Float64)::Bool
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols
    w = m.nlc_infeas_weight
    dbl_max = floatmax(Float64)

    n_infeas = 0
    sum_infeas = 0.0
    cost_changed = false

    @inbounds for seq in 1:ntotal
        st = _sb_get_status(m.status[seq])
        ist = m.nlc_status[seq]
        old_w = _nlc_orig_status(ist)
        lo = m.lower[seq]
        hi = m.upper[seq]
        orig_c = m.nlc_cost2[seq]

        # Restore original bounds if previously shifted by NonLinearCost
        if old_w == CLP_NLC_BELOW_LOWER
            lo = hi
            hi = m.nlc_bound[seq]
        elseif old_w == CLP_NLC_ABOVE_UPPER
            hi = lo
            lo = m.nlc_bound[seq]
        end

        new_w = CLP_NLC_FEASIBLE
        new_c = orig_c
        if st == CLP_STATUS_BASIC
            val = m.solution[seq]
            if val < lo - tol
                new_w = CLP_NLC_BELOW_LOWER
                m.nlc_bound[seq] = hi
                m.lower[seq] = -dbl_max
                m.upper[seq] = lo
                new_c = orig_c - w
                n_infeas += 1
                sum_infeas += (lo - val)
            elseif val > hi + tol
                new_w = CLP_NLC_ABOVE_UPPER
                m.nlc_bound[seq] = lo
                m.lower[seq] = hi
                m.upper[seq] = dbl_max
                new_c = orig_c + w
                n_infeas += 1
                sum_infeas += (val - hi)
            else
                m.lower[seq] = lo
                m.upper[seq] = hi
            end
        else
            m.lower[seq] = lo
            m.upper[seq] = hi
        end

        if m.cost[seq] != new_c
            m.cost[seq] = new_c
            cost_changed = true
        end
        m.nlc_status[seq] = UInt8((CLP_NLC_SAME << 4) | new_w)
    end

    m.nlc_num_infeas = n_infeas
    m.nlc_sum_infeas = sum_infeas
    return cost_changed
end

function _sm_nlc_restore_all!(m::ClpSimplexModel)::Nothing
    ntotal = m.nrows + m.ncols
    @inbounds for seq in 1:ntotal
        ist = m.nlc_status[seq]
        old_w = _nlc_orig_status(ist)
        if old_w == CLP_NLC_BELOW_LOWER
            m.lower[seq] = m.upper[seq]
            m.upper[seq] = m.nlc_bound[seq]
        elseif old_w == CLP_NLC_ABOVE_UPPER
            m.upper[seq] = m.lower[seq]
            m.lower[seq] = m.nlc_bound[seq]
        end
        m.cost[seq] = m.nlc_cost2[seq]
        m.nlc_status[seq] = UInt8((CLP_NLC_SAME << 4) | CLP_NLC_FEASIBLE)
    end
    m.nlc_num_infeas = 0
    m.nlc_sum_infeas = 0.0
    return nothing
end

function _sm_pcs_init_weights!(m::ClpSimplexModel)::Nothing
    ntotal = m.nrows + m.ncols
    fill!(m.pcs_weights, 1.0)
    fill!(m.pcs_ref, UInt32(0))
    @inbounds for seq in 1:ntotal
        if _sb_get_status(m.status[seq]) != CLP_STATUS_BASIC
            _pcs_set_ref!(m.pcs_ref, seq, true)
        end
    end
    m.pcs_devex = 1.0
    m.pcs_pivot_seq = 0
    m.pcs_alt_nnz = 0
    fill!(m.pcs_alt_val, 0.0)
    return nothing
end

function _sm_pcs_refresh_infeas!(m::ClpSimplexModel)::Nothing
    ntotal = m.nrows + m.ncols
    tol = m.dual_tol + min(1.0e-2, m.largest_dual_err)
    m.pcs_infeas_nnz = clp_pcs_redo_infeasibilities!(
        m.pcs_infeas_val, m.pcs_infeas_idx, m.dj, m.status, ntotal, tol,
    )
    return nothing
end

# Handle any_updates == 2 fallback (multiple cost changes in packed r1) in pure Julia
function _sm_primal_update_djs_multi!(m::ClpSimplexModel, r1_nnz::Int)::Nothing
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols
    elems = _sm_active_elements(m)
    dj_rows = @view m.dj[(ncols + 1):ntotal]
    dj_cols = @view m.dj[1:ncols]
    tol = m.dual_tol + min(1.0e-2, m.largest_dual_err)

    r1_nnz = clp_fact_update_column_transpose!(
        m.cf, m.r3_val, m.r3_idx, 0, false, m.r1_val, m.r1_idx, r1_nnz, true,
    )
    c0_nnz = _pl_transpose_times_btran!(
        m.mat_starts, m.mat_indices, elems, false, Float64[], Float64[],
        m.has_row_copy, m.rc_starts, m.rc_cols, m.rc_elements,
        m.r1_val, m.r1_idx, r1_nnz, m.r3_val, m.c0_val, m.c0_idx, m.c1_idx,
        m.marked_buf, m.work_col, m.status, nrows, ncols, m.zero_tol,
    )
    m.pcs_infeas_nnz = clp_pcs_just_djs_section!(
        m.r1_val, m.r1_idx, r1_nnz, dj_rows, m.status,
        m.pcs_infeas_val, m.pcs_infeas_idx, m.pcs_infeas_nnz,
        ncols, CLP_PRIMAL_SLACK_MULT, tol,
    )
    m.pcs_infeas_nnz = clp_pcs_just_djs_section!(
        m.c0_val, m.c0_idx, c0_nnz, dj_cols, m.status,
        m.pcs_infeas_val, m.pcs_infeas_idx, m.pcs_infeas_nnz,
        0, 1.0, tol,
    )
    _pl_clear_packed!(m.r1_val, r1_nnz)
    _pl_clear_packed!(m.c0_val, c0_nnz)
    return nothing
end

"""
    primal!(m::ClpSimplexModel; max_iterations=m.max_iterations, warm_start=false,
            scaling=m.scaling_flag) -> Int

Solve `m` using the standalone pure-Julia Primal Simplex method (`clp_primal_while_iterating!`).
Returns `m.problem_status`: `0` (optimal), `1` (primal infeasible), `2` (primal unbounded),
`3` (max iterations), `4` (error).
"""
function primal!(
    m::ClpSimplexModel;
    max_iterations::Int = m.max_iterations,
    warm_start::Bool = false,
    scaling::Int = m.scaling_flag,
)::Int
    nrows = m.nrows
    ncols = m.ncols
    ntotal = nrows + ncols

    if !_sm_bounds_feasible(m)
        m.number_iterations = 0
        m.problem_status = CLP_STATUS_PRIMAL_INFEAS
        m.secondary_status = 0
        fill!(m.dual_row_sol, 0.0)
        fill!(m.reduced_cost, 0.0)
        return m.problem_status
    end

    if nrows == 0
        return dual!(m; max_iterations = max_iterations, warm_start = warm_start, scaling = scaling)
    end

    m.max_iterations = max_iterations
    m.number_iterations = 0
    m.problem_status = -1
    m.secondary_status = 0
    m.force_factorization = -1
    m.last_bad_iter = -1
    m.last_good_iter = 0
    m.change_made = 0
    m.progress_flag = 0
    m.alpha_accuracy = -1.0
    m.acceptable_pivot = 1.0e-9
    m.drs_weights_stale = true
    m.nlc_infeas_weight = m.infeas_cost
    m.nlc_avg_theta = 1.0
    fill!(m.prog_in, Int32(-1))
    fill!(m.prog_out, Int32(-1))
    fill!(m.prog_way, Int8(0))
    m.prog_bad_times = 0
    _sm_clear_flags!(m.status)

    if !warm_start
        scale_model!(m; mode = scaling)
        m.has_row_copy = false
        all_slack_basis!(m)
    elseif !m.has_row_copy
        build_row_copy!(m)
    end
    create_rim!(m)

    elems = _sm_active_elements(m)
    rs_v = m.has_scale ? m.row_scale : _SM_EMPTY_F64
    cs_v = m.has_scale ? m.col_scale : _SM_EMPTY_F64

    _sm_pcs_init_weights!(m)
    m.pcs_mode = 3
    m.pcs_num_switched = 0

    pivot_row = 0; seq_in = 0; seq_out = 0; dir_in = 0; dir_out = 0
    val_in = 0.0; lo_in = 0.0; hi_in = 0.0; dual_in = 0.0
    val_out = 0.0; lo_out = 0.0; hi_out = 0.0; dual_out = 0.0
    theta = 0.0; alpha = 0.0
    r1_nnz = 0

    outer_iters = 0
    max_outer = min(2000, max(200, max_iterations))
    infeas_bumps = 0
    unbounded_confirm = 0
    stalled_unf = 0
    last_unf_iter = -1

    while outer_iters < max_outer
        outer_iters += 1
        _sm_clear_workspaces!(m)
        r1_nnz = 0
        m.pcs_pivot_seq = 0

        fstat = internal_factorize!(m)
        if fstat != 0
            m.problem_status = CLP_STATUS_ERROR
            break
        end

        compute_primals!(m)
        _sm_nlc_check_infeasibilities!(m, m.primal_tol)
        guts_of_solution!(m; do_primals = false, do_duals = true, check_infeas = true, algorithm = 1)
        _sm_pcs_refresh_infeas!(m)

        # Check termination after fresh factorization
        if m.pcs_infeas_nnz == 0 || m.num_dual_infeas == 0
            n_unf = _sm_clear_flags!(m.status)
            if n_unf > 0
                if m.number_iterations == last_unf_iter
                    stalled_unf += 1
                else
                    stalled_unf = 1
                    last_unf_iter = m.number_iterations
                end
                if stalled_unf >= 2
                    if m.nlc_num_infeas > 0
                        if infeas_bumps < 2 && m.nlc_infeas_weight < 1.0e13
                            infeas_bumps += 1
                            m.nlc_infeas_weight *= 100.0
                            stalled_unf = 0
                            continue
                        else
                            m.problem_status = CLP_STATUS_PRIMAL_INFEAS
                            break
                        end
                    elseif m.num_primal_infeas > 0
                        m.problem_status = CLP_STATUS_PRIMAL_INFEAS
                        break
                    elseif unbounded_confirm >= 1
                        m.problem_status = CLP_STATUS_DUAL_INFEAS
                        break
                    end
                end
                _sm_pcs_refresh_infeas!(m)
            else
                stalled_unf = 0
            end
            if m.pcs_infeas_nnz == 0
                if m.nlc_num_infeas == 0 && m.num_primal_infeas == 0
                    m.problem_status = CLP_STATUS_OPTIMAL
                    break
                elseif m.nlc_num_infeas > 0
                    if infeas_bumps < 2 && m.nlc_infeas_weight < 1.0e13
                        infeas_bumps += 1
                        m.nlc_infeas_weight *= 100.0
                        continue
                    else
                        m.problem_status = CLP_STATUS_PRIMAL_INFEAS
                        break
                    end
                else
                    m.problem_status = CLP_STATUS_PRIMAL_INFEAS
                    break
                end
            end
        end

        if m.number_iterations >= m.max_iterations
            m.problem_status = CLP_STATUS_MAX_ITERATIONS
            break
        end

        # Inner primal loop (re-enter directly on any_updates == 2 fallback without refactoring!)
        prev_outer_iters = m.number_iterations
        m.problem_status = -1
        ret_code = -1
        pcs_size_f = m.cf.lu.len_l + m.cf.lu.len_u
        inner_reentries = 0

        while m.problem_status == -1 && inner_reentries < 500
            inner_reentries += 1
            pl_exit, ret_code, prob_st, n_iters, chg_made, prog_flg, last_bad, ffac,
            pbad, nlc_ninf, nlc_cc, nlc_at, r1_nnz, infeas_nnz, alt_nnz, pcs_pseq, pcs_devex, niw,
            pivot_row, seq_in, seq_out, dir_in, dir_out,
            val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
            theta, alpha, alpha_acc, obj_val,
            ptol, acc_piv, mso, rng_seed = clp_primal_while_iterating!(
                m.cf,
                m.mat_starts, m.mat_lengths, m.mat_indices, m.mat_elements,
                m.mat_starts, m.mat_indices, elems,
                m.has_scale, false, rs_v, cs_v,
                m.has_row_copy, m.rc_starts, m.rc_cols, m.rc_elements,
                m.pivot_var, m.solution, m.lower, m.upper, m.cost, m.dj, m.status,
                m.nlc_status, m.nlc_bound, m.nlc_cost2,
                m.pcs_weights, m.pcs_ref, true,
                m.pcs_infeas_val, m.pcs_infeas_idx, m.pcs_infeas_nnz,
                m.pcs_alt_val, m.pcs_alt_idx, m.pcs_alt_nnz,
                m.pcs_mode, m.pcs_num_switched, pcs_size_f, m.pcs_pivot_seq, m.pcs_devex,
                m.prog_in, m.prog_out, m.prog_way,
                m.r1_val, m.r1_idx, r1_nnz,
                m.r2_val, m.r2_idx, m.r3_val, m.r3_idx,
                m.c0_val, m.c0_idx, m.c1_val, m.c1_idx,
                m.marked_buf, m.work_col,
                nrows, ncols,
                m.primal_tol, m.dual_tol, m.zero_tol, m.infeas_cost, m.large_value,
                m.obj_scale, m.rhs_scale, m.largest_dual_err,
                m.nlc_infeas_weight, m.nlc_sum_infeas,
                m.max_iterations, m.special_options, 1, m.last_good_iter,
                m.problem_status, ret_code, m.number_iterations,
                m.change_made, m.progress_flag, m.last_bad_iter,
                m.force_factorization, m.prog_bad_times,
                m.nlc_num_infeas, m.nlc_change_cost, m.nlc_avg_theta, m.num_primal_infeas,
                pivot_row, seq_in, seq_out, dir_in, dir_out,
                val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                theta, alpha, m.alpha_accuracy, m.objective_value,
                m.acceptable_pivot, m.more_special_options, m.rng_seed,
            )

            m.problem_status = prob_st
            m.number_iterations = n_iters
            m.change_made = chg_made
            m.progress_flag = prog_flg
            m.last_bad_iter = last_bad
            m.force_factorization = ffac
            m.prog_bad_times = pbad
            m.nlc_num_infeas = nlc_ninf
            m.nlc_change_cost = nlc_cc
            m.nlc_avg_theta = nlc_at
            m.pcs_infeas_nnz = infeas_nnz
            m.pcs_alt_nnz = alt_nnz
            m.pcs_pivot_seq = pcs_pseq
            m.pcs_devex = pcs_devex
            m.alpha_accuracy = alpha_acc
            m.objective_value = obj_val
            m.primal_tol = ptol
            m.acceptable_pivot = acc_piv
            m.more_special_options = mso
            m.rng_seed = rng_seed

            if pl_exit == CLP_PL_EXIT_FALLBACK_ITER
                # any_updates == 2: update duals from multi-entry r1 and re-enter!
                if r1_nnz > 0
                    _sm_primal_update_djs_multi!(m, r1_nnz)
                    r1_nnz = 0
                end
                # Update outgoing weight/infeas and continue inner loop
                if m.pcs_pivot_seq > 0
                    # Leave pcs_pivot_seq > 0 so next iteration runs justSteepest!
                end
                continue
            elseif pl_exit == CLP_PL_EXIT_SAVE_DJ_CHECK_INF
                if m.cf.lu.npivots == 0 && seq_in > 0
                    @inbounds m.status[seq_in] |= CLP_STATUS_FLAGGED_BIT
                end
                m.problem_status = -2
                break
            elseif pl_exit == CLP_PL_EXIT_SAVE_DJ_FLAG_W
                m.cf.lu.pivot_tol = min(0.995, max(m.cf.lu.pivot_tol * 1.1, 0.1))
                if m.cf.lu.npivots == 0 && seq_in > 0
                    @inbounds m.status[seq_in] |= CLP_STATUS_FLAGGED_BIT
                end
                m.problem_status = -2
                break
            elseif pl_exit == CLP_PL_EXIT_UNBOUNDED_RAY
                if m.cf.lu.npivots == 0
                    unbounded_confirm += 1
                    if unbounded_confirm >= 2
                        m.problem_status = CLP_STATUS_DUAL_INFEAS
                        break
                    end
                end
                m.problem_status = -2
                break
            else
                # CLP_PL_EXIT_DONE
                if niw != 0
                    _sm_pcs_init_weights!(m)
                end
                if m.problem_status == -5
                    if m.cf.lu.npivots == 0 && m.nlc_num_infeas == 0
                        unbounded_confirm += 1
                        if unbounded_confirm >= 2
                            m.problem_status = CLP_STATUS_DUAL_INFEAS
                            break
                        end
                    end
                    m.problem_status = -2
                end
                break
            end
        end

        if m.number_iterations == prev_outer_iters && m.problem_status == -2 && seq_in > 0
            @inbounds m.status[seq_in] |= CLP_STATUS_FLAGGED_BIT
        end
        if m.problem_status == CLP_STATUS_DUAL_INFEAS ||
           m.problem_status == CLP_STATUS_MAX_ITERATIONS ||
           m.number_iterations >= m.max_iterations
            if m.number_iterations >= m.max_iterations && m.problem_status != CLP_STATUS_DUAL_INFEAS
                m.problem_status = CLP_STATUS_MAX_ITERATIONS
            end
            break
        end
    end

    if m.problem_status < 0
        if m.num_primal_infeas == 0 && m.num_dual_infeas == 0 && m.nlc_num_infeas == 0
            m.problem_status = CLP_STATUS_OPTIMAL
        else
            m.problem_status = CLP_STATUS_MAX_ITERATIONS
        end
    end

    recompute_duals = (m.nlc_num_infeas > 0 || m.problem_status != CLP_STATUS_OPTIMAL)
    _sm_nlc_restore_all!(m)
    if recompute_duals && m.cf.lu.nrows > 0 && m.cf.lu.status == 0
        compute_duals!(m)
    end
    delete_rim!(m)
    if m.cf.lu.nrows == nrows && m.cf.lu.ncols == nrows && m.cf.lu.status == 0 && m.fact_hash != UInt64(0)
        m.basis_hash = _sm_basis_hash(m)
    end
    return m.problem_status
end

# Pure idiomatic 1-based Julia implementation of ClpSimplex autonomous kernels (B11c-1).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Compatible with JuliaC --trim=safe and works both on native Vector{T} and bridge views.

@inline _sb_fmadd(a::Float64, b::Float64, c::Float64) = fma(a, b, c)
@inline _sb_fmsub(a::Float64, b::Float64, c::Float64) = fma(-a, b, c)
@inline _sb_get_status(st::UInt8) = st & CLP_STATUS_MASK
@inline _sb_set_status(st::UInt8, new_st::UInt8) = (st & ~CLP_STATUS_MASK) | new_st
@inline _sb_flagged(st::UInt8) = (st & CLP_STATUS_FLAGGED_BIT) != UInt8(0)

"""
    clp_clean_status!(status, row_act, row_lo, row_hi, col_act, col_lo, col_hi, nrows, ncols, primal_tol)

Matches the status-cleaning loops of `ClpSimplex::cleanStatus()` (lines 1505–1558),
called after `row_act` has been recomputed as `A * col_act`.
"""
function clp_clean_status!(
    status::AbstractVector{UInt8},
    row_act::AbstractVector{Float64},
    row_lo::AbstractVector{Float64},
    row_hi::AbstractVector{Float64},
    col_act::AbstractVector{Float64},
    col_lo::AbstractVector{Float64},
    col_hi::AbstractVector{Float64},
    nrows::Int,
    ncols::Int,
    primal_tol::Float64,
)::Nothing
    n_basic = 0
    @inbounds for i in 1:nrows
        seq = ncols + i
        st = status[seq]
        if _sb_get_status(st) == CLP_STATUS_BASIC
            n_basic += 1
        else
            ra = row_act[i]
            rl = row_lo[i]
            ru = row_hi[i]
            if abs(ra - rl) <= primal_tol
                row_act[i] = rl
                status[seq] = _sb_set_status(st, CLP_STATUS_AT_LOWER)
            elseif abs(ra - ru) <= primal_tol
                row_act[i] = ru
                status[seq] = _sb_set_status(st, CLP_STATUS_AT_UPPER)
            else
                status[seq] = _sb_set_status(st, CLP_STATUS_SUPER_BASIC)
            end
        end
    end
    @inbounds for j in 1:ncols
        st = status[j]
        if _sb_get_status(st) == CLP_STATUS_BASIC
            if n_basic == nrows
                ca = col_act[j]
                cl = col_lo[j]
                cu = col_hi[j]
                if abs(ca - cl) <= primal_tol
                    col_act[j] = cl
                    status[j] = _sb_set_status(st, CLP_STATUS_AT_LOWER)
                elseif abs(ca - cu) <= primal_tol
                    col_act[j] = cu
                    status[j] = _sb_set_status(st, CLP_STATUS_AT_UPPER)
                else
                    status[j] = _sb_set_status(st, CLP_STATUS_SUPER_BASIC)
                end
            else
                n_basic += 1
            end
        elseif col_lo[j] > -1.0e30 || col_hi[j] < 1.0e30
            ca = col_act[j]
            cl = col_lo[j]
            cu = col_hi[j]
            if abs(ca - cl) <= primal_tol
                col_act[j] = cl
                status[j] = _sb_set_status(st, CLP_STATUS_AT_LOWER)
            elseif abs(ca - cu) <= primal_tol
                col_act[j] = cu
                status[j] = _sb_set_status(st, CLP_STATUS_AT_UPPER)
            else
                status[j] = _sb_set_status(st, CLP_STATUS_SUPER_BASIC)
            end
        else
            status[j] = _sb_set_status(st, CLP_STATUS_IS_FREE)
        end
    end
    return nothing
end

"""
    clp_check_primal_solution!(...)

Matches `ClpSimplex::checkPrimalSolution()` (lines 2995–3075) for standard matrices
(no dynamic row expansion). Returns `(objective_value, sum_primal_infeas,
sum_relaxed_primal_infeas, num_primal_infeas)`.
"""
function clp_check_primal_solution(
    row_act::AbstractVector{Float64},
    row_obj::AbstractVector{Float64},
    row_lo::AbstractVector{Float64},
    row_hi::AbstractVector{Float64},
    col_act::AbstractVector{Float64},
    col_obj::AbstractVector{Float64},
    col_lo::AbstractVector{Float64},
    col_hi::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    nrows::Int,
    ncols::Int,
    primal_tol::Float64,
    largest_primal_err::Float64,
    has_rhs_offset::Bool,
    nonlinear_offset::Float64,
    obj_scale::Float64,
    rhs_scale::Float64,
)::Tuple{Float64,Float64,Float64,Int}
    obj_val = 0.0
    sum_inf = 0.0
    sum_rel = 0.0
    num_inf = 0
    err = min(1.0e-2, largest_primal_err)
    rel_tol = primal_tol + err

    @inbounds for i in 1:nrows
        sol = row_act[i]
        obj_val = _sb_fmadd(sol, row_obj[i], obj_val)
        inf = 0.0
        if sol > row_hi[i]
            inf = sol - row_hi[i]
        elseif sol < row_lo[i]
            inf = row_lo[i] - sol
        end
        if inf > primal_tol
            sum_inf += inf - primal_tol
            if inf > rel_tol
                sum_rel += inf - rel_tol
            end
            num_inf += 1
        end
    end

    if !has_rhs_offset
        @inbounds for j in 1:ncols
            sol = col_act[j]
            obj_val = _sb_fmadd(col_obj[j], sol, obj_val)
            inf = 0.0
            if sol > col_hi[j]
                inf = sol - col_hi[j]
            elseif sol < col_lo[j]
                inf = col_lo[j] - sol
            end
            if inf > primal_tol
                sum_inf += inf - primal_tol
                if inf > rel_tol
                    sum_rel += inf - rel_tol
                end
                num_inf += 1
            end
        end
    else
        ip = 0.0
        @inbounds for j in 1:ncols
            ip = _sb_fmadd(col_obj[j], col_act[j], ip)
        end
        obj_val += ip
        @inbounds for i in 1:nrows
            j = Int(pivot_var[i])
            sol = col_act[j]
            inf = 0.0
            if sol > col_hi[j]
                inf = sol - col_hi[j]
            elseif sol < col_lo[j]
                inf = col_lo[j] - sol
            end
            if inf > primal_tol
                sum_inf += inf - primal_tol
                if inf > rel_tol
                    sum_rel += inf - rel_tol
                end
                num_inf += 1
            end
        end
    end

    obj_val += nonlinear_offset
    obj_val /= (obj_scale * rhs_scale)
    return (obj_val, sum_inf, sum_rel, num_inf)
end

"""
    clp_check_dual_solution(...)

Matches `ClpSimplex::checkDualSolution()` (lines 3076–3230) for standard matrices
(after any `matrix_->dualExpanded` call).
Returns `(objective_value, sum_dual_infeas, sum_relaxed_dual_infeas,
          num_dual_infeas, num_dual_infeas_no_free, best_possible_improvement,
          first_free_1based)` where `first_free_1based == 0` means C++ `-1`.
"""
function clp_check_dual_solution(
    col_act::AbstractVector{Float64},
    col_obj::AbstractVector{Float64},
    col_lo::AbstractVector{Float64},
    col_hi::AbstractVector{Float64},
    col_dj::AbstractVector{Float64},
    row_act::AbstractVector{Float64},
    row_obj::AbstractVector{Float64},
    row_lo::AbstractVector{Float64},
    row_hi::AbstractVector{Float64},
    row_dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
    primal_tol::Float64,
    dual_tol::Float64,
    largest_dual_err::Float64,
    algorithm::Int,
    first_free_in::Int,
    last_iter_num0::Int,
    nonlinear_offset::Float64,
    obj_scale::Float64,
    rhs_scale::Float64,
)::Tuple{Float64,Float64,Float64,Int,Int,Float64,Int}
    sum_dual_inf = 0.0
    num_dual_inf = 0
    num_dual_inf_no_free = 0
    first_free_primal = 0
    first_free_dual = 0
    num_sb_with_dj = 0
    best_imp = 0.0
    err = min(1.0e-2, largest_dual_err)
    rel_tol = dual_tol + err
    poss_tol = 5.0 * rel_tol
    sum_rel_dual_inf = 0.0
    obj_val = 0.0

    @inbounds for j in 1:ncols
        sol = col_act[j]
        obj_val = _sb_fmadd(col_obj[j], sol, obj_val)
        st = status[j]
        st_code = _sb_get_status(st)
        if st_code != CLP_STATUS_BASIC && !_sb_flagged(st)
            dist_up = col_hi[j] - sol
            dist_dn = sol - col_lo[j]
            if dist_up > primal_tol
                val = col_dj[j]
                if dist_dn > primal_tol
                    if abs(val) > 1.0e2 * rel_tol
                        num_sb_with_dj += 1
                        if first_free_dual == 0
                            first_free_dual = j
                        end
                    end
                    if first_free_primal == 0
                        first_free_primal = j
                    end
                end
                if val < 0.0
                    val = -val
                    if val > dual_tol
                        if st_code != CLP_STATUS_IS_FREE
                            num_dual_inf_no_free += 1
                            sum_dual_inf += val - dual_tol
                            if val > poss_tol
                                best_imp = _sb_fmadd(min(dist_up, 1.0e10), val, best_imp)
                            end
                            if val > rel_tol
                                sum_rel_dual_inf += val - rel_tol
                            end
                            num_dual_inf += 1
                        else
                            val *= 0.01
                            if val > dual_tol
                                sum_dual_inf += val - dual_tol
                                if val > poss_tol
                                    best_imp = 1.0e100
                                end
                                if val > rel_tol
                                    sum_rel_dual_inf += val - rel_tol
                                end
                                num_dual_inf += 1
                            end
                        end
                    end
                end
            end
            if dist_dn > primal_tol
                val = col_dj[j]
                if val > 0.0 && val > dual_tol
                    sum_dual_inf += val - dual_tol
                    if val > poss_tol
                        best_imp = _sb_fmadd(val, min(dist_dn, 1.0e10), best_imp)
                    end
                    if val > rel_tol
                        sum_rel_dual_inf += val - rel_tol
                    end
                    num_dual_inf += 1
                    if st_code != CLP_STATUS_IS_FREE
                        num_dual_inf_no_free += 1
                    end
                end
            end
        end
    end

    @inbounds for i in 1:nrows
        sol = row_act[i]
        obj_val = _sb_fmadd(sol, row_obj[i], obj_val)
        seq = ncols + i
        st = status[seq]
        st_code = _sb_get_status(st)
        if st_code != CLP_STATUS_BASIC && !_sb_flagged(st)
            dist_up = row_hi[i] - sol
            dist_dn = sol - row_lo[i]
            if dist_up > primal_tol
                val = row_dj[i]
                if dist_dn > primal_tol
                    if abs(val) > 1.0e2 * rel_tol
                        num_sb_with_dj += 1
                        if first_free_dual == 0
                            first_free_dual = seq
                        end
                    end
                    if first_free_primal == 0
                        first_free_primal = seq
                    end
                end
                if val < 0.0
                    val = -val
                    if val > dual_tol
                        sum_dual_inf += val - dual_tol
                        if val > poss_tol
                            best_imp = _sb_fmadd(val, min(dist_up, 1.0e10), best_imp)
                        end
                        if val > rel_tol
                            sum_rel_dual_inf += val - rel_tol
                        end
                        num_dual_inf += 1
                        if st_code != CLP_STATUS_IS_FREE
                            num_dual_inf_no_free += 1
                        end
                    end
                end
            end
            if dist_dn > primal_tol
                val = row_dj[i]
                if val > 0.0 && val > dual_tol
                    sum_dual_inf += val - dual_tol
                    if val > poss_tol
                        best_imp = _sb_fmadd(val, min(dist_dn, 1.0e10), best_imp)
                    end
                    if val > rel_tol
                        sum_rel_dual_inf += val - rel_tol
                    end
                    num_dual_inf += 1
                    if st_code != CLP_STATUS_IS_FREE
                        num_dual_inf_no_free += 1
                    end
                end
            end
        end
    end

    first_free_out = first_free_in
    if algorithm < 0 && first_free_dual > 0
        first_free_out = first_free_dual
    elseif num_sb_with_dj != 0 || last_iter_num0 <= 0
        first_free_out = first_free_primal
    end
    obj_val += nonlinear_offset
    obj_val /= (obj_scale * rhs_scale)
    return (obj_val, sum_dual_inf, sum_rel_dual_inf, num_dual_inf, num_dual_inf_no_free,
            best_imp, first_free_out)
end

"""
    clp_check_both_solutions(...)

Matches the unified fast loop of `ClpSimplex::checkBothSolutions()` (lines 3242–3429)
when `!((skipDualCheck && algorithm > 0 && problemStatus == -2) || rhsOffset)`.
Returns 11-tuple:
`(objective_value, sum_primal_infeas, sum_relaxed_primal_infeas, num_primal_infeas,
  sum_dual_infeas, sum_relaxed_dual_infeas, num_dual_infeas, num_dual_infeas_no_free,
  best_possible_improvement, more_special_options, first_free_1based)`.
"""
function clp_check_both_solutions(
    solution::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    ntotal::Int,
    primal_tol::Float64,
    dual_tol::Float64,
    largest_primal_err::Float64,
    largest_dual_err::Float64,
    algorithm::Int,
    more_special_options::Int,
    first_free_in::Int,
    last_iter_num0::Int,
    nonlinear_offset::Float64,
    obj_scale::Float64,
    rhs_scale::Float64,
)::Tuple{Float64,Float64,Float64,Int,Float64,Float64,Int,Int,Float64,Int,Int}
    obj_val = 0.0
    sum_p_inf = 0.0
    num_p_inf = 0
    err_p = min(1.0e-2, max(largest_primal_err, 0.0))
    rel_tol_p = primal_tol + err_p
    sum_rel_p_inf = 0.0

    sum_d_inf = 0.0
    num_d_inf = 0
    err_d = min(1.0e-2, max(largest_dual_err, 5.0 * dual_tol))
    rel_tol_d = dual_tol + err_d
    poss_tol = 5.0 * rel_tol_d
    sum_rel_d_inf = 0.0
    best_imp = 0.0

    num_d_inf_free = 0
    first_free_primal = 0
    first_free_dual = 0
    num_sb_with_dj = 0

    mso = more_special_options | 8

    @inbounds for seq in 1:ntotal
        val = solution[seq]
        obj_val = _sb_fmadd(val, cost[seq], obj_val)
        dist_up = upper[seq] - val
        dist_dn = val - lower[seq]
        if dist_up < -primal_tol
            inf = -dist_up
            if _sb_get_status(status[seq]) != CLP_STATUS_BASIC
                mso &= ~8
            end
            sum_p_inf += inf - primal_tol
            if inf > rel_tol_p
                sum_rel_p_inf += inf - rel_tol_p
            end
            num_p_inf += 1
        elseif dist_dn < -primal_tol
            inf = -dist_dn
            if _sb_get_status(status[seq]) != CLP_STATUS_BASIC
                mso &= ~8
            end
            sum_p_inf += inf - primal_tol
            if inf > rel_tol_p
                sum_rel_p_inf += inf - rel_tol_p
            end
            num_p_inf += 1
        else
            st = status[seq]
            st_code = _sb_get_status(st)
            if st_code != CLP_STATUS_BASIC && !_sb_flagged(st)
                dj_val = dj[seq]
                if dist_dn < primal_tol
                    if dist_up > primal_tol && dj_val < -dual_tol
                        sum_d_inf -= dj_val + dual_tol
                        if dj_val < -poss_tol
                            best_imp = _sb_fmsub(dist_up, dj_val, best_imp)
                        end
                        if dj_val < -rel_tol_d
                            sum_rel_d_inf -= dj_val + rel_tol_d
                        end
                        num_d_inf += 1
                    end
                elseif dist_up < primal_tol
                    if dj_val > dual_tol
                        sum_d_inf += dj_val - dual_tol
                        if dj_val > poss_tol
                            best_imp = _sb_fmadd(dist_dn, dj_val, best_imp)
                        end
                        if dj_val > rel_tol_d
                            sum_rel_d_inf += dj_val - rel_tol_d
                        end
                        num_d_inf += 1
                    end
                else
                    mso &= ~8
                    dj_val *= 100.0
                    adj = abs(dj_val)
                    if adj > dual_tol
                        if st_code == CLP_STATUS_IS_FREE
                            num_d_inf_free += 1
                        end
                        sum_d_inf += adj - dual_tol
                        best_imp = 1.0e100
                        num_d_inf += 1
                        if adj > rel_tol_d
                            # Exact ClpSimplex.cpp:3364 behavior: uses `val`, not `adj`!
                            sum_rel_d_inf += val - rel_tol_d
                            num_sb_with_dj += 1
                            if first_free_dual == 0
                                first_free_dual = seq
                            end
                            if first_free_primal == 0
                                first_free_primal = seq
                            end
                        end
                    elseif st_code == CLP_STATUS_SUPER_BASIC && first_free_primal == 0
                        first_free_primal = seq
                    end
                end
            end
        end
    end

    obj_val += nonlinear_offset
    obj_val /= (obj_scale * rhs_scale)
    num_d_inf_no_free = num_d_inf - num_d_inf_free

    first_free_out = first_free_in
    if algorithm < 0 && first_free_dual > 0
        first_free_out = first_free_dual
    elseif num_sb_with_dj != 0 || last_iter_num0 <= 0
        first_free_out = first_free_primal
    end

    return (obj_val, sum_p_inf, sum_rel_p_inf, num_p_inf,
            sum_d_inf, sum_rel_d_inf, num_d_inf, num_d_inf_no_free,
            best_imp, mso, first_free_out)
end

"""
    clp_all_slack_basis!(col_act, status, col_lo, col_hi, ncols, solution, inv_col_scale, rhs_scale)

Matches the `resetSolution == true` body of `ClpSimplex::allSlackBasis` (lines 7845–7876).
Pass `solution === nothing` if `solution_` is NULL, and `inv_col_scale === nothing` if unscaled.
"""
function clp_all_slack_basis!(
    col_act::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    col_lo::AbstractVector{Float64},
    col_hi::AbstractVector{Float64},
    ncols::Int,
    solution::Union{AbstractVector{Float64},Nothing},
    inv_col_scale::Union{AbstractVector{Float64},Nothing},
    rhs_scale::Float64,
)::Nothing
    @inbounds for j in 1:ncols
        lo = col_lo[j]
        hi = col_hi[j]
        if lo >= 0.0
            col_act[j] = lo
            status[j] = _sb_set_status(status[j], CLP_STATUS_AT_LOWER)
        elseif hi <= 0.0
            col_act[j] = hi
            status[j] = _sb_set_status(status[j], CLP_STATUS_AT_UPPER)
        elseif lo < -1.0e20 && hi > 1.0e20
            col_act[j] = 0.0
            status[j] = _sb_set_status(status[j], CLP_STATUS_IS_FREE)
        elseif abs(lo) < abs(hi)
            col_act[j] = 0.0
            status[j] = _sb_set_status(status[j], CLP_STATUS_AT_LOWER)
        else
            col_act[j] = 0.0
            status[j] = _sb_set_status(status[j], CLP_STATUS_AT_UPPER)
        end
    end
    if solution !== nothing
        if inv_col_scale === nothing
            @inbounds for j in 1:ncols
                solution[j] = col_act[j]
            end
        else
            @inbounds for j in 1:ncols
                solution[j] = col_act[j] * (rhs_scale * inv_col_scale[j])
            end
        end
    end
    return nothing
end

"""
    clp_check_solution_internal!(...)

Matches `ClpSimplex::checkSolutionInternal()` (lines 11037–11191).
Returns `(objective_value, sum_primal_infeas, num_primal_infeas,
          sum_dual_infeas, num_dual_infeas, problem_status)`.
"""
function clp_check_solution_internal!(
    status::AbstractVector{UInt8},
    dual::AbstractVector{Float64},
    row_act::AbstractVector{Float64},
    row_lo::AbstractVector{Float64},
    row_hi::AbstractVector{Float64},
    reduced_cost::AbstractVector{Float64},
    col_act::AbstractVector{Float64},
    col_lo::AbstractVector{Float64},
    col_hi::AbstractVector{Float64},
    objective::AbstractVector{Float64},
    nrows::Int,
    ncols::Int,
    primal_tol::Float64,
    dual_tol::Float64,
    dual_tol_work::Float64,
    opt_dir::Float64,
    grad_nonlinear_offset::Float64,
    obj_nonlinear_offset::Float64,
)::Tuple{Float64,Float64,Int,Float64,Int,Int}
    obj_val = -grad_nonlinear_offset
    sum_p_inf = 0.0
    num_p_inf = 0
    sum_d_inf = 0.0
    num_d_inf = 0

    @inbounds for i in 1:nrows
        dval = dual[i] * opt_dir
        pval = row_act[i]
        lo = row_lo[i]
        hi = row_hi[i]
        seq = ncols + i
        st = _sb_get_status(status[seq])
        if st != CLP_STATUS_BASIC
            if lo == hi
                st = CLP_STATUS_IS_FIXED
            elseif pval > hi - primal_tol
                st = CLP_STATUS_AT_UPPER
            elseif pval < lo + primal_tol
                st = CLP_STATUS_AT_LOWER
            end
            status[seq] = _sb_set_status(status[seq], st)
        end
        if pval > hi + primal_tol
            sum_p_inf += pval - hi - primal_tol
            num_p_inf += 1
        elseif pval < lo - primal_tol
            sum_p_inf += lo - pval - primal_tol
            num_p_inf += 1
        else
            if st == CLP_STATUS_AT_UPPER
                if dval > dual_tol
                    sum_d_inf += dval - dual_tol_work
                    num_d_inf += 1
                end
            elseif st == CLP_STATUS_AT_LOWER
                if dval < -dual_tol
                    sum_d_inf -= dval + dual_tol_work
                    num_d_inf += 1
                end
            elseif st == CLP_STATUS_SUPER_BASIC || st == CLP_STATUS_IS_FREE
                if pval < hi - primal_tol && dval < -dual_tol
                    sum_d_inf -= dval + dual_tol_work
                    num_d_inf += 1
                end
                if pval > lo + primal_tol && dval > dual_tol
                    sum_d_inf += dval - dual_tol_work
                    num_d_inf += 1
                end
            end
        end
    end

    @inbounds for j in 1:ncols
        dval = reduced_cost[j] * opt_dir
        pval = col_act[j]
        obj_val = _sb_fmadd(objective[j], pval, obj_val)
        lo = col_lo[j]
        hi = col_hi[j]
        st = _sb_get_status(status[j])
        if st != CLP_STATUS_BASIC && lo == hi
            st = CLP_STATUS_IS_FIXED
            status[j] = _sb_set_status(status[j], CLP_STATUS_IS_FIXED)
        end
        if pval > hi + primal_tol
            sum_p_inf += pval - hi - primal_tol
            num_p_inf += 1
        elseif pval < lo - primal_tol
            sum_p_inf += lo - pval - primal_tol
            num_p_inf += 1
        else
            if st == CLP_STATUS_BASIC
                if abs(dval) > 10.0 * dual_tol
                    sum_d_inf += abs(dval) - dual_tol_work
                    num_d_inf += 1
                end
            elseif st == CLP_STATUS_AT_UPPER
                if dval > dual_tol
                    sum_d_inf += dval - dual_tol_work
                    num_d_inf += 1
                end
            elseif st == CLP_STATUS_AT_LOWER
                if dval < -dual_tol
                    sum_d_inf -= dval + dual_tol_work
                    num_d_inf += 1
                end
            elseif st == CLP_STATUS_SUPER_BASIC || st == CLP_STATUS_IS_FREE
                if pval < hi - primal_tol && dval < -dual_tol
                    sum_d_inf -= dval + dual_tol_work
                    num_d_inf += 1
                end
                if pval > lo + primal_tol && dval > dual_tol
                    sum_d_inf += dval - dual_tol_work
                    num_d_inf += 1
                end
            end
        end
    end

    obj_val += obj_nonlinear_offset
    obj_val *= opt_dir
    prob_status = (num_d_inf == 0 && num_p_inf == 0) ? 0 : -1
    return (obj_val, sum_p_inf, num_p_inf, sum_d_inf, num_d_inf, prob_status)
end

# ==============================================================================
# Phase B — B11c-2: createRim1, createRim4, computePrimals/computeDuals kernels
# ==============================================================================

const CLP_STATUS_PIVOTED_BIT = UInt8(32)
@inline _sb_pivoted(st::UInt8) = (st & CLP_STATUS_PIVOTED_BIT) != UInt8(0)

@inline function _sb_scale_bound_pair(
    lo::Float64,
    hi::Float64,
    mult::Float64,
    primal_tol::Float64,
    dbl_max::Float64,
)::Tuple{Float64,Float64}
    if lo > -1.0e20
        lo_w = lo * mult
        if hi >= 1.0e20
            hi_w = dbl_max
        else
            hi_w = hi * mult
            if abs(hi_w - lo_w) <= primal_tol
                if lo_w >= 0.0
                    hi_w = lo_w
                elseif hi_w <= 0.0
                    lo_w = hi_w
                else
                    hi_w = 0.0
                    lo_w = 0.0
                end
            end
        end
        return (lo_w, hi_w)
    elseif hi < 1.0e20
        return (-dbl_max, hi * mult)
    else
        return (-dbl_max, dbl_max)
    end
end

@inline function _sb_unscaled_bound_pair(
    lo::Float64,
    hi::Float64,
    primal_tol::Float64,
    dbl_max::Float64,
)::Tuple{Float64,Float64}
    if lo > -1.0e20
        lo_w = lo
        if hi >= 1.0e20
            hi_w = dbl_max
        else
            hi_w = hi
            if abs(hi_w - lo_w) <= primal_tol
                if lo_w >= 0.0
                    hi_w = lo_w
                elseif hi_w <= 0.0
                    lo_w = hi_w
                else
                    hi_w = 0.0
                    lo_w = 0.0
                end
            end
        end
        return (lo_w, hi_w)
    elseif hi < 1.0e20
        return (-dbl_max, hi)
    else
        return (-dbl_max, dbl_max)
    end
end

"""
    clp_create_rim1!(...)

Matches `ClpSimplex::createRim1(bool initial)` (lines 4511–4721).
"""
function clp_create_rim1!(
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    col_lo::AbstractVector{Float64},
    col_hi::AbstractVector{Float64},
    row_lo::AbstractVector{Float64},
    row_hi::AbstractVector{Float64},
    row_scale::Union{AbstractVector{Float64},Nothing},
    inv_col_scale::Union{AbstractVector{Float64},Nothing},
    nrows::Int,
    nrows_extra::Int,
    ncols::Int,
    max_rows::Int,
    max_cols::Int,
    special_options::Int,
    initial::Bool,
    rhs_scale::Float64,
    primal_tol::Float64,
)::Nothing
    ntotal = nrows + nrows_extra + ncols
    if (special_options & 65536) != 0
        save = max_cols + max_rows
        @inbounds for i in 1:ntotal
            lower[i] = lower[save + i]
            upper[i] = upper[save + i]
        end
        return nothing
    end
    dbl_max = floatmax(Float64)
    if row_scale !== nothing
        if !initial && inv_col_scale !== nothing
            @inbounds for i in 1:ncols
                mult = rhs_scale * inv_col_scale[i]
                lo_w, hi_w = _sb_scale_bound_pair(col_lo[i], col_hi[i], mult, primal_tol, dbl_max)
                lower[i] = lo_w
                upper[i] = hi_w
            end
        end
        @inbounds for i in 1:nrows
            mult = rhs_scale * row_scale[i]
            lo_w, hi_w = _sb_scale_bound_pair(row_lo[i], row_hi[i], mult, primal_tol, dbl_max)
            lower[ncols + i] = lo_w
            upper[ncols + i] = hi_w
        end
    elseif rhs_scale != 1.0
        @inbounds for i in 1:ncols
            lo_w, hi_w = _sb_scale_bound_pair(col_lo[i], col_hi[i], rhs_scale, primal_tol, dbl_max)
            lower[i] = lo_w
            upper[i] = hi_w
        end
        @inbounds for i in 1:nrows
            lo_w, hi_w = _sb_scale_bound_pair(row_lo[i], row_hi[i], rhs_scale, primal_tol, dbl_max)
            lower[ncols + i] = lo_w
            upper[ncols + i] = hi_w
        end
    else
        @inbounds for i in 1:ncols
            lo_w, hi_w = _sb_unscaled_bound_pair(col_lo[i], col_hi[i], primal_tol, dbl_max)
            lower[i] = lo_w
            upper[i] = hi_w
        end
        @inbounds for i in 1:nrows
            lo_w, hi_w = _sb_unscaled_bound_pair(row_lo[i], row_hi[i], primal_tol, dbl_max)
            lower[ncols + i] = lo_w
            upper[ncols + i] = hi_w
        end
    end
    return nothing
end

"""
    clp_create_rim4!(...)

Matches `ClpSimplex::createRim4(bool initial)` (lines 4723–4765).
"""
function clp_create_rim4!(
    cost::AbstractVector{Float64},
    obj::AbstractVector{Float64},
    row_obj::Union{AbstractVector{Float64},Nothing},
    row_scale::Union{AbstractVector{Float64},Nothing},
    col_scale::Union{AbstractVector{Float64},Nothing},
    nrows::Int,
    nrows_extra::Int,
    ncols::Int,
    max_rows::Int,
    max_cols::Int,
    special_options::Int,
    initial::Bool,
    direction::Float64,
)::Nothing
    ntotal = nrows + nrows_extra + ncols
    if (special_options & 65536) != 0
        save = max_cols + max_rows
        @inbounds for i in 1:ntotal
            cost[i] = cost[save + i]
        end
        return nothing
    end
    if row_scale !== nothing
        if row_obj !== nothing
            @inbounds for i in 1:nrows
                cost[ncols + i] = row_obj[i] * direction / row_scale[i]
            end
        else
            @inbounds for i in 1:nrows
                cost[ncols + i] = 0.0
            end
        end
        if !initial && col_scale !== nothing
            @inbounds for i in 1:ncols
                cost[i] = obj[i] * direction * col_scale[i]
            end
        end
    else
        if row_obj !== nothing
            @inbounds for i in 1:nrows
                cost[ncols + i] = row_obj[i] * direction
            end
        else
            @inbounds for i in 1:nrows
                cost[ncols + i] = 0.0
            end
        end
        @inbounds for i in 1:ncols
            cost[i] = obj[i] * direction
        end
    end
    return nothing
end

# --- computePrimals sub-kernels ---

function clp_zero_basic_solution!(
    solution::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    nrows::Int,
)::Nothing
    @inbounds for i in 1:nrows
        solution[Int(pivot_var[i])] = 0.0
    end
    return nothing
end

function clp_pack_sum_nonzero!(
    dst_val::AbstractVector{Float64},
    dst_idx::AbstractVector{Int32},
    a::AbstractVector{Float64},
    b::AbstractVector{Float64},
    nrows::Int,
)::Int
    nnz = 0
    @inbounds for i in 1:nrows
        val = a[i] + b[i]
        if val != 0.0
            dst_val[i] = val
            nnz += 1
            dst_idx[nnz] = Int32(i)
        else
            dst_val[i] = 0.0
        end
    end
    return nnz
end

function clp_scatter_basic_solution!(
    solution::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    src_val::AbstractVector{Float64},
    src_idx::AbstractVector{Int32},
    src_nnz::Int,
    nrows::Int,
    dense_scatter::Bool,
)::Nothing
    if !dense_scatter
        @inbounds for j in 1:src_nnz
            irow = Int(src_idx[j])
            solution[Int(pivot_var[irow])] = src_val[irow]
        end
    else
        @inbounds for irow in 1:nrows
            solution[Int(pivot_var[irow])] = src_val[irow]
        end
    end
    return nothing
end

function clp_primal_residual_scale!(
    work::AbstractVector{Float64},
    row_act::AbstractVector{Float64},
    nrows::Int,
    multiplier::Float64,
)::Float64
    max_err = 0.0
    @inbounds for i in 1:nrows
        val = work[i] + row_act[i]
        work[i] = val * multiplier
        av = abs(val)
        if av > max_err
            max_err = av
        end
    end
    return max_err
end

function clp_refine_prepare!(
    dst_val::AbstractVector{Float64},
    dst_idx::AbstractVector{Int32},
    work::AbstractVector{Float64},
    nrows::Int,
    multiplier::Float64,
)::Int
    nnz = 0
    if multiplier == 1.0
        @inbounds for i in 1:nrows
            val = work[i]
            if val != 0.0
                dst_val[i] = val
                nnz += 1
                dst_idx[nnz] = Int32(i)
                work[i] = 0.0
            end
        end
    else
        @inbounds for i in 1:nrows
            val = multiplier * work[i]
            if val != 0.0
                dst_val[i] = val
                nnz += 1
                dst_idx[nnz] = Int32(i)
            end
            work[i] = 0.0
        end
    end
    return nnz
end

function clp_refine_combine!(
    arr_val::AbstractVector{Float64},
    arr_idx::AbstractVector{Int32},
    prev_val::AbstractVector{Float64},
    nrows::Int,
    inv_multiplier::Float64,
)::Int
    nnz = 0
    @inbounds for i in 1:nrows
        val = _sb_fmadd(inv_multiplier, arr_val[i], prev_val[i])
        if val != 0.0
            arr_val[i] = val
            nnz += 1
            arr_idx[nnz] = Int32(i)
        else
            arr_val[i] = 0.0
        end
    end
    return nnz
end

# --- computeDuals sub-kernels ---

function clp_compute_duals_setup!(
    arr_val::AbstractVector{Float64},
    arr_idx::AbstractVector{Int32},
    cost::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    given_djs::Union{AbstractVector{Float64},Nothing},
    status::AbstractVector{UInt8},
    nrows::Int,
)::Int
    nnz = 0
    if given_djs === nothing
        @inbounds for i in 1:nrows
            ipiv = Int(pivot_var[i])
            val = cost[ipiv]
            if val != 0.0
                arr_val[i] = val
                nnz += 1
                arr_idx[nnz] = Int32(i)
            end
        end
    else
        @inbounds for i in 1:nrows
            ipiv = Int(pivot_var[i])
            if !_sb_pivoted(status[ipiv])
                given_djs[ipiv] = 0.0
            end
            val = cost[ipiv] - given_djs[ipiv]
            if val != 0.0
                arr_val[i] = val
                nnz += 1
                arr_idx[nnz] = Int32(i)
            end
        end
    end
    return nnz
end

function clp_compute_duals_collect_structurals!(
    idx_out::AbstractVector{Int32},
    pivot_var::AbstractVector{Int32},
    nrows::Int,
    ncols::Int,
)::Int
    n = 0
    @inbounds for i in 1:nrows
        ipiv = Int(pivot_var[i])
        if ipiv <= ncols
            n += 1
            idx_out[n] = Int32(ipiv)
        end
    end
    return n
end

function clp_compute_duals_residual!(
    work::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    arr_val::AbstractVector{Float64},
    row_obj::AbstractVector{Float64},
    col_obj::AbstractVector{Float64},
    work2::AbstractVector{Float64},
    given_djs::Union{AbstractVector{Float64},Nothing},
    nrows::Int,
    ncols::Int,
)::Float64
    max_err = 0.0
    ns = 0
    if given_djs === nothing
        @inbounds for i in 1:nrows
            ipiv = Int(pivot_var[i])
            if ipiv > ncols
                r = ipiv - ncols
                val = row_obj[r] + arr_val[r]
            else
                ns += 1
                val = col_obj[ipiv] - work2[ns]
            end
            work[i] = val
            av = abs(val)
            if av > max_err
                max_err = av
            end
        end
    else
        @inbounds for i in 1:nrows
            ipiv = Int(pivot_var[i])
            if ipiv > ncols
                r = ipiv - ncols
                val = row_obj[r] + arr_val[r] - given_djs[ipiv]
            else
                ns += 1
                val = col_obj[ipiv] - work2[ns] - given_djs[ipiv]
            end
            work[i] = val
            av = abs(val)
            if av > max_err
                max_err = av
            end
        end
    end
    return max_err
end

function clp_compute_duals_finalize_prep!(
    dual::AbstractVector{Float64},
    row_dj::AbstractVector{Float64},
    col_dj::AbstractVector{Float64},
    which_col::Union{AbstractVector{Int32},Nothing},
    arr_val::AbstractVector{Float64},
    row_obj::AbstractVector{Float64},
    col_obj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
)::Int
    @inbounds for i in 1:nrows
        val = arr_val[i]
        dual[i] = val
        row_dj[i] = val + row_obj[i]
    end
    if which_col === nothing
        @inbounds for j in 1:ncols
            col_dj[j] = col_obj[j]
        end
        return ncols
    else
        n = 0
        @inbounds for j in 1:ncols
            if _sb_get_status(status[j]) != CLP_STATUS_BASIC
                n += 1
                which_col[n] = Int32(j)
                col_dj[j] = col_obj[j]
            else
                col_dj[j] = 0.0
            end
        end
        return n
    end
end

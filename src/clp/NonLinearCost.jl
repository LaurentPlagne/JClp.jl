# Pure idiomatic 1-based Julia implementation of ClpNonLinearCost (B9c, METHOD2 LP core).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Compatible with JuliaC --trim=safe and works both on native Vector{T} and bridge views.

const CLP_NLC_BELOW_LOWER = UInt8(0)
const CLP_NLC_FEASIBLE    = UInt8(1)
const CLP_NLC_ABOVE_UPPER = UInt8(2)
const CLP_NLC_SAME        = UInt8(4)

@inline _nlc_orig_status(s::UInt8) = s & UInt8(15)
@inline _nlc_curr_status(s::UInt8) = s >> 4
@inline _nlc_set_orig_status(s::UInt8, v::UInt8) = (s & ~UInt8(15)) | v
@inline _nlc_set_curr_status(s::UInt8, v::UInt8) = (s & ~UInt8(15 << 4)) | (v << 4)
@inline _nlc_set_same_status(s::UInt8) = (s & ~UInt8(15 << 4)) | (CLP_NLC_SAME << 4)
@inline _nlc_set_model_status(st::UInt8, new_st::UInt8) = (st & ~CLP_STATUS_MASK) | new_st

function clp_nlc_feasible_bounds!(
    nlc_status::AbstractVector{UInt8},
    bound::AbstractVector{Float64},
    cost2::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    ntotal::Int,
)::Nothing
    @inbounds for iseq in 1:ntotal
        ist = nlc_status[iseq]
        lo = lower[iseq]
        hi = upper[iseq]
        c = cost2[iseq]
        w = _nlc_orig_status(ist)
        if w == CLP_NLC_BELOW_LOWER
            lo = hi
            hi = bound[iseq]
        elseif w == CLP_NLC_ABOVE_UPPER
            hi = lo
            lo = bound[iseq]
        end
        nlc_status[iseq] = _nlc_set_orig_status(ist, CLP_NLC_FEASIBLE)
        lower[iseq] = lo
        upper[iseq] = hi
        cost[iseq] = c
    end
    return nothing
end

function clp_nlc_go_back_all_m2!(
    nlc_status::AbstractVector{UInt8},
    pivot_var::AbstractVector{Int32},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
)::Nothing
    @inbounds for i in 1:n_upd
        irow = Int(upd_idx[i])
        iseq = Int(pivot_var[irow])
        nlc_status[iseq] = _nlc_set_same_status(nlc_status[iseq])
    end
    return nothing
end

function clp_nlc_check_infeas_subset_m2!(
    nlc_status::AbstractVector{UInt8},
    bound::AbstractVector{Float64},
    cost2::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    row_idx::AbstractVector{Int32},
    n_in_arr::Int,
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    primal_tol::Float64,
    infeas_weight::Float64,
    n_infeas::Int,
)::Int
    dbl_max = floatmax(Float64)
    @inbounds for i in 1:n_in_arr
        irow = Int(row_idx[i])
        iseq = Int(pivot_var[irow])
        val = solution[iseq]
        ist = nlc_status[iseq]
        lo = lower[iseq]
        hi = upper[iseq]
        c = cost2[iseq]
        w = _nlc_orig_status(ist)
        if w == CLP_NLC_BELOW_LOWER
            lo = hi
            hi = bound[iseq]
            n_infeas -= 1
        elseif w == CLP_NLC_ABOVE_UPPER
            hi = lo
            lo = bound[iseq]
            n_infeas -= 1
        end
        new_w = CLP_NLC_FEASIBLE
        if val - hi <= primal_tol
            if val - lo < -primal_tol
                new_w = CLP_NLC_BELOW_LOWER
                c -= infeas_weight
                n_infeas += 1
            end
        else
            new_w = CLP_NLC_ABOVE_UPPER
            c += infeas_weight
            n_infeas += 1
        end
        if w != new_w
            nlc_status[iseq] = _nlc_set_orig_status(ist, new_w)
            if new_w == CLP_NLC_BELOW_LOWER
                bound[iseq] = hi
                hi = lo
                lo = -dbl_max
            elseif new_w == CLP_NLC_ABOVE_UPPER
                bound[iseq] = lo
                lo = hi
                hi = dbl_max
            end
            lower[iseq] = lo
            upper[iseq] = hi
            cost[iseq] = c
        end
    end
    return n_infeas
end

function clp_nlc_check_changed_m2!(
    nlc_status::AbstractVector{UInt8},
    bound::AbstractVector{Float64},
    cost2::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    upd_idx::AbstractVector{Int32},
    upd_work::AbstractVector{Float64},
    n_in_arr::Int,
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    primal_tol::Float64,
    infeas_weight::Float64,
    n_infeas::Int,
)::Tuple{Int,Int}
    dbl_max = floatmax(Float64)
    n_changed = 0
    @inbounds for i in 1:n_in_arr
        irow = Int(upd_idx[i])
        iseq = Int(pivot_var[irow])
        val = solution[iseq]
        ist = nlc_status[iseq]
        lo = lower[iseq]
        hi = upper[iseq]
        c = cost2[iseq]
        w = _nlc_orig_status(ist)
        if w == CLP_NLC_BELOW_LOWER
            lo = hi
            hi = bound[iseq]
            n_infeas -= 1
        elseif w == CLP_NLC_ABOVE_UPPER
            hi = lo
            lo = bound[iseq]
            n_infeas -= 1
        end
        new_w = CLP_NLC_FEASIBLE
        if val - hi <= primal_tol
            if val - lo < -primal_tol
                new_w = CLP_NLC_BELOW_LOWER
                c -= infeas_weight
                n_infeas += 1
            end
        else
            new_w = CLP_NLC_ABOVE_UPPER
            c += infeas_weight
            n_infeas += 1
        end
        if w != new_w
            upd_work[irow] = cost[iseq] - c
            n_changed += 1
            upd_idx[n_changed] = Int32(irow)
            nlc_status[iseq] = _nlc_set_orig_status(ist, new_w)
            if new_w == CLP_NLC_BELOW_LOWER
                bound[iseq] = hi
                hi = lo
                lo = -dbl_max
            elseif new_w == CLP_NLC_ABOVE_UPPER
                bound[iseq] = lo
                lo = hi
                hi = dbl_max
            end
            lower[iseq] = lo
            upper[iseq] = hi
            cost[iseq] = c
        end
    end
    return (n_changed, n_infeas)
end

function clp_nlc_set_one_m2!(
    nlc_status::AbstractVector{UInt8},
    bound::AbstractVector{Float64},
    cost2::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    model_status::AbstractVector{UInt8},
    iseq::Int,
    val::Float64,
    primal_tol::Float64,
    infeas_weight::Float64,
    n_infeas::Int,
)::Tuple{Float64,Int}
    dbl_max = floatmax(Float64)
    diff = 0.0
    @inbounds begin
        ist = nlc_status[iseq]
        lo = lower[iseq]
        hi = upper[iseq]
        c = cost2[iseq]
        w = _nlc_orig_status(ist)
        if w == CLP_NLC_BELOW_LOWER
            lo = hi
            hi = bound[iseq]
            n_infeas -= 1
        elseif w == CLP_NLC_ABOVE_UPPER
            hi = lo
            lo = bound[iseq]
            n_infeas -= 1
        end
        new_w = CLP_NLC_FEASIBLE
        if val - hi <= primal_tol
            if val - lo < -primal_tol
                new_w = CLP_NLC_BELOW_LOWER
                c -= infeas_weight
                n_infeas += 1
            end
        else
            new_w = CLP_NLC_ABOVE_UPPER
            c += infeas_weight
            n_infeas += 1
        end
        if w != new_w
            diff = cost[iseq] - c
            nlc_status[iseq] = _nlc_set_orig_status(ist, new_w)
            if new_w == CLP_NLC_BELOW_LOWER
                bound[iseq] = hi
                hi = lo
                lo = -dbl_max
            elseif new_w == CLP_NLC_ABOVE_UPPER
                bound[iseq] = lo
                lo = hi
                hi = dbl_max
            end
            lower[iseq] = lo
            upper[iseq] = hi
            cost[iseq] = c
        end
        st = model_status[iseq] & CLP_STATUS_MASK
        if hi == lo
            if st != CLP_STATUS_BASIC
                model_status[iseq] = _nlc_set_model_status(model_status[iseq], CLP_STATUS_IS_FIXED)
                st = CLP_STATUS_BASIC
            end
        end
        if st == CLP_STATUS_AT_UPPER || st == CLP_STATUS_AT_LOWER || st == CLP_STATUS_IS_FIXED
            if abs(val - lo) <= primal_tol * 1.001
                model_status[iseq] = _nlc_set_model_status(model_status[iseq], CLP_STATUS_AT_LOWER)
            elseif abs(val - hi) <= primal_tol * 1.001
                model_status[iseq] = _nlc_set_model_status(model_status[iseq], CLP_STATUS_AT_UPPER)
            else
                model_status[iseq] = _nlc_set_model_status(model_status[iseq], CLP_STATUS_SUPER_BASIC)
            end
        end
    end
    return (diff, n_infeas)
end

function clp_nlc_nearest_m2(
    nlc_status::AbstractVector{UInt8},
    bound::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    iseq::Int,
    sol_val::Float64,
)::Float64
    @inbounds begin
        lo = lower[iseq]
        hi = upper[iseq]
        w = _nlc_orig_status(nlc_status[iseq])
        if w == CLP_NLC_BELOW_LOWER
            lo = hi
            hi = bound[iseq]
        elseif w == CLP_NLC_ABOVE_UPPER
            hi = lo
            lo = bound[iseq]
        end
        return abs(sol_val - lo) < abs(sol_val - hi) ? lo : hi
    end
end

function clp_nlc_refresh_one!(
    nlc_status::AbstractVector{UInt8},
    bound::AbstractVector{Float64},
    cost2::AbstractVector{Float64},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    iseq::Int,
    primal_tol::Float64,
    infeas_cost::Float64,
)::Nothing
    dbl_max = floatmax(Float64)
    @inbounds begin
        cost2[iseq] = cost[iseq]
        val = solution[iseq]
        lo = lower[iseq]
        hi = upper[iseq]
        if val - hi <= primal_tol
            if val - lo >= -primal_tol
                nlc_status[iseq] = CLP_NLC_FEASIBLE | (CLP_NLC_SAME << 4)
                bound[iseq] = 0.0
            else
                cost[iseq] -= infeas_cost
                nlc_status[iseq] = CLP_NLC_BELOW_LOWER | (CLP_NLC_SAME << 4)
                bound[iseq] = hi
                upper[iseq] = lo
                lower[iseq] = -dbl_max
            end
        else
            cost[iseq] += infeas_cost
            nlc_status[iseq] = CLP_NLC_ABOVE_UPPER | (CLP_NLC_SAME << 4)
            bound[iseq] = lo
            lower[iseq] = hi
            upper[iseq] = dbl_max
        end
    end
    return nothing
end

@inline function clp_nlc_change_in_cost_m2!(
    nlc_status::AbstractVector{UInt8},
    bound::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    iseq::Int,
    alpha::Float64,
    rhs::Float64,
    infeas_weight::Float64,
)::Tuple{Float64,Float64}
    dbl_max = floatmax(Float64)
    @inbounds begin
        ist = nlc_status[iseq]
        iwhere = _nlc_curr_status(ist)
        if iwhere == CLP_NLC_SAME
            iwhere = _nlc_orig_status(ist)
        end
        if iwhere == CLP_NLC_FEASIBLE
            if alpha > 0.0
                iwhere = CLP_NLC_BELOW_LOWER
                rhs = dbl_max
            else
                iwhere = CLP_NLC_ABOVE_UPPER
                rhs = dbl_max
            end
        elseif iwhere == CLP_NLC_BELOW_LOWER
            iwhere = CLP_NLC_FEASIBLE
            rhs += bound[iseq] - upper[iseq]
        else
            iwhere = CLP_NLC_FEASIBLE
            rhs += lower[iseq] - bound[iseq]
        end
        nlc_status[iseq] = _nlc_set_curr_status(ist, iwhere)
        delta_cost = abs(alpha) * infeas_weight
        return (delta_cost, rhs)
    end
end

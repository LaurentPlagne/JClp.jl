# Pure idiomatic 1-based Julia implementation of ClpDualRowDantzig & ClpDualRowSteepest (B9a).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Compatible with JuliaC --trim=safe and works both on native Vector{T} and bridge views.

const CLP_DEVEX_TRY_NORM          = 1.0e-4
const CLP_DUAL_DANTZIG_COL_MULT   = 1.01

@inline _drp_fmadd(a::Float64, b::Float64, c::Float64) = fma(a, b, c)
@inline _drp_fmsub(a::Float64, b::Float64, c::Float64) = fma(-a, b, c)

# ==============================================================================
# 1. ClpDualRowDantzig
# ==============================================================================

"""
    clp_drd_pivot_row(pivot_var, solution, lower, upper, status, nrows, ncols,
                      primal_tol, largest_primal_err) -> Int

Returns 1-based chosen pivot row, or 0 if none.
"""
function clp_drd_pivot_row(
    pivot_var::AbstractVector{Int32},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
    primal_tol::Float64,
    largest_primal_err::Float64,
)::Int
    tol = largest_primal_err > 1.0e-8 ? primal_tol * (largest_primal_err / 1.0e-8) : primal_tol
    largest = 0.0
    chosen = 0
    @inbounds for irow in 1:nrows
        iseq = Int(pivot_var[irow])
        val = solution[iseq]
        lo = lower[iseq]
        hi = upper[iseq]
        infeas = max(val - hi, lo - val)
        if infeas > tol
            if iseq <= ncols
                infeas *= CLP_DUAL_DANTZIG_COL_MULT
            end
            if infeas > largest
                if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                    chosen = irow
                    largest = infeas
                end
            end
        end
    end
    return chosen
end

"""
    clp_drd_extract_alpha(work, which, nnz, packed, pivot_row) -> Float64
"""
function clp_drd_extract_alpha(
    work::AbstractVector{Float64},
    which::AbstractVector{Int32},
    nnz::Int,
    packed::Bool,
    pivot_row::Int,
)::Float64
    @inbounds if packed
        for i in 1:nnz
            if Int(which[i]) == pivot_row
                return work[i]
            end
        end
        return 0.0
    else
        return work[pivot_row]
    end
end

"""
    clp_drd_update_primal_solution!(work, which, nnz, packed, pivot_var,
                                    solution, cost, primal_ratio) -> Float64

Updates `solution`, zeros `work` on active entries, returns `changeObj` delta.
"""
function clp_drd_update_primal_solution!(
    work::AbstractVector{Float64},
    which::AbstractVector{Int32},
    nnz::Int,
    packed::Bool,
    pivot_var::AbstractVector{Int32},
    solution::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    primal_ratio::Float64,
)::Float64
    change_obj = 0.0
    @inbounds if packed
        for i in 1:nnz
            irow = Int(which[i])
            ipiv = Int(pivot_var[irow])
            chg = primal_ratio * work[i]
            solution[ipiv] -= chg
            change_obj = _drp_fmsub(chg, cost[ipiv], change_obj)
            work[i] = 0.0
        end
    else
        for i in 1:nnz
            irow = Int(which[i])
            ipiv = Int(pivot_var[irow])
            chg = primal_ratio * work[irow]
            solution[ipiv] -= chg
            change_obj = _drp_fmsub(chg, cost[ipiv], change_obj)
            work[irow] = 0.0
        end
    end
    return change_obj
end

# ==============================================================================
# 2. ClpDualRowSteepest
# ==============================================================================

"""
    clp_drs_pivot_row!(infeas, infeas_idx, nnz_in, weights, pivot_var,
                       solution, lower, upper, status,
                       nrows, last_pivot_row, mode, n_factor_elems,
                       primal_tol, largest_primal_err, largest_dual_err,
                       n_iters, last_bad_iter, rand_u) -> (chosen_row, nnz_out, tol_changed)

All row/sequence indices are 1-based (`0` means none / `-1` in C++).
Returns `(chosen_row_1based, new_nnz, tol_changed)`.
"""
function clp_drs_pivot_row!(
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    nnz::Int,
    weights::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    last_pivot_row::Int,
    mode::Int,
    n_factor_elems::Int,
    primal_tol::Float64,
    largest_primal_err::Float64,
    largest_dual_err::Float64,
    n_iters::Int,
    last_bad_iter::Int,
    rand_u::Float64,
)::Tuple{Int,Int,Bool}
    err = min(1.0e-2, largest_primal_err)
    tol = min(1000.0, primal_tol + err)
    tol *= tol
    tol_changed = false

    @inbounds if last_pivot_row >= 1 && last_pivot_row <= nrows
        ipiv = Int(pivot_var[last_pivot_row])
        val = solution[ipiv]
        lo = lower[ipiv]
        hi = upper[ipiv]
        if val > hi + tol
            d = val - hi
            d2 = d * d
            if infeas[last_pivot_row] != 0.0
                infeas[last_pivot_row] = d2
            elseif abs(d2) >= CLP_INDEXED_TINY
                nnz += 1
                infeas_idx[nnz] = Int32(last_pivot_row)
                infeas[last_pivot_row] = d2
            end
        elseif val < lo - tol
            d = val - lo
            d2 = d * d
            if infeas[last_pivot_row] != 0.0
                infeas[last_pivot_row] = d2
            else
                nnz += 1
                infeas_idx[nnz] = Int32(last_pivot_row)
                infeas[last_pivot_row] = d2
            end
        else
            if infeas[last_pivot_row] != 0.0
                infeas[last_pivot_row] = CLP_INDEXED_REALLY_TINY
            end
        end
    end

    if n_iters < last_bad_iter + 200
        if largest_dual_err > largest_primal_err
            tol *= min(largest_dual_err / largest_primal_err, 1000.0)
            tol_changed = true
        end
    end

    n_wanted = if mode < 2
        nnz + 1
    elseif mode == 2
        max(2000, div(nnz, 8))
    else
        ratio = Float64(n_factor_elems) / Float64(nrows)
        nw = max(2000, div(nnz, 8))
        if ratio < 1.0
            nw = max(2000, div(nnz, 20))
        elseif ratio > 10.0
            r2 = Float64(nnz) * (ratio / 80.0)
            nw = r2 > Float64(nnz) ? (nnz + 1) : max(2000, trunc(Int, r2))
        end
        nw
    end
    if largest_primal_err > 1.0e-3
        n_wanted = nnz + 1
    end

    start0 = trunc(Int, Float64(nnz) * rand_u) + 1 # 1-based
    largest = 0.0
    chosen = 0

    @inbounds for pass in 1:2
        s_idx = pass == 1 ? start0 : 1
        e_idx = pass == 1 ? nnz : (start0 - 1)
        for i in s_idx:e_idx
            irow = Int(infeas_idx[i])
            val = infeas[irow]
            if val > tol
                w = min(weights[irow], 1.0e50)
                if val > largest * w
                    if irow == last_pivot_row
                        if val * 1.0e-10 < largest * w
                            continue
                        else
                            val *= 1.0e-10
                        end
                    end
                    iseq = Int(pivot_var[irow])
                    if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                        sol = solution[iseq]
                        if sol > upper[iseq] + tol || sol < lower[iseq] - tol
                            chosen = irow
                            largest = val / w
                        end
                    else
                        n_wanted += 1
                    end
                end
                n_wanted -= 1
                n_wanted == 0 && break
            end
        end
        n_wanted == 0 && break
    end

    return (chosen, nnz, tol_changed)
end

"""
    clp_drs_compact_infeasible!(infeas, infeas_idx, nnz) -> Int
"""
function clp_drs_compact_infeasible!(
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    nnz::Int,
)::Int
    n_left = 0
    @inbounds for i in 1:nnz
        irow = Int(infeas_idx[i])
        if abs(infeas[irow]) > 1.0e-50
            n_left += 1
            infeas_idx[n_left] = Int32(irow)
        else
            infeas[irow] = 0.0
        end
    end
    return n_left
end

"""
    clp_drs_prep_ftran!(in_val, in_idx, nnz, permute, spare_val, spare_idx) -> Float64

Permutes `input` (packed) into `spare` (unpacked) and returns `norm = sum(v^2)`.
"""
function clp_drs_prep_ftran!(
    in_val::AbstractVector{Float64},
    in_idx::AbstractVector{Int32},
    nnz::Int,
    permute::Union{Nothing,AbstractVector{Int32}},
    spare_val::AbstractVector{Float64},
    spare_idx::AbstractVector{Int32},
)::Float64
    norm = 0.0
    @inbounds if permute !== nothing
        p = permute::AbstractVector{Int32}
        for i in 1:nnz
            irow = Int(in_idx[i])
            val = in_val[i]
            norm = _drp_fmadd(val, val, norm)
            prow = Int(p[irow])
            spare_val[prow] = val
            spare_idx[i] = Int32(prow)
        end
    else
        for i in 1:nnz
            irow = Int(in_idx[i])
            val = in_val[i]
            norm = _drp_fmadd(val, val, norm)
            spare_val[irow] = val
            spare_idx[i] = Int32(irow)
        end
    end
    return norm
end

"""
    clp_drs_apply_weight_update!(upd_val, upd_idx, nnz, spare_val, pivot_col,
                                 weights, alt_val, alt_idx,
                                 pivot_row, model_alpha, norm_in, is_network) -> (alpha, n_save)
"""
function clp_drs_apply_weight_update!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    nnz::Int,
    spare_val::AbstractVector{Float64},
    pivot_col::Union{Nothing,AbstractVector{Int32}},
    weights::AbstractVector{Float64},
    alt_val::AbstractVector{Float64},
    alt_idx::AbstractVector{Int32},
    pivot_row::Int,
    model_alpha::Float64,
    norm_in::Float64,
    is_network::Bool,
)::Tuple{Float64,Int}
    norm = norm_in / (model_alpha * model_alpha)
    alpha = 0.0
    mult = 2.0 / model_alpha
    n_save = 0
    @inbounds if pivot_col !== nothing
        pc = pivot_col::AbstractVector{Int32}
        for i in 1:nnz
            irow = Int(upd_idx[i])
            theta = upd_val[i]
            if irow == pivot_row
                alpha = theta
            end
            devex = weights[irow]
            n_save += 1
            alt_val[n_save] = devex
            alt_idx[n_save] = Int32(irow)
            jrow = Int(pc[irow])
            val = spare_val[jrow]
            devex = _drp_fmadd(theta, _drp_fmadd(theta, norm, val * mult), devex)
            if devex < CLP_DEVEX_TRY_NORM
                devex = CLP_DEVEX_TRY_NORM
            end
            weights[irow] = devex
        end
    else
        for i in 1:nnz
            irow = Int(upd_idx[i])
            theta = upd_val[i]
            if irow == pivot_row
                alpha = theta
            end
            devex = weights[irow]
            n_save += 1
            alt_val[n_save] = devex
            alt_idx[n_save] = Int32(irow)
            val = spare_val[irow]
            devex = _drp_fmadd(theta, _drp_fmadd(theta, norm, val * mult), devex)
            if devex < CLP_DEVEX_TRY_NORM
                devex = CLP_DEVEX_TRY_NORM
            end
            weights[irow] = devex
        end
    end
    if is_network && alpha == 0.0
        alpha = 1.0e-50
    end
    if norm < CLP_DEVEX_TRY_NORM
        norm = CLP_DEVEX_TRY_NORM
    end
    @inbounds weights[pivot_row] = norm
    return (alpha, n_save)
end

"""
    clp_drs_update_primal_solution!(work, which, nnz, packed, pivot_var,
                                    infeas, infeas_idx, infeas_nnz,
                                    solution, cost, lower, upper,
                                    primal_ratio, primal_tol, pivot_row) -> (change_obj, infeas_nnz)
"""
function clp_drs_update_primal_solution!(
    work::AbstractVector{Float64},
    which::AbstractVector{Int32},
    nnz::Int,
    packed::Bool,
    pivot_var::AbstractVector{Int32},
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    infeas_nnz::Int,
    solution::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    primal_ratio::Float64,
    primal_tol::Float64,
    pivot_row::Int,
)::Tuple{Float64,Int}
    change_obj = 0.0
    @inbounds if packed
        for i in 1:nnz
            irow = Int(which[i])
            ipiv = Int(pivot_var[irow])
            val = solution[ipiv]
            c = cost[ipiv]
            chg = primal_ratio * work[i]
            work[i] = 0.0
            val -= chg
            change_obj = _drp_fmsub(chg, c, change_obj)
            lo = lower[ipiv]
            hi = upper[ipiv]
            solution[ipiv] = val
            if val < lo - primal_tol
                d = val - lo
                d2 = d * d
                if infeas[irow] != 0.0
                    infeas[irow] = d2
                elseif abs(d2) >= CLP_INDEXED_TINY
                    infeas_nnz += 1
                    infeas_idx[infeas_nnz] = Int32(irow)
                    infeas[irow] = d2
                end
            elseif val > hi + primal_tol
                d = val - hi
                d2 = d * d
                if infeas[irow] != 0.0
                    infeas[irow] = d2
                elseif abs(d2) >= CLP_INDEXED_TINY
                    infeas_nnz += 1
                    infeas_idx[infeas_nnz] = Int32(irow)
                    infeas[irow] = d2
                end
            else
                if infeas[irow] != 0.0
                    infeas[irow] = CLP_INDEXED_REALLY_TINY
                end
            end
        end
    else
        for i in 1:nnz
            irow = Int(which[i])
            ipiv = Int(pivot_var[irow])
            val = solution[ipiv]
            c = cost[ipiv]
            chg = primal_ratio * work[irow]
            val -= chg
            change_obj = _drp_fmsub(chg, c, change_obj)
            lo = lower[ipiv]
            hi = upper[ipiv]
            solution[ipiv] = val
            if val < lo - primal_tol
                d = val - lo
                d2 = d * d
                if infeas[irow] != 0.0
                    infeas[irow] = d2
                elseif abs(d2) >= CLP_INDEXED_TINY
                    infeas_nnz += 1
                    infeas_idx[infeas_nnz] = Int32(irow)
                    infeas[irow] = d2
                end
            elseif val > hi + primal_tol
                d = val - hi
                d2 = d * d
                if infeas[irow] != 0.0
                    infeas[irow] = d2
                elseif abs(d2) >= CLP_INDEXED_TINY
                    infeas_nnz += 1
                    infeas_idx[infeas_nnz] = Int32(irow)
                    infeas[irow] = d2
                end
            else
                if infeas[irow] != 0.0
                    infeas[irow] = CLP_INDEXED_REALLY_TINY
                end
            end
            work[irow] = 0.0
        end
    end
    @inbounds if infeas[pivot_row] != 0.0
        infeas[pivot_row] = CLP_INDEXED_REALLY_TINY
    end
    return (change_obj, infeas_nnz)
end

"""
    clp_drs_unroll_weights!(weights, saved_val, saved_idx, nnz, packed) -> Nothing
"""
function clp_drs_unroll_weights!(
    weights::AbstractVector{Float64},
    saved_val::AbstractVector{Float64},
    saved_idx::AbstractVector{Int32},
    nnz::Int,
    packed::Bool,
)::Nothing
    @inbounds if packed
        for i in 1:nnz
            irow = Int(saved_idx[i])
            weights[irow] = saved_val[i]
            saved_val[i] = 0.0
        end
    else
        for i in 1:nnz
            irow = Int(saved_idx[i])
            weights[irow] = saved_val[irow]
            saved_val[irow] = 0.0
        end
    end
    return nothing
end

"""
    clp_drs_looks_optimal(pivot_var, solution, lower, upper, nrows,
                          primal_tol, largest_primal_err) -> Bool
"""
function clp_drs_looks_optimal(
    pivot_var::AbstractVector{Int32},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    nrows::Int,
    primal_tol::Float64,
    largest_primal_err::Float64,
)::Bool
    err = min(1.0e-2, largest_primal_err)
    tol = min(1000.0, primal_tol + err)
    @inbounds for irow in 1:nrows
        ipiv = Int(pivot_var[irow])
        val = solution[ipiv]
        if val < lower[ipiv] - tol || val > upper[ipiv] + tol
            return false
        end
    end
    return true
end

"""
    clp_drs_recompute_infeasibilities!(infeas, infeas_idx, pivot_var,
                                       solution, lower, upper, nrows, primal_tol) -> Int
"""
function clp_drs_recompute_infeasibilities!(
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    pivot_var::AbstractVector{Int32},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    nrows::Int,
    primal_tol::Float64,
)::Int
    nnz = 0
    @inbounds for irow in 1:nrows
        ipiv = Int(pivot_var[irow])
        val = solution[ipiv]
        lo = lower[ipiv]
        hi = upper[ipiv]
        if val < lo - primal_tol
            d = val - lo
            d2 = d * d
            if abs(d2) >= CLP_INDEXED_TINY
                nnz += 1
                infeas_idx[nnz] = Int32(irow)
                infeas[irow] = d2
            end
        elseif val > hi + primal_tol
            d = val - hi
            d2 = d * d
            if abs(d2) >= CLP_INDEXED_TINY
                nnz += 1
                infeas_idx[nnz] = Int32(irow)
                infeas[irow] = d2
            end
        end
    end
    return nnz
end

"""
    clp_drs_restore_weights!(weights, saved_val, saved_idx, back, pivot_var,
                             nrows, ntotal, first_row, last_row) -> Nothing
"""
function clp_drs_restore_weights!(
    weights::AbstractVector{Float64},
    saved_val::AbstractVector{Float64},
    saved_idx::AbstractVector{Int32},
    back::AbstractVector{Int32},
    pivot_var::AbstractVector{Int32},
    nrows::Int,
    ntotal::Int,
    first_row::Int,
    last_row::Int,
)::Nothing
    @inbounds begin
        for i in 1:ntotal
            back[i] = Int32(0)
        end
        for i in 1:nrows
            iseq = Int(saved_idx[i])
            back[iseq] = Int32(i)
        end
        for i in first_row:last_row
            ipiv = Int(pivot_var[i])
            idx = Int(back[ipiv])
            if idx > 0
                w = saved_val[idx]
                weights[i] = w < CLP_DEVEX_TRY_NORM ? CLP_DEVEX_TRY_NORM : w
            else
                weights[i] = 1.0
            end
        end
    end
    return nothing
end

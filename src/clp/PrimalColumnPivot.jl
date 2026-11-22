# Pure idiomatic 1-based Julia implementation of ClpPrimalColumnDantzig & ClpPrimalColumnSteepest (B9b).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Compatible with JuliaC --trim=safe and works both on native Vector{T} and bridge views.

const CLP_PRIMAL_SLACK_MULT = 1.01
const CLP_STEEP_TRY_NORM    = 1.0e-4
const CLP_STEEP_ADD_ONE     = 1.0

@inline _pcp_fmadd(a::Float64, b::Float64, c::Float64) = fma(a, b, c)

@inline function _pcs_ref(ref::AbstractVector{UInt32}, iseq::Int)::Bool
    i0 = iseq - 1
    @inbounds return ((ref[(i0 >> 5) + 1] >> (i0 & 31)) & UInt32(1)) != UInt32(0)
end

@inline function _infeas_zero!(infeas::AbstractVector{Float64}, idx::Int)::Nothing
    @inbounds if infeas[idx] != 0.0
        infeas[idx] = floatmin(Float64)
    end
    return nothing
end

@inline function _infeas_set!(
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    nnz::Int,
    idx::Int,
    val::Float64,
)::Int
    @inbounds if infeas[idx] != 0.0
        infeas[idx] = val
    elseif abs(val) >= CLP_INDEXED_TINY
        nnz += 1
        infeas_idx[nnz] = Int32(idx)
        infeas[idx] = val
    end
    return nnz
end

# ==============================================================================
# 1. ClpPrimalColumnDantzig
# ==============================================================================

function clp_pcd_update_djs!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    rc::AbstractVector{Float64},
)::Nothing
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        rc[iseq] -= upd_val[j]
        upd_val[j] = 0.0
    end
    return nothing
end

function clp_pcd_price(
    rc::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
    dual_tol::Float64,
)::Int
    best_dj = dual_tol
    best_seq = 0
    best_free_dj = dual_tol
    best_free_seq = 0
    ntotal = nrows + ncols

    @inbounds for iseq in 1:ncols
        st_byte = status[iseq]
        if (st_byte & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
            st = st_byte & CLP_STATUS_MASK
            val = rc[iseq]
            if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
                av = abs(val)
                if av > best_free_dj
                    best_free_dj = av
                    best_free_seq = iseq
                end
            elseif st == CLP_STATUS_AT_UPPER
                if val > best_dj
                    best_dj = val
                    best_seq = iseq
                end
            elseif st == CLP_STATUS_AT_LOWER
                if val < -best_dj
                    best_dj = -val
                    best_seq = iseq
                end
            end
        end
    end

    @inbounds for iseq in (ncols + 1):ntotal
        st_byte = status[iseq]
        if (st_byte & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
            st = st_byte & CLP_STATUS_MASK
            val = rc[iseq] * CLP_PRIMAL_SLACK_MULT
            if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
                av = abs(val)
                if av > best_free_dj
                    best_free_dj = av
                    best_free_seq = iseq
                end
            elseif st == CLP_STATUS_AT_UPPER
                if val > best_dj
                    best_dj = val
                    best_seq = iseq
                end
            elseif st == CLP_STATUS_AT_LOWER
                if val < -best_dj
                    best_dj = -val
                    best_seq = iseq
                end
            end
        end
    end

    if best_free_seq > 0 && best_free_dj > 0.1 * best_dj
        best_seq = best_free_seq
    end
    return best_seq
end

# ==============================================================================
# 2. ClpPrimalColumnSteepest
# ==============================================================================

function clp_pcs_price_scan(
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    nnz::Int,
    weights::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    mode::Int,
    switch_type::Int,
    n_wanted::Int,
    tol_sq::Float64,
    rand_u::Float64,
)::Tuple{Int,Float64}
    best_dj = 1.0e-30
    best_seq = 0
    @inbounds if mode > 1
        start0 = trunc(Int, Float64(nnz) * rand_u) + 1
        for pass in 1:2
            s_idx = pass == 1 ? start0 : 1
            e_idx = pass == 1 ? nnz : (start0 - 1)
            if switch_type < 5
                for i in s_idx:e_idx
                    iseq = Int(infeas_idx[i])
                    val = infeas[iseq]
                    w = weights[iseq]
                    if val > tol_sq
                        if val > best_dj * w
                            if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                                best_dj = val / w
                                best_seq = iseq
                            else
                                n_wanted += 1
                            end
                        end
                        n_wanted -= 1
                    end
                    n_wanted == 0 && break
                end
            else
                for i in s_idx:e_idx
                    iseq = Int(infeas_idx[i])
                    val = infeas[iseq]
                    if val > tol_sq
                        if val > best_dj
                            if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                                best_dj = val
                                best_seq = iseq
                            else
                                n_wanted += 1
                            end
                        end
                        n_wanted -= 1
                    end
                    n_wanted == 0 && break
                end
            end
            n_wanted == 0 && break
        end
    else
        for i in 1:nnz
            iseq = Int(infeas_idx[i])
            val = infeas[iseq]
            w = weights[iseq]
            if val > tol_sq
                if val > best_dj * w
                    if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                        best_dj = val / w
                        best_seq = iseq
                    end
                end
            end
        end
    end
    return (best_seq, best_dj)
end

function clp_pcs_just_djs_section!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    rc::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    infeas_nnz::Int,
    add_seq::Int,
    slack_mult::Float64,
    tol::Float64,
)::Int
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        val = rc[iseq] - upd_val[j]
        upd_val[j] = 0.0
        rc[iseq] = val
        gseq = iseq + add_seq
        st = status[gseq] & CLP_STATUS_MASK
        if st == CLP_STATUS_BASIC
            _infeas_zero!(infeas, gseq)
        elseif st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
            if abs(val) > CLP_FREE_ACCEPT * tol
                v = val * CLP_FREE_BIAS
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, v * v)
            else
                _infeas_zero!(infeas, gseq)
            end
        elseif st == CLP_STATUS_AT_UPPER
            if val > tol
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, val * (val * slack_mult))
            else
                _infeas_zero!(infeas, gseq)
            end
        elseif st == CLP_STATUS_AT_LOWER
            if val < -tol
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, val * (val * slack_mult))
            else
                _infeas_zero!(infeas, gseq)
            end
        end
    end
    return infeas_nnz
end

function clp_pcs_devex_section!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    rc::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    weights::AbstractVector{Float64},
    ref_bits::AbstractVector{UInt32},
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    infeas_nnz::Int,
    add_seq::Int,
    scale_factor::Float64,
    devex::Float64,
    slack_mult::Float64,
    tol::Float64,
)::Int
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        val2 = upd_val[j]
        upd_val[j] = 0.0
        val = rc[iseq] - val2
        rc[iseq] = val
        gseq = iseq + add_seq
        st = status[gseq] & CLP_STATUS_MASK
        if st == CLP_STATUS_BASIC
            _infeas_zero!(infeas, gseq)
        elseif st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
            w = weights[gseq]
            piv = val2 * scale_factor
            v3 = piv * piv * devex
            if _pcs_ref(ref_bits, gseq)
                v3 += 1.0
            end
            weights[gseq] = max(0.99 * w, v3)
            if abs(val) > CLP_FREE_ACCEPT * tol
                v = val * CLP_FREE_BIAS
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, v * v)
            else
                _infeas_zero!(infeas, gseq)
            end
        elseif st == CLP_STATUS_AT_UPPER
            w = weights[gseq]
            piv = val2 * scale_factor
            v3 = piv * piv * devex
            if _pcs_ref(ref_bits, gseq)
                v3 += 1.0
            end
            weights[gseq] = max(0.99 * w, v3)
            if val > tol
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, val * (val * slack_mult))
            else
                _infeas_zero!(infeas, gseq)
            end
        elseif st == CLP_STATUS_AT_LOWER
            w = weights[gseq]
            piv = val2 * scale_factor
            v3 = piv * piv * devex
            if _pcs_ref(ref_bits, gseq)
                v3 += 1.0
            end
            weights[gseq] = max(0.99 * w, v3)
            if val < -tol
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, val * (val * slack_mult))
            else
                _infeas_zero!(infeas, gseq)
            end
        end
    end
    return infeas_nnz
end

function clp_pcs_steepest_rows!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    other::AbstractVector{Float64},
    rc::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    weights::AbstractVector{Float64},
    ref_bits::Union{Nothing,AbstractVector{UInt32}},
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    infeas_nnz::Int,
    ncols::Int,
    scale_factor::Float64,
    devex::Float64,
    mode::Int,
    ref_in::Float64,
    tol::Float64,
)::Int
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        val2 = upd_val[j]
        gseq = iseq + ncols
        st = status[gseq] & CLP_STATUS_MASK
        if st == CLP_STATUS_BASIC
            _infeas_zero!(infeas, gseq)
            rc[iseq] = 0.0
        elseif st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC ||
               st == CLP_STATUS_AT_UPPER || st == CLP_STATUS_AT_LOWER
            val = rc[iseq] - val2
            modv = other[iseq]
            w = weights[gseq]
            piv = val2 * scale_factor
            piv2 = piv * piv
            w += _pcp_fmadd(piv2, devex, piv * modv)
            rc[iseq] = val
            if w < CLP_STEEP_TRY_NORM
                if mode == 1
                    w = max(CLP_STEEP_TRY_NORM, CLP_STEEP_ADD_ONE + piv2)
                else
                    w = ref_in * piv2
                    if ref_bits !== nothing && _pcs_ref(ref_bits::AbstractVector{UInt32}, gseq)
                        w += 1.0
                    end
                    w = max(w, CLP_STEEP_TRY_NORM)
                end
            end
            weights[gseq] = w
            if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
                if abs(val) > CLP_FREE_ACCEPT * tol
                    v = val * CLP_FREE_BIAS
                    infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, v * v)
                else
                    _infeas_zero!(infeas, gseq)
                end
            elseif st == CLP_STATUS_AT_UPPER
                if val > tol
                    infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, val * (val * CLP_PRIMAL_SLACK_MULT))
                else
                    _infeas_zero!(infeas, gseq)
                end
            else
                if val < -tol
                    infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, gseq, val * (val * CLP_PRIMAL_SLACK_MULT))
                else
                    _infeas_zero!(infeas, gseq)
                end
            end
        end
    end
    return infeas_nnz
end

function clp_pcs_steepest_cols_rc1!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    infeas_nnz::Int,
)::Int
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        val = upd_val[j]
        if val != 0.0
            upd_val[j] = 0.0
            infeas_nnz = _quick_add!(infeas_idx, infeas, infeas_nnz, iseq, val)
        else
            _infeas_zero!(infeas, iseq)
        end
    end
    return infeas_nnz
end

function clp_pcs_steepest_cols_rc0!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    rc::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    infeas_nnz::Int,
    tol::Float64,
)::Int
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        val = rc[iseq] - upd_val[j]
        upd_val[j] = 0.0
        rc[iseq] = val
        st = status[iseq] & CLP_STATUS_MASK
        if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
            if abs(val) > CLP_FREE_ACCEPT * tol
                v = val * CLP_FREE_BIAS
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, iseq, v * v)
            else
                _infeas_zero!(infeas, iseq)
            end
        elseif st == CLP_STATUS_AT_UPPER
            if val > tol
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, iseq, val * val)
            else
                _infeas_zero!(infeas, iseq)
            end
        elseif st == CLP_STATUS_AT_LOWER
            if val < -tol
                infeas_nnz = _infeas_set!(infeas, infeas_idx, infeas_nnz, iseq, val * val)
            else
                _infeas_zero!(infeas, iseq)
            end
        end
    end
    return infeas_nnz
end

function clp_pcs_just_devex_section!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    weights::AbstractVector{Float64},
    ref_bits::AbstractVector{UInt32},
    add_seq::Int,
    negate_piv::Bool,
    devex::Float64,
)::Nothing
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        gseq = iseq + add_seq
        w = weights[gseq]
        piv = negate_piv ? -upd_val[j] : upd_val[j]
        upd_val[j] = 0.0
        val = piv * piv * devex
        if _pcs_ref(ref_bits, gseq)
            val += 1.0
        end
        weights[gseq] = max(0.99 * w, val)
    end
    return nothing
end

function clp_pcs_just_steepest_rows!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    other::AbstractVector{Float64},
    weights::AbstractVector{Float64},
    ref_bits::Union{Nothing,AbstractVector{UInt32}},
    ncols::Int,
    devex::Float64,
    mode::Int,
    ref_in::Float64,
)::Nothing
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        gseq = iseq + ncols
        w = weights[gseq]
        piv = -upd_val[j]
        upd_val[j] = 0.0
        modv = other[iseq]
        piv2 = piv * piv
        w += _pcp_fmadd(piv2, devex, piv * modv)
        if w < CLP_STEEP_TRY_NORM
            if mode == 1
                w = max(CLP_STEEP_TRY_NORM, CLP_STEEP_ADD_ONE + piv2)
            else
                w = ref_in * piv2
                if ref_bits !== nothing && _pcs_ref(ref_bits::AbstractVector{UInt32}, gseq)
                    w += 1.0
                end
                w = max(w, CLP_STEEP_TRY_NORM)
            end
        end
        weights[gseq] = w
    end
    return nothing
end

function clp_pcs_just_steepest_cols!(
    upd_val::AbstractVector{Float64},
    upd_idx::AbstractVector{Int32},
    n_upd::Int,
    upd2_val::AbstractVector{Float64},
    weights::AbstractVector{Float64},
    ref_bits::Union{Nothing,AbstractVector{UInt32}},
    devex::Float64,
    mode::Int,
    ref_in::Float64,
)::Nothing
    @inbounds for j in 1:n_upd
        iseq = Int(upd_idx[j])
        w = weights[iseq]
        piv = upd_val[j]
        upd_val[j] = 0.0
        modv = upd2_val[j]
        upd2_val[j] = 0.0
        piv2 = piv * piv
        w += _pcp_fmadd(piv2, devex, piv * modv)
        if w < CLP_STEEP_TRY_NORM
            if mode == 1
                w = max(CLP_STEEP_TRY_NORM, CLP_STEEP_ADD_ONE + piv2)
            else
                w = ref_in * piv2
                if ref_bits !== nothing && _pcs_ref(ref_bits::AbstractVector{UInt32}, iseq)
                    w += 1.0
                end
                w = max(w, CLP_STEEP_TRY_NORM)
            end
        end
        weights[iseq] = w
    end
    return nothing
end

function clp_pcs_redo_infeasibilities!(
    infeas::AbstractVector{Float64},
    infeas_idx::AbstractVector{Int32},
    rc::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    ntotal::Int,
    tol::Float64,
)::Int
    neg_tol = -tol
    nnz = 0
    @inbounds for iseq in 1:ntotal
        st = status[iseq] & CLP_STATUS_MASK
        val = rc[iseq]
        infeas[iseq] = 0.0
        if st == CLP_STATUS_AT_LOWER
            # keep val
        elseif (st & UInt8(1)) != UInt8(0)
            val = 0.0
        elseif st == CLP_STATUS_AT_UPPER
            val = -val
        else
            if abs(val) > CLP_FREE_ACCEPT * tol
                val = -abs(val) * CLP_FREE_BIAS
            else
                val = 0.0
            end
        end
        if val < neg_tol
            infeas[iseq] = val * val
            nnz += 1
            infeas_idx[nnz] = Int32(iseq)
        end
    end
    return nnz
end

function clp_pcs_unroll_weights!(
    weights::AbstractVector{Float64},
    saved_val::AbstractVector{Float64},
    saved_idx::AbstractVector{Int32},
    nnz::Int,
)::Nothing
    @inbounds for i in 1:nnz
        irow = Int(saved_idx[i])
        weights[irow] = saved_val[irow]
        saved_val[irow] = 0.0
    end
    return nothing
end

function clp_pcs_looks_optimal_simple(
    rc::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    ntotal::Int,
    tol::Float64,
)::Bool
    @inbounds for iseq in 1:ntotal
        st = status[iseq] & CLP_STATUS_MASK
        val = rc[iseq]
        if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
            if abs(val) > CLP_FREE_ACCEPT * tol
                return false
            end
        elseif st == CLP_STATUS_AT_UPPER
            if val > tol
                return false
            end
        elseif st == CLP_STATUS_AT_LOWER
            if val < -tol
                return false
            end
        end
    end
    return true
end

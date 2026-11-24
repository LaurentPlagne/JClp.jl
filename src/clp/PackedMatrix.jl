# Pure idiomatic 1-based Julia implementation of ClpPackedMatrix core operations (B8).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Compatible with JuliaC --trim=safe and works both on native Vector{T} and bridge views.

const CLP_STATUS_IS_FREE     = UInt8(0)
const CLP_STATUS_BASIC       = UInt8(1)
const CLP_STATUS_AT_UPPER    = UInt8(2)
const CLP_STATUS_AT_LOWER    = UInt8(3)
const CLP_STATUS_SUPER_BASIC = UInt8(4)
const CLP_STATUS_IS_FIXED    = UInt8(5)
const CLP_STATUS_MASK        = UInt8(7)
const CLP_STATUS_FLAGGED_BIT = UInt8(64)

const CLP_FREE_BIAS          = 1.0e1
const CLP_FREE_ACCEPT        = 1.0e2
const CLP_INDEXED_TINY       = 1.0e-50
const CLP_INDEXED_REALLY_TINY = 1.0e-100

@inline _pm_fmadd(a::Float64, b::Float64, c::Float64) = fma(a, b, c)
@inline _pm_fmsub(a::Float64, b::Float64, c::Float64) = fma(-a, b, c)

# ==============================================================================
# Native 1-based ClpPackedMatrix struct
# ==============================================================================

mutable struct ClpPackedMatrix
    nrows::Int
    ncols::Int
    nactive::Int
    col_ordered::Bool
    flags::Int32 # bit 0: has zeros, bit 1: has gaps
    starts::Vector{Int32}
    lengths::Vector{Int32}
    indices::Vector{Int32}
    elements::Vector{Float64}
end

function ClpPackedMatrix(
    nrows::Int,
    ncols::Int,
    starts::Vector{Int32},
    indices::Vector{Int32},
    elements::Vector{Float64};
    lengths::Vector{Int32} = Int32[],
    col_ordered::Bool = true,
    nactive::Int = ncols,
    flags::Int32 = isempty(lengths) ? Int32(0) : Int32(2),
)
    return ClpPackedMatrix(nrows, ncols, nactive, col_ordered, flags, starts, lengths, indices, elements)
end

@inline clp_has_gaps(flags::Int32) = (flags & Int32(2)) != Int32(0)
@inline clp_has_zeros(flags::Int32) = (flags & Int32(1)) != Int32(0)

# ==============================================================================
# 1. Basic SpMV: y += scalar * A * x  (unscaled & scaled)
# ==============================================================================

function clp_mat_times_unscaled!(
    starts::AbstractVector{Int32},
    lengths::Union{Nothing,AbstractVector{Int32}},
    has_gaps::Bool,
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    scalar::Float64,
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
)::Nothing
    @inbounds if !has_gaps
        for icol in 1:nactive
            val = x[icol]
            if val != 0.0
                s = Int(starts[icol])
                e = Int(starts[icol + 1]) - 1
                val *= scalar
                for j in s:e
                    irow = Int(indices[j])
                    y[irow] = _pm_fmadd(val, elements[j], y[irow])
                end
            end
        end
    else
        lens = lengths::AbstractVector{Int32}
        for icol in 1:nactive
            val = x[icol]
            if val != 0.0
                s = Int(starts[icol])
                e = s + Int(lens[icol]) - 1
                val *= scalar
                for j in s:e
                    irow = Int(indices[j])
                    y[irow] = _pm_fmadd(val, elements[j], y[irow])
                end
            end
        end
    end
    return nothing
end

function clp_mat_times_scaled!(
    starts::AbstractVector{Int32},
    lengths::Union{Nothing,AbstractVector{Int32}},
    has_gaps::Bool,
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    scalar::Float64,
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    row_scale::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
)::Nothing
    @inbounds if !has_gaps
        for icol in 1:nactive
            val = x[icol]
            if val != 0.0
                val *= scalar * col_scale[icol]
                s = Int(starts[icol])
                e = Int(starts[icol + 1]) - 1
                for j in s:e
                    irow = Int(indices[j])
                    y[irow] = _pm_fmadd(val * elements[j], row_scale[irow], y[irow])
                end
            end
        end
    else
        lens = lengths::AbstractVector{Int32}
        for icol in 1:nactive
            val = x[icol]
            if val != 0.0
                val *= scalar * col_scale[icol]
                s = Int(starts[icol])
                e = s + Int(lens[icol]) - 1
                for j in s:e
                    irow = Int(indices[j])
                    y[irow] = _pm_fmadd(val * elements[j], row_scale[irow], y[irow])
                end
            end
        end
    end
    return nothing
end

# ==============================================================================
# 2. Basic SpMV-T: y += scalar * x * A  (unscaled, scaled, subset)
# ==============================================================================

function clp_mat_transpose_times_unscaled!(
    starts::AbstractVector{Int32},
    lengths::Union{Nothing,AbstractVector{Int32}},
    has_gaps::Bool,
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    scalar::Float64,
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
)::Nothing
    nactive <= 0 && return nothing
    @inbounds if !has_gaps
        if scalar == -1.0
            s = Int(starts[1])
            for icol in 1:nactive
                nxt = Int(starts[icol + 1])
                val = y[icol]
                for j in s:(nxt - 1)
                    jrow = Int(indices[j])
                    val = _pm_fmsub(x[jrow], elements[j], val)
                end
                s = nxt
                y[icol] = val
            end
        else
            s = Int(starts[1])
            for icol in 1:nactive
                nxt = Int(starts[icol + 1])
                val = 0.0
                for j in s:(nxt - 1)
                    jrow = Int(indices[j])
                    val = _pm_fmadd(x[jrow], elements[j], val)
                end
                s = nxt
                y[icol] = _pm_fmadd(val, scalar, y[icol])
            end
        end
    else
        lens = lengths::AbstractVector{Int32}
        for icol in 1:nactive
            val = 0.0
            s = Int(starts[icol])
            e = s + Int(lens[icol]) - 1
            for j in s:e
                jrow = Int(indices[j])
                val = _pm_fmadd(x[jrow], elements[j], val)
            end
            y[icol] = _pm_fmadd(val, scalar, y[icol])
        end
    end
    return nothing
end

function clp_mat_transpose_times_scaled!(
    starts::AbstractVector{Int32},
    lengths::Union{Nothing,AbstractVector{Int32}},
    has_gaps::Bool,
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nrows::Int,
    nactive::Int,
    scalar::Float64,
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    row_scale::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
    spare::Union{Nothing,AbstractVector{Float64}},
)::Nothing
    nactive <= 0 && return nothing
    @inbounds if spare === nothing
        if !has_gaps
            s = Int(starts[1])
            if scalar == -1.0
                for icol in 1:nactive
                    nxt = Int(starts[icol + 1])
                    val = 0.0
                    for j in s:(nxt - 1)
                        jrow = Int(indices[j])
                        val = _pm_fmadd(x[jrow] * elements[j], row_scale[jrow], val)
                    end
                    s = nxt
                    y[icol] = _pm_fmsub(val, col_scale[icol], y[icol])
                end
            else
                for icol in 1:nactive
                    nxt = Int(starts[icol + 1])
                    val = 0.0
                    for j in s:(nxt - 1)
                        jrow = Int(indices[j])
                        val = _pm_fmadd(x[jrow] * elements[j], row_scale[jrow], val)
                    end
                    s = nxt
                    y[icol] = _pm_fmadd(val * scalar, col_scale[icol], y[icol])
                end
            end
        else
            lens = lengths::AbstractVector{Int32}
            for icol in 1:nactive
                val = 0.0
                s = Int(starts[icol])
                e = s + Int(lens[icol]) - 1
                for j in s:e
                    jrow = Int(indices[j])
                    val = _pm_fmadd(x[jrow] * elements[j], row_scale[jrow], val)
                end
                y[icol] = _pm_fmadd(val * scalar, col_scale[icol], y[icol])
            end
        end
    else
        sp = spare::AbstractVector{Float64}
        for irow in 1:nrows
            val = x[irow]
            sp[irow] = val != 0.0 ? val * row_scale[irow] : 0.0
        end
        if !has_gaps
            s = Int(starts[1])
            for icol in 1:nactive
                nxt = Int(starts[icol + 1])
                val = 0.0
                for j in s:(nxt - 1)
                    jrow = Int(indices[j])
                    val = _pm_fmadd(sp[jrow], elements[j], val)
                end
                s = nxt
                y[icol] = _pm_fmadd(val * scalar, col_scale[icol], y[icol])
            end
        else
            lens = lengths::AbstractVector{Int32}
            for icol in 1:nactive
                val = 0.0
                s = Int(starts[icol])
                e = s + Int(lens[icol]) - 1
                for j in s:e
                    jrow = Int(indices[j])
                    val = _pm_fmadd(sp[jrow], elements[j], val)
                end
                y[icol] = _pm_fmadd(val * scalar, col_scale[icol], y[icol])
            end
        end
    end
    return nothing
end

function clp_mat_transpose_times_subset!(
    starts::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nrows::Int,
    nwhich::Int,
    which::AbstractVector{Int32},
    x::AbstractVector{Float64},
    y::AbstractVector{Float64},
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    spare::Union{Nothing,AbstractVector{Float64}},
)::Nothing
    @inbounds if spare === nothing || row_scale === nothing
        if row_scale !== nothing
            rs = row_scale::AbstractVector{Float64}
            cs = col_scale::AbstractVector{Float64}
            for k in 1:nwhich
                icol = Int(which[k])
                s = Int(starts[icol])
                e = Int(starts[icol + 1]) - 1
                val = 0.0
                for j in s:e
                    jrow = Int(indices[j])
                    val = _pm_fmadd(x[jrow] * elements[j], rs[jrow], val)
                end
                y[icol] = _pm_fmsub(val, cs[icol], y[icol])
            end
        else
            for k in 1:nwhich
                icol = Int(which[k])
                s = Int(starts[icol])
                e = Int(starts[icol + 1]) - 1
                val = 0.0
                for j in s:e
                    jrow = Int(indices[j])
                    val = _pm_fmadd(x[jrow], elements[j], val)
                end
                y[icol] -= val
            end
        end
    else
        rs = row_scale::AbstractVector{Float64}
        cs = col_scale::AbstractVector{Float64}
        sp = spare::AbstractVector{Float64}
        for irow in 1:nrows
            val = x[irow]
            sp[irow] = val != 0.0 ? val * rs[irow] : 0.0
        end
        for k in 1:nwhich
            icol = Int(which[k])
            s = Int(starts[icol])
            e = Int(starts[icol + 1]) - 1
            val = 0.0
            for j in s:e
                jrow = Int(indices[j])
                val = _pm_fmadd(sp[jrow], elements[j], val)
            end
            y[icol] = _pm_fmsub(val, cs[icol], y[icol])
        end
    end
    return nothing
end

# ==============================================================================
# 3. Simplex SpMV-T by column (gutsOfTransposeTimes* & unpacked / gaps)
# ==============================================================================

function clp_mat_guts_tt_unscaled!(
    starts::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    pi::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    zero_tol::Float64,
)::Int
    nactive <= 0 && return 0
    nnz = 0
    @inbounds begin
        s = Int(starts[1])
        for icol in 1:nactive
            nxt = Int(starts[icol + 1])
            val = 0.0
            for j in s:(nxt - 1)
                val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
            end
            s = nxt
            if abs(val) > zero_tol
                nnz += 1
                out_val[nnz] = val
                out_idx[nnz] = Int32(icol)
            end
        end
    end
    return nnz
end

function clp_mat_guts_tt_scaled!(
    starts::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    pi::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    zero_tol::Float64,
)::Int
    nactive <= 0 && return 0
    nnz = 0
    @inbounds begin
        s = Int(starts[1])
        for icol in 1:nactive
            nxt = Int(starts[icol + 1])
            val = 0.0
            for j in s:(nxt - 1)
                val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
            end
            s = nxt
            val *= col_scale[icol]
            if abs(val) > zero_tol
                nnz += 1
                out_val[nnz] = val
                out_idx[nnz] = Int32(icol)
            end
        end
    end
    return nnz
end

function clp_mat_guts_tt_unscaled_status!(
    starts::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    pi::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    zero_tol::Float64,
)::Int
    nnz = 0
    @inbounds for icol in 1:nactive
        if (status[icol] & UInt8(3)) != UInt8(1)
            s = Int(starts[icol])
            e = Int(starts[icol + 1]) - 1
            val = 0.0
            for j in s:e
                val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
            end
            if abs(val) > zero_tol
                nnz += 1
                out_val[nnz] = val
                out_idx[nnz] = Int32(icol)
            end
        end
    end
    return nnz
end

function clp_mat_guts_tt_scaled_status!(
    starts::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    pi::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    zero_tol::Float64,
)::Int
    nnz = 0
    @inbounds for icol in 1:nactive
        if (status[icol] & UInt8(3)) != UInt8(1)
            scale = col_scale[icol]
            s = Int(starts[icol])
            e = Int(starts[icol + 1]) - 1
            val = 0.0
            for j in s:e
                val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
            end
            val *= scale
            if abs(val) > zero_tol
                nnz += 1
                out_val[nnz] = val
                out_idx[nnz] = Int32(icol)
            end
        end
    end
    return nnz
end

function clp_mat_guts_tt_unscaled_dual!(
    starts::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    pi::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    spare_idx::AbstractVector{Int32},
    spare_val::AbstractVector{Float64},
    reduced_cost::AbstractVector{Float64},
    upper_theta::Float64,
    acceptable_pivot::Float64,
    dual_tol::Float64,
    nremaining::Int,
    zero_tol::Float64,
)::Tuple{Int,Int,Float64}
    nnz = 0
    tentative_theta = 1.0e15
    dual_t = -dual_tol
    @inbounds for icol in 1:nactive
        wanted = Int(status[icol] & UInt8(3)) - 1
        if wanted != 0
            s = Int(starts[icol])
            e = Int(starts[icol + 1]) - 1
            val = 0.0
            for j in s:e
                val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
            end
            if abs(val) > zero_tol
                nnz += 1
                out_val[nnz] = val
                out_idx[nnz] = Int32(icol)
                if wanted > 0
                    mult = wanted == 1 ? -1.0 : 1.0
                    alpha = val * mult
                    if alpha > 0.0
                        old_val = reduced_cost[icol] * mult
                        v = _pm_fmsub(tentative_theta, alpha, old_val)
                        if v < dual_t
                            v = _pm_fmsub(upper_theta, alpha, old_val)
                            if v < dual_t && alpha >= acceptable_pivot
                                upper_theta = (old_val - dual_t) / alpha
                            end
                            nremaining += 1
                            spare_val[nremaining] = alpha * mult
                            spare_idx[nremaining] = Int32(icol)
                        end
                    end
                end
            end
        end
    end
    return (nnz, nremaining, upper_theta)
end

# Unpacked transposeTimesByColumn (with or without gaps, scaled or unscaled)
function clp_mat_tt_by_col_unpacked!(
    starts::AbstractVector{Int32},
    lengths::Union{Nothing,AbstractVector{Int32}},
    has_gaps::Bool,
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    scalar::Float64,
    pi::AbstractVector{Float64},
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    zero_tol::Float64,
)::Int
    nactive <= 0 && return 0
    nnz = 0
    @inbounds if row_scale === nothing
        if scalar == -1.0
            if !has_gaps
                s = Int(starts[1])
                for icol in 1:nactive
                    nxt = Int(starts[icol + 1])
                    val = 0.0
                    for j in s:(nxt - 1)
                        val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
                    end
                    s = nxt
                    if abs(val) > zero_tol
                        nnz += 1
                        out_idx[nnz] = Int32(icol)
                        out_val[icol] = -val
                    end
                end
            else
                lens = lengths::AbstractVector{Int32}
                for icol in 1:nactive
                    s = Int(starts[icol])
                    e = s + Int(lens[icol]) - 1
                    val = 0.0
                    for j in s:e
                        val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
                    end
                    if abs(val) > zero_tol
                        nnz += 1
                        out_idx[nnz] = Int32(icol)
                        out_val[icol] = -val
                    end
                end
            end
        else
            if !has_gaps
                s = Int(starts[1])
                for icol in 1:nactive
                    nxt = Int(starts[icol + 1])
                    val = 0.0
                    for j in s:(nxt - 1)
                        val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
                    end
                    s = nxt
                    val *= scalar
                    if abs(val) > zero_tol
                        nnz += 1
                        out_idx[nnz] = Int32(icol)
                        out_val[icol] = val
                    end
                end
            else
                lens = lengths::AbstractVector{Int32}
                for icol in 1:nactive
                    s = Int(starts[icol])
                    e = s + Int(lens[icol]) - 1
                    val = 0.0
                    for j in s:e
                        val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
                    end
                    val *= scalar
                    if abs(val) > zero_tol
                        nnz += 1
                        out_idx[nnz] = Int32(icol)
                        out_val[icol] = val
                    end
                end
            end
        end
    else
        rs = row_scale::AbstractVector{Float64}
        cs = col_scale::AbstractVector{Float64}
        if scalar == -1.0
            if !has_gaps
                s = Int(starts[1])
                for icol in 1:nactive
                    nxt = Int(starts[icol + 1])
                    val = 0.0
                    for j in s:(nxt - 1)
                        irow = Int(indices[j])
                        val = _pm_fmadd(pi[irow] * elements[j], rs[irow], val)
                    end
                    s = nxt
                    val *= cs[icol]
                    if abs(val) > zero_tol
                        nnz += 1
                        out_idx[nnz] = Int32(icol)
                        out_val[icol] = -val
                    end
                end
            else
                lens = lengths::AbstractVector{Int32}
                for icol in 1:nactive
                    s = Int(starts[icol])
                    e = s + Int(lens[icol]) - 1
                    val = 0.0
                    for j in s:e
                        irow = Int(indices[j])
                        val = _pm_fmadd(pi[irow] * elements[j], rs[irow], val)
                    end
                    val *= cs[icol]
                    if abs(val) > zero_tol
                        nnz += 1
                        out_idx[nnz] = Int32(icol)
                        out_val[icol] = -val
                    end
                end
            end
        else
            if !has_gaps
                s = Int(starts[1])
                for icol in 1:nactive
                    nxt = Int(starts[icol + 1])
                    val = 0.0
                    for j in s:(nxt - 1)
                        irow = Int(indices[j])
                        val = _pm_fmadd(pi[irow] * elements[j], rs[irow], val)
                    end
                    s = nxt
                    val *= cs[icol] * scalar
                    if abs(val) > zero_tol
                        nnz += 1
                        out_idx[nnz] = Int32(icol)
                        out_val[icol] = val
                    end
                end
            else
                lens = lengths::AbstractVector{Int32}
                for icol in 1:nactive
                    s = Int(starts[icol])
                    e = s + Int(lens[icol]) - 1
                    val = 0.0
                    for j in s:e
                        irow = Int(indices[j])
                        val = _pm_fmadd(pi[irow] * elements[j], rs[irow], val)
                    end
                    val *= scalar * cs[icol]
                    if abs(val) > zero_tol
                        nnz += 1
                        out_idx[nnz] = Int32(icol)
                        out_val[icol] = val
                    end
                end
            end
        end
    end
    return nnz
end

# Packed transposeTimes with gaps (by column)
function clp_mat_tt_by_col_packed_gaps!(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    scalar::Float64,
    pi_old::AbstractVector{Float64},
    which_row::AbstractVector{Int32},
    n_in_row::Int,
    pi_work::AbstractVector{Float64},
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    zero_tol::Float64,
)::Int
    nnz = 0
    @inbounds begin
        if row_scale === nothing
            if scalar == -1.0
                for i in 1:n_in_row
                    pi_work[Int(which_row[i])] = -pi_old[i]
                end
            else
                for i in 1:n_in_row
                    pi_work[Int(which_row[i])] = scalar * pi_old[i]
                end
            end
            for icol in 1:nactive
                s = Int(starts[icol])
                e = s + Int(lengths[icol]) - 1
                val = 0.0
                for j in s:e
                    val = _pm_fmadd(pi_work[Int(indices[j])], elements[j], val)
                end
                if abs(val) > zero_tol
                    nnz += 1
                    out_val[nnz] = val
                    out_idx[nnz] = Int32(icol)
                end
            end
        else
            rs = row_scale::AbstractVector{Float64}
            cs = col_scale::AbstractVector{Float64}
            if scalar == -1.0
                for i in 1:n_in_row
                    irow = Int(which_row[i])
                    pi_work[irow] = -pi_old[i] * rs[irow]
                end
            else
                for i in 1:n_in_row
                    irow = Int(which_row[i])
                    pi_work[irow] = scalar * pi_old[i] * rs[irow]
                end
            end
            for icol in 1:nactive
                s = Int(starts[icol])
                e = s + Int(lengths[icol]) - 1
                val = 0.0
                for j in s:e
                    val = _pm_fmadd(pi_work[Int(indices[j])], elements[j], val)
                end
                val *= cs[icol]
                if abs(val) > zero_tol
                    nnz += 1
                    out_val[nnz] = val
                    out_idx[nnz] = Int32(icol)
                end
            end
        end
        for i in 1:n_in_row
            pi_work[Int(which_row[i])] = 0.0
        end
    end
    return nnz
end

# ==============================================================================
# 4. Simplex SpMV-T by row (rowCopy is CSR and already scaled)
# ==============================================================================

function clp_mat_tt_by_row_eq1_packed!(
    row_starts::AbstractVector{Int32},
    cols::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    irow::Int,
    pi0::Float64,
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    zero_tol::Float64,
    scalar::Float64,
)::Int
    nnz = 0
    val = pi0 * scalar
    @inbounds begin
        s = Int(row_starts[irow])
        e = Int(row_starts[irow + 1]) - 1
        for j in s:e
            v2 = val * elements[j]
            if abs(v2) > zero_tol
                nnz += 1
                out_val[nnz] = v2
                out_idx[nnz] = cols[j]
            end
        end
    end
    return nnz
end

function clp_mat_tt_by_row_eq2_packed!(
    row_starts::AbstractVector{Int32},
    cols::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    irow0::Int,
    irow1::Int,
    pi0::Float64,
    pi1::Float64,
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    marked::AbstractVector{UInt8},
    lookup::AbstractVector{Int32},
    zero_tol::Float64,
    scalar::Float64,
)::Int
    @inbounds begin
        s0 = Int(row_starts[irow0])
        e0 = Int(row_starts[irow0 + 1]) - 1
        s1 = Int(row_starts[irow1])
        e1 = Int(row_starts[irow1 + 1]) - 1
        if (e0 - s0) > (e1 - s1)
            s0, s1 = s1, s0
            e0, e1 = e1, e0
            pi0, pi1 = pi1, pi0
        end
        nnz = 0
        val = pi0 * scalar
        for j in s0:e0
            icol = Int(cols[j])
            v2 = val * elements[j]
            nnz += 1
            out_val[nnz] = v2
            marked[icol] = UInt8(1)
            lookup[icol] = Int32(nnz)
            out_idx[nnz] = Int32(icol)
        end
        val = pi1 * scalar
        for j in s1:e1
            icol = Int(cols[j])
            v2 = val * elements[j]
            if marked[icol] != UInt8(0)
                k = Int(lookup[icol])
                out_val[k] += v2
            else
                if abs(v2) > zero_tol
                    nnz += 1
                    out_val[nnz] = v2
                    out_idx[nnz] = Int32(icol)
                end
            end
        end
        n = nnz
        nnz = 0
        for i in 1:n
            icol = Int(out_idx[i])
            marked[icol] = UInt8(0)
            vi = out_val[i]
            if abs(vi) > zero_tol
                nnz += 1
                out_val[nnz] = vi
                out_idx[nnz] = Int32(icol)
            end
        end
        for i in (nnz + 1):n
            out_val[i] = 0.0
        end
    end
    return nnz
end

function clp_mat_tt_by_row_ge3_packed!(
    row_starts::AbstractVector{Int32},
    cols::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    which_row::AbstractVector{Int32},
    pi::AbstractVector{Float64},
    n_in_row::Int,
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    work::AbstractVector{Float64},
    zero_tol::Float64,
    scalar::Float64,
)::Int
    nnz = 0
    @inbounds begin
        for i in 1:n_in_row
            irow = Int(which_row[i])
            val = pi[i] * scalar
            s = Int(row_starts[irow])
            e = Int(row_starts[irow + 1]) - 1
            for j in s:e
                icol = Int(cols[j])
                in_val = work[icol]
                el_val = elements[j] * val
                if in_val != 0.0
                    out_v = in_val + el_val
                    if out_v == 0.0
                        out_v = CLP_INDEXED_REALLY_TINY
                    end
                    work[icol] = out_v
                else
                    work[icol] = el_val
                    nnz += 1
                    out_idx[nnz] = Int32(icol)
                end
            end
        end
        save_n = nnz
        nnz = 0
        for i in 1:save_n
            icol = Int(out_idx[i])
            v = work[icol]
            work[icol] = 0.0
            if abs(v) > zero_tol
                nnz += 1
                out_val[nnz] = v
                out_idx[nnz] = Int32(icol)
            end
        end
    end
    return nnz
end

function clp_mat_tt_by_row_gek_packed!(
    row_starts::AbstractVector{Int32},
    cols::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    which_row::AbstractVector{Int32},
    pi::AbstractVector{Float64},
    n_in_row::Int,
    ncols::Int,
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    zero_tol::Float64,
    scalar::Float64,
)::Int
    @inbounds begin
        for i in 1:n_in_row
            irow = Int(which_row[i])
            val = pi[i] * scalar
            s = Int(row_starts[irow])
            e = Int(row_starts[irow + 1]) - 1
            for j in s:e
                icol = Int(cols[j])
                out_val[icol] += elements[j] * val
            end
        end
        nnz = 0
        for i in 1:ncols
            v = out_val[i]
            if v != 0.0
                out_val[i] = 0.0
                if abs(v) > zero_tol
                    nnz += 1
                    out_val[nnz] = v
                    out_idx[nnz] = Int32(i)
                end
            end
        end
    end
    return nnz
end

function clp_mat_tt_by_row_unpacked!(
    row_starts::AbstractVector{Int32},
    cols::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    which_row::AbstractVector{Int32},
    pi::AbstractVector{Float64},
    n_in_row::Int,
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    marked::AbstractVector{UInt8},
    zero_tol::Float64,
    scalar::Float64,
)::Int
    nnz = 0
    @inbounds if n_in_row > 2
        for i in 1:n_in_row
            irow = Int(which_row[i])
            val = pi[irow] * scalar
            s = Int(row_starts[irow])
            e = Int(row_starts[irow + 1]) - 1
            for j in s:e
                icol = Int(cols[j])
                if marked[icol] == UInt8(0)
                    marked[icol] = UInt8(1)
                    nnz += 1
                    out_idx[nnz] = Int32(icol)
                end
                out_val[icol] = _pm_fmadd(val, elements[j], out_val[icol])
            end
        end
        n_orig = nnz
        nnz = 0
        for i in 1:n_orig
            icol = Int(out_idx[i])
            marked[icol] = UInt8(0)
            if abs(out_val[icol]) > zero_tol
                nnz += 1
                out_idx[nnz] = Int32(icol)
            else
                out_val[icol] = 0.0
            end
        end
    elseif n_in_row == 2
        irow0 = Int(which_row[1])
        val = pi[irow0] * scalar
        s0 = Int(row_starts[irow0])
        e0 = Int(row_starts[irow0 + 1]) - 1
        for j in s0:e0
            icol = Int(cols[j])
            nnz += 1
            out_idx[nnz] = Int32(icol)
            out_val[icol] = val * elements[j]
        end
        irow1 = Int(which_row[2])
        val = pi[irow1] * scalar
        s1 = Int(row_starts[irow1])
        e1 = Int(row_starts[irow1 + 1]) - 1
        for j in s1:e1
            icol = Int(cols[j])
            v2 = val * elements[j]
            cur = out_val[icol]
            if cur != 0.0
                v2 += cur
            else
                nnz += 1
                out_idx[nnz] = Int32(icol)
            end
            out_val[icol] = v2
        end
        n_orig = nnz
        nnz = 0
        for i in 1:n_orig
            icol = Int(out_idx[i])
            if abs(out_val[icol]) > zero_tol
                nnz += 1
                out_idx[nnz] = Int32(icol)
            else
                out_val[icol] = 0.0
            end
        end
    elseif n_in_row == 1
        irow = Int(which_row[1])
        val = pi[irow] * scalar
        s = Int(row_starts[irow])
        e = Int(row_starts[irow + 1]) - 1
        for j in s:e
            icol = Int(cols[j])
            v2 = val * elements[j]
            if abs(v2) > zero_tol
                nnz += 1
                out_idx[nnz] = Int32(icol)
                out_val[icol] = v2
            end
        end
    end
    return nnz
end

# ==============================================================================
# 5. Subset SpMV-T (subsetTransposeTimes)
# ==============================================================================

# Matches clang++ -O3 arm64 ordered 4-wide vectorization of standalone dot loops:
# groups of 4 use separate fmul + fadd, trailing <4 elements use fmadd.
@inline function _pm_dot_unscaled_standalone(
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    pi::AbstractVector{Float64},
    s::Int,
    e::Int,
)::Float64
    val = 0.0
    j = s
    @inbounds while j + 3 <= e
        p0 = pi[Int(indices[j])] * elements[j]
        p1 = pi[Int(indices[j + 1])] * elements[j + 1]
        p2 = pi[Int(indices[j + 2])] * elements[j + 2]
        p3 = pi[Int(indices[j + 3])] * elements[j + 3]
        val = (((val + p0) + p1) + p2) + p3
        j += 4
    end
    @inbounds while j <= e
        val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
        j += 1
    end
    return val
end

@inline function _pm_dot_scaled_standalone(
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    pi::AbstractVector{Float64},
    rs::AbstractVector{Float64},
    s::Int,
    e::Int,
)::Float64
    val = 0.0
    j = s
    @inbounds while j + 3 <= e
        r0 = Int(indices[j]);     p0 = (pi[r0] * elements[j]) * rs[r0]
        r1 = Int(indices[j + 1]); p1 = (pi[r1] * elements[j + 1]) * rs[r1]
        r2 = Int(indices[j + 2]); p2 = (pi[r2] * elements[j + 2]) * rs[r2]
        r3 = Int(indices[j + 3]); p3 = (pi[r3] * elements[j + 3]) * rs[r3]
        val = (((val + p0) + p1) + p2) + p3
        j += 4
    end
    @inbounds while j <= e
        r = Int(indices[j])
        val = _pm_fmadd(pi[r] * elements[j], rs[r], val)
        j += 1
    end
    return val
end

function clp_mat_subset_tt!(
    starts::AbstractVector{Int32},
    lengths::Union{Nothing,AbstractVector{Int32}},
    has_gaps::Bool,
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    which::AbstractVector{Int32},
    n_to_do::Int,
    pi::AbstractVector{Float64},
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    out_val::AbstractVector{Float64},
)::Nothing
    n_to_do <= 0 && return nothing
    @inbounds if !has_gaps && n_to_do > 2
        if row_scale === nothing
            icol0 = Int(which[1])
            out_val[1] = _pm_dot_unscaled_standalone(indices, elements, pi, Int(starts[icol0]), Int(starts[icol0 + 1]) - 1)
            for k in 2:(n_to_do - 1)
                icol = Int(which[k])
                s = Int(starts[icol])
                e = Int(starts[icol + 1]) - 1
                val = 0.0
                for j in s:e
                    val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
                end
                out_val[k] = val
            end
            icol_last = Int(which[n_to_do])
            out_val[n_to_do] = _pm_dot_unscaled_standalone(indices, elements, pi, Int(starts[icol_last]), Int(starts[icol_last + 1]) - 1)
        else
            rs = row_scale::AbstractVector{Float64}
            cs = col_scale::AbstractVector{Float64}
            icol0 = Int(which[1])
            out_val[1] = _pm_dot_scaled_standalone(indices, elements, pi, rs, Int(starts[icol0]), Int(starts[icol0 + 1]) - 1) * cs[icol0]
            for k in 2:n_to_do
                icol = Int(which[k])
                s = Int(starts[icol])
                e = Int(starts[icol + 1]) - 1
                val = 0.0
                for j in s:e
                    irow = Int(indices[j])
                    val = _pm_fmadd(pi[irow] * elements[j], rs[irow], val)
                end
                out_val[k] = val * cs[icol]
            end
        end
    elseif !has_gaps
        if row_scale === nothing
            for k in 1:n_to_do
                icol = Int(which[k])
                s = Int(starts[icol])
                e = Int(starts[icol + 1]) - 1
                val = 0.0
                for j in s:e
                    val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
                end
                out_val[k] = val
            end
        else
            rs = row_scale::AbstractVector{Float64}
            cs = col_scale::AbstractVector{Float64}
            for k in 1:n_to_do
                icol = Int(which[k])
                s = Int(starts[icol])
                e = Int(starts[icol + 1]) - 1
                val = 0.0
                for j in s:e
                    irow = Int(indices[j])
                    val = _pm_fmadd(pi[irow] * elements[j], rs[irow], val)
                end
                out_val[k] = val * cs[icol]
            end
        end
    else
        lens = lengths::AbstractVector{Int32}
        if row_scale === nothing
            for k in 1:n_to_do
                icol = Int(which[k])
                s = Int(starts[icol])
                e = s + Int(lens[icol]) - 1
                val = 0.0
                for j in s:e
                    val = _pm_fmadd(pi[Int(indices[j])], elements[j], val)
                end
                out_val[k] = val
            end
        else
            rs = row_scale::AbstractVector{Float64}
            cs = col_scale::AbstractVector{Float64}
            for k in 1:n_to_do
                icol = Int(which[k])
                s = Int(starts[icol])
                e = s + Int(lens[icol]) - 1
                val = 0.0
                for j in s:e
                    irow = Int(indices[j])
                    val = _pm_fmadd(pi[irow] * elements[j], rs[irow], val)
                end
                out_val[k] = val * cs[icol]
            end
        end
    end
    return nothing
end

# ==============================================================================
# 6. Column unpack, unpackPacked, add (indexed & dense)
# ==============================================================================

@inline function _quick_add!(
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    nnz::Int,
    irow::Int,
    el::Float64,
)::Int
    @inbounds cur = out_val[irow]
    if cur != 0.0
        s = el + cur
        @inbounds out_val[irow] = abs(s) >= CLP_INDEXED_TINY ? s : CLP_INDEXED_REALLY_TINY
    elseif abs(el) >= CLP_INDEXED_TINY
        nnz += 1
        @inbounds out_idx[nnz] = Int32(irow)
        @inbounds out_val[irow] = el
    end
    return nnz
end

function clp_mat_unpack!(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    icol::Int,
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    nnz::Int,
)::Int
    @inbounds begin
        s = Int(starts[icol])
        e = s + Int(lengths[icol]) - 1
        if row_scale === nothing
            for j in s:e
                nnz = _quick_add!(out_idx, out_val, nnz, Int(indices[j]), elements[j])
            end
        else
            rs = row_scale::AbstractVector{Float64}
            sc = (col_scale::AbstractVector{Float64})[icol]
            for j in s:e
                irow = Int(indices[j])
                nnz = _quick_add!(out_idx, out_val, nnz, irow, elements[j] * sc * rs[irow])
            end
        end
    end
    return nnz
end

function clp_mat_unpack_packed!(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    icol::Int,
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
)::Int
    nnz = 0
    @inbounds begin
        s = Int(starts[icol])
        e = s + Int(lengths[icol]) - 1
        if row_scale === nothing
            for j in s:e
                val = elements[j]
                if val != 0.0
                    nnz += 1
                    out_val[nnz] = val
                    out_idx[nnz] = indices[j]
                end
            end
        else
            rs = row_scale::AbstractVector{Float64}
            sc = (col_scale::AbstractVector{Float64})[icol]
            for j in s:e
                irow = Int(indices[j])
                val = elements[j] * sc * rs[irow]
                if val != 0.0
                    nnz += 1
                    out_val[nnz] = val
                    out_idx[nnz] = Int32(irow)
                end
            end
        end
    end
    return nnz
end

function clp_mat_add_indexed!(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    icol::Int,
    multiplier::Float64,
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    out_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    nnz::Int,
)::Int
    @inbounds begin
        s = Int(starts[icol])
        e = s + Int(lengths[icol]) - 1
        if row_scale === nothing
            for j in s:e
                nnz = _quick_add!(out_idx, out_val, nnz, Int(indices[j]), multiplier * elements[j])
            end
        else
            rs = row_scale::AbstractVector{Float64}
            sc = (col_scale::AbstractVector{Float64})[icol] * multiplier
            for j in s:e
                irow = Int(indices[j])
                nnz = _quick_add!(out_idx, out_val, nnz, irow, elements[j] * sc * rs[irow])
            end
        end
    end
    return nnz
end

function clp_mat_add_dense!(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    icol::Int,
    multiplier::Float64,
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    array::AbstractVector{Float64},
)::Nothing
    @inbounds begin
        s = Int(starts[icol])
        e = s + Int(lengths[icol]) - 1
        if row_scale === nothing
            for j in s:e
                irow = Int(indices[j])
                array[irow] = _pm_fmadd(multiplier, elements[j], array[irow])
            end
        else
            rs = row_scale::AbstractVector{Float64}
            sc = (col_scale::AbstractVector{Float64})[icol] * multiplier
            for j in s:e
                irow = Int(indices[j])
                array[irow] = _pm_fmadd(elements[j] * sc, rs[irow], array[irow])
            end
        end
    end
    return nothing
end

# ==============================================================================
# 7. Basis extraction, rangeOfElements, scaleRowCopy, reallyScale, partialPricing
# ==============================================================================

function clp_mat_count_basis(
    lengths::AbstractVector{Int32},
    which_col::AbstractVector{Int32},
    n_col_basic::Int,
)::Int
    total = 0
    @inbounds for i in 1:n_col_basic
        total += Int(lengths[Int(which_col[i])])
    end
    return total
end

function clp_mat_fill_basis!(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    has_zeros::Bool,
    which_col::AbstractVector{Int32},
    n_col_basic::Int,
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    index_row_u::AbstractVector{Int32},
    start_u::AbstractVector{Int32},
    row_count::AbstractVector{Int32},
    col_count::AbstractVector{Int32},
    element_u::AbstractVector{Float64},
)::Nothing
    @inbounds begin
        pos = Int(start_u[1]) # 1-based next free position in index_row_u / element_u
        if !has_zeros
            if row_scale === nothing
                for i in 1:n_col_basic
                    icol = Int(which_col[i])
                    len = Int(lengths[icol])
                    s = Int(starts[icol])
                    col_count[i] = Int32(len)
                    e = s + len - 1
                    for j in s:e
                        irow = Int(indices[j])
                        index_row_u[pos] = Int32(irow)
                        row_count[irow] += Int32(1)
                        element_u[pos] = elements[j]
                        pos += 1
                    end
                    start_u[i + 1] = Int32(pos)
                end
            else
                rs = row_scale::AbstractVector{Float64}
                cs = col_scale::AbstractVector{Float64}
                for i in 1:n_col_basic
                    icol = Int(which_col[i])
                    sc = cs[icol]
                    len = Int(lengths[icol])
                    s = Int(starts[icol])
                    col_count[i] = Int32(len)
                    e = s + len - 1
                    for j in s:e
                        irow = Int(indices[j])
                        index_row_u[pos] = Int32(irow)
                        row_count[irow] += Int32(1)
                        element_u[pos] = elements[j] * sc * rs[irow]
                        pos += 1
                    end
                    start_u[i + 1] = Int32(pos)
                end
            end
        else
            if row_scale === nothing
                for i in 1:n_col_basic
                    icol = Int(which_col[i])
                    s = Int(starts[icol])
                    e = s + Int(lengths[icol]) - 1
                    pos0 = pos
                    for j in s:e
                        val = elements[j]
                        if val != 0.0
                            irow = Int(indices[j])
                            index_row_u[pos] = Int32(irow)
                            row_count[irow] += Int32(1)
                            element_u[pos] = val
                            pos += 1
                        end
                    end
                    start_u[i + 1] = Int32(pos)
                    col_count[i] = Int32(pos - pos0)
                end
            else
                rs = row_scale::AbstractVector{Float64}
                cs = col_scale::AbstractVector{Float64}
                for i in 1:n_col_basic
                    icol = Int(which_col[i])
                    sc = cs[icol]
                    s = Int(starts[icol])
                    e = s + Int(lengths[icol]) - 1
                    pos0 = pos
                    for j in s:e
                        val = elements[j]
                        if val != 0.0
                            irow = Int(indices[j])
                            index_row_u[pos] = Int32(irow)
                            row_count[irow] += Int32(1)
                            element_u[pos] = val * sc * rs[irow]
                            pos += 1
                        end
                    end
                    start_u[i + 1] = Int32(pos)
                    col_count[i] = Int32(pos - pos0)
                end
            end
        end
    end
    return nothing
end

function clp_mat_range_of_elements(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    ncols::Int,
)::NTuple{4,Float64}
    smallest_neg = -floatmax(Float64)
    largest_neg = 0.0
    smallest_pos = floatmax(Float64)
    largest_pos = 0.0
    @inbounds for icol in 1:ncols
        s = Int(starts[icol])
        e = s + Int(lengths[icol]) - 1
        for j in s:e
            val = elements[j]
            if val > 0.0
                if val < smallest_pos
                    smallest_pos = val
                end
                if val > largest_pos
                    largest_pos = val
                end
            elseif val < 0.0
                if val > smallest_neg
                    smallest_neg = val
                end
                if val < largest_neg
                    largest_neg = val
                end
            end
        end
    end
    return (smallest_neg, largest_neg, smallest_pos, largest_pos)
end

function clp_mat_scale_row_copy!(
    row_starts::AbstractVector{Int32},
    cols::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nrows::Int,
    row_scale::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
)::Nothing
    @inbounds for irow in 1:nrows
        sc = row_scale[irow]
        s = Int(row_starts[irow])
        e = Int(row_starts[irow + 1]) - 1
        for j in s:e
            icol = Int(cols[j])
            elements[j] *= sc * col_scale[icol]
        end
    end
    return nothing
end

function clp_mat_really_scale!(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    ncols::Int,
    row_scale::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
)::Nothing
    @inbounds for icol in 1:ncols
        sc = col_scale[icol]
        s = Int(starts[icol])
        e = s + Int(lengths[icol]) - 1
        for j in s:e
            irow = Int(indices[j])
            elements[j] *= sc * row_scale[irow]
        end
    end
    return nothing
end

function clp_mat_partial_pricing!(
    starts::AbstractVector{Int32},
    lengths::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    nactive::Int,
    start_frac::Float64,
    end_frac::Float64,
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    duals::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    reduced_cost::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    seq_out::Int,           # 1-based (or 0 if none)
    best_seq::Int,          # 1-based (or 0 if none)
    best_dj::Float64,
    n_wanted::Int,
    orig_wanted::Int,
    min_scan::Int,
    min_good::Int,
    tol::Float64,
)::Tuple{Int,Int,Float64}
    # 1-based range matching C++: start0 = trunc(Int, start_frac * nactive), end0 = min(trunc(Int, end_frac * nactive + 1), nactive)
    start_idx = trunc(Int, start_frac * nactive) + 1
    end_idx = min(trunc(Int, end_frac * nactive + 1.0), nactive)
    save_seq = best_seq
    last_scan = min_scan < 0 ? (end_idx + 1) : (start_idx + min_scan)
    min_neg = min_good == -1 ? n_wanted : min_good
    saved_dj = 0.0

    @inbounds if row_scale !== nothing
        rs = row_scale::AbstractVector{Float64}
        cs = col_scale::AbstractVector{Float64}
        for iseq in start_idx:end_idx
            if iseq != seq_out
                st = status[iseq] & CLP_STATUS_MASK
                if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
                    val = 0.0
                    s = Int(starts[iseq])
                    e = s + Int(lengths[iseq]) - 1
                    for j in s:e
                        jrow = Int(indices[j])
                        val = _pm_fmsub(duals[jrow] * elements[j], rs[jrow], val)
                    end
                    val = abs(_pm_fmadd(val, cs[iseq], cost[iseq]))
                    if val > CLP_FREE_ACCEPT * tol
                        n_wanted -= 1
                        val *= CLP_FREE_BIAS
                        if val > best_dj
                            if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                                best_dj = val
                                best_seq = iseq
                            else
                                n_wanted += 1
                            end
                        end
                    end
                elseif st == CLP_STATUS_AT_UPPER
                    val = 0.0
                    s = Int(starts[iseq])
                    e = s + Int(lengths[iseq]) - 1
                    for j in s:e
                        jrow = Int(indices[j])
                        val = _pm_fmsub(duals[jrow] * elements[j], rs[jrow], val)
                    end
                    val = _pm_fmadd(val, cs[iseq], cost[iseq])
                    if val > tol
                        n_wanted -= 1
                        if val > best_dj
                            if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                                best_dj = val
                                best_seq = iseq
                            else
                                n_wanted += 1
                            end
                        end
                    end
                elseif st == CLP_STATUS_AT_LOWER
                    val = 0.0
                    s = Int(starts[iseq])
                    e = s + Int(lengths[iseq]) - 1
                    for j in s:e
                        jrow = Int(indices[j])
                        val = _pm_fmsub(duals[jrow] * elements[j], rs[jrow], val)
                    end
                    val = -_pm_fmadd(val, cs[iseq], cost[iseq])
                    if val > tol
                        n_wanted -= 1
                        if val > best_dj
                            if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                                best_dj = val
                                best_seq = iseq
                            else
                                n_wanted += 1
                            end
                        end
                    end
                end
            end
            if n_wanted + min_neg < orig_wanted && iseq > last_scan
                break
            end
            n_wanted == 0 && break
        end
        if best_seq != save_seq
            val = 0.0
            s = Int(starts[best_seq])
            e = s + Int(lengths[best_seq]) - 1
            for j in s:e
                jrow = Int(indices[j])
                val = _pm_fmsub(duals[jrow] * elements[j], rs[jrow], val)
            end
            saved_dj = _pm_fmadd(val, cs[best_seq], cost[best_seq])
            reduced_cost[best_seq] = saved_dj
        end
    else
        for iseq in start_idx:end_idx
            if iseq != seq_out
                st = status[iseq] & CLP_STATUS_MASK
                if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
                    val = cost[iseq]
                    s = Int(starts[iseq])
                    e = s + Int(lengths[iseq]) - 1
                    for j in s:e
                        jrow = Int(indices[j])
                        val = _pm_fmsub(duals[jrow], elements[j], val)
                    end
                    val = abs(val)
                    if val > CLP_FREE_ACCEPT * tol
                        n_wanted -= 1
                        val *= CLP_FREE_BIAS
                        if val > best_dj
                            if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                                best_dj = val
                                best_seq = iseq
                            else
                                n_wanted += 1
                            end
                        end
                    end
                elseif st == CLP_STATUS_AT_UPPER
                    val = cost[iseq]
                    s = Int(starts[iseq])
                    e = s + Int(lengths[iseq]) - 1
                    for j in s:e
                        jrow = Int(indices[j])
                        val = _pm_fmsub(duals[jrow], elements[j], val)
                    end
                    if val > tol
                        n_wanted -= 1
                        if val > best_dj
                            if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                                best_dj = val
                                best_seq = iseq
                            else
                                n_wanted += 1
                            end
                        end
                    end
                elseif st == CLP_STATUS_AT_LOWER
                    val = cost[iseq]
                    s = Int(starts[iseq])
                    e = s + Int(lengths[iseq]) - 1
                    for j in s:e
                        jrow = Int(indices[j])
                        val = _pm_fmsub(duals[jrow], elements[j], val)
                    end
                    val = -val
                    if val > tol
                        n_wanted -= 1
                        if val > best_dj
                            if (status[iseq] & CLP_STATUS_FLAGGED_BIT) == UInt8(0)
                                best_dj = val
                                best_seq = iseq
                            else
                                n_wanted += 1
                            end
                        end
                    end
                end
            end
            if n_wanted + min_neg < orig_wanted && iseq > last_scan
                break
            end
            n_wanted == 0 && break
        end
        if best_seq != save_seq
            val = cost[best_seq]
            s = Int(starts[best_seq])
            e = s + Int(lengths[best_seq]) - 1
            for j in s:e
                jrow = Int(indices[j])
                val = _pm_fmsub(duals[jrow], elements[j], val)
            end
            saved_dj = val
            reduced_cost[best_seq] = saved_dj
        end
    end
    return (best_seq, n_wanted, saved_dj)
end

"""
    clp_mat_transpose_times2!(starts, indices, elements, ncols,
                              pi1_val, pi1_idx, pi1_nnz,
                              pi2_val, pi2_idx, pi2_nnz,
                              spare_val, dj1_val, dj1_idx,
                              infeas, rc, weights, ref, has_ref, status,
                              has_scale, row_scale, col_scale,
                              ref_in, devex, scale_factor, zero_tol, dual_tol) -> Int

Exact 1-based Julia translation of `ClpPackedMatrix::transposeTimes2` (packed `pi1`, unpacked `pi2`,
no gaps, no `columnCopy_`, `infeas != NULL`, `scaleFactor != 0`). Returns `dj1_nnz` (with `returnCode = 1`).
"""
function clp_mat_transpose_times2!(
    starts::AbstractVector{Int32},
    indices::AbstractVector{Int32},
    elements::AbstractVector{Float64},
    ncols::Int,
    pi1_val::AbstractVector{Float64},
    pi1_idx::AbstractVector{Int32},
    pi1_nnz::Int,
    pi2_val::AbstractVector{Float64},
    pi2_idx::AbstractVector{Int32},
    pi2_nnz::Int,
    spare_val::AbstractVector{Float64},
    dj1_val::AbstractVector{Float64},
    dj1_idx::AbstractVector{Int32},
    infeas::AbstractVector{Float64},
    rc::AbstractVector{Float64},
    weights::AbstractVector{Float64},
    ref::AbstractVector{UInt32},
    has_ref::Bool,
    status::AbstractVector{UInt8},
    has_scale::Bool,
    row_scale::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
    ref_in::Float64,
    devex::Float64,
    scale_factor::Float64,
    zero_tol::Float64,
    dual_tol::Float64,
)::Int
    nnz = 0
    dbl_min = floatmin(Float64)
    @inbounds begin
        if !has_scale
            for i in 1:pi1_nnz
                spare_val[Int(pi1_idx[i])] = pi1_val[i]
            end
            e_col = Int(starts[1])
            for icol in 1:ncols
                s_col = e_col
                e_col = Int(starts[icol + 1])
                st = status[icol] & CLP_STATUS_MASK
                if st == CLP_STATUS_BASIC || st == CLP_STATUS_IS_FIXED
                    continue
                end
                val = 0.0
                modv = 0.0
                for j in s_col:(e_col - 1)
                    irow = Int(indices[j])
                    el = elements[j]
                    val = fma(-spare_val[irow], el, val)
                    modv = fma(pi2_val[irow], el, modv)
                end
                if abs(val) > zero_tol
                    w = weights[icol]
                    piv = val * scale_factor
                    piv2 = piv * piv
                    w += fma(piv2, devex, piv * modv)
                    if w < CLP_STEEP_TRY_NORM
                        if ref_in < 0.0
                            w = max(CLP_STEEP_TRY_NORM, CLP_STEEP_ADD_ONE + piv2)
                        else
                            w = ref_in * piv2
                            if has_ref
                                i0 = icol - 1
                                if ((ref[(i0 >> 5) + 1] >> (i0 & 31)) & UInt32(1)) != UInt32(0)
                                    w += 1.0
                                end
                            end
                            w = max(w, CLP_STEEP_TRY_NORM)
                        end
                    end
                    weights[icol] = w
                    val = rc[icol] - val
                    rc[icol] = val
                    if st == CLP_STATUS_AT_LOWER
                        if val < -dual_tol
                            v2 = val * val
                            if infeas[icol] != 0.0
                                infeas[icol] = v2
                            else
                                nnz += 1
                                dj1_val[nnz] = v2
                                dj1_idx[nnz] = Int32(icol)
                            end
                        elseif infeas[icol] != 0.0
                            infeas[icol] = dbl_min
                        end
                    elseif st == CLP_STATUS_AT_UPPER
                        if val > dual_tol
                            v2 = val * val
                            if infeas[icol] != 0.0
                                infeas[icol] = v2
                            else
                                nnz += 1
                                dj1_val[nnz] = v2
                                dj1_idx[nnz] = Int32(icol)
                            end
                        else
                            nnz += 1
                            dj1_val[nnz] = 0.0
                            dj1_idx[nnz] = Int32(icol)
                        end
                    elseif st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
                        if abs(val) > CLP_FREE_ACCEPT * dual_tol
                            vb = val * CLP_FREE_BIAS
                            v2 = vb * vb
                            if infeas[icol] != 0.0
                                infeas[icol] = v2
                            else
                                nnz += 1
                                dj1_val[nnz] = v2
                                dj1_idx[nnz] = Int32(icol)
                            end
                        else
                            nnz += 1
                            dj1_val[nnz] = 0.0
                            dj1_idx[nnz] = Int32(icol)
                        end
                    end
                end
            end
        else
            for i in 1:pi1_nnz
                ir = Int(pi1_idx[i])
                spare_val[ir] = pi1_val[i] * row_scale[ir]
            end
            for i in 1:pi2_nnz
                ir = Int(pi2_idx[i])
                pi2_val[ir] *= row_scale[ir]
            end
            e_col = Int(starts[1])
            for icol in 1:ncols
                s_col = e_col
                e_col = Int(starts[icol + 1])
                st = status[icol] & CLP_STATUS_MASK
                if st == CLP_STATUS_BASIC || st == CLP_STATUS_IS_FIXED
                    continue
                end
                sc = col_scale[icol]
                val = 0.0
                modv = 0.0
                for j in s_col:(e_col - 1)
                    irow = Int(indices[j])
                    el = elements[j]
                    val = fma(-spare_val[irow], el, val)
                    modv = fma(pi2_val[irow], el, modv)
                end
                val *= sc
                if abs(val) > zero_tol
                    modv *= sc
                    w = weights[icol]
                    piv = val * scale_factor
                    piv2 = piv * piv
                    w += fma(piv2, devex, piv * modv)
                    if w < CLP_STEEP_TRY_NORM
                        if ref_in < 0.0
                            w = max(CLP_STEEP_TRY_NORM, CLP_STEEP_ADD_ONE + piv2)
                        else
                            w = ref_in * piv2
                            if has_ref
                                i0 = icol - 1
                                if ((ref[(i0 >> 5) + 1] >> (i0 & 31)) & UInt32(1)) != UInt32(0)
                                    w += 1.0
                                end
                            end
                            w = max(w, CLP_STEEP_TRY_NORM)
                        end
                    end
                    weights[icol] = w
                    val = rc[icol] - val
                    rc[icol] = val
                    if st == CLP_STATUS_AT_LOWER
                        if val < -dual_tol
                            v2 = val * val
                            if infeas[icol] != 0.0
                                infeas[icol] = v2
                            else
                                nnz += 1
                                dj1_val[nnz] = v2
                                dj1_idx[nnz] = Int32(icol)
                            end
                        elseif infeas[icol] != 0.0
                            infeas[icol] = dbl_min
                        end
                    elseif st == CLP_STATUS_AT_UPPER
                        if val > dual_tol
                            v2 = val * val
                            if infeas[icol] != 0.0
                                infeas[icol] = v2
                            else
                                nnz += 1
                                dj1_val[nnz] = v2
                                dj1_idx[nnz] = Int32(icol)
                            end
                        elseif infeas[icol] != 0.0
                            infeas[icol] = dbl_min
                        end
                    elseif st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
                        if abs(val) > CLP_FREE_ACCEPT * dual_tol
                            vb = val * CLP_FREE_BIAS
                            v2 = vb * vb
                            if infeas[icol] != 0.0
                                infeas[icol] = v2
                            else
                                nnz += 1
                                dj1_val[nnz] = v2
                                dj1_idx[nnz] = Int32(icol)
                            end
                        else
                            nnz += 1
                            dj1_val[nnz] = 0.0
                            dj1_idx[nnz] = Int32(icol)
                        end
                    end
                end
            end
        end
        for i in 1:pi1_nnz
            spare_val[Int(pi1_idx[i])] = 0.0
        end
    end
    return nnz
end

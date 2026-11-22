# Pure idiomatic 1-based Julia implementation of ClpSimplexDual hot kernels (B11a-1).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Compatible with JuliaC --trim=safe and works both on native Vector{T} and bridge views.

const CLP_FAKE_NO    = UInt8(0)
const CLP_FAKE_LOWER = UInt8(1)
const CLP_FAKE_UPPER = UInt8(2)
const CLP_FAKE_BOTH  = UInt8(3)

const _SD_TENTATIVE_THETA = 1.0e25
const _SD_UPPER_THETA_INF = 1.0e31
const _SD_LARGE_THETA     = 1.0e50
const _SD_MINIMUM_THETA   = 1.0e-18
const _SD_MAX_TRY         = 100

@inline _sd_fmadd(a::Float64, b::Float64, c::Float64) = fma(a, b, c)
@inline _sd_fmsub(a::Float64, b::Float64, c::Float64) = fma(-a, b, c)
@inline _sd_fms(a::Float64, b::Float64, c::Float64) = fma(a, b, -c)

@inline _sd_get_status(st_byte::UInt8) = st_byte & CLP_STATUS_MASK
@inline _sd_set_status(st_byte::UInt8, new_st::UInt8) = (st_byte & ~CLP_STATUS_MASK) | new_st
@inline _sd_get_fake_bound(st_byte::UInt8) = (st_byte >> 3) & UInt8(3)
@inline _sd_set_fake_bound(st_byte::UInt8, fb::UInt8) = (st_byte & ~UInt8(24)) | (fb << 3)

# ==============================================================================
# 1. ClpSimplexDual::dualColumn0
# ==============================================================================

function _dual_col0_fast_section!(
    work::AbstractVector{Float64},
    which::AbstractVector{Int32},
    number::Int,
    add_seq::Int,
    spare_val::AbstractVector{Float64},
    spare_idx::AbstractVector{Int32},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    sequence_out::Int,
    dual_t::Float64,
    acceptable_pivot::Float64,
    upper_theta::Float64,
    number_remaining::Int,
)::Tuple{Int,Float64}
    tentative_theta = _SD_TENTATIVE_THETA
    @inbounds for i in 1:number
        iseq = Int(which[i])
        jseq = iseq + add_seq
        if jseq == sequence_out
            continue
        end
        st3 = status[jseq] & UInt8(3)
        if st3 == UInt8(2) || st3 == UInt8(3)
            mult = st3 == UInt8(2) ? -1.0 : 1.0
            alpha_val = work[i] * mult
            if alpha_val > 0.0
                old_val = dj[jseq] * mult
                val = _sd_fmsub(tentative_theta, alpha_val, old_val)
                if val < dual_t
                    val2 = _sd_fmsub(upper_theta, alpha_val, old_val)
                    if val2 < dual_t && alpha_val >= acceptable_pivot
                        upper_theta = (old_val - dual_t) / alpha_val
                    end
                    number_remaining += 1
                    spare_val[number_remaining] = alpha_val * mult
                    spare_idx[number_remaining] = Int32(jseq)
                end
            end
        end
    end
    return (number_remaining, upper_theta)
end

function _dual_col0_full_section!(
    work::AbstractVector{Float64},
    which::AbstractVector{Int32},
    number::Int,
    add_seq::Int,
    spare_val::AbstractVector{Float64},
    spare_idx::AbstractVector{Int32},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    sequence_out::Int,
    dual_tol::Float64,
    dual_bound::Float64,
    acceptable_pivot::Float64,
    upper_theta::Float64,
    free_pivot::Float64,
    bad_free::Float64,
    number_remaining::Int,
    sequence_in::Int,
    theta::Float64,
    alpha::Float64,
    number_fake::Int,
)::Tuple{Int,Float64,Float64,Float64,Int,Float64,Float64,Int}
    tentative_theta = _SD_TENTATIVE_THETA
    @inbounds for i in 1:number
        iseq = Int(which[i])
        jseq = iseq + add_seq
        if jseq == sequence_out
            continue
        end
        st = _sd_get_status(status[jseq])
        if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
            alpha_val = work[i]
            old_val = dj[jseq]
            keep = if old_val > dual_tol || old_val < -dual_tol
                true
            elseif abs(alpha_val) > max(10.0 * acceptable_pivot, 1.0e-5)
                true
            else
                bad_free = max(bad_free, abs(alpha_val))
                false
            end
            if keep
                if abs(alpha_val) > free_pivot
                    free_pivot = abs(alpha_val)
                    sequence_in = jseq
                    theta = old_val / alpha_val
                    alpha = alpha_val
                end
                if 2.0 * abs(solution[jseq]) < dual_bound
                    status[jseq] = _sd_set_fake_bound(status[jseq], CLP_FAKE_BOTH)
                    number_fake += 1
                    val = _sd_fmsub(tentative_theta, alpha_val, old_val)
                    if val > dual_tol
                        upper[jseq] = solution[jseq]
                        lower[jseq] = upper[jseq] - dual_bound
                        status[jseq] = _sd_set_status(status[jseq], CLP_STATUS_AT_UPPER)
                    else
                        lower[jseq] = solution[jseq]
                        upper[jseq] = lower[jseq] + dual_bound
                        status[jseq] = _sd_set_status(status[jseq], CLP_STATUS_AT_LOWER)
                    end
                end
            end
        elseif st == CLP_STATUS_AT_UPPER
            alpha_val = work[i]
            old_val = dj[jseq]
            val = _sd_fmsub(tentative_theta, alpha_val, old_val)
            if val > dual_tol
                val2 = _sd_fmsub(upper_theta, alpha_val, old_val)
                if val2 > dual_tol && -alpha_val >= acceptable_pivot
                    upper_theta = (old_val - dual_tol) / alpha_val
                end
                number_remaining += 1
                spare_val[number_remaining] = alpha_val
                spare_idx[number_remaining] = Int32(jseq)
            end
        elseif st == CLP_STATUS_AT_LOWER
            alpha_val = work[i]
            old_val = dj[jseq]
            val = _sd_fmsub(tentative_theta, alpha_val, old_val)
            if val < -dual_tol
                val2 = _sd_fmsub(upper_theta, alpha_val, old_val)
                if val2 < -dual_tol && alpha_val >= acceptable_pivot
                    upper_theta = (old_val + dual_tol) / alpha_val
                end
                number_remaining += 1
                spare_val[number_remaining] = alpha_val
                spare_idx[number_remaining] = Int32(jseq)
            end
        end
    end
    return (number_remaining, upper_theta, free_pivot, bad_free, sequence_in, theta, alpha, number_fake)
end

"""
    clp_dual_column0!(row_val, row_idx, row_nnz, col_val, col_idx, col_nnz,
                      spare_val, spare_idx, dj, status, solution, lower, upper,
                      ncols, more_special_options, sequence_out,
                      dual_tol, dual_bound, acceptable_pivot,
                      sequence_in, theta, alpha, number_fake)
        -> (number_remaining, upper_theta, bad_free, sequence_in, theta, alpha, number_fake)

First-pass candidate filter for dual ratio test (`ClpSimplexDual::dualColumn0`).
All indices are 1-based (`0` denotes `-1` / unset).
"""
function clp_dual_column0!(
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    col_val::AbstractVector{Float64},
    col_idx::AbstractVector{Int32},
    col_nnz::Int,
    spare_val::AbstractVector{Float64},
    spare_idx::AbstractVector{Int32},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    ncols::Int,
    more_special_options::Int,
    sequence_out::Int,
    dual_tol::Float64,
    dual_bound::Float64,
    acceptable_pivot::Float64,
    sequence_in::Int,
    theta::Float64,
    alpha::Float64,
    number_fake::Int,
)::Tuple{Int,Float64,Float64,Int,Float64,Float64,Int}
    upper_theta = _SD_UPPER_THETA_INF
    bad_free = 0.0
    number_remaining = 0

    if (more_special_options & 8) != 0
        dual_t = -dual_tol
        number_remaining, upper_theta = _dual_col0_fast_section!(
            row_val, row_idx, row_nnz, ncols,
            spare_val, spare_idx, dj, status,
            sequence_out, dual_t, acceptable_pivot, upper_theta, number_remaining,
        )
        number_remaining, upper_theta = _dual_col0_fast_section!(
            col_val, col_idx, col_nnz, 0,
            spare_val, spare_idx, dj, status,
            sequence_out, dual_t, acceptable_pivot, upper_theta, number_remaining,
        )
    else
        free_pivot = acceptable_pivot
        number_remaining, upper_theta, free_pivot, bad_free, sequence_in, theta, alpha, number_fake =
            _dual_col0_full_section!(
                row_val, row_idx, row_nnz, ncols,
                spare_val, spare_idx, dj, status, solution, lower, upper,
                sequence_out, dual_tol, dual_bound, acceptable_pivot,
                upper_theta, free_pivot, bad_free, number_remaining,
                sequence_in, theta, alpha, number_fake,
            )
        number_remaining, upper_theta, free_pivot, bad_free, sequence_in, theta, alpha, number_fake =
            _dual_col0_full_section!(
                col_val, col_idx, col_nnz, 0,
                spare_val, spare_idx, dj, status, solution, lower, upper,
                sequence_out, dual_tol, dual_bound, acceptable_pivot,
                upper_theta, free_pivot, bad_free, number_remaining,
                sequence_in, theta, alpha, number_fake,
            )
    end
    return (number_remaining, upper_theta, bad_free, sequence_in, theta, alpha, number_fake)
end

# ==============================================================================
# 2. ClpSimplexDual::dualColumn
# ==============================================================================

function _dual_col_best_possible_section(
    work::AbstractVector{Float64},
    which::AbstractVector{Int32},
    number::Int,
    add_seq::Int,
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    dual_t::Float64,
    best_possible::Float64,
)::Float64
    tentative_theta = _SD_TENTATIVE_THETA
    @inbounds for i in 1:number
        jseq = Int(which[i]) + add_seq
        st = _sd_get_status(status[jseq])
        if st == CLP_STATUS_IS_FREE || st == CLP_STATUS_SUPER_BASIC
            best_possible = max(best_possible, abs(work[i]))
        elseif st == CLP_STATUS_AT_UPPER || st == CLP_STATUS_AT_LOWER
            mult = st == CLP_STATUS_AT_UPPER ? -1.0 : 1.0
            alpha_val = work[i] * mult
            if alpha_val > 0.0
                old_val = dj[jseq] * mult
                val = _sd_fmsub(tentative_theta, alpha_val, old_val)
                if val < dual_t
                    best_possible = max(best_possible, alpha_val)
                end
            end
        end
    end
    return best_possible
end

@inline function _dual_col_clear_spare!(
    v::AbstractVector{Float64},
    m0::Int,
    m1::Int,
    ncols::Int,
)::Nothing
    @inbounds begin
        for k in 1:m0
            v[k] = 0.0
        end
        for k in m1:ncols
            v[k] = 0.0
        end
    end
    return nothing
end

"""
    clp_dual_column!(row_val, row_idx, row_nnz, col_val, col_idx, col_nnz,
                     spare_val, spare_idx, spare2_val, spare2_idx,
                     dj, cost, solution, lower, upper, status,
                     nrows, ncols,
                     dual_tol, dual_bound, dual_out, lower_out, upper_out,
                     large_value, objective_value,
                     special_options, more_special_options, factor_pivots, sequence_out,
                     acceptable_pivot, acceptable_pivot_member,
                     spare_int0, spare_int1, spare_nnz_in,
                     spare_dbl0, spare_dbl2, spare_dbl3,
                     sequence_in, theta, number_fake, number_changed)
        -> (best_possible, sequence_in, direction_in, theta, alpha,
            lower_in, upper_in, value_in, dual_in,
            acceptable_pivot_member, number_fake, number_changed)

Dual flip-flop ratio test (`ClpSimplexDual::dualColumn`).
All indices are 1-based (`0` denotes `-1` / unset). `spare2` must be cleared on entry.
"""
function clp_dual_column!(
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    col_val::AbstractVector{Float64},
    col_idx::AbstractVector{Int32},
    col_nnz::Int,
    spare_val::V,
    spare_idx::I,
    spare2_val::V,
    spare2_idx::I,
    dj::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
    dual_tol::Float64,
    dual_bound::Float64,
    dual_out::Float64,
    lower_out::Float64,
    upper_out::Float64,
    large_value::Float64,
    objective_value::Float64,
    special_options::Int,
    more_special_options::Int,
    factor_pivots::Int,
    sequence_out::Int,
    acceptable_pivot::Float64,
    acceptable_pivot_member::Float64,
    spare_int0::Int,
    spare_int1::Int,
    spare_nnz_in::Int,
    spare_dbl0::Float64,
    spare_dbl2::Float64,
    spare_dbl3::Float64,
    sequence_in::Int,
    theta::Float64,
    number_fake::Int,
    number_changed::Int,
)::Tuple{Float64,Int,Int,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Int,Int} where {
    V<:AbstractVector{Float64},
    I<:AbstractVector{Int32},
}
    number_possibly_swapped = 0
    number_remaining = 0
    total_thru = 0.0
    best_ever_pivot = acceptable_pivot
    last_sequence = 0
    last_pivot = 0.0
    upper_theta = _SD_UPPER_THETA_INF
    new_tolerance = dual_tol
    modify_costs = false
    increase_in_objective = 0.0

    i_flip = 0
    interesting_0 = 0
    interesting_1 = 0
    swapped_0 = ncols + 1
    swapped_1 = ncols + 1
    marker_0_0 = 0
    marker_0_1 = ncols + 1
    marker_1_0 = 0
    marker_1_1 = ncols + 1

    best_possible = 1.0
    bad_free = 0.0
    alpha = 0.0

    if spare_int0 >= 0
        number_remaining, upper_theta, bad_free, sequence_in, theta, alpha, number_fake =
            clp_dual_column0!(
                row_val, row_idx, row_nnz,
                col_val, col_idx, col_nnz,
                spare_val, spare_idx,
                dj, status, solution, lower, upper,
                ncols, more_special_options, sequence_out,
                dual_tol, dual_bound, acceptable_pivot,
                sequence_in, theta, alpha, number_fake,
            )
    else
        number_remaining = spare_nnz_in
        upper_theta = spare_dbl0
        if spare_int0 == -1
            theta = spare_dbl2
            alpha = spare_dbl3
            sequence_in = spare_int1
        end
    end

    interesting_0 = number_remaining
    marker_0_0 = number_remaining

    if number_remaining == 0 && sequence_in == 0
        return (0.0, 0, 0, theta, 0.0, 0.0, 0.0, 0.0, 0.0, acceptable_pivot_member, number_fake, number_changed)
    end

    bad_sum_pivots = false
    abs_dual_out = abs(dual_out)

    @inbounds if sequence_in == 0
        theta = _SD_LARGE_THETA
        tentative_theta = max(10.0 * upper_theta, 1.0e-7)

        while tentative_theta < 1.0e22
            thru_this = 0.0
            best_pivot = acceptable_pivot
            best_sequence = 0
            number_possibly_swapped = ncols + 1
            number_remaining = 0
            upper_theta = _SD_LARGE_THETA

            cur_val = i_flip == 0 ? spare_val : spare2_val
            cur_idx = i_flip == 0 ? spare_idx : spare2_idx
            next_val = i_flip == 0 ? spare2_val : spare_val
            next_idx = i_flip == 0 ? spare2_idx : spare_idx

            increase_in_this = 0.0
            int_cur = i_flip == 0 ? interesting_0 : interesting_1

            for i in 1:int_cur
                iseq = Int(cur_idx[i])
                alpha_val = cur_val[i]
                old_val = dj[iseq]
                val = _sd_fmsub(tentative_theta, alpha_val, old_val)

                if alpha_val < 0.0
                    if val > new_tolerance
                        rng = upper[iseq] - lower[iseq]
                        thru_this = _sd_fmsub(rng, alpha_val, thru_this)
                        increase_in_this = _sd_fmsub(old_val + dual_tol, rng, increase_in_this)
                        number_possibly_swapped -= 1
                        next_val[number_possibly_swapped] = alpha_val
                        next_idx[number_possibly_swapped] = Int32(iseq)
                        if abs(alpha_val) > best_pivot
                            best_pivot = abs(alpha_val)
                            best_sequence = number_possibly_swapped
                        end
                    else
                        val2 = _sd_fmsub(upper_theta, alpha_val, old_val)
                        if val2 > new_tolerance && -alpha_val >= acceptable_pivot
                            upper_theta = (old_val - new_tolerance) / alpha_val
                        end
                        number_remaining += 1
                        next_val[number_remaining] = alpha_val
                        next_idx[number_remaining] = Int32(iseq)
                    end
                else
                    if val < -new_tolerance
                        rng = upper[iseq] - lower[iseq]
                        thru_this = _sd_fmadd(rng, alpha_val, thru_this)
                        increase_in_this = _sd_fmadd(old_val - dual_tol, rng, increase_in_this)
                        number_possibly_swapped -= 1
                        next_val[number_possibly_swapped] = alpha_val
                        next_idx[number_possibly_swapped] = Int32(iseq)
                        if abs(alpha_val) > best_pivot
                            best_pivot = abs(alpha_val)
                            best_sequence = number_possibly_swapped
                        end
                    else
                        val2 = _sd_fmsub(upper_theta, alpha_val, old_val)
                        if val2 < -new_tolerance && alpha_val >= acceptable_pivot
                            upper_theta = (old_val + new_tolerance) / alpha_val
                        end
                        number_remaining += 1
                        next_val[number_remaining] = alpha_val
                        next_idx[number_remaining] = Int32(iseq)
                    end
                end
            end

            if i_flip == 0
                swapped_1 = number_possibly_swapped
                interesting_1 = number_remaining
                marker_1_0 = max(marker_1_0, number_remaining)
                marker_1_1 = min(marker_1_1, number_possibly_swapped)
            else
                swapped_0 = number_possibly_swapped
                interesting_0 = number_remaining
                marker_0_0 = max(marker_0_0, number_remaining)
                marker_0_1 = min(marker_0_1, number_possibly_swapped)
            end

            check = abs(total_thru + thru_this)
            check += _sd_fmadd(1.0e-10, check, 1.0e-8)

            if check >= abs_dual_out || increase_in_objective + increase_in_this < 0.0
                sw_other = i_flip == 0 ? swapped_1 : swapped_0
                number_remaining = 0
                for i in ncols:-1:sw_other
                    number_remaining += 1
                    cur_val[number_remaining] = next_val[i]
                    cur_idx[number_remaining] = next_idx[i]
                end
                if i_flip == 0
                    interesting_0 = number_remaining
                else
                    interesting_1 = number_remaining
                end

                hit_max_try = true
                for _ in 1:_SD_MAX_TRY
                    upper_theta = _SD_LARGE_THETA
                    number_possibly_swapped = ncols + 1
                    number_remaining = 0
                    increase_in_this = 0.0
                    thru_this = 0.0

                    cur_val = i_flip == 0 ? spare_val : spare2_val
                    cur_idx = i_flip == 0 ? spare_idx : spare2_idx
                    next_val = i_flip == 0 ? spare2_val : spare_val
                    next_idx = i_flip == 0 ? spare2_idx : spare_idx
                    int_cur = i_flip == 0 ? interesting_0 : interesting_1

                    for i in 1:int_cur
                        iseq = Int(cur_idx[i])
                        alpha_val = cur_val[i]
                        old_val = dj[iseq]
                        val = _sd_fmsub(upper_theta, alpha_val, old_val)
                        if alpha_val < 0.0
                            if val > new_tolerance && -alpha_val >= acceptable_pivot
                                upper_theta = (old_val - new_tolerance) / alpha_val
                            end
                        else
                            if val < -new_tolerance && alpha_val >= acceptable_pivot
                                upper_theta = (old_val + new_tolerance) / alpha_val
                            end
                        end
                    end

                    best_pivot = acceptable_pivot
                    sequence_in = 0
                    largest_pivot = acceptable_pivot
                    sum_bad_pivots = 0.0
                    bad_sum_pivots = false
                    upper_theta *= 1.0000000001

                    for i in 1:int_cur
                        iseq = Int(cur_idx[i])
                        alpha_val = cur_val[i]
                        dj_val = dj[iseq]
                        val = _sd_fmsub(upper_theta, alpha_val, dj_val)
                        bad_dj = 0.0
                        add_to_swapped = false

                        if alpha_val < 0.0
                            if val >= 0.0
                                add_to_swapped = true
                                bad_dj = -dj_val - dual_tol
                            end
                        else
                            if val <= 0.0
                                add_to_swapped = true
                                bad_dj = dj_val - dual_tol
                            end
                        end

                        if !add_to_swapped
                            number_remaining += 1
                            next_val[number_remaining] = alpha_val
                            next_idx[number_remaining] = Int32(iseq)
                        else
                            number_possibly_swapped -= 1
                            next_val[number_possibly_swapped] = alpha_val
                            next_idx[number_possibly_swapped] = Int32(iseq)
                            abs_alpha = abs(alpha_val)
                            take = abs_alpha > best_pivot

                            if abs_alpha < acceptable_pivot && upper_theta < 1.0e20
                                if alpha_val < 0.0
                                    if val > dual_tol
                                        gap = upper[iseq] - lower[iseq]
                                        if gap < 1.0e20
                                            sum_bad_pivots = _sd_fmadd(val, gap, sum_bad_pivots)
                                        else
                                            sum_bad_pivots += 1.0e20
                                        end
                                    end
                                else
                                    if val < -dual_tol
                                        gap = upper[iseq] - lower[iseq]
                                        if gap < 1.0e20
                                            sum_bad_pivots = _sd_fmsub(val, gap, sum_bad_pivots)
                                        else
                                            sum_bad_pivots += 1.0e20
                                        end
                                    end
                                end
                            end

                            if take
                                sequence_in = number_possibly_swapped
                                best_pivot = abs_alpha
                                theta = dj_val / alpha_val
                                largest_pivot = max(largest_pivot, 0.5 * best_pivot)
                            end
                            rng = upper[iseq] - lower[iseq]
                            thru_this = _sd_fmadd(rng, abs_alpha, thru_this)
                            increase_in_this = _sd_fmadd(bad_dj, rng, increase_in_this)
                        end
                    end

                    if i_flip == 0
                        marker_1_0 = max(marker_1_0, number_remaining)
                        marker_1_1 = min(marker_1_1, number_possibly_swapped)
                        swapped_1 = number_possibly_swapped
                        interesting_1 = number_remaining
                    else
                        marker_0_0 = max(marker_0_0, number_remaining)
                        marker_0_1 = min(marker_0_1, number_possibly_swapped)
                        swapped_0 = number_possibly_swapped
                        interesting_0 = number_remaining
                    end

                    if sum_bad_pivots > 1.0e4 && factor_pivots > 3
                        bad_sum_pivots = true
                        hit_max_try = false
                        break
                    end

                    increase = (abs_dual_out - total_thru) * theta
                    increase += increase_in_objective
                    if theta < 0.0
                        thru_this += abs_dual_out
                    end
                    if increase_in_objective < 0.0 && increase < 0.0 && last_sequence > 0
                        best_pivot = 0.0
                    else
                        total_thru += thru_this
                        increase_in_objective += increase_in_this
                    end

                    if best_pivot < 0.1 * best_ever_pivot && best_ever_pivot > 1.0e-6 &&
                       (best_pivot < 1.0e-3 || total_thru * 2.0 > abs_dual_out)
                        sequence_in = last_sequence
                        i_flip = 1 - i_flip
                        hit_max_try = false
                        break
                    elseif sequence_in == 0 && upper_theta > large_value
                        if last_pivot > acceptable_pivot
                            sequence_in = last_sequence
                            i_flip = 1 - i_flip
                        end
                        hit_max_try = false
                        break
                    elseif total_thru >= abs_dual_out
                        modify_costs = true
                        hit_max_try = false
                        break
                    else
                        last_sequence = sequence_in
                        if best_pivot > best_ever_pivot
                            best_ever_pivot = best_pivot
                        end
                        i_flip = 1 - i_flip
                        modify_costs = true
                    end
                end
                if hit_max_try
                    i_flip = 1 - i_flip
                end
                break
            else
                if best_pivot > 1.0e-3 || best_pivot > best_ever_pivot
                    best_ever_pivot = best_pivot
                    last_sequence = best_sequence
                else
                    sw_cur = i_flip == 0 ? swapped_0 : swapped_1
                    for k in sw_cur:ncols
                        next_val[k] = cur_val[k]
                        next_idx[k] = cur_idx[k]
                    end
                    if i_flip == 0
                        marker_1_1 = min(marker_1_1, sw_cur)
                        swapped_1 = sw_cur
                    else
                        marker_0_1 = min(marker_0_1, sw_cur)
                        swapped_0 = sw_cur
                    end
                end
                increase_in_objective += increase_in_this
                i_flip = 1 - i_flip
                tentative_theta = 2.0 * upper_theta
                total_thru += thru_this
            end
        end

        if sequence_in == 0 && last_sequence > 0
            sequence_in = last_sequence
            i_flip = 1 - i_flip
        end

        minimum_theta = upper_out > lower_out ? _SD_MINIMUM_THETA : 0.0
        if sequence_in > 0
            i_flip = 1 - i_flip
            cur_val = i_flip == 0 ? spare_val : spare2_val
            cur_idx = i_flip == 0 ? spare_idx : spare2_idx
            alpha = cur_val[sequence_in]
            sequence_in = Int(cur_idx[sequence_in])
            old_val = dj[sequence_in]
            theta = max(old_val / alpha, 0.0)
            if theta < minimum_theta && abs(alpha) < 1.0e5
                theta = minimum_theta
            end

            if modify_costs && !bad_sum_pivots
                sw_cur = i_flip == 0 ? swapped_0 : swapped_1
                for i in ncols:-1:sw_cur
                    iseq = Int(cur_idx[i])
                    alpha_i = cur_val[i]
                    val = _sd_fmsub(theta, alpha_i, dj[iseq])
                    if alpha_i < 0.0
                        if val > dual_tol
                            mod_val = _sd_fms(alpha_i, theta, dj[iseq]) + new_tolerance
                            if (special_options & (2048 + 4096 + 16384)) != 0
                                if (special_options & 16384) != 0
                                    abs(mod_val) < 1.0e-8 && (mod_val = 0.0)
                                elseif (special_options & 2048) != 0
                                    abs(mod_val) < 1.0e-10 && (mod_val = 0.0)
                                else
                                    abs(mod_val) < 1.0e-12 && (mod_val = 0.0)
                                end
                            end
                            dj[iseq] += mod_val
                            cost[iseq] += mod_val
                            if mod_val != 0.0
                                number_changed += 1
                            end
                        end
                    else
                        if -val > dual_tol
                            mod_val = _sd_fms(alpha_i, theta, dj[iseq]) - new_tolerance
                            if (special_options & (2048 + 4096)) != 0
                                if (special_options & 2048) != 0
                                    abs(mod_val) < 1.0e-10 && (mod_val = 0.0)
                                else
                                    abs(mod_val) < 1.0e-12 && (mod_val = 0.0)
                                end
                            end
                            dj[iseq] += mod_val
                            cost[iseq] += mod_val
                            if mod_val != 0.0
                                number_changed += 1
                            end
                        end
                    end
                end
            end
        end
    end

    if (bad_sum_pivots || abs(theta * bad_free) > 10.0 * dual_tol) && factor_pivots > 0
        sequence_in = 0
        acceptable_pivot_member = -acceptable_pivot_member
    end

    direction_in = 0
    lower_in = 0.0
    upper_in = 0.0
    value_in = 0.0
    dual_in = 0.0

    @inbounds if sequence_in > 0
        lower_in = lower[sequence_in]
        upper_in = upper[sequence_in]
        value_in = solution[sequence_in]
        dual_in = dj[sequence_in]

        mod_val = _sd_fms(theta, alpha, dual_in)
        move_obj = abs(mod_val * solution[sequence_in])
        small_move = max(abs(objective_value), 1.0e-3)
        if move_obj > small_move
            mod_val *= small_move / move_obj
        end
        if bad_sum_pivots
            mod_val = 0.0
        end
        if (special_options & (2048 + 4096)) != 0
            if (special_options & 16384) != 0
                abs(mod_val) < 1.0e-7 && (mod_val = 0.0)
            elseif (special_options & 2048) != 0
                abs(mod_val) < 1.0e-10 && (mod_val = 0.0)
            else
                abs(mod_val) < 1.0e-12 && (mod_val = 0.0)
            end
        end
        dual_in += mod_val
        dj[sequence_in] = dual_in
        cost[sequence_in] += mod_val
        if mod_val != 0.0
            number_changed += 1
        end

        if alpha < 0.0
            direction_in = -1
            upper_in = value_in
        else
            direction_in = 1
            lower_in = value_in
        end

        if abs(alpha) < 1.0e-6
            dual_t = -dual_tol
            best_possible = 0.0
            best_possible = _dual_col_best_possible_section(
                row_val, row_idx, row_nnz, ncols, dj, status, dual_t, best_possible,
            )
            best_possible = _dual_col_best_possible_section(
                col_val, col_idx, col_nnz, 0, dj, status, dual_t, best_possible,
            )
        else
            best_possible = abs(alpha)
        end
    else
        best_possible = 0.0
        alpha = 0.0
    end

    _dual_col_clear_spare!(spare_val, marker_0_0, marker_0_1, ncols)
    _dual_col_clear_spare!(spare2_val, marker_1_0, marker_1_1, ncols)

    return (
        best_possible, sequence_in, direction_in, theta, alpha,
        lower_in, upper_in, value_in, dual_in,
        acceptable_pivot_member, number_fake, number_changed,
    )
end

# ==============================================================================
# 3. ClpSimplexDual::flipBounds
# ==============================================================================

@inline function _flip_bounds_section!(
    which::AbstractVector{Int32},
    number::Int,
    add_seq::Int,
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    status::AbstractVector{UInt8},
)::Nothing
    @inbounds for i in 1:number
        iseq = Int(which[i]) + add_seq
        st = _sd_get_status(status[iseq])
        if st == CLP_STATUS_AT_UPPER
            status[iseq] = _sd_set_status(status[iseq], CLP_STATUS_AT_LOWER)
            solution[iseq] = lower[iseq]
        elseif st == CLP_STATUS_AT_LOWER
            status[iseq] = _sd_set_status(status[iseq], CLP_STATUS_AT_UPPER)
            solution[iseq] = upper[iseq]
        end
    end
    return nothing
end

"""
    clp_flip_bounds!(row_idx, row_nnz, col_idx, col_nnz,
                     solution, lower, upper, status, ncols) -> Nothing

Flips non-basic variables between bounds (`ClpSimplexDual::flipBounds`,
for `ClpPackedMatrix` where `correctSequence` is a no-op).
"""
function clp_flip_bounds!(
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    col_idx::AbstractVector{Int32},
    col_nnz::Int,
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    ncols::Int,
)::Nothing
    _flip_bounds_section!(row_idx, row_nnz, ncols, solution, lower, upper, status)
    _flip_bounds_section!(col_idx, col_nnz, 0, solution, lower, upper, status)
    return nothing
end

# ==============================================================================
# 4. ClpSimplexDual::updateDualsInValuesPass
# ==============================================================================

"""
    clp_update_duals_in_values_pass!(row_val, row_idx, row_nnz,
                                     col_val, col_idx, col_nnz,
                                     dj, status, ncols, theta, dual_tol) -> Nothing

Updates duals and zeros packed work arrays during values pass
(`ClpSimplexDual::updateDualsInValuesPass`).
"""
function clp_update_duals_in_values_pass!(
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    col_val::AbstractVector{Float64},
    col_idx::AbstractVector{Int32},
    col_nnz::Int,
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    ncols::Int,
    theta::Float64,
    dual_tol::Float64,
)::Nothing
    @inbounds for i in 1:row_nnz
        irow = Int(row_idx[i])
        iseq = irow + ncols
        alpha_i = row_val[i]
        val = _sd_fmsub(theta, alpha_i, dj[iseq])
        row_val[i] = 0.0
        dj[iseq] = val
        st = _sd_get_status(status[iseq])
        if st == CLP_STATUS_AT_UPPER
            if val > dual_tol
                dj[iseq] = 0.0
            end
        elseif st == CLP_STATUS_AT_LOWER
            if val < -dual_tol
                dj[iseq] = 0.0
            end
        end
    end
    @inbounds for i in 1:col_nnz
        icol = Int(col_idx[i])
        alpha_i = col_val[i]
        val = _sd_fmsub(theta, alpha_i, dj[icol])
        col_val[i] = 0.0
        dj[icol] = val
        st = _sd_get_status(status[icol])
        if st == CLP_STATUS_AT_LOWER
            if val < -dual_tol
                dj[icol] = 0.0
            end
        elseif st == CLP_STATUS_AT_UPPER
            if val > dual_tol
                dj[icol] = 0.0
            end
        end
    end
    return nothing
end

# ==============================================================================
# 5. ClpSimplexDual::updateDualsInDual
# ==============================================================================

function _update_duals_inc_rows!(
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    out_val::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_nnz::Int,
    dj::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    ncols::Int,
    theta::Float64,
    tol::Float64,
    change_obj::Float64,
)::Tuple{Int,Int,Float64}
    n_infeas = 0
    @inbounds for i in 1:row_nnz
        irow = Int(row_idx[i])
        alpha_i = row_val[i]
        row_val[i] = 0.0
        true_seq = irow + ncols
        st3 = status[true_seq] & UInt8(3)
        if st3 != UInt8(1)
            val = _sd_fmsub(theta, alpha_i, dj[true_seq])
            dj[true_seq] = val
            if st3 != UInt8(0)
                mult = st3 == UInt8(2) ? -1.0 : 1.0
                if val * mult < -tol
                    movement = mult * (lower[true_seq] - upper[true_seq])
                    n_infeas += 1
                    row_idx[n_infeas] = Int32(irow)
                    change_obj = _sd_fmsub(movement, cost[true_seq], change_obj)
                    out_nnz = _quick_add!(out_idx, out_val, out_nnz, irow, movement)
                end
            end
        end
    end
    return (n_infeas, out_nnz, change_obj)
end

function _update_duals_inc_cols!(
    col_val::AbstractVector{Float64},
    col_idx::AbstractVector{Int32},
    col_nnz::Int,
    out_val::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_nnz::Int,
    mat_starts::AbstractVector{Int32},
    mat_lengths::AbstractVector{Int32},
    mat_indices::AbstractVector{Int32},
    mat_elements::AbstractVector{Float64},
    row_scale::RS,
    col_scale::CS,
    dj::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    theta::Float64,
    tol::Float64,
    more_special_options::Int,
    change_obj::Float64,
)::Tuple{Int,Int,Float64} where {RS,CS}
    n_infeas = 0
    @inbounds if (more_special_options & 8) != 0
        for i in 1:col_nnz
            icol = Int(col_idx[i])
            alpha_i = col_val[i]
            col_val[i] = 0.0
            st3 = status[icol] & UInt8(3)
            if st3 != UInt8(1)
                val = _sd_fmsub(theta, alpha_i, dj[icol])
                dj[icol] = val
                if st3 != UInt8(0)
                    mult = st3 == UInt8(2) ? -1.0 : 1.0
                    if val * mult < -tol
                        movement = mult * (upper[icol] - lower[icol])
                        n_infeas += 1
                        col_idx[n_infeas] = Int32(icol)
                        change_obj = _sd_fmadd(movement, cost[icol], change_obj)
                        out_nnz = clp_mat_add_indexed!(
                            mat_starts, mat_lengths, mat_indices, mat_elements,
                            icol, movement, row_scale, col_scale,
                            out_idx, out_val, out_nnz,
                        )
                    end
                end
            end
        end
    else
        for i in 1:col_nnz
            icol = Int(col_idx[i])
            alpha_i = col_val[i]
            col_val[i] = 0.0
            st = _sd_get_status(status[icol])
            if st == CLP_STATUS_AT_LOWER
                val = _sd_fmsub(theta, alpha_i, dj[icol])
                dj[icol] = val
                if val < -tol
                    n_infeas += 1
                    col_idx[n_infeas] = Int32(icol)
                    movement = upper[icol] - lower[icol]
                    change_obj = _sd_fmadd(movement, cost[icol], change_obj)
                    out_nnz = clp_mat_add_indexed!(
                        mat_starts, mat_lengths, mat_indices, mat_elements,
                        icol, movement, row_scale, col_scale,
                        out_idx, out_val, out_nnz,
                    )
                end
            elseif st == CLP_STATUS_AT_UPPER
                val = _sd_fmsub(theta, alpha_i, dj[icol])
                dj[icol] = val
                if val > tol
                    n_infeas += 1
                    col_idx[n_infeas] = Int32(icol)
                    movement = lower[icol] - upper[icol]
                    change_obj = _sd_fmadd(movement, cost[icol], change_obj)
                    out_nnz = clp_mat_add_indexed!(
                        mat_starts, mat_lengths, mat_indices, mat_elements,
                        icol, movement, row_scale, col_scale,
                        out_idx, out_val, out_nnz,
                    )
                end
            elseif st == CLP_STATUS_IS_FREE
                val = _sd_fmsub(theta, alpha_i, dj[icol])
                dj[icol] = val
            end
        end
    end
    return (n_infeas, out_nnz, change_obj)
end

@inline function _full_at_upper!(
    seq::Int,
    val::Float64,
    tol::Float64,
    dual_bound::Float64,
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    change_obj::Float64,
    n_fake::Int,
)::Tuple{Bool,Float64,Float64,Int}
    @inbounds if val > tol
        movement = lower[seq] - upper[seq]
        if abs(movement) > dual_bound
            if _sd_get_fake_bound(status[seq]) == CLP_FAKE_NO
                status[seq] = _sd_set_fake_bound(status[seq], CLP_FAKE_LOWER)
                lower[seq] = upper[seq] - dual_bound
                movement = lower[seq] - upper[seq]
                n_fake += 1
            end
        end
        change_obj = _sd_fmadd(movement, cost[seq], change_obj)
        return (true, movement, change_obj, n_fake)
    elseif val > -tol
        if _sd_get_fake_bound(status[seq]) == CLP_FAKE_UPPER
            movement = lower[seq] - upper[seq]
            status[seq] = _sd_set_status(status[seq], CLP_STATUS_AT_LOWER)
            solution[seq] = lower[seq]
            change_obj = _sd_fmadd(movement, cost[seq], change_obj)
        end
    end
    return (false, 0.0, change_obj, n_fake)
end

@inline function _full_at_lower!(
    seq::Int,
    val::Float64,
    tol::Float64,
    dual_bound::Float64,
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    change_obj::Float64,
    n_fake::Int,
)::Tuple{Bool,Float64,Float64,Int}
    @inbounds if val < -tol
        movement = upper[seq] - lower[seq]
        if abs(movement) > dual_bound
            if _sd_get_fake_bound(status[seq]) == CLP_FAKE_NO
                status[seq] = _sd_set_fake_bound(status[seq], CLP_FAKE_UPPER)
                upper[seq] = lower[seq] + dual_bound
                movement = upper[seq] - lower[seq]
                n_fake += 1
            end
        end
        change_obj = _sd_fmadd(movement, cost[seq], change_obj)
        return (true, movement, change_obj, n_fake)
    elseif val < tol
        if _sd_get_fake_bound(status[seq]) == CLP_FAKE_LOWER
            movement = upper[seq] - lower[seq]
            status[seq] = _sd_set_status(status[seq], CLP_STATUS_AT_UPPER)
            solution[seq] = upper[seq]
            change_obj = _sd_fmadd(movement, cost[seq], change_obj)
        end
    end
    return (false, 0.0, change_obj, n_fake)
end

function _update_duals_full!(
    row_idx::AbstractVector{Int32},
    col_idx::AbstractVector{Int32},
    out_val::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_nnz::Int,
    mat_starts::AbstractVector{Int32},
    mat_lengths::AbstractVector{Int32},
    mat_indices::AbstractVector{Int32},
    mat_elements::AbstractVector{Float64},
    row_scale::RS,
    col_scale::CS,
    dj::AbstractVector{Float64},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
    tol::Float64,
    dual_bound::Float64,
    number_fake::Int,
    change_obj::Float64,
)::Tuple{Int,Int,Int,Float64,Int} where {RS,CS}
    n_row_infeas = 0
    @inbounds for irow in 1:nrows
        true_seq = irow + ncols
        val = dj[true_seq]
        st = _sd_get_status(status[true_seq])
        if st == CLP_STATUS_AT_UPPER
            flipped, movement, change_obj, number_fake = _full_at_upper!(
                true_seq, val, tol, dual_bound,
                solution, lower, upper, cost, status, change_obj, number_fake,
            )
            if flipped
                n_row_infeas += 1
                row_idx[n_row_infeas] = Int32(irow)
                out_nnz = _quick_add!(out_idx, out_val, out_nnz, irow, -movement)
            end
        elseif st == CLP_STATUS_AT_LOWER
            flipped, movement, change_obj, number_fake = _full_at_lower!(
                true_seq, val, tol, dual_bound,
                solution, lower, upper, cost, status, change_obj, number_fake,
            )
            if flipped
                n_row_infeas += 1
                row_idx[n_row_infeas] = Int32(irow)
                out_nnz = _quick_add!(out_idx, out_val, out_nnz, irow, -movement)
            end
        end
    end

    n_col_infeas = 0
    @inbounds for icol in 1:ncols
        val = dj[icol]
        st = _sd_get_status(status[icol])
        if st == CLP_STATUS_AT_LOWER
            flipped, movement, change_obj, number_fake = _full_at_lower!(
                icol, val, tol, dual_bound,
                solution, lower, upper, cost, status, change_obj, number_fake,
            )
            if flipped
                n_col_infeas += 1
                col_idx[n_col_infeas] = Int32(icol)
                out_nnz = clp_mat_add_indexed!(
                    mat_starts, mat_lengths, mat_indices, mat_elements,
                    icol, movement, row_scale, col_scale,
                    out_idx, out_val, out_nnz,
                )
            end
        elseif st == CLP_STATUS_AT_UPPER
            flipped, movement, change_obj, number_fake = _full_at_upper!(
                icol, val, tol, dual_bound,
                solution, lower, upper, cost, status, change_obj, number_fake,
            )
            if flipped
                n_col_infeas += 1
                col_idx[n_col_infeas] = Int32(icol)
                out_nnz = clp_mat_add_indexed!(
                    mat_starts, mat_lengths, mat_indices, mat_elements,
                    icol, movement, row_scale, col_scale,
                    out_idx, out_val, out_nnz,
                )
            end
        end
    end
    return (n_row_infeas, n_col_infeas, out_nnz, change_obj, number_fake)
end

"""
    clp_update_duals_in_dual!(row_val, row_idx, row_nnz, col_val, col_idx, col_nnz,
                              out_val, out_idx, out_nnz,
                              mat_starts, mat_lengths, mat_indices, mat_elements,
                              row_scale, col_scale,
                              dj, solution, lower, upper, cost, status,
                              nrows, ncols, theta, dual_tol, largest_dual_err,
                              dual_bound, more_special_options, number_fake,
                              full_recompute)
        -> (total_infeas, out_row_nnz, out_col_nnz, out_nnz, change_obj, number_fake)

Updates reduced costs and detects bound flips (`ClpSimplexDual::updateDualsInDual`).
`out_val`/`out_idx` is expected cleared (`out_nnz == 0`) on entry. When `full_recompute == true`,
`clp_flip_bounds!` is invoked at the end so `out_row_nnz == 0` and `out_col_nnz == 0` while
`total_infeas` still reports the number of flipped variables.
"""
function clp_update_duals_in_dual!(
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    col_val::AbstractVector{Float64},
    col_idx::AbstractVector{Int32},
    col_nnz::Int,
    out_val::AbstractVector{Float64},
    out_idx::AbstractVector{Int32},
    out_nnz::Int,
    mat_starts::AbstractVector{Int32},
    mat_lengths::AbstractVector{Int32},
    mat_indices::AbstractVector{Int32},
    mat_elements::AbstractVector{Float64},
    row_scale::RS,
    col_scale::CS,
    dj::AbstractVector{Float64},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
    theta::Float64,
    dual_tol::Float64,
    largest_dual_err::Float64,
    dual_bound::Float64,
    more_special_options::Int,
    number_fake::Int,
    full_recompute::Bool,
)::Tuple{Int,Int,Int,Int,Float64,Int} where {
    RS<:Union{Nothing,AbstractVector{Float64}},
    CS<:Union{Nothing,AbstractVector{Float64}},
}
    err = min(1.0e-2, largest_dual_err)
    tol = dual_tol + err
    change_obj = 0.0

    if !full_recompute
        n_row_infeas, out_nnz, change_obj = _update_duals_inc_rows!(
            row_val, row_idx, row_nnz, out_val, out_idx, out_nnz,
            dj, lower, upper, cost, status, ncols, theta, tol, change_obj,
        )
        n_col_infeas, out_nnz, change_obj = _update_duals_inc_cols!(
            col_val, col_idx, col_nnz, out_val, out_idx, out_nnz,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            row_scale, col_scale,
            dj, lower, upper, cost, status, theta, tol, more_special_options, change_obj,
        )
        total_infeas = n_row_infeas + n_col_infeas
        return (total_infeas, n_row_infeas, n_col_infeas, out_nnz, change_obj, number_fake)
    else
        n_row_infeas, n_col_infeas, out_nnz, change_obj, number_fake = _update_duals_full!(
            row_idx, col_idx, out_val, out_idx, out_nnz,
            mat_starts, mat_lengths, mat_indices, mat_elements,
            row_scale, col_scale,
            dj, solution, lower, upper, cost, status,
            nrows, ncols, tol, dual_bound, number_fake, change_obj,
        )
        total_infeas = n_row_infeas + n_col_infeas
        clp_flip_bounds!(
            row_idx, n_row_infeas, col_idx, n_col_infeas,
            solution, lower, upper, status, ncols,
        )
        return (total_infeas, 0, 0, out_nnz, change_obj, number_fake)
    end
end

# Pure idiomatic 1-based Julia implementation of ClpSimplexPrimal hot kernels (B11b).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Compatible with JuliaC --trim=safe and works both on native Vector{T} and bridge views.

const CLP_STATUS_ACTIVE_BIT = UInt8(128)
const _SP_MINIMUM_THETA     = 1.0e-12
const _SP_MAX_TRY           = 1000
const _SP_SORT_VALUES_BIT   = 67108864

@inline _sp_fmadd(a::Float64, b::Float64, c::Float64) = fma(a, b, c)
@inline _sp_fmsub(a::Float64, b::Float64, c::Float64) = fma(-a, b, c)

@inline _sp_get_status(st_byte::UInt8) = st_byte & CLP_STATUS_MASK
@inline _sp_set_status(st_byte::UInt8, new_st::UInt8) = (st_byte & ~CLP_STATUS_MASK) | new_st
@inline _sp_is_active(st_byte::UInt8) = (st_byte & CLP_STATUS_ACTIVE_BIT) != UInt8(0)
@inline _sp_set_active(st_byte::UInt8) = st_byte | CLP_STATUS_ACTIVE_BIT
@inline _sp_clear_active(st_byte::UInt8) = st_byte & ~CLP_STATUS_ACTIVE_BIT
@inline _sp_is_flagged(st_byte::UInt8) = (st_byte & CLP_STATUS_FLAGGED_BIT) != UInt8(0)
@inline _sp_clear_flagged(st_byte::UInt8) = st_byte & ~CLP_STATUS_FLAGGED_BIT

# ==============================================================================
# Pure 1-based CoinSort_2 (exact EKK quicksort + libc++ introsort fallback)
# ==============================================================================

@inline function _sp_swap2!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    i::Int,
    j::Int,
)::Nothing
    @inbounds begin
        ki = k[i]
        kj = k[j]
        k[i] = kj
        k[j] = ki
        vi = v[i]
        vj = v[j]
        v[i] = vj
        v[j] = vi
    end
    return nothing
end

function _sp_sift_down_2!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    lo::Int,
    start::Int,
    hi::Int,
)::Nothing
    root = start
    @inbounds while true
        child = lo + 2 * (root - lo) + 1
        child > hi && break
        swap_idx = root
        if k[swap_idx] < k[child]
            swap_idx = child
        end
        if child + 1 <= hi && k[swap_idx] < k[child + 1]
            swap_idx = child + 1
        end
        swap_idx == root && break
        _sp_swap2!(k, v, root, swap_idx)
        root = swap_idx
    end
    return nothing
end

function _sp_heapsort_2!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    lo::Int,
    hi::Int,
)::Nothing
    len = hi - lo + 1
    len <= 1 && return nothing
    start = lo + (len >> 1) - 1
    while start >= lo
        _sp_sift_down_2!(k, v, lo, start, hi)
        start -= 1
    end
    end_idx = hi
    while end_idx > lo
        _sp_swap2!(k, v, lo, end_idx)
        end_idx -= 1
        _sp_sift_down_2!(k, v, lo, lo, end_idx)
    end
    return nothing
end

function _sp_ekk_qsort_2!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    lo::Int,
    hi::Int,
    depth_limit::Int,
)::Nothing
    @inbounds while hi - lo > 10
        if depth_limit <= 0
            _sp_heapsort_2!(k, v, lo, hi)
            return nothing
        end
        depth_limit -= 1
        l = lo
        r = hi
        m = l + ((r - l) >> 1)
        if k[l] > k[m]
            _sp_swap2!(k, v, l, m)
        end
        if k[m] > k[r]
            _sp_swap2!(k, v, m, r)
            if k[l] > k[m]
                _sp_swap2!(k, v, l, m)
            end
        end
        c = k[m]
        while r - l > 1
            l += 1
            while k[l] < c
                l += 1
            end
            r -= 1
            while k[r] > c
                r -= 1
            end
            _sp_swap2!(k, v, l, r)
        end
        l = r - 1
        if l < m
            _sp_ekk_qsort_2!(k, v, lo, l, depth_limit)
            lo = r
        else
            _sp_ekk_qsort_2!(k, v, r, hi, depth_limit)
            hi = l
        end
    end
    return nothing
end

@inline function _sp_lc_sort3!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    x::Int,
    y::Int,
    z::Int,
)::Bool
    @inbounds begin
        if !(k[y] < k[x])
            if !(k[z] < k[y])
                return false
            end
            _sp_swap2!(k, v, y, z)
            if k[y] < k[x]
                _sp_swap2!(k, v, x, y)
            end
            return true
        end
        if k[z] < k[y]
            _sp_swap2!(k, v, x, z)
            return true
        end
        _sp_swap2!(k, v, x, y)
        if k[z] < k[y]
            _sp_swap2!(k, v, y, z)
        end
        return true
    end
end

@inline function _sp_lc_sort4!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    x1::Int,
    x2::Int,
    x3::Int,
    x4::Int,
)::Nothing
    _sp_lc_sort3!(k, v, x1, x2, x3)
    @inbounds if k[x4] < k[x3]
        _sp_swap2!(k, v, x3, x4)
        if k[x3] < k[x2]
            _sp_swap2!(k, v, x2, x3)
            if k[x2] < k[x1]
                _sp_swap2!(k, v, x1, x2)
            end
        end
    end
    return nothing
end

@inline function _sp_lc_sort5!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    x1::Int,
    x2::Int,
    x3::Int,
    x4::Int,
    x5::Int,
)::Nothing
    _sp_lc_sort4!(k, v, x1, x2, x3, x4)
    @inbounds if k[x5] < k[x4]
        _sp_swap2!(k, v, x4, x5)
        if k[x4] < k[x3]
            _sp_swap2!(k, v, x3, x4)
            if k[x3] < k[x2]
                _sp_swap2!(k, v, x2, x3)
                if k[x2] < k[x1]
                    _sp_swap2!(k, v, x1, x2)
                end
            end
        end
    end
    return nothing
end

function _sp_lc_insertion_sort!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    first::Int,
    last::Int,
)::Nothing
    first == last && return nothing
    i = first + 1
    @inbounds while i < last
        j = i - 1
        if k[i] < k[j]
            tk = k[i]
            tv = v[i]
            kk = j
            j = i
            while true
                k[j] = k[kk]
                v[j] = v[kk]
                j = kk
                j == first && break
                kk -= 1
                !(tk < k[kk]) && break
            end
            k[j] = tk
            v[j] = tv
        end
        i += 1
    end
    return nothing
end

function _sp_lc_insertion_sort_unguarded!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    first::Int,
    last::Int,
)::Nothing
    first == last && return nothing
    i = first + 1
    @inbounds while i < last
        j = i - 1
        if k[i] < k[j]
            tk = k[i]
            tv = v[i]
            kk = j
            j = i
            while true
                k[j] = k[kk]
                v[j] = v[kk]
                j = kk
                kk -= 1
                !(tk < k[kk]) && break
            end
            k[j] = tk
            v[j] = tv
        end
        i += 1
    end
    return nothing
end

function _sp_lc_insertion_sort_incomplete!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    first::Int,
    last::Int,
)::Bool
    len = last - first
    if len <= 1
        return true
    elseif len == 2
        @inbounds if k[last - 1] < k[first]
            _sp_swap2!(k, v, first, last - 1)
        end
        return true
    elseif len == 3
        _sp_lc_sort3!(k, v, first, first + 1, last - 1)
        return true
    elseif len == 4
        _sp_lc_sort4!(k, v, first, first + 1, first + 2, last - 1)
        return true
    elseif len == 5
        _sp_lc_sort5!(k, v, first, first + 1, first + 2, first + 3, last - 1)
        return true
    end
    j = first + 2
    _sp_lc_sort3!(k, v, first, first + 1, j)
    count = 0
    i = j + 1
    @inbounds while i < last
        if k[i] < k[j]
            tk = k[i]
            tv = v[i]
            kk = j
            j = i
            while true
                k[j] = k[kk]
                v[j] = v[kk]
                j = kk
                j == first && break
                kk -= 1
                !(tk < k[kk]) && break
            end
            k[j] = tk
            v[j] = tv
            count += 1
            if count == 8
                return (i + 1) == last
            end
        end
        j = i
        i += 1
    end
    return true
end

function _sp_lc_partition_equals_right!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    first::Int,
    last::Int,
)::Tuple{Int,Bool}
    @inbounds begin
        begin_idx = first
        pk = k[first]
        pv = v[first]

        first += 1
        while k[first] < pk
            first += 1
        end

        if begin_idx == first - 1
            while first < last
                last -= 1
                k[last] < pk && break
            end
        else
            while true
                last -= 1
                k[last] < pk && break
            end
        end

        already_partitioned = first >= last
        while first < last
            _sp_swap2!(k, v, first, last)
            first += 1
            while k[first] < pk
                first += 1
            end
            last -= 1
            while !(k[last] < pk)
                last -= 1
            end
        end

        pivot_pos = first - 1
        if begin_idx != pivot_pos
            k[begin_idx] = k[pivot_pos]
            v[begin_idx] = v[pivot_pos]
        end
        k[pivot_pos] = pk
        v[pivot_pos] = pv
        return (pivot_pos, already_partitioned)
    end
end

function _sp_lc_partition_equals_left!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    first::Int,
    last::Int,
)::Int
    @inbounds begin
        begin_idx = first
        pk = k[first]
        pv = v[first]

        if pk < k[last - 1]
            first += 1
            while !(pk < k[first])
                first += 1
            end
        else
            first += 1
            while first < last && !(pk < k[first])
                first += 1
            end
        end

        if first < last
            last -= 1
            while pk < k[last]
                last -= 1
            end
        end

        while first < last
            _sp_swap2!(k, v, first, last)
            first += 1
            while !(pk < k[first])
                first += 1
            end
            last -= 1
            while pk < k[last]
                last -= 1
            end
        end

        pivot_pos = first - 1
        if begin_idx != pivot_pos
            k[begin_idx] = k[pivot_pos]
            v[begin_idx] = v[pivot_pos]
        end
        k[pivot_pos] = pk
        v[pivot_pos] = pv
        return first
    end
end

function _sp_lc_introsort!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    first::Int,
    last::Int,
    depth::Int,
    leftmost::Bool,
)::Nothing
    @inbounds while true
        len = last - first
        if len <= 1
            return nothing
        elseif len == 2
            if k[last - 1] < k[first]
                _sp_swap2!(k, v, first, last - 1)
            end
            return nothing
        elseif len == 3
            _sp_lc_sort3!(k, v, first, first + 1, last - 1)
            return nothing
        elseif len == 4
            _sp_lc_sort4!(k, v, first, first + 1, first + 2, last - 1)
            return nothing
        elseif len == 5
            _sp_lc_sort5!(k, v, first, first + 1, first + 2, first + 3, last - 1)
            return nothing
        end

        if len < 24
            if leftmost
                _sp_lc_insertion_sort!(k, v, first, last)
            else
                _sp_lc_insertion_sort_unguarded!(k, v, first, last)
            end
            return nothing
        end

        if depth == 0
            _sp_heapsort_2!(k, v, first, last - 1)
            return nothing
        end
        depth -= 1

        half_len = len >> 1
        if len > 128
            _sp_lc_sort3!(k, v, first, first + half_len, last - 1)
            _sp_lc_sort3!(k, v, first + 1, first + (half_len - 1), last - 2)
            _sp_lc_sort3!(k, v, first + 2, first + (half_len + 1), last - 3)
            _sp_lc_sort3!(k, v, first + (half_len - 1), first + half_len, first + (half_len + 1))
            _sp_swap2!(k, v, first, first + half_len)
        else
            _sp_lc_sort3!(k, v, first + half_len, first, last - 1)
        end

        if !leftmost && !(k[first - 1] < k[first])
            first = _sp_lc_partition_equals_left!(k, v, first, last)
            continue
        end

        pivot_pos, already_part = _sp_lc_partition_equals_right!(k, v, first, last)
        if already_part
            fs = _sp_lc_insertion_sort_incomplete!(k, v, first, pivot_pos)
            if _sp_lc_insertion_sort_incomplete!(k, v, pivot_pos + 1, last)
                fs && return nothing
                last = pivot_pos
                continue
            else
                if fs
                    first = pivot_pos + 1
                    continue
                end
            end
        end

        _sp_lc_introsort!(k, v, first, pivot_pos, depth, leftmost)
        leftmost = false
        first = pivot_pos + 1
    end
end

function _sp_coin_sort_2!(
    k::AbstractVector{Float64},
    v::AbstractVector{Int32},
    n::Int,
)::Nothing
    n <= 1 && return nothing
    if n > 10000
        depth_limit = 2 * (63 - leading_zeros(UInt64(n)))
        _sp_lc_introsort!(k, v, 1, n + 1, depth_limit, true)
        return nothing
    end
    @inbounds begin
        last = k[1]
        j = 2
        while j <= n
            kj = k[j]
            if kj >= last
                last = kj
                j += 1
            else
                break
            end
        end
        j > n && return nothing

        depth_limit = 2 * (64 - leading_zeros(UInt64(n))) + 4
        _sp_ekk_qsort_2!(k, v, 1, n, depth_limit)

        l = 1
        while l < n
            kl = k[l]
            kn = k[l + 1]
            if kl > kn
                c = kn
                it = v[l + 1]
                r = l
                while r >= 1
                    kr = k[r]
                    if kr > c
                        k[r + 1] = kr
                        v[r + 1] = v[r]
                        r -= 1
                    else
                        break
                    end
                end
                k[r + 1] = c
                v[r + 1] = it
            end
            l += 1
        end
    end
    return nothing
end

# ==============================================================================
# 1. ClpSimplexPrimal::primalRow
# ==============================================================================

"""
    clp_primal_row!(row_val, row_idx, row_nnz,
                    rhs_val, rhs_idx, spare_val, spare_idx,
                    pivot_variable, cost, solution, lower, upper, dj, status,
                    nlc_status, nlc_bound, nlc_infeas_weight, nlc_num_infeas, nlc_avg_theta,
                    values_pass, obj_type,
                    sequence_in, direction_in, lower_in, upper_in, value_in, dual_in,
                    primal_tol, dual_tol, infeas_cost, large_value, acceptable_pivot,
                    factor_pivots, number_iterations, special_options,
                    sequence_out, direction_out, alpha, value_out, lower_out, upper_out, dual_out)
        -> (pivot_row, sequence_out, direction_in, direction_out, theta, alpha,
            dual_in, value_out, lower_out, upper_out, dual_out, primal_tol, nlc_avg_theta)

Primal ratio test (`ClpSimplexPrimal::primalRow`, `CLP_METHOD2`, `CLP_MOVEMENT == 2`).
`row_val`/`row_idx` is packed (`1:row_nnz`); `rhs_val` and `spare_val` must be cleared on entry
and are cleared on exit. All row/sequence indices are 1-based (`0` denotes C++ `-1`,
and `pivot_row == -1` denotes C++ `-2` / bound flip).
"""
function clp_primal_row!(
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    rhs_val::AbstractVector{Float64},
    rhs_idx::AbstractVector{Int32},
    spare_val::AbstractVector{Float64},
    spare_idx::AbstractVector{Int32},
    pivot_variable::AbstractVector{Int32},
    cost::AbstractVector{Float64},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nlc_status::AbstractVector{UInt8},
    nlc_bound::AbstractVector{Float64},
    nlc_infeas_weight::Float64,
    nlc_num_infeas::Int,
    nlc_avg_theta::Float64,
    values_pass::Int,
    obj_type::Int,
    sequence_in::Int,
    direction_in::Int,
    lower_in::Float64,
    upper_in::Float64,
    value_in::Float64,
    dual_in::Float64,
    primal_tol::Float64,
    dual_tol::Float64,
    infeas_cost::Float64,
    large_value::Float64,
    acceptable_pivot::Float64,
    factor_pivots::Int,
    number_iterations::Int,
    special_options::Int,
    sequence_out::Int,
    direction_out::Int,
    alpha::Float64,
    value_out::Float64,
    lower_out::Float64,
    upper_out::Float64,
    dual_out::Float64,
)::Tuple{Int,Int,Int,Int,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64}
    save_dj = dual_in
    @inbounds if values_pass != 0 && obj_type < 2
        dual_in = cost[sequence_in]
        for i_index in 1:row_nnz
            i_row = Int(row_idx[i_index])
            a = row_val[i_index]
            i_pivot = Int(pivot_variable[i_row])
            dual_in = _sp_fmsub(a, cost[i_pivot], dual_in)
        end
        if dual_in < -dual_tol
            direction_in = 1
        elseif dual_in > dual_tol
            direction_in = -1
        else
            if value_in - lower_in < upper_in - value_in
                direction_in = -1
                dual_in = dual_tol
            else
                direction_in = 1
                dual_in = -dual_tol
            end
        end
    end

    pivot_row = 0
    number_remaining = 0
    total_thru = 0.0

    acc_piv = 1.0e-1 * acceptable_pivot
    if number_iterations > 100
        acc_piv = acceptable_pivot
    end
    if factor_pivots > 10
        acc_piv = 1.0e+3 * acceptable_pivot
    elseif factor_pivots > 5
        acc_piv = 1.0e+2 * acceptable_pivot
    elseif factor_pivots != 0
        acc_piv = acceptable_pivot
    end
    best_ever_pivot = acc_piv
    last_pivot_row = 0
    last_pivot = 0.0
    last_theta = 1.0e50

    way = Float64(direction_in)
    maximum_movement = if way > 0.0
        min(1.0e30, upper_in - value_in)
    else
        min(1.0e30, value_in - lower_in)
    end

    average_theta = nlc_avg_theta
    tentative_theta = min(10.0 * average_theta, maximum_movement)
    upper_theta = maximum_movement
    if tentative_theta > 0.5 * maximum_movement
        tentative_theta = maximum_movement
    end
    theta_at_maximum = tentative_theta == maximum_movement
    if maximum_movement < 1.0
        tentative_theta *= 1.1
    end
    dual_check = abs(dual_in)
    dual_check = max(_sp_fmsub(100.0, dual_tol, dual_check), 0.99 * dual_check)

    @inbounds while true
        total_thru = 0.0
        number_remaining = 0
        dual_in = cost[sequence_in]
        largest_alpha = 0.0
        for i_index in 1:row_nnz
            i_row = Int(row_idx[i_index])
            a = row_val[i_index]
            i_pivot = Int(pivot_variable[i_row])
            dual_in = _sp_fmsub(a, cost[i_pivot], dual_in)
            a *= way
            old_val = solution[i_pivot]
            if a > 0.0
                bnd = lower[i_pivot]
                change = tentative_theta * a
                possible = (old_val - change) <= bnd + primal_tol
                old_val -= bnd
            else
                bnd = upper[i_pivot]
                change = tentative_theta * a
                possible = (old_val - change) >= bnd - primal_tol
                old_val = bnd - old_val
                a = -a
            end
            if possible
                largest_alpha = max(largest_alpha, a)
                val = _sp_fmsub(upper_theta, a, old_val)
                if val < -primal_tol && a >= acc_piv
                    upper_theta = (old_val + primal_tol) / a
                end
                number_remaining += 1
                spare_val[number_remaining] = a
                rhs_val[number_remaining] = old_val
                rhs_idx[number_remaining] = Int32(i_index)
                spare_idx[number_remaining] = Int32(i_row)
                total_thru += a
                status[i_row] = _sp_set_active(status[i_row])
            end
        end

        if largest_alpha < 1.0e-3
            total_thru = 0.0
            tentative_theta = maximum_movement
            number_remaining = 0
            for i_index in 1:row_nnz
                i_row = Int(row_idx[i_index])
                a = row_val[i_index] * way
                i_pivot = Int(pivot_variable[i_row])
                old_val = solution[i_pivot]
                if a > 0.0
                    bnd = lower[i_pivot]
                    change = tentative_theta * a
                    possible = (old_val - change) <= bnd + primal_tol
                    old_val -= bnd
                else
                    bnd = upper[i_pivot]
                    change = tentative_theta * a
                    possible = (old_val - change) >= bnd - primal_tol
                    old_val = bnd - old_val
                    a = -a
                end
                if possible
                    val = _sp_fmsub(upper_theta, a, old_val)
                    if val < -primal_tol && a >= acc_piv
                        upper_theta = (old_val + primal_tol) / a
                    end
                    number_remaining += 1
                    spare_val[number_remaining] = a
                    rhs_val[number_remaining] = old_val
                    rhs_idx[number_remaining] = Int32(i_index)
                    spare_idx[number_remaining] = Int32(i_row)
                    total_thru += a
                    status[i_row] = _sp_set_active(status[i_row])
                end
            end
        end

        if upper_theta < maximum_movement && total_thru * infeas_cost >= 1.0001 * dual_check
            break
        elseif !theta_at_maximum
            tentative_theta = maximum_movement
            theta_at_maximum = true
        else
            break
        end
    end

    total_thru = 0.0
    theta = maximum_movement
    go_back_one = false
    if obj_type > 1
        dual_in = save_dj
    end

    @inbounds if number_remaining > 0 && upper_theta < maximum_movement
        for _ in 1:_SP_MAX_TRY
            upper_theta = maximum_movement
            i_best = 0
            for i_index in 1:number_remaining
                a = spare_val[i_index]
                old_val = rhs_val[i_index]
                val = _sp_fmsub(upper_theta, a, old_val)
                if val < -primal_tol && a >= acc_piv
                    upper_theta = (old_val + primal_tol) / a
                    i_best = i_index
                end
            end

            sum_infeas = 0.0
            best_pivot = acc_piv
            pivot_row = 0
            for i_index in 1:number_remaining
                i_row = Int(spare_idx[i_index])
                a = spare_val[i_index]
                old_val = rhs_val[i_index]
                val = _sp_fmsub(upper_theta, a, old_val)
                if val <= 0.0 || i_best == i_index
                    true_alpha = way * row_val[Int(rhs_idx[i_index])]
                    j_seq = Int(pivot_variable[i_row])
                    delta_cost, new_rhs = clp_nlc_change_in_cost_m2!(
                        nlc_status, nlc_bound, lower, upper,
                        j_seq, true_alpha, old_val, nlc_infeas_weight,
                    )
                    rhs_val[i_index] = new_rhs
                    total_thru += delta_cost
                    status[i_row] = _sp_set_active(status[i_row])
                    if a > best_pivot
                        best_pivot = a
                        theta = old_val / best_pivot
                        pivot_row = i_index
                    elseif a < acc_piv
                        if val < -primal_tol
                            sum_infeas += -val - primal_tol
                        end
                    end
                end
            end

            if best_pivot < 0.1 * best_ever_pivot && best_ever_pivot > 1.0e-6 && best_pivot < 1.0e-3
                go_back_one = true
                break
            elseif pivot_row == 0 && upper_theta > large_value
                if last_pivot > acc_piv
                    go_back_one = true
                end
                break
            elseif total_thru >= dual_check
                if sum_infeas > primal_tol && nlc_num_infeas == 0
                    if last_pivot > acc_piv
                        go_back_one = true
                    else
                        dual_in = 0.0
                    end
                end
                break
            else
                last_pivot_row = pivot_row
                last_theta = theta
                if best_pivot > best_ever_pivot
                    best_ever_pivot = best_pivot
                end
            end
        end
        if go_back_one || (pivot_row == 0 && last_pivot_row > 0)
            pivot_row = last_pivot_row
            theta = last_theta
        end
    elseif pivot_row == 0 && maximum_movement > 1.0e20
        value_out = floatmax(Float64)
        if nlc_num_infeas != 0
            tentative_theta = 1.0e50
            for i_index in 1:row_nnz
                i_row = Int(row_idx[i_index])
                a = row_val[i_index] * way
                i_pivot = Int(pivot_variable[i_row])
                old_val = solution[i_pivot]
                if a > 0.0
                    old_val -= lower[i_pivot]
                else
                    old_val = upper[i_pivot] - old_val
                    a = -a
                end
                if _sp_fmsub(tentative_theta, a, old_val) < 0.0
                    tentative_theta = old_val / a
                end
            end
            if lower_in < -1.0e20 && upper_in > 1.0e20
                if dual_in * value_in > 0.0
                    if abs(value_in) < 1.0e-2 && (tentative_theta < abs(value_in) || tentative_theta > 1.0e20)
                        tentative_theta = abs(value_in)
                    end
                end
            end
            if tentative_theta < 1.0e10
                value_out = _sp_fmadd(way, tentative_theta, value_in)
            end
        end
    end

    @inbounds if pivot_row > 0
        position = pivot_row
        pivot_row = Int(spare_idx[position])
        alpha = row_val[Int(rhs_idx[position])]
        sequence_out = Int(pivot_variable[pivot_row])
        value_out = solution[sequence_out]
        lower_out = lower[sequence_out]
        upper_out = upper[sequence_out]
        minimum_theta = upper_out > lower_out ? _SP_MINIMUM_THETA : 0.0
        distance = (alpha * way > 0.0) ? (value_out - lower_out) : (upper_out - value_out)
        if _sp_fmsub(minimum_theta, abs(alpha), distance) < -primal_tol
            minimum_theta = max(0.0, _sp_fmadd(0.5, primal_tol, distance) / abs(alpha))
        end
        if theta < minimum_theta && (special_options & 4) == 0 && values_pass == 0
            theta = minimum_theta
        end
        if theta > tentative_theta
            for i_index in 1:row_nnz
                i_row = Int(row_idx[i_index])
                status[i_row] = _sp_set_active(status[i_row])
            end
        end
        if way < 0.0
            theta = -theta
        end
        new_value = _sp_fmsub(theta, alpha, value_out)
        if alpha * way < 0.0
            direction_out = -1
            if abs(theta) > 1.0e-6 || (special_options & 4) != 0
                upper_out = clp_nlc_nearest_m2(nlc_status, nlc_bound, lower, upper, sequence_out, new_value)
            else
                upper_out = new_value
            end
        else
            direction_out = 1
            if abs(theta) > 1.0e-6 || (special_options & 4) != 0
                lower_out = clp_nlc_nearest_m2(nlc_status, nlc_bound, lower, upper, sequence_out, new_value)
            else
                lower_out = new_value
            end
        end
        dual_out = dj[sequence_out]
    elseif maximum_movement < 1.0e20
        pivot_row = -1 # C++ -2 (bound flip)
        sequence_out = sequence_in
        value_out = value_in
        dual_out = dual_in
        lower_out = lower_in
        upper_out = upper_in
        alpha = 0.0
        if way < 0.0
            direction_out = 1
            theta = lower_out - value_out
        else
            direction_out = -1
            theta = upper_out - value_out
        end
    end

    theta1 = max(abs(theta), _SP_MINIMUM_THETA)
    theta2 = Float64(number_iterations) * nlc_avg_theta
    if average_theta != 0.987e40
        nlc_avg_theta = (theta1 + theta2) / Float64(number_iterations + 1)
    end

    @inbounds for i in 1:number_remaining
        spare_val[i] = 0.0
        rhs_val[i] = 0.0
    end
    clp_nlc_go_back_all_m2!(nlc_status, pivot_variable, spare_idx, number_remaining)

    return (
        pivot_row, sequence_out, direction_in, direction_out,
        theta, alpha, dual_in, value_out, lower_out, upper_out, dual_out,
        primal_tol, nlc_avg_theta,
    )
end

# ==============================================================================
# 2. ClpSimplexPrimal::primalColumn tail
# ==============================================================================

"""
    clp_primal_column_tail!(nlc_status, nlc_bound, nlc_cost2,
                            solution, lower, upper, cost, dj, status,
                            sequence_in, look_both_ways, primal_tol, nlc_infeas_weight, n_infeas)
        -> (sequence_in, value_in, dual_in, lower_in, upper_in, direction_in, n_infeas)

Post-`pivotColumn` tail of `ClpSimplexPrimal::primalColumn` (`CLP_METHOD2`).
`sequence_in` is 1-based (`0` denotes C++ `-1`).
"""
function clp_primal_column_tail!(
    nlc_status::AbstractVector{UInt8},
    nlc_bound::AbstractVector{Float64},
    nlc_cost2::AbstractVector{Float64},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    sequence_in::Int,
    look_both_ways::Bool,
    primal_tol::Float64,
    nlc_infeas_weight::Float64,
    n_infeas::Int,
)::Tuple{Int,Float64,Float64,Float64,Float64,Int,Int}
    sequence_in <= 0 && return (0, 0.0, 0.0, 0.0, 0.0, 0, n_infeas)
    @inbounds begin
        value_in = solution[sequence_in]
        dual_in = dj[sequence_in]
        if look_both_ways
            st = _sp_get_status(status[sequence_in])
            if st == CLP_STATUS_AT_UPPER
                if dual_in < 0.0
                    dual_in -= -nlc_infeas_weight
                    target = _sp_fmadd(2.0, primal_tol, upper[sequence_in])
                    _, n_infeas = clp_nlc_set_one_m2!(
                        nlc_status, nlc_bound, nlc_cost2,
                        lower, upper, cost, status,
                        sequence_in, target, primal_tol, nlc_infeas_weight, n_infeas,
                    )
                    status[sequence_in] = _sp_set_status(status[sequence_in], CLP_STATUS_AT_LOWER)
                end
            elseif st == CLP_STATUS_AT_LOWER
                if dual_in > 0.0
                    dual_in -= nlc_infeas_weight
                    target = _sp_fmsub(2.0, primal_tol, lower[sequence_in])
                    _, n_infeas = clp_nlc_set_one_m2!(
                        nlc_status, nlc_bound, nlc_cost2,
                        lower, upper, cost, status,
                        sequence_in, target, primal_tol, nlc_infeas_weight, n_infeas,
                    )
                    status[sequence_in] = _sp_set_status(status[sequence_in], CLP_STATUS_AT_UPPER)
                end
            end
        end
        lower_in = lower[sequence_in]
        upper_in = upper[sequence_in]
        direction_in = dual_in > 0.0 ? -1 : 1
        return (sequence_in, value_in, dual_in, lower_in, upper_in, direction_in, n_infeas)
    end
end

# ==============================================================================
# 3. ClpSimplexPrimal::updatePrimalsInPrimal
# ==============================================================================

"""
    clp_update_primals_in_primal!(row_val, row_idx, row_nnz, pivot_variable,
                                  solution, lower, upper, cost, dj, status,
                                  nlc_status, nlc_bound, nlc_cost2, nlc_infeas_weight, n_infeas,
                                  pivot_row, sequence_out, dual_in, theta, objective_change,
                                  primal_tol, values_pass)
        -> (new_nnz, objective_change, nlc_change_cost, n_infeas)

Updates basic primal solution and accumulates cost changes (`ClpSimplexPrimal::updatePrimalsInPrimal`,
`CLP_METHOD2`). `pivot_row` and `sequence_out` are 1-based (`0` denotes C++ `-1`,
`pivot_row == -1` denotes C++ `-2`).
"""
function clp_update_primals_in_primal!(
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    pivot_variable::AbstractVector{Int32},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nlc_status::AbstractVector{UInt8},
    nlc_bound::AbstractVector{Float64},
    nlc_cost2::AbstractVector{Float64},
    nlc_infeas_weight::Float64,
    n_infeas::Int,
    pivot_row::Int,
    sequence_out::Int,
    dual_in::Float64,
    theta::Float64,
    objective_change::Float64,
    primal_tol::Float64,
    values_pass::Int,
)::Tuple{Int,Float64,Float64,Int}
    old_cost = 0.0
    @inbounds if pivot_row > 0
        old_cost = cost[sequence_out]
    end

    new_number = 0
    pivot_pos = 0
    nlc_change_cost = 0.0
    relaxed_tol = 1.001 * primal_tol

    @inbounds if values_pass == 0
        for i_index in 1:row_nnz
            i_row = Int(row_idx[i_index])
            a = row_val[i_index]
            row_val[i_index] = 0.0
            i_pivot = Int(pivot_variable[i_row])
            change = theta * a
            val = solution[i_pivot] - change
            solution[i_pivot] = val
            if _sp_is_active(status[i_row]) || theta < 0.0
                status[i_row] = _sp_clear_active(status[i_row])
                if change > 0.0
                    if val <= lower[i_pivot] + primal_tol
                        if i_pivot == sequence_out && val >= lower[i_pivot] - relaxed_tol
                            val = lower[i_pivot]
                        end
                        diff, n_infeas = clp_nlc_set_one_m2!(
                            nlc_status, nlc_bound, nlc_cost2,
                            lower, upper, cost, status,
                            i_pivot, val, primal_tol, nlc_infeas_weight, n_infeas,
                        )
                        nlc_change_cost = _sp_fmadd(val, diff, nlc_change_cost)
                        if diff != 0.0
                            new_number += 1
                            if i_row == pivot_row
                                pivot_pos = new_number
                            end
                            row_val[new_number] = diff
                            dj[i_pivot] = -diff
                            row_idx[new_number] = Int32(i_row)
                        end
                    end
                else
                    if val >= upper[i_pivot] - primal_tol
                        if i_pivot == sequence_out && val < upper[i_pivot] + relaxed_tol
                            val = upper[i_pivot]
                        end
                        diff, n_infeas = clp_nlc_set_one_m2!(
                            nlc_status, nlc_bound, nlc_cost2,
                            lower, upper, cost, status,
                            i_pivot, val, primal_tol, nlc_infeas_weight, n_infeas,
                        )
                        nlc_change_cost = _sp_fmadd(val, diff, nlc_change_cost)
                        if diff != 0.0
                            new_number += 1
                            if i_row == pivot_row
                                pivot_pos = new_number
                            end
                            row_val[new_number] = diff
                            dj[i_pivot] = -diff
                            row_idx[new_number] = Int32(i_row)
                        end
                    end
                end
            end
        end
    else
        for i_index in 1:row_nnz
            i_row = Int(row_idx[i_index])
            a = row_val[i_index]
            row_val[i_index] = 0.0
            i_pivot = Int(pivot_variable[i_row])
            change = theta * a
            val = solution[i_pivot] - change
            solution[i_pivot] = val
            status[i_row] = _sp_clear_active(status[i_row])
            if change > 0.0
                if val <= lower[i_pivot] + primal_tol
                    if i_pivot == sequence_out && val > lower[i_pivot] - relaxed_tol
                        val = lower[i_pivot]
                    end
                    diff, n_infeas = clp_nlc_set_one_m2!(
                        nlc_status, nlc_bound, nlc_cost2,
                        lower, upper, cost, status,
                        i_pivot, val, primal_tol, nlc_infeas_weight, n_infeas,
                    )
                    nlc_change_cost = _sp_fmadd(val, diff, nlc_change_cost)
                    if diff != 0.0
                        new_number += 1
                        if i_row == pivot_row
                            pivot_pos = new_number
                        end
                        row_val[new_number] = diff
                        dj[i_pivot] = -diff
                        row_idx[new_number] = Int32(i_row)
                    end
                end
            else
                if val >= upper[i_pivot] - primal_tol
                    if i_pivot == sequence_out && val < upper[i_pivot] + relaxed_tol
                        val = upper[i_pivot]
                    end
                    diff, n_infeas = clp_nlc_set_one_m2!(
                        nlc_status, nlc_bound, nlc_cost2,
                        lower, upper, cost, status,
                        i_pivot, val, primal_tol, nlc_infeas_weight, n_infeas,
                    )
                    nlc_change_cost = _sp_fmadd(val, diff, nlc_change_cost)
                    if diff != 0.0
                        new_number += 1
                        if i_row == pivot_row
                            pivot_pos = new_number
                        end
                        row_val[new_number] = diff
                        dj[i_pivot] = -diff
                        row_idx[new_number] = Int32(i_row)
                    end
                end
            end
        end
    end

    objective_change += nlc_change_cost
    @inbounds if pivot_row > 0
        d_in = dual_in + (old_cost - cost[sequence_out])
        if pivot_pos > 0
            row_val[pivot_pos] -= d_in
        else
            new_number += 1
            row_val[new_number] = -d_in
            row_idx[new_number] = Int32(pivot_row)
        end
    end

    return (new_number, objective_change, nlc_change_cost, n_infeas)
end

# ==============================================================================
# 4. ClpSimplexPrimal::primalRay
# ==============================================================================

"""
    clp_primal_ray!(ray, row_val, row_idx, row_nnz, packed,
                    pivot_variable, ncols, sequence_in, direction_in) -> Nothing

Builds primal unbounded ray of length `ncols` (`ClpSimplexPrimal::primalRay`).
All indices are 1-based.
"""
function clp_primal_ray!(
    ray::AbstractVector{Float64},
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    packed::Bool,
    pivot_variable::AbstractVector{Int32},
    ncols::Int,
    sequence_in::Int,
    direction_in::Int,
)::Nothing
    @inbounds for i in 1:ncols
        ray[i] = 0.0
    end
    way = -Float64(direction_in)
    zero_tol = 1.0e-12
    @inbounds if sequence_in >= 1 && sequence_in <= ncols
        ray[sequence_in] = Float64(direction_in)
    end
    @inbounds if !packed
        for i in 1:row_nnz
            i_row = Int(row_idx[i])
            i_pivot = Int(pivot_variable[i_row])
            arr_val = row_val[i_row]
            if i_pivot <= ncols && abs(arr_val) >= zero_tol
                ray[i_pivot] = way * arr_val
            end
        end
    else
        for i in 1:row_nnz
            i_row = Int(row_idx[i])
            i_pivot = Int(pivot_variable[i_row])
            arr_val = row_val[i]
            if i_pivot <= ncols && abs(arr_val) >= zero_tol
                ray[i_pivot] = way * arr_val
            end
        end
    end
    return nothing
end

# ==============================================================================
# 5. ClpSimplexPrimal::nextSuperBasic
# ==============================================================================

"""
    clp_primal_next_super_basic!(col_val, col_idx, col_nnz,
                                 dj, solution, lower, upper, status,
                                 ntotal, super_basic_type, first_free,
                                 more_special_options, primal_tol, dual_tol)
        -> (return_value, first_free, col_nnz, more_special_options)

Selects the next superbasic variable (`ClpSimplexPrimal::nextSuperBasic`, `TRY_VALUES_PASS == 0`).
All indices are 1-based (`0` denotes C++ `-1` / none).
"""
function clp_primal_next_super_basic!(
    col_val::AbstractVector{Float64},
    col_idx::AbstractVector{Int32},
    col_nnz::Int,
    dj::AbstractVector{Float64},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    ntotal::Int,
    super_basic_type::Int,
    first_free::Int,
    more_special_options::Int,
    primal_tol::Float64,
    dual_tol::Float64,
)::Tuple{Int,Int,Int,Int}
    return_value = 0
    finished = false
    @inbounds while !finished
        if first_free > 0
            st_byte = status[first_free]
            if !_sp_is_flagged(st_byte) && _sp_get_status(st_byte) == CLP_STATUS_SUPER_BASIC
                return_value = first_free
            end
        end
        i_col = first_free + 1
        if super_basic_type > 1 || (more_special_options & _SP_SORT_VALUES_BIT) != 0
            if super_basic_type > 2 || (col_nnz == 0 && (more_special_options & _SP_SORT_VALUES_BIT) != 0)
                number = 0
                for j_col in 1:ntotal
                    st_byte = status[j_col]
                    if !_sp_is_flagged(st_byte) && _sp_get_status(st_byte) == CLP_STATUS_SUPER_BASIC
                        sol_v = solution[j_col]
                        lo_v = lower[j_col]
                        hi_v = upper[j_col]
                        if abs(sol_v - lo_v) <= primal_tol
                            solution[j_col] = lo_v
                            status[j_col] = _sp_set_status(st_byte, CLP_STATUS_AT_LOWER)
                        elseif abs(sol_v - hi_v) <= primal_tol
                            solution[j_col] = hi_v
                            status[j_col] = _sp_set_status(st_byte, CLP_STATUS_AT_UPPER)
                        elseif lo_v < -1.0e30 && hi_v > 1.0e30
                            status[j_col] = _sp_set_status(st_byte, CLP_STATUS_IS_FREE)
                        else
                            number += 1
                            col_val[number] = -min(sol_v - lo_v, 0.8 * (hi_v - sol_v))
                            col_idx[number] = Int32(j_col)
                        end
                    end
                end
                _sp_coin_sort_2!(col_val, col_idx, number)
                col_nnz = number
                for j in 1:number
                    col_val[j] = 0.0
                end
            end
            if col_nnz == 0
                i_col = ntotal + 1
                return_value = 0
                more_special_options &= ~_SP_SORT_VALUES_BIT
            else
                return_value = Int(col_idx[col_nnz])
                i_col = return_value
                col_nnz -= 1
            end
        else
            while i_col <= ntotal
                st_byte = status[i_col]
                if !_sp_is_flagged(st_byte) && _sp_get_status(st_byte) == CLP_STATUS_SUPER_BASIC
                    sol_v = solution[i_col]
                    lo_v = lower[i_col]
                    hi_v = upper[i_col]
                    if abs(sol_v - lo_v) <= primal_tol
                        solution[i_col] = lo_v
                        status[i_col] = _sp_set_status(st_byte, CLP_STATUS_AT_LOWER)
                    elseif abs(sol_v - hi_v) <= primal_tol
                        solution[i_col] = hi_v
                        status[i_col] = _sp_set_status(st_byte, CLP_STATUS_AT_UPPER)
                    elseif lo_v < -1.0e20 && hi_v > 1.0e20
                        status[i_col] = _sp_set_status(st_byte, CLP_STATUS_IS_FREE)
                        if abs(dj[i_col]) > dual_tol
                            break
                        end
                    else
                        break
                    end
                end
                i_col += 1
            end
        end
        first_free = i_col
        finished = true
        if first_free == ntotal + 1
            first_free = 0
        end
        if return_value > 0
            st_ret = _sp_get_status(status[return_value])
            if st_ret != CLP_STATUS_SUPER_BASIC && st_ret != CLP_STATUS_IS_FREE
                finished = false
            end
        end
    end
    return (return_value, first_free, col_nnz, more_special_options)
end

# ==============================================================================
# 6. ClpSimplexPrimal::unflag, clearAll, alwaysOptimal, exactOutgoing
# ==============================================================================

"""
    clp_primal_unflag!(dj, status, ntotal, dual_tol, largest_dual_err) -> Int

Clears `CLP_STATUS_FLAGGED_BIT` on `1:ntotal` and returns the number of unflagged
variables with `abs(dj[i]) > dual_tol + min(1e-2, 10 * largest_dual_err)`
(`ClpSimplexPrimal::unflag` core loop).
"""
function clp_primal_unflag!(
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    ntotal::Int,
    dual_tol::Float64,
    largest_dual_err::Float64,
)::Int
    number_flagged = 0
    relaxed_tol_d = dual_tol + min(1.0e-2, 10.0 * largest_dual_err)
    @inbounds for i in 1:ntotal
        st_byte = status[i]
        if _sp_is_flagged(st_byte)
            status[i] = _sp_clear_flagged(st_byte)
            if abs(dj[i]) > relaxed_tol_d
                number_flagged += 1
            end
        end
    end
    return number_flagged
end

"""
    clp_primal_clear_all!(row_val, row_idx, row_nnz, row_cap, packed, status) -> Int

Clears active bits on `row_idx[1:row_nnz]`, clears dense entries in `row_val`, and
returns `0` (`ClpSimplexPrimal::clearAll` core loop).
"""
function clp_primal_clear_all!(
    row_val::AbstractVector{Float64},
    row_idx::AbstractVector{Int32},
    row_nnz::Int,
    row_cap::Int,
    packed::Bool,
    status::AbstractVector{UInt8},
)::Int
    @inbounds if packed
        for i in 1:row_nnz
            i_row = Int(row_idx[i])
            status[i_row] = _sp_clear_active(status[i_row])
            row_val[i] = 0.0
        end
    else
        for i in 1:row_nnz
            i_row = Int(row_idx[i])
            status[i_row] = _sp_clear_active(status[i_row])
            row_val[i_row] = 0.0
        end
        if 3 * row_nnz >= row_cap
            for i in 1:row_cap
                row_val[i] = 0.0
            end
        end
    end
    return 0
end

@inline clp_primal_always_optimal(special_options::Int)::Bool = (special_options & 1) != 0
@inline clp_primal_always_optimal!(special_options::Int, on_off::Bool)::Int =
    on_off ? (special_options | 1) : (special_options & ~1)

@inline clp_primal_exact_outgoing(special_options::Int)::Bool = (special_options & 4) != 0
@inline clp_primal_exact_outgoing!(special_options::Int, on_off::Bool)::Int =
    on_off ? (special_options | 4) : (special_options & ~4)

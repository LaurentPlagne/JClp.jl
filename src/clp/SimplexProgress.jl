# Pure idiomatic 1-based Julia implementation of ClpSimplexProgress (B11c-1).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Compatible with JuliaC --trim=safe and works both on native Vector{T} and bridge views.

const CLP_PROGRESS = 5
const CLP_CYCLE = 12

# Exact bit-level double comparison matching C++ ClpSolve.cpp::equalDouble.
@inline _spg_equal_double(a::Float64, b::Float64)::Bool =
    reinterpret(UInt64, a) == reinterpret(UInt64, b)

"""
    ClpSimplexProgressState

Pure Julia representation of `ClpSimplexProgress` (for standalone Julia simplex drivers).
"""
mutable struct ClpSimplexProgressState
    objective::Vector{Float64}
    infeasibility::Vector{Float64}
    real_infeasibility::Vector{Float64}
    number_infeasibilities::Vector{Int32}
    iteration_number::Vector{Int32}
    in_seq::Vector{Int32}
    out_seq::Vector{Int32}
    way::Vector{Int8}
    initial_weight::Float64
    number_times::Int
    number_bad_times::Int
    number_really_bad_times::Int
    number_times_flagged::Int
    odd_state::Int
    check_scaling_after::Int
end

function ClpSimplexProgressState()
    big = floatmax(Float64) * 1.0e-50
    return ClpSimplexProgressState(
        fill(big, CLP_PROGRESS),
        fill(-1.0, CLP_PROGRESS),
        fill(big, CLP_PROGRESS),
        fill(Int32(-1), CLP_PROGRESS),
        fill(Int32(-1), CLP_PROGRESS),
        fill(Int32(-1), CLP_CYCLE),
        fill(Int32(-1), CLP_CYCLE),
        fill(Int8(0), CLP_CYCLE),
        0.0, 0, 0, 0, 0, 0, 0,
    )
end

function clp_progress_start_check!(
    in_arr::AbstractVector{Int32},
    out_arr::AbstractVector{Int32},
    way_arr::AbstractVector{Int8},
)::Nothing
    @inbounds for i in 1:CLP_CYCLE
        in_arr[i] = Int32(-1)
        out_arr[i] = Int32(-1)
        way_arr[i] = Int8(0)
    end
    return nothing
end

function clp_progress_reset!(
    objective::AbstractVector{Float64},
    infeasibility::AbstractVector{Float64},
    real_infeasibility::AbstractVector{Float64},
    number_infeasibilities::AbstractVector{Int32},
    iteration_number::AbstractVector{Int32},
    in_arr::AbstractVector{Int32},
    out_arr::AbstractVector{Int32},
    way_arr::AbstractVector{Int8},
    algorithm::Int,
)::Nothing
    big = floatmax(Float64) * 1.0e-50
    obj_init = algorithm >= 0 ? big : -big
    @inbounds for i in 1:CLP_PROGRESS
        objective[i] = obj_init
        infeasibility[i] = -1.0
        real_infeasibility[i] = big
        number_infeasibilities[i] = Int32(-1)
        iteration_number[i] = Int32(-1)
    end
    clp_progress_start_check!(in_arr, out_arr, way_arr)
    return nothing
end

"""
    clp_progress_cycle!(in_arr, out_arr, way_arr, in_seq, out_seq, way_in, way_out) -> Int

Matches `ClpSimplexProgress::cycle(int in, int out, int wayIn, int wayOut)`.
Note: `in_arr`/`out_arr` store raw C++ sequence numbers (with `-1` as empty sentinel)
so equality checks are invariant to 0-based vs 1-based offsets.
"""
function clp_progress_cycle!(
    in_arr::AbstractVector{Int32},
    out_arr::AbstractVector{Int32},
    way_arr::AbstractVector{Int8},
    in_seq::Int32,
    out_seq::Int32,
    way_in::Int,
    way_out::Int,
)::Int
    matched = 0
    @inbounds begin
        for i in 2:CLP_CYCLE
            if in_seq == out_arr[i]
                matched = -1
                break
            end
        end
        if matched != 0 && in_arr[1] >= Int32(0)
            matched = 0
            n_matched = 0
            way0 = way_arr[1]
            in0 = in_arr[1]
            out0 = out_arr[1]
            # k in C++ is 1 .. CLP_CYCLE - 5 (0-based), so 1-based index is k+1 in 2 .. CLP_CYCLE - 4
            for k in 1:(CLP_CYCLE - 5)
                if in0 == in_arr[k + 1] && out0 == out_arr[k + 1] && way0 == way_arr[k + 1]
                    n_matched += 1
                    end_j = CLP_CYCLE - k
                    j = 1
                    while j < end_j
                        if in_arr[j + k + 1] != in_arr[j + 1] ||
                           out_arr[j + k + 1] != out_arr[j + 1] ||
                           way_arr[j + k + 1] != way_arr[j + 1]
                            break
                        end
                        j += 1
                    end
                    if j == end_j
                        matched = k
                        break
                    end
                end
            end
            if matched <= 0 && n_matched > 1
                matched = 100
            end
        end
        for i in 1:(CLP_CYCLE - 1)
            in_arr[i] = in_arr[i + 1]
            out_arr[i] = out_arr[i + 1]
            way_arr[i] = way_arr[i + 1]
        end
        way_val = Int8(1 - way_in + 4 * (1 - way_out))
        in_arr[CLP_CYCLE] = in_seq
        out_arr[CLP_CYCLE] = out_seq
        way_arr[CLP_CYCLE] = way_val
    end
    return matched
end

# Action flags returned by `clp_progress_looping_step!` to caller/C++ bridge:
const CLP_LOOP_ACT_NONE          = 0
const CLP_LOOP_ACT_DUAL_TOL      = 1  # dual: tol *= 1.05; if dual_bound < 1e17: dual_bound *= 1.1 + resetFakeBounds(0)
const CLP_LOOP_ACT_PRIMAL_TOL    = 2  # primal: if nlc_num_infeas > 0 && infeas_cost < 1e17: infeas_cost *= 1.1
const CLP_LOOP_ACT_FLAG_SEQ      = 3  # flag out_flag_seq (0-based C++ sequence in in_[CLP_CYCLE-1] or out_[CLP_CYCLE-1])
const CLP_LOOP_ACT_GIVE_UP_FLAG  = 4  # all flagged -> return 4
const CLP_LOOP_ACT_VICTORY       = 5  # infeasibility < 1e-4 -> return 0
const CLP_LOOP_ACT_ABORT_LOOP    = 6  # print CLP_LOOP -> return 3

"""
    clp_progress_looping_step!(...)

Core state transition of `ClpSimplexProgress::looping()`.
Returns `(ret_code, action, number_matched, matched_mask, number_times, number_bad_times,
          number_really_bad_times, force_factorization, flag_seq)`.
"""
function clp_progress_looping_step!(
    objective_hist::AbstractVector{Float64},
    infeas_hist::AbstractVector{Float64},
    real_infeas_hist::AbstractVector{Float64},
    num_infeas_hist::AbstractVector{Int32},
    iter_num_hist::AbstractVector{Int32},
    in_arr::AbstractVector{Int32},
    out_arr::AbstractVector{Int32},
    way_arr::AbstractVector{Int8},
    algorithm::Int,
    objective::Float64,
    infeasibility::Float64,
    real_infeasibility::Float64,
    number_infeasibilities::Int,
    iteration_number::Int,
    nrows::Int,
    ncols::Int,
    progress_flag::Bool,
    special_options::Int,
    matrix_type::Int,
    number_times::Int,
    number_bad_times::Int,
    number_really_bad_times::Int,
    force_factorization::Int,
)::Tuple{Int,Int,Int,Int,Int,Int,Int,Int,Int32}
    if algorithm >= 0 && iteration_number > 3 * nrows + 3 * ncols
        return (1, CLP_LOOP_ACT_NONE, 0, 0, number_times, number_bad_times,
                number_really_bad_times, force_factorization, Int32(-1))
    end

    number_matched = 0
    matched = 0
    nsame = 0
    @inbounds for i in 1:CLP_PROGRESS
        matched_obj = _spg_equal_double(objective, objective_hist[i])
        matched_inf = _spg_equal_double(infeasibility, infeas_hist[i])
        matched_ninf = (Int32(number_infeasibilities) == num_infeas_hist[i])
        if matched_obj && matched_inf && matched_ninf
            matched |= (1 << (i - 1))
            if Int32(iteration_number) != iter_num_hist[i]
                number_matched += 1
            else
                nsame += 1
            end
        end
        if i > 1
            objective_hist[i - 1] = objective_hist[i]
            infeas_hist[i - 1] = infeas_hist[i]
            real_infeas_hist[i - 1] = real_infeas_hist[i]
            num_infeas_hist[i - 1] = num_infeas_hist[i]
            iter_num_hist[i - 1] = iter_num_hist[i]
        end
    end
    @inbounds begin
        objective_hist[CLP_PROGRESS] = objective
        infeas_hist[CLP_PROGRESS] = infeasibility
        real_infeas_hist[CLP_PROGRESS] = real_infeasibility
        num_infeas_hist[CLP_PROGRESS] = Int32(number_infeasibilities)
        iter_num_hist[CLP_PROGRESS] = Int32(iteration_number)
    end

    if nsame == CLP_PROGRESS
        number_matched = CLP_PROGRESS
    end
    if progress_flag
        number_matched = 0
    end
    number_times += 1
    if number_times < 10
        number_matched = 0
    end
    if matched == (1 << (CLP_PROGRESS - 1))
        number_matched = 0
    end

    if iteration_number > 20 * nrows + 5 * ncols + 100 && (special_options & 0x03000000) != 0
        if iteration_number < 25 * nrows + 8 * ncols + 300 && number_really_bad_times < 100
            force_factorization = min(force_factorization, 5)
            number_really_bad_times += 1
        else
            number_matched = 1000
            number_bad_times = 100
        end
    end

    if number_matched != 0 && matrix_type < 15
        number_bad_times += 1
        if number_bad_times < 10
            force_factorization = 1
            if number_bad_times < 2
                clp_progress_start_check!(in_arr, out_arr, way_arr)
                act = algorithm < 0 ? CLP_LOOP_ACT_DUAL_TOL : CLP_LOOP_ACT_PRIMAL_TOL
                return (-2, act, number_matched, matched, number_times, number_bad_times,
                        number_really_bad_times, force_factorization, Int32(-1))
            else
                seq = @inbounds (algorithm < 0 ? in_arr[CLP_CYCLE] : out_arr[CLP_CYCLE])
                if seq >= Int32(0)
                    clp_progress_start_check!(in_arr, out_arr, way_arr)
                    number_bad_times = 2
                    return (-2, CLP_LOOP_ACT_FLAG_SEQ, number_matched, matched, number_times,
                            number_bad_times, number_really_bad_times, force_factorization, seq)
                else
                    return (4, CLP_LOOP_ACT_GIVE_UP_FLAG, number_matched, matched, number_times,
                            number_bad_times, number_really_bad_times, force_factorization, Int32(-1))
                end
            end
        else
            if infeasibility < 1.0e-4
                return (0, CLP_LOOP_ACT_VICTORY, number_matched, matched, number_times,
                        number_bad_times, number_really_bad_times, force_factorization, Int32(-1))
            else
                return (3, CLP_LOOP_ACT_ABORT_LOOP, number_matched, matched, number_times,
                        number_bad_times, number_really_bad_times, force_factorization, Int32(-1))
            end
        end
    end

    return (-1, CLP_LOOP_ACT_NONE, number_matched, matched, number_times, number_bad_times,
            number_really_bad_times, force_factorization, Int32(-1))
end

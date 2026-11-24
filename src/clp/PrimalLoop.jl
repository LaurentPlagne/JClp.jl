# Pure idiomatic 1-based Julia implementation of ClpSimplexPrimal::pivotResult (B12b-1).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Composes B8 (PackedMatrix), B9b (PrimalColumnPivot), B9c (NonLinearCost),
# B10 (Factorization/SparseLU), B11b (SimplexPrimal), and B11c (SimplexBase + SimplexProgress)
# in a single pure-Julia call per primal pivot.

const CLP_PL_EXIT_DONE                = 0  # pivotResult finished normally
const CLP_PL_EXIT_SAVE_DJ_CHECK_INF   = 1  # saveDj reject -> C++ calls nonLinearCost_->checkInfeasibilities(0.0)
const CLP_PL_EXIT_SAVE_DJ_FLAG_W      = 2  # saveDj reject at lastGoodIteration -> C++ calls saveWeights(this,6) + saferTolerances
const CLP_PL_EXIT_UNBOUNDED_RAY       = 3  # pivotRow == -1 unbounded -> C++ builds primalRay(rowArray_[1])
const CLP_PL_EXIT_FALLBACK_ITER       = 4  # whileIterating fallback to C++ for 1 iteration (e.g. Dantzig/mode switch/anyUpdates==2)

@inline function _pl_clear_packed!(val::AbstractVector{Float64}, nnz::Int)::Int
    @inbounds for i in 1:nnz
        val[i] = 0.0
    end
    return 0
end

"""
    clp_primal_pivot_result!(...) -> Tuple

Pure 1-based Julia translation of `ClpSimplexPrimal::pivotResult(0)` for standard primal simplex
(`solveType_ == 1`, `ifValuesPass == 0`, `ClpNonLinearCost` method 2, `ClpFactorization`,
`ClpPackedMatrix`, `ClpPrimalColumnSteepest` or no-op weight update).

All row/sequence indices are 1-based (`0` denotes C++ `-1`, `-1` for `pivot_row` denotes C++ `-2` bound flip).
"""
function clp_primal_pivot_result!(
    cf::ClpFactorization,
    mat_starts::AbstractVector{Int32},
    mat_lengths::AbstractVector{Int32},
    mat_indices::AbstractVector{Int32},
    mat_elements::AbstractVector{Float64},
    has_scale::Bool,
    row_scale::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nlc_status::AbstractVector{UInt8},
    nlc_bound::AbstractVector{Float64},
    nlc_cost2::AbstractVector{Float64},
    has_pcs::Bool,
    pcs_weights::AbstractVector{Float64},
    pcs_ref::AbstractVector{UInt32},
    pcs_has_ref::Bool,
    pcs_alt_val::AbstractVector{Float64},
    pcs_alt_idx::AbstractVector{Int32},
    pcs_alt_nnz::Int,
    pcs_mode::Int,
    pcs_num_switched::Int,
    pcs_pivot_seq::Int,
    pcs_devex::Float64,
    prog_in::AbstractVector{Int32},
    prog_out::AbstractVector{Int32},
    prog_way::AbstractVector{Int8},
    r1_val::AbstractVector{Float64},
    r1_idx::AbstractVector{Int32},
    r2_val::AbstractVector{Float64},
    r2_idx::AbstractVector{Int32},
    r3_val::AbstractVector{Float64},
    r3_idx::AbstractVector{Int32},
    nrows::Int,
    ncols::Int,
    primal_tol::Float64,
    dual_tol::Float64,
    infeas_cost::Float64,
    large_value::Float64,
    obj_scale::Float64,
    rhs_scale::Float64,
    largest_dual_err::Float64,
    nlc_infeas_weight::Float64,
    nlc_sum_infeas::Float64,
    max_iterations::Int,
    special_options::Int,
    obj_type::Int,
    last_good_iter::Int,
    # mutable scalars
    problem_status::Int,
    number_iterations::Int,
    change_made::Int,
    progress_flag::Int,
    last_bad_iter::Int,
    force_factorization::Int,
    prog_bad_times::Int,
    nlc_num_infeas::Int,
    nlc_change_cost::Float64,
    nlc_avg_theta::Float64,
    pivot_row::Int,
    seq_in::Int,
    seq_out::Int,
    dir_in::Int,
    dir_out::Int,
    val_in::Float64,
    lo_in::Float64,
    hi_in::Float64,
    dual_in::Float64,
    val_out::Float64,
    lo_out::Float64,
    hi_out::Float64,
    dual_out::Float64,
    theta::Float64,
    alpha::Float64,
    alpha_accuracy::Float64,
    objective_value::Float64,
    acceptable_pivot::Float64,
    more_special_options::Int,
    rng_seed::UInt32,
)
    return_code = -1
    need_init_weights = 0
    r1_nnz = 0
    dbl_max = floatmax(Float64)

    # 1. unpackPacked(rowArray_[1])
    if seq_in > ncols
        @inbounds begin
            r1_val[1] = -1.0
            r1_idx[1] = Int32(seq_in - ncols)
        end
        r1_nnz = 1
    else
        if has_scale
            r1_nnz = clp_mat_unpack_packed!(
                mat_starts, mat_lengths, mat_indices, mat_elements,
                seq_in, row_scale, col_scale, r1_idx, r1_val,
            )
        else
            r1_nnz = clp_mat_unpack_packed!(
                mat_starts, mat_lengths, mat_indices, mat_elements,
                seq_in, nothing, nothing, r1_idx, r1_val,
            )
        end
    end

    save_dj = dual_in

    # 2. factorization_->updateColumnFT(rowArray_[2], rowArray_[1])
    r1_nnz = clp_fact_update_column_ft!(
        cf, r2_val, r2_idx,
        r1_val, r1_idx, r1_nnz, true,
    )

    # 3. primalRow(rowArray_[1], rowArray_[3], rowArray_[2], 0)
    fpiv = cf.lu.npivots
    pivot_row, seq_out, dir_in, dir_out,
    theta, alpha, dual_in, val_out, lo_out, hi_out, dual_out,
    primal_tol, nlc_avg_theta = clp_primal_row!(
        r1_val, r1_idx, r1_nnz,
        r3_val, r3_idx, r2_val, r2_idx,
        pivot_var, cost, solution, lower, upper, dj, status,
        nlc_status, nlc_bound, nlc_infeas_weight, nlc_num_infeas, nlc_avg_theta,
        0, obj_type, seq_in, dir_in, lo_in, hi_in, val_in, dual_in,
        primal_tol, dual_tol, infeas_cost, large_value, acceptable_pivot,
        fpiv, number_iterations, special_options,
        seq_out, dir_out, alpha, val_out, lo_out, hi_out, dual_out,
    )

    # 4. saveDj vs dualIn accuracy check (solveType_ == 1, ifValuesPass == 0)
    check_val = largest_dual_err > 1.0e-5 ? 1.0e-1 : 1.0e-2
    if (save_dj * dual_in < 1.0e-20 ||
        abs(save_dj - dual_in) > check_val * (1.0 + abs(save_dj)) ||
        abs(dual_in) < dual_tol)
        if !(save_dj * dual_in > 0.0 && min(abs(save_dj), abs(dual_in)) > 1.0e5)
            if last_good_iter != number_iterations
                r1_nnz = _pl_clear_packed!(r1_val, r1_nnz)
                pivot_row = 0
                return_code = -4
                seq_out = 0
                exit_c = (last_good_iter + 1 == number_iterations) ? CLP_PL_EXIT_SAVE_DJ_CHECK_INF : CLP_PL_EXIT_DONE
                return (exit_c, return_code, problem_status, number_iterations,
                        change_made, progress_flag, last_bad_iter, force_factorization,
                        prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                        r1_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                        pivot_row, seq_out, dir_in, dir_out,
                        val_in, val_out, lo_out, hi_out, dual_in, dual_out,
                        theta, alpha, alpha_accuracy, objective_value,
                        primal_tol, acceptable_pivot, rng_seed)
            else
                if ((save_dj * dual_in < 1.0e-20 ||
                     abs(save_dj - dual_in) > 2.0e-1 * (1.0 + abs(dual_in)) ||
                     abs(dual_in) < dual_tol) &&
                    (abs(save_dj) > abs(dual_in) || save_dj * dual_in < 1.0e-4 || fpiv != 0))
                    @inbounds status[seq_in] |= CLP_STATUS_FLAGGED_BIT
                    return (CLP_PL_EXIT_SAVE_DJ_FLAG_W, return_code, problem_status, number_iterations,
                            change_made, progress_flag, last_bad_iter, force_factorization,
                            prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                            r1_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                            pivot_row, seq_out, dir_in, dir_out,
                            val_in, val_out, lo_out, hi_out, dual_in, dual_out,
                            theta, alpha, alpha_accuracy, objective_value,
                            primal_tol, acceptable_pivot, rng_seed)
                end
            end
        end
    end

    # 5. Basis update (pivot_row > 0) or unbounded / bound flip
    if pivot_row > 0
        n_new = seq_in <= ncols ? @inbounds(Int(mat_lengths[seq_in])) : 1
        n_old = seq_out <= ncols ? @inbounds(Int(mat_lengths[seq_out])) : 1
        upd_status = clp_fact_replace_column!(
            cf, r2_val, r2_idx, r1_val, r1_idx, r1_nnz, true,
            pivot_row, alpha, (more_special_options & 16) != 0, n_new, n_old,
        )
        if upd_status == 2 && last_good_iter == number_iterations && abs(alpha) > 1.0e-5
            upd_status = 4
        end
        if upd_status == 1 || upd_status == 4
            if cf.lu.npivots > 5 || upd_status == 4
                return_code = -3
            end
        elseif upd_status == 2
            cf.lu.zero_tol = min(cf.lu.zero_tol, 1.0e-15)
            max_fac = cf.lu.max_pivots
            if max_fac > 10
                if force_factorization < 0
                    force_factorization = max_fac
                end
                force_factorization = max(1, force_factorization >> 1)
            end
            if last_good_iter != number_iterations
                r1_nnz = _pl_clear_packed!(r1_val, r1_nnz)
                pivot_row = 0
                return_code = -4
                return (CLP_PL_EXIT_DONE, return_code, problem_status, number_iterations,
                        change_made, progress_flag, last_bad_iter, force_factorization,
                        prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                        r1_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                        pivot_row, seq_out, dir_in, dir_out,
                        val_in, val_out, lo_out, hi_out, dual_in, dual_out,
                        theta, alpha, alpha_accuracy, objective_value,
                        primal_tol, acceptable_pivot, rng_seed)
            else
                @inbounds status[seq_in] |= CLP_STATUS_FLAGGED_BIT
                prog_bad_times = 0
                last_bad_iter = number_iterations
                r1_nnz = _pl_clear_packed!(r1_val, r1_nnz)
                pivot_row = 0
                seq_out = 0
                return_code = -5
                return (CLP_PL_EXIT_DONE, return_code, problem_status, number_iterations,
                        change_made, progress_flag, last_bad_iter, force_factorization,
                        prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                        r1_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                        pivot_row, seq_out, dir_in, dir_out,
                        val_in, val_out, lo_out, hi_out, dual_in, dual_out,
                        theta, alpha, alpha_accuracy, objective_value,
                        primal_tol, acceptable_pivot, rng_seed)
            end
        elseif upd_status == 3
            if cf.lu.npivots < 0.5 * cf.lu.max_pivots && cf.lu.npivots < 200
                cf.lu.area_factor *= 1.1
            end
            return_code = -2
        elseif upd_status == 5
            problem_status = -2
        end

        # primalColumnPivot_->updateWeights(rowArray_[1])
        if has_pcs
            pcs_devex, pcs_alt_nnz, pcs_pivot_seq, niw = clp_pcs_update_weights_packed!(
                r1_val, r1_idx, r1_nnz, pivot_var, status, nrows + ncols,
                pcs_weights, pcs_ref, pcs_has_ref,
                pcs_alt_val, pcs_alt_idx, pcs_alt_nnz,
                pcs_mode, pcs_num_switched, pivot_row, pcs_pivot_seq, seq_in, seq_out,
                alpha, pcs_devex,
            )
            if niw
                need_init_weights = 1
            end
        end
    else
        if pivot_row == 0
            # no outgoing row is valid (C++ pivotRow_ == -1)
            if val_out != dbl_max
                theta = val_out - val_in
                r1_nnz, _, d_cc, nlc_num_infeas = clp_update_primals_in_primal!(
                    r1_val, r1_idx, r1_nnz, pivot_var,
                    solution, lower, upper, cost, dj, status,
                    nlc_status, nlc_bound, nlc_cost2, nlc_infeas_weight, nlc_num_infeas,
                    pivot_row, seq_out, dual_in, theta, 0.0, primal_tol, 0,
                )
                nlc_change_cost += d_cc
                @inbounds solution[seq_in] += theta
            end
            if cf.lu.npivots == 0 && acceptable_pivot <= 1.0e-8
                return_code = 2
                exit_c = (nlc_sum_infeas == 0.0) ? CLP_PL_EXIT_UNBOUNDED_RAY : CLP_PL_EXIT_DONE
                return (exit_c, return_code, problem_status, number_iterations,
                        change_made, progress_flag, last_bad_iter, force_factorization,
                        prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                        r1_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                        pivot_row, seq_out, dir_in, dir_out,
                        val_in, val_out, lo_out, hi_out, dual_in, dual_out,
                        theta, alpha, alpha_accuracy, objective_value,
                        primal_tol, acceptable_pivot, rng_seed)
            else
                acceptable_pivot = 1.0e-8
                return_code = 4
                return (CLP_PL_EXIT_DONE, return_code, problem_status, number_iterations,
                        change_made, progress_flag, last_bad_iter, force_factorization,
                        prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                        r1_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                        pivot_row, seq_out, dir_in, dir_out,
                        val_in, val_out, lo_out, hi_out, dual_in, dual_out,
                        theta, alpha, alpha_accuracy, objective_value,
                        primal_tol, acceptable_pivot, rng_seed)
            end
        end
        # else pivot_row == -1 (C++ -2): bound flip
    end

    # 6. updatePrimalsInPrimal
    old_cost = seq_out > 0 ? @inbounds(cost[seq_out]) : 0.0
    r1_nnz, obj_chg, d_cc, nlc_num_infeas = clp_update_primals_in_primal!(
        r1_val, r1_idx, r1_nnz, pivot_var,
        solution, lower, upper, cost, dj, status,
        nlc_status, nlc_bound, nlc_cost2, nlc_infeas_weight, nlc_num_infeas,
        pivot_row, seq_out, dual_in, theta, 0.0, primal_tol, 0,
    )
    nlc_change_cost += d_cc

    # 7. Update incoming/outgoing values & NonLinearCost
    old_val = val_in
    if dir_in == -1
        val_in = (seq_in != seq_out) ? (val_in - abs(theta)) : lo_in
    else
        val_in = (seq_in != seq_out) ? (val_in + abs(theta)) : hi_in
    end
    obj_chg += dual_in * (val_in - old_val)

    if seq_in != seq_out
        val_out = dir_out > 0 ? lo_out : hi_out
        @inbounds begin
            lo_so = lower[seq_out]
            hi_so = upper[seq_out]
        end
        if val_out < lo_so - primal_tol
            val_out = lo_so - 0.9 * primal_tol
        elseif val_out > hi_so + primal_tol
            val_out = hi_so + 0.9 * primal_tol
        end
        dir_out, val_out, d_cc_out, nlc_num_infeas = clp_nlc_set_one_outgoing_m2!(
            nlc_status, nlc_bound, nlc_cost2, lower, upper, cost,
            seq_out, val_out, primal_tol, nlc_infeas_weight, nlc_num_infeas,
        )
        nlc_change_cost += d_cc_out
        @inbounds begin
            dj[seq_out] = cost[seq_out] - old_cost
            solution[seq_out] = val_out
        end
    end

    d_cc_in, nlc_num_infeas = clp_nlc_set_one_m2!(
        nlc_status, nlc_bound, nlc_cost2, lower, upper, cost, status,
        seq_in, val_in, primal_tol, nlc_infeas_weight, nlc_num_infeas,
    )
    nlc_change_cost += val_in * d_cc_in

    # 8. housekeeping (exact short-circuit timeToRefactorize & RNG in Julia)
    hit_max = (number_iterations + 1 >= max_iterations)
    hk_piv_row = pivot_row > 0 ? pivot_row : 0
    hk_ret, hk_act, hk_cyc, number_iterations, change_made, progress_flag,
    alpha_accuracy, objective_value, _ = clp_housekeeping!(
        pivot_var, solution, lower, upper, status,
        prog_in, prog_out, prog_way,
        nrows, ncols, hk_piv_row, seq_in, seq_out, dir_in, dir_out,
        val_in, val_out, alpha, alpha_accuracy, objective_value,
        obj_chg, obj_scale, rhs_scale,
        number_iterations, max_iterations, change_made, progress_flag,
        special_options, obj_type, 1, hit_max,
        cf.lu.npivots, cf.lu.max_pivots,
        false, false, -1,
    )
    what_next = hk_ret
    if hk_ret != 0
        if hk_act == CLP_HK_ACT_CYCLE
            rand_c, rng_seed = _dl_rand_double(rng_seed)
            extra = unsafe_trunc(Int, 9.999 * rand_c)
            off_val = extra < 4 ? 1 : (extra < 7 ? 2 : (extra < 9 ? 3 : 4))
            if cf.lu.npivots > hk_cyc
                force_factorization = max(1, hk_cyc - off_val)
            else
                @inbounds status[seq_out] |= CLP_STATUS_FLAGGED_BIT
            end
        end
    else
        dont_invert = ((special_options & 16384) != 0 && number_iterations * 3 > 2 * max_iterations)
        if clp_fact_time_to_refactorize!(cf) && !dont_invert
            what_next = 1
        elseif force_factorization > 0 && cf.lu.npivots == force_factorization
            force_factorization = (3 + 5 * force_factorization) ÷ 4
            if force_factorization > cf.lu.max_pivots
                force_factorization = -1
            end
            what_next = 1
        elseif hk_act == CLP_HK_ACT_RAND_REFAC
            rand_r, rng_seed = _dl_rand_double(rng_seed)
            while rand_r < 0.45
                rand_r *= 2.0
            end
            max_num = force_factorization < 0 ? cf.lu.max_pivots : min(force_factorization, cf.lu.max_pivots)
            if Float64(cf.lu.npivots) >= rand_r * Float64(max_num)
                what_next = 1
            elseif number_iterations > 1000000 + 10 * (nrows + (ncols >> 2)) &&
                   number_iterations < 1001000 + 10 * (nrows + (ncols >> 2))
                what_next = 1
            end
        end
    end

    if what_next == 1
        return_code = -2
    elseif what_next == 2
        return_code = 3
    elseif number_iterations == last_good_iter + 2 * cf.lu.max_pivots
        return_code = -2
    end

    return (CLP_PL_EXIT_DONE, return_code, problem_status, number_iterations,
            change_made, progress_flag, last_bad_iter, force_factorization,
            prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
            r1_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
            pivot_row, seq_out, dir_in, dir_out,
            val_in, val_out, lo_out, hi_out, dual_in, dual_out,
            theta, alpha, alpha_accuracy, objective_value,
            primal_tol, acceptable_pivot, rng_seed)
end

# ==============================================================================
# B12b-2: ClpSimplexPrimal::whileIterating pure-Julia inner loop
# ==============================================================================

# Matches ClpPackedMatrix::transposeTimes(model, -1.0, pi1 (packed), y, c0) when !columnCopy_ && !(flags_ & 2)
function _pl_transpose_times_btran!(
    tt_starts::AbstractVector{Int32},
    tt_indices::AbstractVector{Int32},
    tt_elements::AbstractVector{Float64},
    tt_has_scale::Bool,
    row_scale::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
    has_row_copy::Bool,
    rc_starts::AbstractVector{Int32},
    rc_cols::AbstractVector{Int32},
    rc_elements::AbstractVector{Float64},
    r1_val::AbstractVector{Float64},
    r1_idx::AbstractVector{Int32},
    r1_nnz::Int,
    y_val::AbstractVector{Float64},
    c0_val::AbstractVector{Float64},
    c0_idx::AbstractVector{Int32},
    c1_idx::AbstractVector{Int32},
    marked_buf::Vector{UInt8},
    work_col::Vector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
    zero_tol::Float64,
)::Int
    factor = 0.5
    if ncols * 8 > 1000000
        if nrows * 10 < ncols
            factor *= 0.333333333
        elseif nrows * 4 < ncols
            factor *= 0.5
        elseif nrows * 2 < ncols
            factor *= 0.66666666667
        end
    end

    if Float64(r1_nnz) > factor * Float64(nrows) || !has_row_copy
        if !tt_has_scale
            @inbounds for i in 1:r1_nnz
                y_val[Int(r1_idx[i])] = -r1_val[i]
            end
            c0_nnz = clp_mat_guts_tt_unscaled_status!(
                tt_starts, tt_indices, tt_elements, ncols,
                y_val, c0_idx, c0_val, status, zero_tol,
            )
        else
            @inbounds for i in 1:r1_nnz
                irow = Int(r1_idx[i])
                y_val[irow] = -r1_val[i] * row_scale[irow]
            end
            c0_nnz = clp_mat_guts_tt_scaled_status!(
                tt_starts, tt_indices, tt_elements, ncols,
                y_val, col_scale, c0_idx, c0_val, status, zero_tol,
            )
        end
        @inbounds for i in 1:r1_nnz
            y_val[Int(r1_idx[i])] = 0.0
        end
        return c0_nnz
    else
        if r1_nnz == 1
            return clp_mat_tt_by_row_eq1_packed!(
                rc_starts, rc_cols, rc_elements,
                @inbounds(Int(r1_idx[1])), @inbounds(r1_val[1]),
                c0_idx, c0_val, zero_tol, -1.0,
            )
        elseif r1_nnz == 2
            return clp_mat_tt_by_row_eq2_packed!(
                rc_starts, rc_cols, rc_elements,
                @inbounds(Int(r1_idx[1])), @inbounds(Int(r1_idx[2])),
                @inbounds(r1_val[1]), @inbounds(r1_val[2]),
                c0_idx, c0_val, marked_buf, c1_idx, zero_tol, -1.0,
            )
        else
            n_cov = 0
            sparse = true
            @inbounds for i in 1:r1_nnz
                ir = Int(r1_idx[i])
                n_cov += Int(rc_starts[ir + 1]) - Int(rc_starts[ir])
                if n_cov > ncols
                    sparse = false
                    break
                end
            end
            if sparse
                return clp_mat_tt_by_row_ge3_packed!(
                    rc_starts, rc_cols, rc_elements,
                    r1_idx, r1_val, r1_nnz,
                    c0_idx, c0_val, work_col, zero_tol, -1.0,
                )
            else
                return clp_mat_tt_by_row_gek_packed!(
                    rc_starts, rc_cols, rc_elements,
                    r1_idx, r1_val, r1_nnz, ncols,
                    c0_idx, c0_val, zero_tol, -1.0,
                )
            end
        end
    end
end

@inline function _pl_can_combine(r1_nnz::Int, nrows::Int, ncols::Int, has_row_copy::Bool)::Bool
    factor = 0.30
    if ncols * 8 > 1000000
        if nrows * 10 < ncols
            factor *= 0.333333333
        elseif nrows * 4 < ncols
            factor *= 0.5
        elseif nrows * 2 < ncols
            factor *= 0.66666666667
        end
    end
    return Float64(r1_nnz) > factor * Float64(nrows) || !has_row_copy
end

"""
    clp_primal_while_iterating!(...) -> Tuple

Pure 1-based Julia inner loop of `ClpSimplexPrimal::whileIterating(0)` (standard primal simplex,
`solveType_ == 1`, `ifValuesPass == 0`, `ClpPrimalColumnSteepest` with `switchType < 5`,
`ClpNonLinearCost` method 2, `ClpFactorization`, `ClpPackedMatrix` without `columnCopy_`).
Runs all pivots between factorizations with zero FFI crossings per pivot.
"""
function clp_primal_while_iterating!(
    cf::ClpFactorization,
    mat_starts::AbstractVector{Int32},
    mat_lengths::AbstractVector{Int32},
    mat_indices::AbstractVector{Int32},
    mat_elements::AbstractVector{Float64},
    tt_starts::AbstractVector{Int32},
    tt_indices::AbstractVector{Int32},
    tt_elements::AbstractVector{Float64},
    has_scale::Bool,
    tt_has_scale::Bool,
    row_scale::AbstractVector{Float64},
    col_scale::AbstractVector{Float64},
    has_row_copy::Bool,
    rc_starts::AbstractVector{Int32},
    rc_cols::AbstractVector{Int32},
    rc_elements::AbstractVector{Float64},
    pivot_var::AbstractVector{Int32},
    solution::AbstractVector{Float64},
    lower::AbstractVector{Float64},
    upper::AbstractVector{Float64},
    cost::AbstractVector{Float64},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nlc_status::AbstractVector{UInt8},
    nlc_bound::AbstractVector{Float64},
    nlc_cost2::AbstractVector{Float64},
    pcs_weights::AbstractVector{Float64},
    pcs_ref::AbstractVector{UInt32},
    pcs_has_ref::Bool,
    pcs_infeas_val::AbstractVector{Float64},
    pcs_infeas_idx::AbstractVector{Int32},
    pcs_infeas_nnz::Int,
    pcs_alt_val::AbstractVector{Float64},
    pcs_alt_idx::AbstractVector{Int32},
    pcs_alt_nnz::Int,
    pcs_mode::Int,
    pcs_num_switched::Int,
    pcs_size_fact::Int,
    pcs_pivot_seq::Int,
    pcs_devex::Float64,
    prog_in::AbstractVector{Int32},
    prog_out::AbstractVector{Int32},
    prog_way::AbstractVector{Int8},
    r1_val::AbstractVector{Float64},
    r1_idx::AbstractVector{Int32},
    r1_nnz::Int,
    r2_val::AbstractVector{Float64},
    r2_idx::AbstractVector{Int32},
    r3_val::AbstractVector{Float64},
    r3_idx::AbstractVector{Int32},
    c0_val::AbstractVector{Float64},
    c0_idx::AbstractVector{Int32},
    c1_val::AbstractVector{Float64},
    c1_idx::AbstractVector{Int32},
    marked_buf::Vector{UInt8},
    work_col::Vector{Float64},
    nrows::Int,
    ncols::Int,
    primal_tol::Float64,
    dual_tol::Float64,
    zero_tol::Float64,
    infeas_cost::Float64,
    large_value::Float64,
    obj_scale::Float64,
    rhs_scale::Float64,
    largest_dual_err::Float64,
    nlc_infeas_weight::Float64,
    nlc_sum_infeas::Float64,
    max_iterations::Int,
    special_options::Int,
    obj_type::Int,
    last_good_iter::Int,
    # mutable scalars
    problem_status::Int,
    return_code::Int,
    number_iterations::Int,
    change_made::Int,
    progress_flag::Int,
    last_bad_iter::Int,
    force_factorization::Int,
    prog_bad_times::Int,
    nlc_num_infeas::Int,
    nlc_change_cost::Float64,
    nlc_avg_theta::Float64,
    num_primal_infeas::Int,
    pivot_row::Int,
    seq_in::Int,
    seq_out::Int,
    dir_in::Int,
    dir_out::Int,
    val_in::Float64,
    lo_in::Float64,
    hi_in::Float64,
    dual_in::Float64,
    val_out::Float64,
    lo_out::Float64,
    hi_out::Float64,
    dual_out::Float64,
    theta::Float64,
    alpha::Float64,
    alpha_accuracy::Float64,
    objective_value::Float64,
    acceptable_pivot::Float64,
    more_special_options::Int,
    rng_seed::UInt32,
)
    ntotal = nrows + ncols
    dj_rows = @view dj[(ncols + 1):ntotal]
    dj_cols = @view dj[1:ncols]
    need_init_weights = 0

    switch_type = pcs_mode == 4 ? (5 - pcs_num_switched) : (pcs_mode >= 10 ? 3 : pcs_mode)
    ratio = Float64(pcs_size_fact) / Float64(nrows)

    while problem_status == -1
        # Check if PCS pivotColumn needs fallback (Dantzig, mode switch 4->3, or anyUpdates == 2)
        if switch_type >= 5 || (switch_type == 4 && ratio >= 7.0)
            return (CLP_PL_EXIT_FALLBACK_ITER, return_code, problem_status, number_iterations,
                    change_made, progress_flag, last_bad_iter, force_factorization,
                    prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                    r1_nnz, pcs_infeas_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                    pivot_row, seq_in, seq_out, dir_in, dir_out,
                    val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                    theta, alpha, alpha_accuracy, objective_value,
                    primal_tol, acceptable_pivot, more_special_options, rng_seed)
        end

        any_updates = if r1_nnz > 1
            2
        elseif r1_nnz == 1
            (@inbounds(Int(r1_idx[1])) == pivot_row && abs(@inbounds(r1_val[1])) > 1.0e-6) ? 1 : 2
        elseif pcs_pivot_seq > 0
            -1
        else
            0
        end

        if any_updates == 2
            return (CLP_PL_EXIT_FALLBACK_ITER, return_code, problem_status, number_iterations,
                    change_made, progress_flag, last_bad_iter, force_factorization,
                    prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                    r1_nnz, pcs_infeas_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                    pivot_row, seq_in, seq_out, dir_in, dir_out,
                    val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                    theta, alpha, alpha_accuracy, objective_value,
                    primal_tol, acceptable_pivot, more_special_options, rng_seed)
        end

        # 1. ClpPrimalColumnSteepest::pivotColumn
        tol = dual_tol + min(1.0e-2, largest_dual_err)
        outgoing_weight = seq_out > 0 ? @inbounds(pcs_weights[seq_out]) : 0.0

        if any_updates == 1
            if switch_type < 4
                # djsAndSteepest
                scale_factor = 1.0 / @inbounds(r1_val[1])
                pcs_pivot_seq = 0
                r1_nnz = clp_fact_update_column_transpose!(
                    cf, r3_val, r3_idx, 0, false, r1_val, r1_idx, r1_nnz, true,
                )
                pcs_alt_nnz = clp_fact_update_column_transpose!(
                    cf, r3_val, r3_idx, 0, false, pcs_alt_val, pcs_alt_idx, pcs_alt_nnz, false,
                )
                ref_in = pcs_mode != 1 ? ((pcs_has_ref && _pcs_ref(pcs_ref, seq_in)) ? 1.0 : 0.0) : -1.0
                if pcs_mode != 1 && pcs_has_ref
                    pcs_infeas_nnz = clp_pcs_steepest_rows!(
                        r1_val, r1_idx, r1_nnz, pcs_alt_val, dj_rows, status,
                        pcs_weights, pcs_ref, pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz,
                        ncols, scale_factor, pcs_devex, pcs_mode, ref_in, tol,
                    )
                else
                    pcs_infeas_nnz = clp_pcs_steepest_rows!(
                        r1_val, r1_idx, r1_nnz, pcs_alt_val, dj_rows, status,
                        pcs_weights, nothing, pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz,
                        ncols, scale_factor, pcs_devex, pcs_mode, ref_in, tol,
                    )
                end

                if _pl_can_combine(r1_nnz, nrows, ncols, has_row_copy)
                    c0_nnz = clp_mat_transpose_times2!(
                        tt_starts, tt_indices, tt_elements, ncols,
                        r1_val, r1_idx, r1_nnz,
                        pcs_alt_val, pcs_alt_idx, pcs_alt_nnz,
                        r3_val, c0_val, c0_idx,
                        pcs_infeas_val, dj_cols, pcs_weights, pcs_ref, pcs_has_ref, status,
                        tt_has_scale, row_scale, col_scale,
                        ref_in, pcs_devex, -scale_factor, zero_tol, tol,
                    )
                    r1_nnz = _pl_clear_packed!(r1_val, r1_nnz)
                    pcs_alt_nnz = _dl_clear_unpacked!(pcs_alt_val, pcs_alt_idx, pcs_alt_nnz)
                    pcs_infeas_nnz = clp_pcs_steepest_cols_rc1!(
                        c0_val, c0_idx, c0_nnz, pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz,
                    )
                else
                    c0_nnz = _pl_transpose_times_btran!(
                        tt_starts, tt_indices, tt_elements, tt_has_scale, row_scale, col_scale,
                        has_row_copy, rc_starts, rc_cols, rc_elements,
                        r1_val, r1_idx, r1_nnz, r3_val, c0_val, c0_idx, c1_idx,
                        marked_buf, work_col, status, nrows, ncols, zero_tol,
                    )
                    if tt_has_scale
                        clp_mat_subset_tt!(
                            tt_starts, nothing, false, tt_indices, tt_elements,
                            c0_idx, c0_nnz, pcs_alt_val, row_scale, col_scale, c1_val,
                        )
                    else
                        clp_mat_subset_tt!(
                            tt_starts, nothing, false, tt_indices, tt_elements,
                            c0_idx, c0_nnz, pcs_alt_val, nothing, nothing, c1_val,
                        )
                    end
                    sf_col = -scale_factor
                    @inbounds for j in 1:c0_nnz
                        iseq = Int(c0_idx[j])
                        val2 = c0_val[j]
                        modv = c1_val[j]
                        c1_val[j] = 0.0
                        st = status[iseq] & CLP_STATUS_MASK
                        if st != CLP_STATUS_BASIC && st != CLP_STATUS_IS_FIXED
                            w = pcs_weights[iseq]
                            piv = val2 * sf_col
                            piv2 = piv * piv
                            w += _pcp_fmadd(piv2, pcs_devex, piv * modv)
                            if w < CLP_STEEP_TRY_NORM
                                if ref_in < 0.0
                                    w = max(CLP_STEEP_TRY_NORM, CLP_STEEP_ADD_ONE + piv2)
                                else
                                    w = ref_in * piv2
                                    if pcs_has_ref && _pcs_ref(pcs_ref, iseq)
                                        w += 1.0
                                    end
                                    w = max(w, CLP_STEEP_TRY_NORM)
                                end
                            end
                            pcs_weights[iseq] = w
                        end
                    end
                    r1_nnz = _pl_clear_packed!(r1_val, r1_nnz)
                    pcs_alt_nnz = _dl_clear_unpacked!(pcs_alt_val, pcs_alt_idx, pcs_alt_nnz)
                    pcs_infeas_nnz = clp_pcs_steepest_cols_rc0!(
                        c0_val, c0_idx, c0_nnz, dj_cols, status,
                        pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz, tol,
                    )
                end
                if seq_out > 0
                    @inbounds pcs_weights[seq_out] = outgoing_weight
                end
                _infeas_zero!(pcs_infeas_val, seq_in)
            else
                # djsAndDevex (switch_type == 4)
                pcs_pivot_seq = 0
                r1_nnz = clp_fact_update_column_transpose!(
                    cf, r3_val, r3_idx, 0, false, r1_val, r1_idx, r1_nnz, true,
                )
                scale_factor = 1.0 / @inbounds(r1_val[1])
                c0_nnz = _pl_transpose_times_btran!(
                    tt_starts, tt_indices, tt_elements, tt_has_scale, row_scale, col_scale,
                    has_row_copy, rc_starts, rc_cols, rc_elements,
                    r1_val, r1_idx, r1_nnz, r3_val, c0_val, c0_idx, c1_idx,
                    marked_buf, work_col, status, nrows, ncols, zero_tol,
                )
                pcs_infeas_nnz = clp_pcs_devex_section!(
                    r1_val, r1_idx, r1_nnz, dj_rows, status, pcs_weights, pcs_ref,
                    pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz,
                    ncols, scale_factor, pcs_devex, CLP_PRIMAL_SLACK_MULT, tol,
                )
                pcs_infeas_nnz = clp_pcs_devex_section!(
                    c0_val, c0_idx, c0_nnz, dj_cols, status, pcs_weights, pcs_ref,
                    pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz,
                    0, -scale_factor, pcs_devex, 1.0, tol,
                )
                r1_nnz = 0
                if seq_out > 0
                    @inbounds pcs_weights[seq_out] = outgoing_weight
                end
                _infeas_zero!(pcs_infeas_val, seq_in)
            end
        elseif any_updates == -1
            piv_r = pcs_pivot_seq
            pcs_pivot_seq = 0
            seq_in_prev = @inbounds Int(pivot_var[piv_r])
            _infeas_zero!(pcs_infeas_val, seq_in_prev)
            @inbounds begin
                r1_val[1] = -1.0
                r1_idx[1] = Int32(piv_r)
            end
            r1_nnz = clp_fact_update_column_transpose!(
                cf, r3_val, r3_idx, 0, false, r1_val, r1_idx, 1, true,
            )
            c0_nnz = _pl_transpose_times_btran!(
                tt_starts, tt_indices, tt_elements, tt_has_scale, row_scale, col_scale,
                has_row_copy, rc_starts, rc_cols, rc_elements,
                r1_val, r1_idx, r1_nnz, r3_val, c0_val, c0_idx, c1_idx,
                marked_buf, work_col, status, nrows, ncols, zero_tol,
            )
            if switch_type < 4
                # justSteepest
                ref_in = (pcs_mode != 1 && pcs_has_ref && _pcs_ref(pcs_ref, seq_in_prev)) ? 1.0 : 0.0
                pcs_alt_nnz = clp_fact_update_column_transpose!(
                    cf, r3_val, r3_idx, 0, false, pcs_alt_val, pcs_alt_idx, pcs_alt_nnz, false,
                )
                if tt_has_scale
                    clp_mat_subset_tt!(
                        tt_starts, nothing, false, tt_indices, tt_elements,
                        c0_idx, c0_nnz, pcs_alt_val, row_scale, col_scale, c1_val,
                    )
                else
                    clp_mat_subset_tt!(
                        tt_starts, nothing, false, tt_indices, tt_elements,
                        c0_idx, c0_nnz, pcs_alt_val, nothing, nothing, c1_val,
                    )
                end
                if pcs_mode != 1 && pcs_has_ref
                    clp_pcs_just_steepest_rows!(
                        r1_val, r1_idx, r1_nnz, pcs_alt_val, pcs_weights, pcs_ref,
                        ncols, pcs_devex, pcs_mode, ref_in,
                    )
                    clp_pcs_just_steepest_cols!(
                        c0_val, c0_idx, c0_nnz, c1_val, pcs_weights, pcs_ref,
                        pcs_devex, pcs_mode, ref_in,
                    )
                else
                    clp_pcs_just_steepest_rows!(
                        r1_val, r1_idx, r1_nnz, pcs_alt_val, pcs_weights, nothing,
                        ncols, pcs_devex, pcs_mode, ref_in,
                    )
                    clp_pcs_just_steepest_cols!(
                        c0_val, c0_idx, c0_nnz, c1_val, pcs_weights, nothing,
                        pcs_devex, pcs_mode, ref_in,
                    )
                end
                pcs_alt_nnz = _dl_clear_unpacked!(pcs_alt_val, pcs_alt_idx, pcs_alt_nnz)
                r1_nnz = 0
            else
                # justDevex
                clp_pcs_just_devex_section!(
                    r1_val, r1_idx, r1_nnz, pcs_weights, pcs_ref, ncols, true, pcs_devex,
                )
                clp_pcs_just_devex_section!(
                    c0_val, c0_idx, c0_nnz, pcs_weights, pcs_ref, 0, false, pcs_devex,
                )
                r1_nnz = 0
            end
            if seq_out > 0
                @inbounds pcs_weights[seq_out] = outgoing_weight
            end
        end

        # Check outgoing from last iteration
        if seq_out > 0
            @inbounds begin
                st_out = status[seq_out] & CLP_STATUS_MASK
                v_out = dj[seq_out]
            end
            if st_out == CLP_STATUS_IS_FREE || st_out == CLP_STATUS_SUPER_BASIC
                if abs(v_out) > CLP_FREE_ACCEPT * tol
                    vb = v_out * CLP_FREE_BIAS
                    pcs_infeas_nnz = _infeas_set!(pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz, seq_out, vb * vb)
                else
                    _infeas_zero!(pcs_infeas_val, seq_out)
                end
            elseif st_out == CLP_STATUS_AT_UPPER
                if v_out > tol
                    pcs_infeas_nnz = _infeas_set!(pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz, seq_out, v_out * v_out)
                else
                    _infeas_zero!(pcs_infeas_val, seq_out)
                end
            elseif st_out == CLP_STATUS_AT_LOWER
                if v_out < -tol
                    pcs_infeas_nnz = _infeas_set!(pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz, seq_out, v_out * v_out)
                else
                    _infeas_zero!(pcs_infeas_val, seq_out)
                end
            end
        end

        # Determine numberWanted
        n_wanted = 10
        if switch_type == 4
            if ratio < 5.0
                n_wanted = max(2000, pcs_infeas_nnz ÷ 10)
                n_wanted = max(n_wanted, ncols ÷ 20)
            else
                n_wanted = max(2000, pcs_infeas_nnz ÷ 5)
                n_wanted = max(n_wanted, ncols ÷ 10)
            end
        else
            if switch_type < 2
                n_wanted = typemax(Int32) - 1
            elseif switch_type == 2
                n_wanted = max(2000, pcs_infeas_nnz ÷ 8)
            else
                if ratio < 1.0
                    n_wanted = max(2000, pcs_infeas_nnz ÷ 20)
                elseif ratio < 5.0
                    n_wanted = max(2000, pcs_infeas_nnz ÷ 10)
                    n_wanted = max(n_wanted, ncols ÷ 40)
                elseif ratio < 10.0
                    n_wanted = max(2000, pcs_infeas_nnz ÷ 8)
                    n_wanted = max(n_wanted, ncols ÷ 20)
                else
                    r_sc = Float64(pcs_infeas_nnz) * (ratio / 80.0)
                    if r_sc > Float64(pcs_infeas_nnz)
                        n_wanted = pcs_infeas_nnz + 1
                    else
                        n_wanted = max(2000, unsafe_trunc(Int, r_sc))
                        n_wanted = max(n_wanted, ncols ÷ 10)
                    end
                end
            end
        end

        # Pricing tolerance adjustments
        fpiv = cf.lu.npivots
        if number_iterations < last_bad_iter + 200 && fpiv > 10
            if largest_dual_err > 1.0e-8
                tol *= largest_dual_err / 1.0e-8
            end
            tol = min(1000.0, tol)
        end
        save_out_infeas = 0.0
        if seq_out > 0
            @inbounds begin
                save_out_infeas = pcs_infeas_val[seq_out]
                pcs_infeas_val[seq_out] = 0.0
            end
        end
        if fpiv != 0 && num_primal_infeas != 0
            tol = max(tol, 1.0e-15 * infeas_cost)
        end
        tol_sq = tol * tol

        rand_u = 0.0
        if pcs_mode > 1
            rand_u, rng_seed = _dl_rand_double(rng_seed)
        end
        best_seq, _ = clp_pcs_price_scan(
            pcs_infeas_val, pcs_infeas_idx, pcs_infeas_nnz,
            pcs_weights, status, pcs_mode, switch_type, n_wanted, tol_sq, rand_u,
        )
        if seq_out > 0
            @inbounds pcs_infeas_val[seq_out] = save_out_infeas
        end

        # 2. primalColumn tail (lookBothWays == false)
        seq_in, val_in, dual_in, lo_in, hi_in, dir_in, nlc_num_infeas = clp_primal_column_tail!(
            nlc_status, nlc_bound, nlc_cost2,
            solution, lower, upper, cost, dj, status,
            best_seq, false, primal_tol, nlc_infeas_weight, nlc_num_infeas,
        )

        pivot_row = 0
        seq_out = 0
        r1_nnz = _pl_clear_packed!(r1_val, r1_nnz)

        if seq_in > 0
            # 3. pivotResult(0)
            pr_exit, return_code, problem_status, number_iterations,
            change_made, progress_flag, last_bad_iter, force_factorization,
            prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
            r1_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, niw,
            pivot_row, seq_out, dir_in, dir_out,
            val_in, val_out, lo_out, hi_out, dual_in, dual_out,
            theta, alpha, alpha_accuracy, objective_value,
            primal_tol, acceptable_pivot, rng_seed = clp_primal_pivot_result!(
                cf, mat_starts, mat_lengths, mat_indices, mat_elements,
                has_scale, row_scale, col_scale,
                pivot_var, solution, lower, upper, cost, dj, status,
                nlc_status, nlc_bound, nlc_cost2,
                true, pcs_weights, pcs_ref, pcs_has_ref,
                pcs_alt_val, pcs_alt_idx, pcs_alt_nnz,
                pcs_mode, pcs_num_switched, pcs_pivot_seq, pcs_devex,
                prog_in, prog_out, prog_way,
                r1_val, r1_idx, r2_val, r2_idx, r3_val, r3_idx,
                nrows, ncols, primal_tol, dual_tol, infeas_cost, large_value,
                obj_scale, rhs_scale, largest_dual_err, nlc_infeas_weight, nlc_sum_infeas,
                max_iterations, special_options, obj_type, last_good_iter,
                problem_status, number_iterations, change_made, progress_flag,
                last_bad_iter, force_factorization, prog_bad_times,
                nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                pivot_row, seq_in, seq_out, dir_in, dir_out,
                val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                theta, alpha, alpha_accuracy, objective_value,
                acceptable_pivot, more_special_options, rng_seed,
            )

            if pr_exit != CLP_PL_EXIT_DONE || niw != 0
                need_init_weights = niw
                return (pr_exit, return_code, problem_status, number_iterations,
                        change_made, progress_flag, last_bad_iter, force_factorization,
                        prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
                        r1_nnz, pcs_infeas_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
                        pivot_row, seq_in, seq_out, dir_in, dir_out,
                        val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                        theta, alpha, alpha_accuracy, objective_value,
                        primal_tol, acceptable_pivot, more_special_options, rng_seed)
            end

            # Post-pivotResult status dispatch (matches whileIterating lines 784-800)
            if return_code < -1 && return_code > -5
                problem_status = -2
            elseif return_code == -5
                if (more_special_options & 16) == 0 && cf.lu.npivots != 0
                    more_special_options |= 16
                    problem_status = -2
                end
            elseif return_code == 2
                problem_status = -5
            elseif return_code == 4
                problem_status = -2
            elseif return_code != -1
                if problem_status != 5
                    problem_status = 3
                end
            end
        else
            # No pivot column
            if nlc_num_infeas != 0
                problem_status = -4
            end
            return_code = 0
            force_factorization = min(force_factorization, (cf.lu.npivots + 1) >> 1)
            break
        end
    end

    return (CLP_PL_EXIT_DONE, return_code, problem_status, number_iterations,
            change_made, progress_flag, last_bad_iter, force_factorization,
            prog_bad_times, nlc_num_infeas, nlc_change_cost, nlc_avg_theta,
            r1_nnz, pcs_infeas_nnz, pcs_alt_nnz, pcs_pivot_seq, pcs_devex, need_init_weights,
            pivot_row, seq_in, seq_out, dir_in, dir_out,
            val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
            theta, alpha, alpha_accuracy, objective_value,
            primal_tol, acceptable_pivot, more_special_options, rng_seed)
end

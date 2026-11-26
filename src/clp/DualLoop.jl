# Pure idiomatic 1-based Julia implementation of ClpSimplexDual::whileIterating (B12a).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, zero heap allocations on hot paths.
# Composes B8 (PackedMatrix), B9a (DualRowPivot), B10 (Factorization/SparseLU),
# B11a (SimplexDual), and B11c (SimplexBase + SimplexProgress) with zero FFI per pivot.

const CLP_DL_EXIT_DONE          = 0  # clean exit (refactorize / infeasible / etc.)
const CLP_DL_EXIT_PIVOT_ROW_NEG = 1  # pivot_row == 0 (-1 in C++): optimal / unflag / fake bounds
const CLP_DL_EXIT_FALLBACK_ITER = 2  # nextSuperBasic found a superbasic before pivot started
const CLP_DL_EXIT_SEQ_IN_NEG    = 3  # sequence_in == 0 (-1 in C++): dual unbounded / ray
const CLP_DL_EXIT_HIT_MAX       = 4  # hit maximum iterations (whatNext == 2)

@inline function _dl_rand_double(seed::UInt32)::Tuple{Float64,UInt32}
    s = UInt32(1664525) * seed + UInt32(1013904223)
    return (Float64(s) / 4294967296.0, s)
end

@inline function _dl_clear_packed!(val::AbstractVector{Float64}, nnz::Int)::Int
    @inbounds for i in 1:nnz
        val[i] = 0.0
    end
    return 0
end

@inline function _dl_clear_unpacked!(
    val::AbstractVector{Float64},
    idx::AbstractVector{Int32},
    nnz::Int,
)::Int
    cap = length(val)
    if 3 * nnz < cap
        @inbounds for i in 1:nnz
            val[Int(idx[i])] = 0.0
        end
    else
        @inbounds for i in 1:cap
            val[i] = 0.0
        end
    end
    return 0
end

# Matches ClpPackedMatrix::transposeTimes(model, -1.0, rowArray[0] (packed), rowArray[1] (y), columnArray[0])
# when !columnCopy_ && !(flags_ & 2).
function _dl_transpose_times_btran!(
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
    r0_val::AbstractVector{Float64},
    r0_idx::AbstractVector{Int32},
    r0_nnz::Int,
    y_val::AbstractVector{Float64},
    y_idx::AbstractVector{Int32},
    c0_val::AbstractVector{Float64},
    c0_idx::AbstractVector{Int32},
    r3_val::AbstractVector{Float64},
    r3_idx::AbstractVector{Int32},
    c1_idx::AbstractVector{Int32},
    marked_buf::Vector{UInt8},
    work_col::Vector{Float64},
    dj::AbstractVector{Float64},
    status::AbstractVector{UInt8},
    nrows::Int,
    ncols::Int,
    zero_tol::Float64,
    dual_tol::Float64,
    acceptable_pivot::Float64,
    spare_int0::Int,
    spare_dbl0::Float64,
)::Tuple{Int,Int,Int,Float64}
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

    r3_nnz = 0
    if Float64(r0_nnz) > factor * Float64(nrows) || !has_row_copy
        if !tt_has_scale
            @inbounds for i in 1:r0_nnz
                y_val[Int(r0_idx[i])] = -r0_val[i]
            end
            if spare_int0 > 0
                dual_t = -dual_tol
                tentative_theta = 1.0e15
                upper_theta = 1.0e31
                n_rem = 0
                @inbounds for i in 1:r0_nnz
                    iseq_r = Int(r0_idx[i])
                    seq = ncols + iseq_r
                    ist = Int(status[seq] & UInt8(3)) - 1
                    if ist > 0
                        mult = ist == 1 ? -1.0 : 1.0
                        alpha = r0_val[i] * mult
                        if alpha > 0.0
                            old_val = dj[seq] * mult
                            val = old_val - tentative_theta * alpha
                            if val < dual_t
                                val2 = old_val - upper_theta * alpha
                                if val2 < dual_t && alpha >= acceptable_pivot
                                    upper_theta = (old_val - dual_t) / alpha
                                end
                                n_rem += 1
                                r3_val[n_rem] = alpha * mult
                                r3_idx[n_rem] = Int32(seq)
                            end
                        end
                    end
                end
                c0_nnz, n_rem_out, upper_theta_out = clp_mat_guts_tt_unscaled_dual!(
                    tt_starts, tt_indices, tt_elements, ncols,
                    y_val, c0_idx, c0_val, status,
                    r3_idx, r3_val, dj,
                    upper_theta, acceptable_pivot, dual_tol, n_rem, zero_tol,
                )
                r3_nnz = n_rem_out
                spare_dbl0 = upper_theta_out
                spare_int0 = -2
            else
                c0_nnz = clp_mat_guts_tt_unscaled_status!(
                    tt_starts, tt_indices, tt_elements, ncols,
                    y_val, c0_idx, c0_val, status, zero_tol,
                )
            end
        else
            @inbounds for i in 1:r0_nnz
                irow = Int(r0_idx[i])
                y_val[irow] = -r0_val[i] * row_scale[irow]
            end
            c0_nnz = clp_mat_guts_tt_scaled_status!(
                tt_starts, tt_indices, tt_elements, ncols,
                y_val, col_scale, c0_idx, c0_val, status, zero_tol,
            )
        end
        @inbounds for i in 1:r0_nnz
            y_val[Int(r0_idx[i])] = 0.0
        end
        return (c0_nnz, r3_nnz, spare_int0, spare_dbl0)
    else
        if r0_nnz == 1
            c0_nnz = clp_mat_tt_by_row_eq1_packed!(
                rc_starts, rc_cols, rc_elements,
                @inbounds(Int(r0_idx[1])), @inbounds(r0_val[1]),
                c0_idx, c0_val, zero_tol, -1.0,
            )
        elseif r0_nnz == 2
            c0_nnz = clp_mat_tt_by_row_eq2_packed!(
                rc_starts, rc_cols, rc_elements,
                @inbounds(Int(r0_idx[1])), @inbounds(Int(r0_idx[2])),
                @inbounds(r0_val[1]), @inbounds(r0_val[2]),
                c0_idx, c0_val, marked_buf, c1_idx, zero_tol, -1.0,
            )
        else
            n_cov = 0
            sparse = true
            @inbounds for i in 1:r0_nnz
                ir = Int(r0_idx[i])
                n_cov += Int(rc_starts[ir + 1]) - Int(rc_starts[ir])
                if n_cov > ncols
                    sparse = false
                    break
                end
            end
            if sparse
                c0_nnz = clp_mat_tt_by_row_ge3_packed!(
                    rc_starts, rc_cols, rc_elements,
                    r0_idx, r0_val, r0_nnz,
                    c0_idx, c0_val, work_col, zero_tol, -1.0,
                )
            else
                c0_nnz = clp_mat_tt_by_row_gek_packed!(
                    rc_starts, rc_cols, rc_elements,
                    r0_idx, r0_val, r0_nnz, ncols,
                    c0_idx, c0_val, zero_tol, -1.0,
                )
            end
        end
        return (c0_nnz, r3_nnz, spare_int0, spare_dbl0)
    end
end

"""
    clp_dual_while_iterating!(...)

Pure 1-based Julia inner loop of `ClpSimplexDual::whileIterating` (normal dual simplex,
no values pass, `ClpDualRowSteepest`, `ClpFactorization`, `ClpPackedMatrix` without `columnCopy_`).
Runs all pivots between factorizations without crossing FFI.
"""
function clp_dual_while_iterating!(
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
    inv_col_scale::AbstractVector{Float64},
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
    col_lower_orig::AbstractVector{Float64},
    col_upper_orig::AbstractVector{Float64},
    row_lower_orig::AbstractVector{Float64},
    row_upper_orig::AbstractVector{Float64},
    drs_weights::AbstractVector{Float64},
    drs_infeas_val::AbstractVector{Float64},
    drs_infeas_idx::AbstractVector{Int32},
    drs_infeas_nnz::Int,
    drs_alt_val::AbstractVector{Float64},
    drs_alt_idx::AbstractVector{Int32},
    drs_alt_nnz::Int,
    drs_mode::Int,
    prog_in::AbstractVector{Int32},
    prog_out::AbstractVector{Int32},
    prog_way::AbstractVector{Int8},
    r0_val::AbstractVector{Float64},
    r0_idx::AbstractVector{Int32},
    r1_val::AbstractVector{Float64},
    r1_idx::AbstractVector{Int32},
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
    dual_bound::Float64,
    large_value::Float64,
    acceptable_pivot_member::Float64,
    obj_scale::Float64,
    rhs_scale::Float64,
    largest_primal_err::Float64,
    largest_dual_err::Float64,
    save_sum_dual::Float64,
    max_iterations::Int,
    special_options::Int,
    obj_type::Int,
    number_threads::Int,
    problem_status::Int,
    return_code::Int,
    number_iterations::Int,
    change_made::Int,
    progress_flag::Int,
    more_special_options::Int,
    first_free::Int,
    last_bad_iter::Int,
    force_factorization::Int,
    prog_bad_times::Int,
    number_fake::Int,
    number_changed::Int,
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
    spare_int0::Int,
    spare_int1::Int,
    spare_dbl0::Float64,
    spare_dbl2::Float64,
    spare_dbl3::Float64,
    rng_seed::UInt32,
    max_zero_step::Int = 0,
)
    ntotal = nrows + ncols
    r0_nnz = 0
    c0_nnz = 0
    best_possible = 1.0
    consec_zero_steps = 0

    while problem_status == -1
        # 1. Check nextSuperBasic
        if first_free > 0
            nf, first_free = clp_dual_next_super_basic!(dj, status, ntotal, first_free, dual_tol)
            if nf > 0
                first_free = nf
                return (CLP_DL_EXIT_FALLBACK_ITER, problem_status, return_code, number_iterations,
                        change_made, progress_flag, more_special_options, first_free, last_bad_iter,
                        force_factorization, prog_bad_times, number_fake, number_changed, num_primal_infeas,
                        drs_infeas_nnz, drs_alt_nnz, r0_nnz, c0_nnz,
                        pivot_row, seq_in, seq_out, dir_in, dir_out,
                        val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                        theta, alpha, alpha_accuracy, objective_value,
                        spare_int0, spare_int1, spare_dbl0, acceptable_pivot_member, best_possible, rng_seed)
            end
        end

        # 2. DualRowSteepest::pivotRow()
        last_piv = pivot_row
        n_fact_elems = drs_mode >= 3 ? cf.lu.total_elements : 0
        rand_u, rng_seed = _dl_rand_double(rng_seed)
        chosen, drs_infeas_nnz, tol_changed = clp_drs_pivot_row!(
            drs_infeas_val, drs_infeas_idx, drs_infeas_nnz,
            drs_weights, pivot_var, solution, lower, upper, status,
            nrows, last_piv, drs_mode, n_fact_elems,
            primal_tol, largest_primal_err, largest_dual_err,
            number_iterations, last_bad_iter, rand_u,
        )
        if chosen == 0 && tol_changed
            rand_u, rng_seed = _dl_rand_double(rng_seed)
            chosen, drs_infeas_nnz, _ = clp_drs_pivot_row!(
                drs_infeas_val, drs_infeas_idx, drs_infeas_nnz,
                drs_weights, pivot_var, solution, lower, upper, status,
                nrows, last_piv, drs_mode, n_fact_elems,
                primal_tol, largest_primal_err, 0.0,
                number_iterations, last_bad_iter, rand_u,
            )
        end
        if chosen == 0 && last_piv == 0
            n_left = clp_drs_compact_infeasible!(drs_infeas_val, drs_infeas_idx, drs_infeas_nnz)
            drs_infeas_nnz = n_left
            num_primal_infeas = n_left
        end
        pivot_row = chosen
        if pivot_row == 0
            return (CLP_DL_EXIT_PIVOT_ROW_NEG, problem_status, return_code, number_iterations,
                    change_made, progress_flag, more_special_options, first_free, last_bad_iter,
                    force_factorization, prog_bad_times, number_fake, number_changed, num_primal_infeas,
                    drs_infeas_nnz, drs_alt_nnz, r0_nnz, c0_nnz,
                    pivot_row, seq_in, seq_out, dir_in, dir_out,
                    val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                    theta, alpha, alpha_accuracy, objective_value,
                    spare_int0, spare_int1, spare_dbl0, acceptable_pivot_member, best_possible, rng_seed)
        end

        # 3. dualRow finish
        seq_out, val_out, lo_out, hi_out, dir_out, dual_out = clp_dual_row_finish!(
            pivot_var, solution, lower, upper, dj, pivot_row, false,
        )

        # 4. acceptablePivot
        fpiv = cf.lu.npivots
        acc_piv = 1.0e-1 * acceptable_pivot_member
        if number_iterations > 100
            acc_piv = acceptable_pivot_member
        end
        if fpiv > 10 || (fpiv != 0 && save_sum_dual != 0.0)
            acc_piv = 1.0e+3 * acceptable_pivot_member
        elseif fpiv > 5
            acc_piv = 1.0e+2 * acceptable_pivot_member
        elseif fpiv != 0
            acc_piv = acceptable_pivot_member
        end

        # 5. BTRAN: rowArray_[0] = e_{pivotRow} * directionOut
        @inbounds begin
            r0_val[1] = Float64(dir_out)
            r0_idx[1] = Int32(pivot_row)
        end
        r0_nnz = clp_fact_update_column_transpose!(
            cf, r1_val, r1_idx, 0, false, r0_val, r0_idx, 1, true,
        )

        # 6. Matrix transposeTimes
        if number_threads < -1
            spare_int0 = 1
        end
        spare_dbl0 = acc_piv
        seq_in = 0
        if (more_special_options & 8) != 0 && !tt_has_scale
            spare_int0 = 1
        end
        c0_nnz, r3_nnz, spare_int0, spare_dbl0 = _dl_transpose_times_btran!(
            tt_starts, tt_indices, tt_elements, tt_has_scale, row_scale, col_scale,
            has_row_copy, rc_starts, rc_cols, rc_elements,
            r0_val, r0_idx, r0_nnz, r1_val, r1_idx, c0_val, c0_idx, r3_val, r3_idx,
            c1_idx, marked_buf, work_col,
            dj, status, nrows, ncols, zero_tol, dual_tol, acc_piv, spare_int0, spare_dbl0,
        )

        # 7. Dual ratio test (dualColumn)
        best_possible, seq_in, dir_in, theta, alpha, lo_in, hi_in, val_in, dual_in,
        acceptable_pivot_member, number_fake, number_changed =
            clp_dual_column!(
                r0_val, r0_idx, r0_nnz,
                c0_val, c0_idx, c0_nnz,
                r3_val, r3_idx, c1_val, c1_idx,
                dj, cost, solution, lower, upper, status,
                nrows, ncols, dual_tol, dual_bound, dual_out, lo_out, hi_out,
                large_value, objective_value, special_options, more_special_options,
                fpiv, seq_out, acc_piv, acceptable_pivot_member,
                spare_int0, spare_int1, r3_nnz, spare_dbl0, spare_dbl2, spare_dbl3,
                seq_in, theta, number_fake, number_changed,
            )

        if seq_in == 0
            if acc_piv <= acceptable_pivot_member && fpiv == 0
                problem_status = 1
            end
            return (CLP_DL_EXIT_SEQ_IN_NEG, problem_status, return_code, number_iterations,
                    change_made, progress_flag, more_special_options, first_free, last_bad_iter,
                    force_factorization, prog_bad_times, number_fake, number_changed, num_primal_infeas,
                    drs_infeas_nnz, drs_alt_nnz, r0_nnz, c0_nnz,
                    pivot_row, seq_in, seq_out, dir_in, dir_out,
                    val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                    theta, alpha, alpha_accuracy, objective_value,
                    spare_int0, spare_int1, spare_dbl0, acceptable_pivot_member, best_possible, rng_seed)
        end

        # 8. Unpack incoming column into rowArray_[1] (packed)
        btran_alpha = -alpha * Float64(dir_out)
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

        # 9. DualRowSteepest::updateWeights (FTRAN + weight update)
        drs_alt_nnz = _dl_clear_packed!(drs_alt_val, drs_alt_nnz)
        has_perm = !isempty(cf.lu.permute)
        norm_in = has_perm ?
            clp_drs_prep_ftran!(r0_val, r0_idx, r0_nnz, cf.lu.permute, r2_val, r2_idx) :
            clp_drs_prep_ftran!(r0_val, r0_idx, r0_nnz, nothing, r2_val, r2_idx)
        r1_nnz, r2_nnz = clp_fact_update_two_columns_ft!(
            cf, r3_val, r3_idx,
            r1_val, r1_idx, r1_nnz, true,
            r2_val, r2_idx, r0_nnz, false,
            has_perm,
        )
        alpha, drs_alt_nnz = has_perm ?
            clp_drs_apply_weight_update!(r1_val, r1_idx, r1_nnz, r2_val, cf.lu.pivot_col,
                                         drs_weights, drs_alt_val, drs_alt_idx,
                                         pivot_row, alpha, norm_in, false) :
            clp_drs_apply_weight_update!(r1_val, r1_idx, r1_nnz, r2_val, nothing,
                                         drs_weights, drs_alt_val, drs_alt_idx,
                                         pivot_row, alpha, norm_in, false)
        r2_nnz = _dl_clear_unpacked!(r2_val, r2_idx, r2_nnz)

        # 10. btranAlpha vs alpha accuracy check
        check_val = largest_primal_err > 10.0 ? min(1.0e-4, 1.0e-8 * largest_primal_err) : 1.0e-7
        if abs(btran_alpha) < 1.0e-12 || abs(alpha) < 1.0e-12 ||
           abs(btran_alpha - alpha) > check_val * (1.0 + abs(alpha))
            if fpiv != 0
                clp_drs_unroll_weights!(drs_weights, drs_alt_val, drs_alt_idx, drs_alt_nnz, true)
                drs_alt_nnz = 0
                problem_status = -2
                r0_nnz = _dl_clear_packed!(r0_val, r0_nnz)
                r1_nnz = _dl_clear_packed!(r1_val, r1_nnz)
                c0_nnz = _dl_clear_packed!(c0_val, c0_nnz)
                return_code = -2
                break
            else
                test_val = (abs(btran_alpha) < 1.0e-8 || abs(alpha) < 1.0e-8) ?
                    (1.0e-1 * abs(alpha)) : (1.0e-4 * (1.0 + abs(alpha)))
                if abs(btran_alpha) < 1.0e-12 || abs(alpha) < 1.0e-12 ||
                   abs(btran_alpha - alpha) > test_val
                    clp_drs_unroll_weights!(drs_weights, drs_alt_val, drs_alt_idx, drs_alt_nnz, true)
                    drs_alt_nnz = 0
                    clp_fact_safer_tolerances!(cf, -0.99, -1.03)
                    @inbounds status[seq_out] |= CLP_STATUS_FLAGGED_BIT
                    prog_bad_times = 0
                    last_bad_iter = number_iterations
                    r0_nnz = _dl_clear_packed!(r0_val, r0_nnz)
                    r1_nnz = _dl_clear_packed!(r1_val, r1_nnz)
                    c0_nnz = _dl_clear_packed!(c0_val, c0_nnz)
                    if abs(alpha) < 1.0e-10 && abs(btran_alpha) < 1.0e-8 && number_iterations > 100
                        problem_status = 1
                        return_code = 1
                        break
                    end
                    continue
                end
            end
        end

        # 11. updateDualsInDual
        obj_chg = 0.0
        @inbounds begin
            save_st_in = status[seq_in]
            status[seq_in] = _sb_set_status(save_st_in, CLP_STATUS_BASIC)
        end
        if has_scale
            nswapped, r0_nnz, c0_nnz, r2_nnz, obj_chg, number_fake = clp_update_duals_in_dual!(
                r0_val, r0_idx, r0_nnz, c0_val, c0_idx, c0_nnz, r2_val, r2_idx, 0,
                mat_starts, mat_lengths, mat_indices, mat_elements, row_scale, col_scale,
                dj, solution, lower, upper, cost, status,
                nrows, ncols, theta, dual_tol, largest_dual_err, dual_bound,
                more_special_options, number_fake, false,
            )
        else
            nswapped, r0_nnz, c0_nnz, r2_nnz, obj_chg, number_fake = clp_update_duals_in_dual!(
                r0_val, r0_idx, r0_nnz, c0_val, c0_idx, c0_nnz, r2_val, r2_idx, 0,
                mat_starts, mat_lengths, mat_indices, mat_elements, nothing, nothing,
                dj, solution, lower, upper, cost, status,
                nrows, ncols, theta, dual_tol, largest_dual_err, dual_bound,
                more_special_options, number_fake, false,
            )
        end
        @inbounds status[seq_in] = save_st_in

        old_dual_out = dual_out
        if nswapped != 0
            if r2_nnz > 0
                r2_nnz = clp_fact_update_column!(cf, r3_val, r3_idx, 0, r2_val, r2_idx, r2_nnz, false, false)
                d_obj, drs_infeas_nnz = clp_drs_update_primal_solution!(
                    r2_val, r2_idx, r2_nnz, false, pivot_var,
                    drs_infeas_val, drs_infeas_idx, drs_infeas_nnz,
                    solution, cost, lower, upper, 1.0, primal_tol, pivot_row,
                )
                obj_chg += d_obj
                r2_nnz = 0
            end
            @inbounds val_out = solution[seq_out]
            dual_out = dir_out < 0 ? (val_out - hi_out) : (lo_out - val_out)
        end

        movement = -dual_out * Float64(dir_out) / alpha
        movement_old = old_dual_out * Float64(dir_out) / alpha
        if obj_chg + abs(movement_old * dual_in) < -max(1.0e-5, 1.0e-12 * abs(objective_value))
            if fpiv != 0
                clp_drs_unroll_weights!(drs_weights, drs_alt_val, drs_alt_idx, drs_alt_nnz, true)
                drs_alt_nnz = 0
                r1_nnz = _dl_clear_packed!(r1_val, r1_nnz)
                r0_nnz = 0
                c0_nnz = 0
                problem_status = -2
                return_code = -2
                break
            end
        end

        # 12. replaceColumn
        n_new = seq_in <= ncols ? @inbounds(Int(mat_lengths[seq_in])) : 1
        n_old = seq_out <= ncols ? @inbounds(Int(mat_lengths[seq_out])) : 1
        upd_status = clp_fact_replace_column!(
            cf, r2_val, r2_idx, r1_val, r1_idx, r1_nnz, true,
            pivot_row, alpha, (more_special_options & 16) != 0, n_new, n_old,
        )
        if abs(dual_out) > 1.0e50
            upd_status = 2
        end
        if upd_status == 2 && cf.lu.npivots == 0 && abs(alpha) > 1.0e-5
            upd_status = 4
        end
        if upd_status == 1 || upd_status == 4
            if cf.lu.npivots > 5 || upd_status == 4
                problem_status = -2
                return_code = -3
            end
        elseif upd_status == 2
            clp_drs_unroll_weights!(drs_weights, drs_alt_val, drs_alt_idx, drs_alt_nnz, true)
            drs_alt_nnz = 0
            if cf.lu.npivots != 0 && ((more_special_options & 16) == 0 || cf.lu.npivots > 4)
                problem_status = -2
                return_code = -2
                more_special_options |= 16
                pt = cf.lu.pivot_tol
                if pt < 0.4 && cf.lu.npivots < 100
                    cf.lu.pivot_tol = 1.05 * pt
                end
                r1_nnz = _dl_clear_packed!(r1_val, r1_nnz)
                r0_nnz = 0
                c0_nnz = 0
                break
            else
                @inbounds status[seq_out] |= CLP_STATUS_FLAGGED_BIT
                prog_bad_times = 0
                last_bad_iter = number_iterations
                r0_nnz = 0
                r1_nnz = _dl_clear_packed!(r1_val, r1_nnz)
                c0_nnz = 0
                if has_scale
                    _, r0_nnz, c0_nnz, r2_nnz, _, number_fake = clp_update_duals_in_dual!(
                        r0_val, r0_idx, 0, c0_val, c0_idx, 0, r1_val, r1_idx, 0,
                        mat_starts, mat_lengths, mat_indices, mat_elements, row_scale, col_scale,
                        dj, solution, lower, upper, cost, status,
                        nrows, ncols, 0.0, dual_tol, largest_dual_err, dual_bound,
                        more_special_options, number_fake, true,
                    )
                else
                    _, r0_nnz, c0_nnz, r2_nnz, _, number_fake = clp_update_duals_in_dual!(
                        r0_val, r0_idx, 0, c0_val, c0_idx, 0, r1_val, r1_idx, 0,
                        mat_starts, mat_lengths, mat_indices, mat_elements, nothing, nothing,
                        dj, solution, lower, upper, cost, status,
                        nrows, ncols, 0.0, dual_tol, largest_dual_err, dual_bound,
                        more_special_options, number_fake, true,
                    )
                end
                r1_nnz = _dl_clear_unpacked!(r1_val, r1_idx, r2_nnz)
                r0_nnz = 0
                c0_nnz = 0
                continue
            end
        elseif upd_status == 3
            if cf.lu.npivots < 0.5 * cf.lu.max_pivots && cf.lu.npivots < 200
                cf.lu.area_factor *= 1.1
            end
            problem_status = -2
        elseif upd_status == 5
            problem_status = -2
        end

        # 13. flipBounds + updatePrimalSolution
        if theta < 0.0
            theta = 0.0
        end
        clp_flip_bounds!(r0_idx, r0_nnz, c0_idx, c0_nnz, solution, lower, upper, status, ncols)
        r0_nnz = 0
        c0_nnz = 0

        d_obj, drs_infeas_nnz = clp_drs_update_primal_solution!(
            r1_val, r1_idx, r1_nnz, true, pivot_var,
            drs_infeas_val, drs_infeas_idx, drs_infeas_nnz,
            solution, cost, lower, upper, movement, primal_tol, pivot_row,
        )
        obj_chg += d_obj
        r1_nnz = 0

        dual_out /= alpha
        dual_out *= -Float64(dir_out)
        @inbounds dj[seq_in] = 0.0
        old_val = val_in
        val_in = (dir_in == -1 ? hi_in : lo_in) + dual_out
        @inbounds obj_chg += cost[seq_in] * (val_in - old_val)
        if dir_out > 0
            val_out = lo_out
            @inbounds dj[seq_out] = theta
        else
            val_out = hi_out
            @inbounds dj[seq_out] = -theta
        end
        @inbounds solution[seq_out] = val_out

        # 14. housekeeping (exact short-circuit timeToRefactorize & RNG in Julia)
        hit_max = (number_iterations + 1 >= max_iterations)
        hk_ret, hk_act, hk_cyc, number_iterations, change_made, progress_flag,
        alpha_accuracy, objective_value, _ = clp_housekeeping!(
            pivot_var, solution, lower, upper, status,
            prog_in, prog_out, prog_way,
            nrows, ncols, pivot_row, seq_in, seq_out, dir_in, dir_out,
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

        # 15. originalBound(seq_in) + changeBound(seq_out)
        if has_scale
            number_fake = clp_dual_original_bound!(
                lower, upper, status, col_lower_orig, col_upper_orig,
                row_lower_orig, row_upper_orig, row_scale, inv_col_scale,
                ncols, rhs_scale, seq_in, number_fake,
            )
            _, number_fake = clp_dual_change_bound!(
                lower, upper, status, solution, col_lower_orig, col_upper_orig,
                row_lower_orig, row_upper_orig, row_scale, inv_col_scale,
                ncols, rhs_scale, dual_bound, seq_out, number_fake,
            )
        else
            number_fake = clp_dual_original_bound!(
                lower, upper, status, col_lower_orig, col_upper_orig,
                row_lower_orig, row_upper_orig, nothing, nothing,
                ncols, rhs_scale, seq_in, number_fake,
            )
            _, number_fake = clp_dual_change_bound!(
                lower, upper, status, solution, col_lower_orig, col_upper_orig,
                row_lower_orig, row_upper_orig, nothing, nothing,
                ncols, rhs_scale, dual_bound, seq_out, number_fake,
            )
        end

        if what_next == 1
            problem_status = -2
        elseif what_next == 0 && max_zero_step > 0
            if theta <= dual_tol
                consec_zero_steps += 1
                if consec_zero_steps >= max_zero_step
                    problem_status = -2
                    return_code = -4
                end
            else
                consec_zero_steps = 0
            end
        elseif what_next == 2
            problem_status = 3
            return_code = 3
            return (CLP_DL_EXIT_HIT_MAX, problem_status, return_code, number_iterations,
                    change_made, progress_flag, more_special_options, first_free, last_bad_iter,
                    force_factorization, prog_bad_times, number_fake, number_changed, num_primal_infeas,
                    drs_infeas_nnz, drs_alt_nnz, r0_nnz, c0_nnz,
                    pivot_row, seq_in, seq_out, dir_in, dir_out,
                    val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
                    theta, alpha, alpha_accuracy, objective_value,
                    spare_int0, spare_int1, spare_dbl0, acceptable_pivot_member, best_possible, rng_seed)
        end
    end

    return (CLP_DL_EXIT_DONE, problem_status, return_code, number_iterations,
            change_made, progress_flag, more_special_options, first_free, last_bad_iter,
            force_factorization, prog_bad_times, number_fake, number_changed, num_primal_infeas,
            drs_infeas_nnz, drs_alt_nnz, r0_nnz, c0_nnz,
            pivot_row, seq_in, seq_out, dir_in, dir_out,
            val_in, lo_in, hi_in, dual_in, val_out, lo_out, hi_out, dual_out,
            theta, alpha, alpha_accuracy, objective_value,
            spare_int0, spare_int1, spare_dbl0, acceptable_pivot_member, best_possible, rng_seed)
end

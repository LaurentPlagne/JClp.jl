# ==============================================================================
# B10: ClpFactorization — Pure idiomatic 1-based Julia implementation
# ==============================================================================
# Direct integration of ClpFactorization with pure 1-based Julia `SparseLU`
# (zero Ptr, zero unsafe_load/unsafe_store!, zero PodView, --trim=safe clean).
#
# Eliminates C++ CoinFactorization array allocations and C++<->Julia copies
# during `factorize` (for ClpPackedMatrix), `timeToRefactorize`, `replaceColumn`,
# `updateColumn*`, `updateTwoColumns*`, `goSparse`, `cleanUp`, `getWeights`,
# `saferTolerances`, and `emptyRows`.
# ==============================================================================

const _CF_WEIGHT_INC_U        = 1.0
const _CF_WEIGHT_R            = 2.0
const _CF_WEIGHT_REST         = 1.0
const _CF_WEIGHT_FACT_L       = 30.0
const _CF_WEIGHT_FACT_DENSE   = 0.1
const _CF_WEIGHT_NROWS        = 10.0
const _CF_INCREASE_NEEDED     = 1.1
const _CF_CONST_WEIGHT_ITER   = 1.0
const _CF_WEIGHT_NROWS_ITER   = 3.0

"""
    ClpFactorization

Pure 1-based Julia factorization state matching Clp's `ClpFactorization`
(wrapping `SparseLU` directly without any C++ `CoinFactorization` mirror).
"""
mutable struct ClpFactorization
    lu::SparseLU
    # CLP_FACTORIZATION_NEW_TIMING fields
    shortest_average::Float64
    total_in_r::Float64
    total_in_increasing_u::Float64
    end_length_u::Int
    last_number_pivots::Int
    effective_start_number_u::Int
    # Control flags
    do_statistics::Bool
    message_level::Int
    persistence_flag::Int
    # Reusable workspace for emptyRows
    del_row_buf::Vector{UInt8}
end

function ClpFactorization()
    return ClpFactorization(
        SparseLU(),
        0.0, 0.0, 0.0,
        0, 0, 0,
        true, 0, 0,
        UInt8[],
    )
end

function copy_clp_factorization(src::ClpFactorization)::ClpFactorization
    return ClpFactorization(
        copy_lu(src.lu),
        src.shortest_average,
        src.total_in_r,
        src.total_in_increasing_u,
        src.end_length_u,
        src.last_number_pivots,
        src.effective_start_number_u,
        src.do_statistics,
        src.message_level,
        src.persistence_flag,
        UInt8[],
    )
end

@inline function clp_fact_clear_arrays!(cf::ClpFactorization)::Nothing
    lu = cf.lu
    lu.nrows = 0
    lu.ncols = 0
    lu.nrows_extra = 0
    lu.ncols_extra = 0
    lu.npivots = 0
    lu.ndense = 0
    lu.len_u = 0
    lu.len_l = 0
    lu.len_r = 0
    lu.total_elements = 0
    lu.factor_elements = 0
    lu.ncompressions = 0
    lu.status = -1
    lu.sparse_threshold = 0
    lu.sparse_threshold2 = 0
    return nothing
end

@inline function clp_fact_almost_destructor!(cf::ClpFactorization)::Nothing
    clp_fact_clear_arrays!(cf)
    return nothing
end

@inline function clp_fact_set_default_values!(cf::ClpFactorization)::Nothing
    cf.lu.zero_tol = 1.0e-13
    cf.lu.slack_value = -1.0
    return nothing
end

@inline function clp_fact_adjusted_area_factor(cf::ClpFactorization)::Float64
    lu = cf.lu
    factor = lu.area_factor
    if lu.ndense != 0 && factor > 1.0
        dense = Float64(lu.ndense)
        dense *= dense
        without_dense = Float64(lu.total_elements) - dense + 1.0
        factor *= 1.0 + dense / without_dense
    end
    return factor
end

@inline function clp_fact_space_for_ft(cf::ClpFactorization)::Bool
    lu = cf.lu
    if !lu.do_forrest_tomlin
        return false
    end
    idx = lu.max_cols_extra + 1
    if idx > length(lu.start_col_u)
        return false
    end
    @inbounds start_0 = Int(lu.start_col_u[idx]) - 1
    space = lu.len_area_u - (start_0 + lu.nrows_extra)
    return space >= 0
end

# ==============================================================================
# Adaptive refactorization heuristic (ClpFactorization::timeToRefactorize)
# ==============================================================================

function clp_fact_time_to_refactorize!(cf::ClpFactorization)::Bool
    lu = cf.lu
    re_factor = false
    re_factor3 = false
    n_pivots = lu.npivots
    if n_pivots > cf.last_number_pivots
        if cf.last_number_pivots == 0
            cf.total_in_r = 0.0
            cf.total_in_increasing_u = 0.0
            cf.shortest_average = floatmax(Float64)
        end
        cf.last_number_pivots = n_pivots
        n_dense = lu.ndense
        nnd = Float64(n_dense * n_dense)
        len_l = lu.len_l
        len_r = lu.len_r
        n_rows = lu.nrows
        len_u = lu.len_u - (n_rows - n_dense)
        cf.total_in_r += Float64(len_r)
        effective_u = len_u - cf.effective_start_number_u
        cf.total_in_increasing_u += Float64(effective_u)

        rest = Float64(len_l) + 0.05 * nnd
        const_weight_factor =
            _CF_WEIGHT_FACT_L * Float64(len_l) +
            _CF_WEIGHT_FACT_DENSE * nnd +
            _CF_WEIGHT_NROWS * Float64(n_rows)
        const_weight_iterate_x =
            _CF_CONST_WEIGHT_ITER * Float64(len_l + cf.end_length_u) +
            _CF_WEIGHT_NROWS_ITER * Float64(n_rows)
        variable_weight =
            _CF_WEIGHT_INC_U * cf.total_in_increasing_u +
            _CF_WEIGHT_R * cf.total_in_r +
            _CF_WEIGHT_REST * rest
        average = const_weight_iterate_x + (const_weight_factor + variable_weight) / Float64(n_pivots)

        if average < cf.shortest_average
            cf.shortest_average = average
        end
        if average > _CF_INCREASE_NEEDED * cf.shortest_average && n_pivots > 30
            re_factor3 = true
        end
    end
    return re_factor || re_factor3
end

# ==============================================================================
# Tolerance & area-factor helpers
# ==============================================================================

function clp_fact_safer_tolerances!(
    cf::ClpFactorization,
    zero_value::Float64,
    pivot_value::Float64,
)::Nothing
    lu = cf.lu
    # Match ClpFactorization::saferTolerances (L3254 uses min(zeroTolerance(), zeroValue))
    lu.zero_tol = min(lu.zero_tol, zero_value)
    new_val = pivot_value > 0.0 ? pivot_value : (-lu.pivot_tol * pivot_value)
    lu.pivot_tol = min(max(lu.pivot_tol, new_val), 0.999)
    return nothing
end

@inline function clp_fact_adjust_area_factor!(cf::ClpFactorization)::Nothing
    lu = cf.lu
    if lu.npivots > 1 && lu.ncompressions * 10 > lu.npivots + 10
        lu.area_factor *= 1.1
    end
    return nothing
end

# ==============================================================================
# Direct zero-copy factorization from ClpPackedMatrix (1-based)
# ==============================================================================

"""
    clp_fact_factorize_packed!(...)

Runs the complete inner LU loop of `ClpFactorization::factorize` directly on
`cf.lu` using `clp_mat_count_basis` and `clp_mat_fill_basis!` with zero C++
buffer allocations and zero matrix copies:
  1. `allocate_areas!` directly in `cf.lu`
  2. Insert `n_row_basic` slacks (`-1.0`) directly into `lu.element_u` / `lu.index_row_u`
  3. Fill basic structural columns directly into `lu` via `clp_mat_fill_basis!`
  4. `preprocess!` + `factor!` + retry on `-99` (`area_factor *= 2`) or `-1` (`dense_threshold = 0`)
  5. On success (`status == 0`): reorder `pivot_variable`, `finalize_basis_permutation!`, `check_sparse!`

All index arrays (`pivot_temp`, `mat_starts`, `mat_indices`, `pivot_variable`)
are 1-based. Returns `lu.status` (`0` = nonsingular, `-1` = singular).
"""
function clp_fact_factorize_packed!(
    cf::ClpFactorization,
    nrows::Int,
    ncols_model::Int,
    n_row_basic::Int,
    n_basic::Int,
    pivot_temp::AbstractVector{Int32},
    mat_starts::AbstractVector{Int32},
    mat_lengths::AbstractVector{Int32},
    mat_indices::AbstractVector{Int32},
    mat_elements::AbstractVector{Float64},
    has_zeros::Bool,
    row_scale::Union{Nothing,AbstractVector{Float64}},
    col_scale::Union{Nothing,AbstractVector{Float64}},
    number_iterations::Int,
    n_times_round::Int,
    pivot_variable::AbstractVector{Int32},
    use_number_rows::Int,
)::Int
    lu = cf.lu
    lu.status = -99
    saved_dense_threshold = lu.dense_threshold
    n_col_basic = n_basic - n_row_basic
    col_slice = @view pivot_temp[(n_row_basic + 1):n_basic]

    while lu.status == -99
        lu.sparse_threshold = 0
        lu.sparse_threshold2 = 0
        lu.len_area_r = 0

        n_elems = n_row_basic + clp_mat_count_basis(mat_lengths, col_slice, n_col_basic)
        area_elems = 3 * n_basic + 3 * n_elems + 20000
        allocate_areas!(lu, nrows, n_basic, area_elems, 2 * area_elems; already_scaled = false)

        len_row = lu.len_row_u
        len_col = lu.len_col_u
        el_u = lu.element_u
        idx_row = lu.index_row_u
        start_col = lu.start_col_u

        @inbounds for i in 1:(nrows + 1)
            len_row[i] = Int32(0)
        end
        @inbounds for i in 1:(lu.max_cols_extra + 1)
            len_col[i] = Int32(0)
        end

        slack_val = -1.0 # COIN_FAST_CODE
        @inbounds for i in 1:n_row_basic
            irow = Int(pivot_temp[i])
            idx_row[i] = Int32(irow)
            start_col[i] = Int32(i)
            el_u[i] = slack_val
            len_row[irow] = Int32(1)
            len_col[i] = Int32(1)
        end
        @inbounds start_col[n_row_basic + 1] = Int32(n_row_basic + 1)

        if n_col_basic > 0
            clp_mat_fill_basis!(
                mat_starts, mat_lengths, mat_indices, mat_elements, has_zeros,
                col_slice, n_col_basic,
                row_scale, col_scale,
                idx_row,
                @view(start_col[(n_row_basic + 1):(n_basic + 1)]),
                len_row,
                @view(len_col[(n_row_basic + 1):n_basic]),
                el_u,
            )
        end

        if n_basic > 0
            @inbounds actual_elems = Int(start_col[n_basic]) - 1 + Int(len_col[n_basic])
        else
            actual_elems = 0
        end

        cf.last_number_pivots = 0
        cf.effective_start_number_u = actual_elems - nrows
        lu.len_u = actual_elems

        state = (lu.bias_lu >= 3 || lu.nrows != lu.ncols) ? 2 : 3
        preprocess!(lu, state)
        st = factor!(lu)

        cf.end_length_u = lu.total_elements - lu.ndense * lu.ndense - lu.len_l

        if st == -99
            lu.area_factor *= 2.0
        elseif st == -1 && lu.dense_threshold != 0
            lu.dense_threshold = 0
            lu.status = -99
        end
    end

    if lu.status == 0
        perm_back = lu.permute_back
        piv_back = lu.pivot_col_back
        @inbounds begin
            for i in 1:n_row_basic
                k = Int(pivot_temp[i])
                j = Int(perm_back[Int(piv_back[i])])
                pivot_variable[j] = Int32(k + ncols_model)
            end
            for i in (n_row_basic + 1):use_number_rows
                k = Int(pivot_temp[i])
                j = Int(perm_back[Int(piv_back[i])])
                pivot_variable[j] = Int32(k)
            end
        end
        finalize_basis_permutation!(lu)
        check_sparse!(lu)
    elseif lu.status == -1
        # Sparse LU confirmed genuine singularity: restore dense_threshold for future bases
        if saved_dense_threshold != 0
            lu.dense_threshold = saved_dense_threshold
        end
        # Leave pivot_variable in useful form for cleaning singular basis:
        # pivot_variable[r] = sequence of basic variable that pivoted on row r,
        # or Int32(-1) if row r was unpivoted.
        piv_col = lu.pivot_col
        @inbounds for r in 1:use_number_rows
            pivot_variable[r] = Int32(-1)
        end
        @inbounds for i in 1:n_row_basic
            r = Int(piv_col[i])
            if r > 0
                pivot_variable[r] = Int32(Int(pivot_temp[i]) + ncols_model)
            end
        end
        @inbounds for i in (n_row_basic + 1):n_basic
            r = Int(piv_col[i])
            if r > 0
                pivot_variable[r] = pivot_temp[i]
            end
        end
    end

    return lu.status
end

"""
    clp_fact_factorize_csc!(...)

Generic 1-based CSC entry point for non-`ClpPackedMatrix` matrices (e.g.,
`ClpNetworkMatrix`, `ClpPlusMinusOneMatrix`, `ClpGubMatrix`, `ClpDynamicMatrix`).
Loads a pre-filled basis CSC into `cf.lu`, runs `preprocess!` + `factor!`,
and updates `end_length_u` / `effective_start_number_u` / `last_number_pivots`.
"""
function clp_fact_factorize_csc!(
    cf::ClpFactorization,
    nrows::Int,
    ncols::Int,
    nnz::Int,
    area_l::Int,
    area_u::Int,
    start_col::AbstractVector{Int32},
    len_col::AbstractVector{Int32},
    idx_row::AbstractVector{Int32},
    el_u::AbstractVector{Float64},
    len_row::AbstractVector{Int32},
    state::Int,
)::Int
    lu = cf.lu
    lu.sparse_threshold = 0
    lu.sparse_threshold2 = 0
    lu.len_area_r = 0

    allocate_areas!(lu, nrows, ncols, area_l, area_u; already_scaled = false)
    load_csc!(lu, nrows, ncols, nnz, start_col, len_col, idx_row, el_u, len_row; zero_based = false)

    cf.last_number_pivots = 0
    cf.effective_start_number_u = nnz - nrows

    preprocess!(lu, state)
    st = factor!(lu)
    cf.end_length_u = lu.total_elements - lu.ndense * lu.ndense - lu.len_l
    return st
end

"""
    clp_fact_reorder_and_finalize!(cf, ncols_model, n_row_basic, use_number_rows, pivot_temp, pivot_variable, do_check_sparse)

Reorders `pivot_variable` after a successful generic CSC factorization,
finalizes `pivot_col`/`pivot_col_back`, and optionally calls `check_sparse!`.
"""
function clp_fact_reorder_and_finalize!(
    cf::ClpFactorization,
    ncols_model::Int,
    n_row_basic::Int,
    use_number_rows::Int,
    pivot_temp::AbstractVector{Int32},
    pivot_variable::AbstractVector{Int32},
    do_check_sparse::Bool,
)::Nothing
    lu = cf.lu
    perm_back = lu.permute_back
    piv_back = lu.pivot_col_back
    @inbounds begin
        for i in 1:n_row_basic
            k = Int(pivot_temp[i])
            j = Int(perm_back[Int(piv_back[i])])
            pivot_variable[j] = Int32(k + ncols_model)
        end
        for i in (n_row_basic + 1):use_number_rows
            k = Int(pivot_temp[i])
            j = Int(perm_back[Int(piv_back[i])])
            pivot_variable[j] = Int32(k)
        end
    end
    finalize_basis_permutation!(lu)
    if do_check_sparse
        check_sparse!(lu)
    end
    return nothing
end

# ==============================================================================
# QP helper: CoinFactorization::emptyRows (1-based)
# ==============================================================================

function clp_fact_empty_rows!(
    cf::ClpFactorization,
    n_to_empty::Int,
    which::AbstractVector{Int32},
)::Nothing
    lu = cf.lu
    mre = lu.max_rows_extra
    _ensure_zero!(cf.del_row_buf, mre)
    del_row = cf.del_row_buf
    idx_row = lu.index_row_u
    el_u = lu.element_u
    start_col = lu.start_col_u
    len_row = lu.len_row_u
    len_col = lu.len_col_u

    @inbounds for i in 1:n_to_empty
        irow = Int(which[i])
        del_row[irow] = UInt8(1)
        len_row[irow] = Int32(0)
    end

    @inbounds for i in 1:lu.nu
        s = Int(start_col[i])
        e = s + Int(len_col[i]) - 1
        j = s
        for k in s:e
            irow = Int(idx_row[k])
            if del_row[irow] == UInt8(0)
                idx_row[j] = idx_row[k]
                el_u[j] = el_u[k]
                j += 1
            end
        end
        len_col[i] = Int32(j - s)
    end

    # Rebuild row-wise cross reference of U
    nr = lu.nrows
    start_row = lu.start_row_u
    idx_col = lu.index_col_u
    r2c = lu.row_to_col_u

    pos = 1
    @inbounds for irow in 1:nr
        start_row[irow] = pos % Int32
        pos += Int(len_row[irow])
    end
    lu.factor_elements = pos - 1

    @inbounds for irow in 1:nr
        len_row[irow] = Int32(0)
    end

    @inbounds for i in 1:nr
        s = Int(start_col[i])
        e = s + Int(len_col[i]) - 1
        for j in s:e
            irow = Int(idx_row[j])
            ilook = Int(len_row[irow])
            len_row[irow] = (ilook + 1) % Int32
            k = Int(start_row[irow]) + ilook
            idx_col[k] = i % Int32
            r2c[k] = j % Int32
        end
    end
    return nothing
end

# ==============================================================================
# Hot simplex operations: replaceColumn, FTRAN, BTRAN, goSparse, getWeights
# ==============================================================================

function clp_fact_replace_column!(
    cf::ClpFactorization,
    w_val::AbstractVector{Float64},
    w_idx::AbstractVector{Int32},
    tab_val::AbstractVector{Float64},
    tab_idx::AbstractVector{Int32},
    tab_nnz::Int,
    tab_packed::Bool,
    pivot_row::Int,
    pivot_check::Float64,
    check_before::Bool,
    n_new::Int,
    n_old::Int,
)::Int
    cf.effective_start_number_u += n_new - n_old
    lu = cf.lu
    if lu.do_forrest_tomlin
        return replace_column!(lu, w_val, w_idx, pivot_row, pivot_check, check_before)
    else
        return replace_column_pfi!(lu, tab_val, tab_idx, tab_nnz, tab_packed, pivot_row, pivot_check)
    end
end

function clp_fact_update_column_ft!(
    cf::ClpFactorization,
    w_val::AbstractVector{Float64},
    w_idx::AbstractVector{Int32},
    r_val::AbstractVector{Float64},
    r_idx::AbstractVector{Int32},
    r_nnz::Int,
    r_packed::Bool,
)::Int
    lu = cf.lu
    if lu.nrows == 0
        return 0
    end
    lu.collect_statistics = true
    rc = ftran_ft!(lu, w_val, w_idx, r_val, r_idx, r_nnz, r_packed)
    lu.collect_statistics = false
    return rc
end

function clp_fact_update_column!(
    cf::ClpFactorization,
    w_val::AbstractVector{Float64},
    w_idx::AbstractVector{Int32},
    w_nnz::Int,
    r_val::AbstractVector{Float64},
    r_idx::AbstractVector{Int32},
    r_nnz::Int,
    r_packed::Bool,
    no_permute::Bool,
)::Int
    lu = cf.lu
    if lu.nrows == 0
        return 0
    end
    lu.collect_statistics = cf.do_statistics
    rc = ftran!(lu, w_val, w_idx, w_nnz, r_val, r_idx, r_nnz, r_packed; no_permute = no_permute)
    lu.collect_statistics = false
    return rc
end

function clp_fact_update_two_columns_ft!(
    cf::ClpFactorization,
    r1_val::AbstractVector{Float64},
    r1_idx::AbstractVector{Int32},
    r2_val::AbstractVector{Float64},
    r2_idx::AbstractVector{Int32},
    r2_nnz::Int,
    r2_packed::Bool,
    r3_val::AbstractVector{Float64},
    r3_idx::AbstractVector{Int32},
    r3_nnz::Int,
    r3_packed::Bool,
    no_permute_r3::Bool,
)::Tuple{Int,Int}
    lu = cf.lu
    if lu.nrows == 0
        return (0, r3_nnz)
    end
    lu.collect_statistics = true
    if clp_fact_space_for_ft(cf)
        out2, out3 = ftran_two_ft!(
            lu, r1_val, r1_idx,
            r2_val, r2_idx, r2_nnz, r2_packed,
            r3_val, r3_idx, r3_nnz, r3_packed;
            no_permute_r3 = no_permute_r3,
        )
    else
        out2 = ftran_ft!(lu, r1_val, r1_idx, r2_val, r2_idx, r2_nnz, r2_packed)
        if no_permute_r3
            out3 = ftran!(lu, r3_val, r3_idx, r3_nnz, r3_val, r3_idx, r3_nnz, r3_packed; no_permute = true)
        else
            out3 = ftran!(lu, r1_val, r1_idx, 0, r3_val, r3_idx, r3_nnz, r3_packed; no_permute = false)
        end
    end
    lu.collect_statistics = false
    return (out2, out3)
end

function clp_fact_update_column_for_debug!(
    cf::ClpFactorization,
    w_val::AbstractVector{Float64},
    w_idx::AbstractVector{Int32},
    w_nnz::Int,
    r_val::AbstractVector{Float64},
    r_idx::AbstractVector{Int32},
    r_nnz::Int,
    r_packed::Bool,
    no_permute::Bool,
)::Int
    lu = cf.lu
    if lu.nrows == 0
        return 0
    end
    # Save and restore FTRAN/BTRAN statistics (matching ClpFactorization::updateColumnForDebug)
    s_nf  = lu.nftran_counts
    s_fi  = lu.ftran_count_input
    s_fl  = lu.ftran_count_after_l
    s_fr  = lu.ftran_count_after_r
    s_fu  = lu.ftran_count_after_u
    s_fal = lu.ftran_avg_after_l
    s_far = lu.ftran_avg_after_r
    s_fau = lu.ftran_avg_after_u
    s_nb  = lu.nbtran_counts
    s_bi  = lu.btran_count_input
    s_bu  = lu.btran_count_after_u
    s_br  = lu.btran_count_after_r
    s_bl  = lu.btran_count_after_l
    s_bau = lu.btran_avg_after_u
    s_bar = lu.btran_avg_after_r
    s_bal = lu.btran_avg_after_l

    lu.collect_statistics = false
    rc = ftran!(lu, w_val, w_idx, w_nnz, r_val, r_idx, r_nnz, r_packed; no_permute = no_permute)

    lu.nftran_counts       = s_nf
    lu.ftran_count_input   = s_fi
    lu.ftran_count_after_l = s_fl
    lu.ftran_count_after_r = s_fr
    lu.ftran_count_after_u = s_fu
    lu.ftran_avg_after_l   = s_fal
    lu.ftran_avg_after_r   = s_far
    lu.ftran_avg_after_u   = s_fau
    lu.nbtran_counts       = s_nb
    lu.btran_count_input   = s_bi
    lu.btran_count_after_u = s_bu
    lu.btran_count_after_r = s_br
    lu.btran_count_after_l = s_bl
    lu.btran_avg_after_u   = s_bau
    lu.btran_avg_after_r   = s_bar
    lu.btran_avg_after_l   = s_bal
    return rc
end

function clp_fact_update_column_transpose!(
    cf::ClpFactorization,
    w_val::AbstractVector{Float64},
    w_idx::AbstractVector{Int32},
    w_nnz::Int,
    w_packed::Bool,
    r_val::AbstractVector{Float64},
    r_idx::AbstractVector{Int32},
    r_nnz::Int,
    r_packed::Bool,
)::Int
    lu = cf.lu
    if lu.nrows == 0
        return 0
    end
    lu.collect_statistics = cf.do_statistics
    rc = btran!(lu, w_val, w_idx, w_nnz, w_packed, r_val, r_idx, r_nnz, r_packed)
    lu.collect_statistics = false
    return rc
end

function clp_fact_update_two_columns_transpose!(
    cf::ClpFactorization,
    w_val::AbstractVector{Float64},
    w_idx::AbstractVector{Int32},
    w_nnz::Int,
    w_packed::Bool,
    r2_val::AbstractVector{Float64},
    r2_idx::AbstractVector{Int32},
    r2_nnz::Int,
    r2_packed::Bool,
    r3_val::AbstractVector{Float64},
    r3_idx::AbstractVector{Int32},
    r3_nnz::Int,
    r3_packed::Bool,
)::Tuple{Int,Int}
    lu = cf.lu
    if lu.nrows == 0
        return (r2_nnz, r3_nnz)
    end
    lu.collect_statistics = cf.do_statistics
    out3 = btran!(lu, w_val, w_idx, w_nnz, w_packed, r3_val, r3_idx, r3_nnz, r3_packed)
    out2 = btran!(lu, w_val, w_idx, 0, false, r2_val, r2_idx, r2_nnz, r2_packed)
    lu.collect_statistics = false
    return (out2, out3)
end

@inline function clp_fact_go_sparse!(cf::ClpFactorization)::Nothing
    go_sparse!(cf.lu)
    return nothing
end

@inline function clp_fact_clean_up!(cf::ClpFactorization)::Nothing
    reset_statistics!(cf.lu)
    return nothing
end

@inline function clp_fact_get_weights!(
    cf::ClpFactorization,
    weights::AbstractVector{Int32},
)::Nothing
    get_weights!(cf.lu, weights)
    return nothing
end

# C ABI entrypoints compiled AOT via JuliaC (--trim=safe --compile-ccallable).
# All functions here must be strictly type-stable, exception-free on hot paths,
# and zero-allocation when called inside solver loops.

const JCLP_BRIDGE_API_VERSION = Cint(1)

# --- B0: Smoke kernels ---

Base.@ccallable function jclp_bridge_version()::Cint
    return JCLP_BRIDGE_API_VERSION
end

Base.@ccallable function jclp_smoke_add(a::Cdouble, b::Cdouble)::Cdouble
    return a + b
end

Base.@ccallable function jclp_smoke_dot(
    n::Cint,
    x::Ptr{Cdouble},
    y::Ptr{Cdouble},
)::Cdouble
    acc = 0.0
    i = Cint(0)
    while i < n
        xi = unsafe_load(x, Int(i) + 1)
        yi = unsafe_load(y, Int(i) + 1)
        acc += xi * yi
        i += Cint(1)
    end
    return acc
end

# --- B1: CoinFinite, CoinThreadRandom, CoinSort ---

Base.@ccallable function jclp_coin_finite(val::Cdouble)::Cint
    return coin_finite(val) ? Cint(1) : Cint(0)
end

Base.@ccallable function jclp_coin_isnan(val::Cdouble)::Cint
    return coin_isnan(val) ? Cint(1) : Cint(0)
end

Base.@ccallable function jclp_random_double(seed_ptr::Ptr{UInt32})::Cdouble
    return coin_random_double_ptr!(seed_ptr)
end

Base.@ccallable function jclp_randomize(seed_ptr::Ptr{UInt32}, n::Cint)::Cvoid
    coin_randomize_ptr!(seed_ptr, Int32(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_2_i32_f64(
    k::Ptr{Cint},
    v::Ptr{Cdouble},
    n::Cint,
)::Cvoid
    coin_sort_2!(k, v, Int(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_2_f64_i32(
    k::Ptr{Cdouble},
    v::Ptr{Cint},
    n::Cint,
)::Cvoid
    coin_sort_2!(k, v, Int(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_2_i32_i32(
    k::Ptr{Cint},
    v::Ptr{Cint},
    n::Cint,
)::Cvoid
    coin_sort_2!(k, v, Int(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_2_std_i32_f64(
    k::Ptr{Cint},
    v::Ptr{Cdouble},
    n::Cint,
)::Cvoid
    coin_sort_2_std!(k, v, Int(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_2_std_f64_i32(
    k::Ptr{Cdouble},
    v::Ptr{Cint},
    n::Cint,
)::Cvoid
    coin_sort_2_std!(k, v, Int(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_2_std_i32_i32(
    k::Ptr{Cint},
    v::Ptr{Cint},
    n::Cint,
)::Cvoid
    coin_sort_2_std!(k, v, Int(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_2_desc_f64_i32(
    k::Ptr{Cdouble},
    v::Ptr{Cint},
    n::Cint,
)::Cvoid
    coin_sort_2_desc!(k, v, Int(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_2_desc_i32_f64(
    k::Ptr{Cint},
    v::Ptr{Cdouble},
    n::Cint,
)::Cvoid
    coin_sort_2_desc!(k, v, Int(n))
    return nothing
end

Base.@ccallable function jclp_coin_sort_3_i32_i32_f64(
    k::Ptr{Cint},
    v::Ptr{Cint},
    w::Ptr{Cdouble},
    n::Cint,
)::Cvoid
    coin_sort_3!(k, v, w, Int(n))
    return nothing
end

# --- B2: CoinIndexedVector ---

Base.@ccallable function jclp_civ_clear(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    n_elements::Cint,
    capacity::Cint,
    packed_mode::Cint,
)::Cvoid
    civ_clear_ptr!(elements, indices, Int32(n_elements), Int32(capacity), packed_mode != Cint(0))
    return nothing
end

Base.@ccallable function jclp_civ_clean(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    n_elements::Cint,
    tolerance::Cdouble,
)::Cint
    return Cint(civ_clean_ptr!(elements, indices, Int32(n_elements), tolerance))
end

Base.@ccallable function jclp_civ_scan(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    n_elements::Cint,
    capacity::Cint,
    start_idx::Cint,
    end_idx::Cint,
)::Cint
    return Cint(civ_scan_ptr!(elements, indices, Int32(n_elements), Int32(capacity), Int32(start_idx), Int32(end_idx)))
end

Base.@ccallable function jclp_civ_scan_tol(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    n_elements::Cint,
    capacity::Cint,
    start_idx::Cint,
    end_idx::Cint,
    tolerance::Cdouble,
)::Cint
    return Cint(civ_scan_tol_ptr!(elements, indices, Int32(n_elements), Int32(capacity), Int32(start_idx), Int32(end_idx), tolerance))
end

Base.@ccallable function jclp_civ_clean_and_pack(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    n_elements::Cint,
    tolerance::Cdouble,
)::Cint
    return Cint(civ_clean_and_pack_ptr!(elements, indices, Int32(n_elements), tolerance))
end

Base.@ccallable function jclp_civ_scan_and_pack(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    n_elements::Cint,
    capacity::Cint,
    start_idx::Cint,
    end_idx::Cint,
)::Cint
    return Cint(civ_scan_and_pack_ptr!(elements, indices, Int32(n_elements), Int32(capacity), Int32(start_idx), Int32(end_idx)))
end

Base.@ccallable function jclp_civ_scan_and_pack_tol(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    n_elements::Cint,
    capacity::Cint,
    start_idx::Cint,
    end_idx::Cint,
    tolerance::Cdouble,
)::Cint
    return Cint(civ_scan_and_pack_tol_ptr!(elements, indices, Int32(n_elements), Int32(capacity), Int32(start_idx), Int32(end_idx), tolerance))
end

Base.@ccallable function jclp_civ_create_packed(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    number::Cint,
    src_indices::Ptr{Cint},
    src_elements::Ptr{Cdouble},
)::Cvoid
    civ_create_packed_ptr!(elements, indices, Int32(number), src_indices, src_elements)
    return nothing
end

Base.@ccallable function jclp_civ_create_unpacked(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    number::Cint,
    src_indices::Ptr{Cint},
    src_elements::Ptr{Cdouble},
)::Cvoid
    civ_create_unpacked_ptr!(elements, indices, Int32(number), src_indices, src_elements)
    return nothing
end

Base.@ccallable function jclp_civ_copy_fast(
    dst_elements::Ptr{Cdouble},
    dst_indices::Ptr{Cint},
    dst_n::Cint,
    capacity::Cint,
    dst_packed::Cint,
    src_elements::Ptr{Cdouble},
    src_indices::Ptr{Cint},
    src_n::Cint,
    src_packed::Cint,
    multiplier::Cdouble,
)::Cint
    return Cint(civ_copy_fast_ptr!(
        dst_elements, dst_indices, Int32(dst_n), Int32(capacity), dst_packed != Cint(0),
        src_elements, src_indices, Int32(src_n), src_packed != Cint(0), multiplier,
    ))
end

Base.@ccallable function jclp_civ_add(
    elements::Ptr{Cdouble},
    indices::Ptr{Cint},
    n_elements::Cint,
    index::Cint,
    element::Cdouble,
)::Cint
    return Cint(civ_add_ptr!(elements, indices, Int32(n_elements), Int32(index), element))
end

# --- B3: CoinPackedMatrix ---

Base.@ccallable function jclp_cpm_count_ortho_length(
    major_dim::Cint,
    minor_dim::Cint,
    start::Ptr{Cint},
    length_arr::Ptr{Cint},
    index::Ptr{Cint},
    ortho_length::Ptr{Cint},
)::Cvoid
    cpm_count_ortho_length_ptr!(Int32(major_dim), Int32(minor_dim), start, length_arr, index, ortho_length)
    return nothing
end

Base.@ccallable function jclp_cpm_reverse_scatter(
    src_major_dim::Cint,
    src_start::Ptr{Cint},
    src_length::Ptr{Cint},
    src_index::Ptr{Cint},
    src_element::Ptr{Cdouble},
    dst_major_dim::Cint,
    dst_start::Ptr{Cint},
    dst_length::Ptr{Cint},
    dst_index::Ptr{Cint},
    dst_element::Ptr{Cdouble},
)::Cvoid
    cpm_reverse_scatter_ptr!(
        Int32(src_major_dim), src_start, src_length, src_index, src_element,
        Int32(dst_major_dim), dst_start, dst_length, dst_index, dst_element,
    )
    return nothing
end

Base.@ccallable function jclp_cpm_remove_gaps(
    major_dim::Cint,
    size_in::Cint,
    start::Ptr{Cint},
    length_arr::Ptr{Cint},
    index::Ptr{Cint},
    element::Ptr{Cdouble},
    remove_value::Cdouble,
)::Cint
    return Cint(cpm_remove_gaps_ptr!(
        Int32(major_dim), Int32(size_in), start, length_arr, index, element, remove_value,
    ))
end

Base.@ccallable function jclp_cpm_times_major(
    major_dim::Cint,
    minor_dim::Cint,
    start::Ptr{Cint},
    length_arr::Ptr{Cint},
    index::Ptr{Cint},
    element::Ptr{Cdouble},
    x::Ptr{Cdouble},
    y::Ptr{Cdouble},
)::Cvoid
    cpm_times_major_ptr!(Int32(major_dim), Int32(minor_dim), start, length_arr, index, element, x, y)
    return nothing
end

Base.@ccallable function jclp_cpm_times_minor(
    major_dim::Cint,
    start::Ptr{Cint},
    length_arr::Ptr{Cint},
    index::Ptr{Cint},
    element::Ptr{Cdouble},
    x::Ptr{Cdouble},
    y::Ptr{Cdouble},
)::Cvoid
    cpm_times_minor_ptr!(Int32(major_dim), start, length_arr, index, element, x, y)
    return nothing
end

Base.@ccallable function jclp_cpm_eliminate_duplicates(
    major_dim::Cint,
    minor_dim::Cint,
    start::Ptr{Cint},
    length_arr::Ptr{Cint},
    index::Ptr{Cint},
    element::Ptr{Cdouble},
    threshold::Cdouble,
    mark::Ptr{Cint},
)::Cint
    return Cint(cpm_eliminate_duplicates_ptr!(
        Int32(major_dim), Int32(minor_dim), start, length_arr, index, element, threshold, mark,
    ))
end

Base.@ccallable function jclp_cpm_clean_matrix_core(
    major_dim::Cint,
    minor_dim::Cint,
    start::Ptr{Cint},
    length_arr::Ptr{Cint},
    index::Ptr{Cint},
    element::Ptr{Cdouble},
    threshold::Cdouble,
    mark::Ptr{Cint},
)::Cint
    return Cint(cpm_clean_matrix_core_ptr!(
        Int32(major_dim), Int32(minor_dim), start, length_arr, index, element, threshold, mark,
    ))
end

# --- B4: CoinDenseFactorization ---

Base.@ccallable function jclp_cdf_factor(
    n_rows::Cint,
    n_cols::Cint,
    elements::Ptr{Cdouble},
    pivot_row::Ptr{Cint},
    work_area::Ptr{Cdouble},
    zero_tol::Cdouble,
    io_solve_mode::Ptr{Cint},
    out_status::Ptr{Cint},
)::Cint
    return Cint(cdf_factor_ptr!(
        Int32(n_rows), Int32(n_cols), elements, pivot_row, work_area,
        zero_tol, io_solve_mode, out_status,
    ))
end

Base.@ccallable function jclp_cdf_replace_column(
    n_rows::Cint,
    n_cols::Cint,
    n_pivots::Cint,
    max_pivots::Cint,
    solve_mode::Cint,
    elements_base::Ptr{Cdouble},
    pivot_row_arr::Ptr{Cint},
    zero_tol::Cdouble,
    region::Ptr{Cdouble},
    region_index::Ptr{Cint},
    n_nonzero::Cint,
    packed_mode::Cint,
    pivot_row_idx::Cint,
    pivot_check::Cdouble,
)::Cint
    return Cint(cdf_replace_column_ptr!(
        Int32(n_rows), Int32(n_cols), Int32(n_pivots), Int32(max_pivots), Int32(solve_mode),
        elements_base, pivot_row_arr, zero_tol, region, region_index,
        Int32(n_nonzero), packed_mode != Cint(0), Int32(pivot_row_idx), pivot_check,
    ))
end

Base.@ccallable function jclp_cdf_update_column(
    n_rows::Cint,
    n_pivots::Cint,
    solve_mode::Cint,
    elements_base::Ptr{Cdouble},
    pivot_row_arr::Ptr{Cint},
    zero_tol::Cdouble,
    region1::Ptr{Cdouble},
    region2::Ptr{Cdouble},
    region_index::Ptr{Cint},
    n_nonzero_in::Cint,
    packed_mode::Cint,
    no_permute::Cint,
)::Cint
    return Cint(cdf_update_column_ptr!(
        Int32(n_rows), Int32(n_pivots), Int32(solve_mode), elements_base, pivot_row_arr, zero_tol,
        region1, region2, region_index, Int32(n_nonzero_in),
        packed_mode != Cint(0), no_permute != Cint(0),
    ))
end

Base.@ccallable function jclp_cdf_update_column_transpose(
    n_rows::Cint,
    n_pivots::Cint,
    solve_mode::Cint,
    elements_base::Ptr{Cdouble},
    pivot_row_arr::Ptr{Cint},
    zero_tol::Cdouble,
    region1::Ptr{Cdouble},
    region2::Ptr{Cdouble},
    region_index::Ptr{Cint},
    n_nonzero_in::Cint,
    packed_mode::Cint,
)::Cint
    return Cint(cdf_update_column_transpose_ptr!(
        Int32(n_rows), Int32(n_pivots), Int32(solve_mode), elements_base, pivot_row_arr, zero_tol,
        region1, region2, region_index, Int32(n_nonzero_in), packed_mode != Cint(0),
    ))
end

# --- B5a: CoinFactorization BTRAN / FTRAN ---

Base.@ccallable function jclp_cf_pod_sizeof()::Cint
    return Cint(sizeof(CoinFactorizationPodView))
end

Base.@ccallable function jclp_cf_update_column_transpose(
    cf_ptr::Ptr{CoinFactorizationPodView},
    reg1_ptr::Ptr{CoinIndexedVecPod},
    reg2_ptr::Ptr{CoinIndexedVecPod},
)::Cint
    cf = unsafe_load(cf_ptr)
    return Cint(cf_update_column_transpose!(cf, reg1_ptr, reg2_ptr))
end

Base.@ccallable function jclp_cf_update_column(
    cf_ptr::Ptr{CoinFactorizationPodView},
    reg1_ptr::Ptr{CoinIndexedVecPod},
    reg2_ptr::Ptr{CoinIndexedVecPod},
    no_permute::Cint,
)::Cint
    cf = unsafe_load(cf_ptr)
    return Cint(cf_update_column!(cf, reg1_ptr, reg2_ptr, no_permute != Cint(0)))
end

Base.@ccallable function jclp_cf_update_column_ft(
    cf_ptr::Ptr{CoinFactorizationPodView},
    reg1_ptr::Ptr{CoinIndexedVecPod},
    reg2_ptr::Ptr{CoinIndexedVecPod},
)::Cint
    cf = unsafe_load(cf_ptr)
    return Cint(cf_update_column_ft!(cf, reg1_ptr, reg2_ptr))
end

Base.@ccallable function jclp_cf_update_two_columns_ft(
    cf_ptr::Ptr{CoinFactorizationPodView},
    reg1_ptr::Ptr{CoinIndexedVecPod},
    reg2_ptr::Ptr{CoinIndexedVecPod},
    reg3_ptr::Ptr{CoinIndexedVecPod},
    no_permute_reg3::Cint,
    out_n3_ptr::Ptr{Cint},
)::Cint
    cf = unsafe_load(cf_ptr)
    return Cint(cf_update_two_columns_ft!(cf, reg1_ptr, reg2_ptr, reg3_ptr, no_permute_reg3 != Cint(0), out_n3_ptr))
end

# --- B5b: CoinFactorization Forrest-Tomlin replaceColumn ---

Base.@ccallable function jclp_cf_replace_state_sizeof()::Cint
    return Cint(sizeof(CoinFactorizationReplaceStatePod))
end

Base.@ccallable function jclp_cf_replace_column(
    cf_ptr::Ptr{CoinFactorizationPodView},
    st_ptr::Ptr{CoinFactorizationReplaceStatePod},
    reg_ptr::Ptr{CoinIndexedVecPod},
    pivot_row::Cint,
    pivot_check::Cdouble,
    check_before_modifying::Cint,
)::Cint
    cf = unsafe_load(cf_ptr)
    return Cint(cf_replace_column!(cf, st_ptr, reg_ptr, Int32(pivot_row), pivot_check, check_before_modifying != Cint(0)))
end

# --- B5c: CoinFactorization Markowitz Sparse LU ---

Base.@ccallable function jclp_cf_factor_state_sizeof()::Cint
    return Cint(sizeof(CoinFactorizationFactorStatePod))
end

Base.@ccallable function jclp_cf_pre_process(
    st_ptr::Ptr{CoinFactorizationFactorStatePod},
    state::Cint,
)::Cvoid
    cf_pre_process!(st_ptr, Int32(state))
    return nothing
end

Base.@ccallable function jclp_cf_factor_sparse(
    st_ptr::Ptr{CoinFactorizationFactorStatePod},
)::Cint
    return Cint(cf_factor_sparse!(st_ptr))
end

Base.@ccallable function jclp_cf_factor_dense(
    st_ptr::Ptr{CoinFactorizationFactorStatePod},
)::Cint
    return Cint(cf_factor_dense!(st_ptr))
end

Base.@ccallable function jclp_cf_factor_singular(
    st_ptr::Ptr{CoinFactorizationFactorStatePod},
)::Cint
    return Cint(cf_factor_singular!(st_ptr))
end

Base.@ccallable function jclp_cf_cleanup(
    st_ptr::Ptr{CoinFactorizationFactorStatePod},
)::Cvoid
    cf_cleanup!(st_ptr)
    return nothing
end

Base.@ccallable function jclp_cf_go_sparse_thresholds(
    st_ptr::Ptr{CoinFactorizationFactorStatePod},
)::Cint
    return Cint(cf_go_sparse_thresholds!(st_ptr))
end

Base.@ccallable function jclp_cf_go_sparse_build(
    st_ptr::Ptr{CoinFactorizationFactorStatePod},
)::Cvoid
    cf_go_sparse_build!(st_ptr)
    return nothing
end

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

# --- B6: CoinMpsIO ---

Base.@ccallable function jclp_mps_reader_pod_sizeof()::Cint
    return Cint(sizeof(CoinMpsCardReaderPod))
end

Base.@ccallable function jclp_mps_osi_strtod(
    ptr::Ptr{UInt8},
    out_ptr::Ptr{Ptr{UInt8}},
    fmt_type::Cint,
    strings_allowed::Cint,
    value_string::Ptr{UInt8},
)::Cdouble
    return mps_osi_strtod(ptr, out_ptr, Int32(fmt_type), strings_allowed != Cint(0), value_string)
end

Base.@ccallable function jclp_mps_read_to_next_section(
    r_ptr::Ptr{CoinMpsCardReaderPod},
)::Cint
    return Cint(mps_read_to_next_section!(r_ptr))
end

Base.@ccallable function jclp_mps_next_field(
    r_ptr::Ptr{CoinMpsCardReaderPod},
)::Cint
    return Cint(mps_next_field!(r_ptr))
end

Base.@ccallable function jclp_mps_clean_card(
    r_ptr::Ptr{CoinMpsCardReaderPod},
)::Cint
    return Cint(mps_clean_card!(r_ptr))
end

Base.@ccallable function jclp_mps_start_hash(
    names::Ptr{Ptr{UInt8}},
    number::Cint,
    hash_tbl::Ptr{CoinHashLinkPod},
)::Cvoid
    mps_start_hash!(names, Int32(number), hash_tbl)
    return nothing
end

Base.@ccallable function jclp_mps_find_hash(
    name::Ptr{UInt8},
    names::Ptr{Ptr{UInt8}},
    number::Cint,
    hash_tbl::Ptr{CoinHashLinkPod},
)::Cint
    return Cint(mps_find_hash(name, names, Int32(number), hash_tbl))
end

Base.@ccallable function jclp_mps_convert_double(
    section::Cint,
    format_type::Cint,
    value::Cdouble,
    out::Ptr{UInt8},
)::Cvoid
    coin_convert_double!(Int32(section), Int32(format_type), value, out)
    return nothing
end

# --- B7a: CoinPresolve Core Helpers + Zeros + Empty + Isolated + Useless + Fixed ---

Base.@ccallable function jclp_ps_pod_sizeof()::Cint
    return Cint(sizeof(CoinPresolvePodView))
end

Base.@ccallable function jclp_post_pod_sizeof()::Cint
    return Cint(sizeof(CoinPostsolvePodView))
end

Base.@ccallable function jclp_ps_make_memlists(
    lengths::Ptr{Cint},
    lnk::Ptr{PresolveHLinkPod},
    n::Cint,
)::Cvoid
    ps_make_memlists!(lengths, lnk, Int32(n))
    return nothing
end

Base.@ccallable function jclp_ps_expand_major(
    majstrts::Ptr{Cint},
    els::Ptr{Cdouble},
    minndxs::Ptr{Cint},
    majlens::Ptr{Cint},
    majlinks::Ptr{PresolveHLinkPod},
    nmaj::Cint,
    k::Cint,
)::Cint
    return ps_expand_major!(majstrts, els, minndxs, majlens, majlinks, Int32(nmaj), Int32(k)) ? Cint(1) : Cint(0)
end

Base.@ccallable function jclp_ps_delete_from_major2(
    majndx::Cint,
    minndx::Cint,
    majstrts::Ptr{Cint},
    majlens::Ptr{Cint},
    minndxs::Ptr{Cint},
    majlinks::Ptr{Cint},
    free_list_ptr::Ptr{Cint},
)::Cvoid
    ps_delete_from_major2!(Int32(majndx), Int32(minndx), majstrts, majlens, minndxs, majlinks, free_list_ptr)
    return nothing
end

Base.@ccallable function jclp_ps_recompute_sums(
    p_ptr::Ptr{CoinPresolvePodView},
    one_row::Cint,
)::Cint
    return Cint(ps_recompute_sums!(p_ptr, Int32(one_row)))
end

Base.@ccallable function jclp_ps_init_cols_todo(p_ptr::Ptr{CoinPresolvePodView})::Cvoid
    ps_init_cols_todo!(p_ptr)
    return nothing
end

Base.@ccallable function jclp_ps_init_rows_todo(p_ptr::Ptr{CoinPresolvePodView})::Cvoid
    ps_init_rows_todo!(p_ptr)
    return nothing
end

Base.@ccallable function jclp_ps_step_cols_todo(p_ptr::Ptr{CoinPresolvePodView})::Cint
    return Cint(ps_step_cols_todo!(p_ptr))
end

Base.@ccallable function jclp_ps_step_rows_todo(p_ptr::Ptr{CoinPresolvePodView})::Cint
    return Cint(ps_step_rows_todo!(p_ptr))
end

Base.@ccallable function jclp_ps_zeros_count(
    p_ptr::Ptr{CoinPresolvePodView},
    checkcols::Ptr{Cint},
    ncheck_ptr::Ptr{Cint},
)::Cint
    return Cint(ps_zeros_count!(p_ptr, checkcols, ncheck_ptr))
end

Base.@ccallable function jclp_ps_zeros_drop(
    p_ptr::Ptr{CoinPresolvePodView},
    checkcols::Ptr{Cint},
    ncheckcols::Cint,
    actions::Ptr{DroppedZeroPod},
)::Cint
    return Cint(ps_zeros_drop!(p_ptr, checkcols, Int32(ncheckcols), actions))
end

Base.@ccallable function jclp_ps_zeros_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    zeros::Ptr{DroppedZeroPod},
    nzeros::Cint,
)::Cvoid
    ps_zeros_postsolve!(p_ptr, zeros, Int32(nzeros))
    return nothing
end

Base.@ccallable function jclp_ps_drop_empty_cols_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    ecols::Ptr{Cint},
    necols::Cint,
    actions::Ptr{DropEmptyColActionPod},
    colmapping::Ptr{Cint},
    newclink::Ptr{PresolveHLinkPod},
    out_bad_col::Ptr{Cint},
)::Cint
    return Cint(ps_drop_empty_cols_presolve!(
        p_ptr, ecols, Int32(necols), actions, colmapping, newclink, out_bad_col,
    ))
end

Base.@ccallable function jclp_ps_drop_empty_cols_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{DropEmptyColActionPod},
    nactions::Cint,
    colmapping::Ptr{Cint},
)::Cvoid
    ps_drop_empty_cols_postsolve!(p_ptr, actions, Int32(nactions), colmapping)
    return nothing
end

Base.@ccallable function jclp_ps_drop_empty_rows_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    actions::Ptr{DropEmptyRowActionPod},
    rowmapping::Ptr{Cint},
    out_infeas_row::Ptr{Cint},
)::Cint
    return Cint(ps_drop_empty_rows_presolve!(p_ptr, actions, rowmapping, out_infeas_row))
end

Base.@ccallable function jclp_ps_drop_empty_rows_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{DropEmptyRowActionPod},
    nactions::Cint,
    rowmapping::Ptr{Cint},
)::Cvoid
    ps_drop_empty_rows_postsolve!(p_ptr, actions, Int32(nactions), rowmapping)
    return nothing
end

Base.@ccallable function jclp_ps_isolated_check(
    p_ptr::Ptr{CoinPresolvePodView},
    irow::Cint,
)::Cint
    return Cint(ps_isolated_check(p_ptr, Int32(irow)))
end

Base.@ccallable function jclp_ps_isolated_apply(
    p_ptr::Ptr{CoinPresolvePodView},
    irow::Cint,
    out_rowcols::Ptr{Cint},
    out_rowels::Ptr{Cdouble},
    out_costs::Ptr{Cdouble},
)::Cvoid
    ps_isolated_apply!(p_ptr, Int32(irow), out_rowcols, out_rowels, out_costs)
    return nothing
end

Base.@ccallable function jclp_ps_isolated_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    irow::Cint,
    ninrow::Cint,
    rlo_val::Cdouble,
    rup_val::Cdouble,
    rowcols::Ptr{Cint},
    rowels::Ptr{Cdouble},
)::Cvoid
    ps_isolated_postsolve!(p_ptr, Int32(irow), Int32(ninrow), rlo_val, rup_val, rowcols, rowels)
    return nothing
end

Base.@ccallable function jclp_ps_test_redundant_core(
    p_ptr::Ptr{CoinPresolvePodView},
    col_lower::Ptr{Cdouble},
    col_upper::Ptr{Cdouble},
    out_nuseless::Ptr{Cint},
    out_nfixed::Ptr{Cint},
    out_infeas_row::Ptr{Cint},
)::Cint
    return Cint(ps_test_redundant_core!(p_ptr, col_lower, col_upper, out_nuseless, out_nfixed, out_infeas_row))
end

Base.@ccallable function jclp_ps_useless_remove_rows(
    p_ptr::Ptr{CoinPresolvePodView},
    useless_rows::Ptr{Cint},
    nuseless::Cint,
)::Cvoid
    ps_useless_remove_rows!(p_ptr, useless_rows, Int32(nuseless))
    return nothing
end

Base.@ccallable function jclp_ps_useless_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{UselessActionPod},
    nactions::Cint,
)::Cvoid
    ps_useless_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

Base.@ccallable function jclp_ps_remove_fixed_filter(
    p_ptr::Ptr{CoinPresolvePodView},
    fcols::Ptr{Cint},
    nfcols_ptr::Ptr{Cint},
)::Cint
    return Cint(ps_remove_fixed_filter!(p_ptr, fcols, nfcols_ptr))
end

Base.@ccallable function jclp_ps_remove_fixed_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    fcols::Ptr{Cint},
    nfcols::Cint,
    actions::Ptr{RemoveFixedActionPod},
    els_action::Ptr{Cdouble},
    rows_action::Ptr{Cint},
    rstrt::Ptr{Cint},
    column_buf::Ptr{Cint},
    mark_buf::Ptr{UInt8},
)::Cint
    return Cint(ps_remove_fixed_presolve!(
        p_ptr, fcols, Int32(nfcols), actions, els_action, rows_action, rstrt, column_buf, mark_buf,
    ))
end

Base.@ccallable function jclp_ps_remove_fixed_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{RemoveFixedActionPod},
    nactions::Cint,
    els_action::Ptr{Cdouble},
    rows_action::Ptr{Cint},
)::Cvoid
    ps_remove_fixed_postsolve!(p_ptr, actions, Int32(nactions), els_action, rows_action)
    return nothing
end

Base.@ccallable function jclp_ps_make_fixed_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    fcols::Ptr{Cint},
    nfcols::Cint,
    fix_to_lower::Cint,
    actions::Ptr{MakeFixedActionPod},
)::Cvoid
    ps_make_fixed_presolve!(p_ptr, fcols, Int32(nfcols), fix_to_lower != Cint(0), actions)
    return nothing
end

Base.@ccallable function jclp_ps_make_fixed_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{MakeFixedActionPod},
    nactions::Cint,
    fix_to_lower::Cint,
)::Cvoid
    ps_make_fixed_postsolve!(p_ptr, actions, Int32(nactions), fix_to_lower != Cint(0))
    return nothing
end

Base.@ccallable function jclp_ps_transfer_costs(
    p_ptr::Ptr{CoinPresolvePodView},
)::Cvoid
    ps_transfer_costs!(p_ptr)
    return nothing
end

# -----------------------------------------------------------------------------
# B7b: CoinPresolveTighten, Forcing, Singleton, Dual
# -----------------------------------------------------------------------------

Base.@ccallable function jclp_ps_tighten_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    actions_out::Ptr{TightenActionPod},
    alloc_cb::Ptr{Cvoid},
    infeas_cb::Ptr{Cvoid},
    cb_ctx::Ptr{Cvoid},
    out_nactions::Ptr{Cint},
    out_nuseless::Ptr{Cint},
    out_nfixup::Ptr{Cint},
    out_nfixdown::Ptr{Cint},
)::Cvoid
    ps_tighten_presolve!(
        p_ptr, actions_out, alloc_cb, infeas_cb, cb_ctx,
        out_nactions, out_nuseless, out_nfixup, out_nfixdown,
    )
    return nothing
end

Base.@ccallable function jclp_ps_tighten_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{TightenActionPod},
    nactions::Cint,
)::Cvoid
    ps_tighten_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

Base.@ccallable function jclp_ps_can_be_feasible(
    p_ptr::Ptr{CoinPresolvePodView},
    irow::Cint,
)::Cint
    return Cint(ps_can_be_feasible(p_ptr, Int32(irow)))
end

Base.@ccallable function jclp_ps_forcing_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    actions_out::Ptr{ForcingActionPod},
    fixed_cols::Ptr{Cint},
    useless_rows::Ptr{Cint},
    alloc_cb::Ptr{Cvoid},
    out_nactions::Ptr{Cint},
    out_nuseless::Ptr{Cint},
    out_nfixed::Ptr{Cint},
    out_infeas_row::Ptr{Cint},
)::Cvoid
    ps_forcing_presolve!(
        p_ptr, actions_out, fixed_cols, useless_rows, alloc_cb,
        out_nactions, out_nuseless, out_nfixed, out_infeas_row,
    )
    return nothing
end

Base.@ccallable function jclp_ps_forcing_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{ForcingActionPod},
    nactions::Cint,
)::Cvoid
    ps_forcing_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

Base.@ccallable function jclp_ps_slack_doubleton_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    actions_out::Ptr{SlackActionPod},
    out_nactions::Ptr{Cint},
    out_nfixed::Ptr{Cint},
    out_infeas_col::Ptr{Cint},
    out_infeas_lo::Ptr{Cdouble},
    out_infeas_up::Ptr{Cdouble},
)::Cvoid
    ps_slack_doubleton_presolve!(
        p_ptr, actions_out, out_nactions, out_nfixed,
        out_infeas_col, out_infeas_lo, out_infeas_up,
    )
    return nothing
end

Base.@ccallable function jclp_ps_slack_doubleton_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{SlackActionPod},
    nactions::Cint,
)::Cvoid
    ps_slack_doubleton_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

Base.@ccallable function jclp_ps_slack_singleton_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    row_objective::Ptr{Cdouble},
    actions_ptr_ref::Ptr{Ptr{SlackActionPod}},
    max_actions::Cint,
    fixed_cols::Ptr{Cint},
    grow_cb::Ptr{Cvoid},
    out_nactions::Ptr{Cint},
    out_nfixed::Ptr{Cint},
)::Cvoid
    ps_slack_singleton_presolve!(
        p_ptr, row_objective, actions_ptr_ref, Int32(max_actions),
        fixed_cols, grow_cb, out_nactions, out_nfixed,
    )
    return nothing
end

Base.@ccallable function jclp_ps_slack_singleton_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{SlackActionPod},
    nactions::Cint,
)::Cvoid
    ps_slack_singleton_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

Base.@ccallable function jclp_ps_dual_presolve_phase1(
    p_ptr::Ptr{CoinPresolvePodView},
    cbarmin::Ptr{Cdouble},
    cbarmax::Ptr{Cdouble},
    msg_cb::Ptr{Cvoid},
    cb_ctx::Ptr{Cvoid},
    out_nfixup::Ptr{Cint},
    out_nfixdown::Ptr{Cint},
)::Cvoid
    ps_dual_presolve_phase1!(
        p_ptr, cbarmin, cbarmax, msg_cb, cb_ctx, out_nfixup, out_nfixdown,
    )
    return nothing
end

Base.@ccallable function jclp_ps_dual_presolve_phase2(
    p_ptr::Ptr{CoinPresolvePodView},
)::Cint
    return Cint(ps_dual_presolve_phase2!(p_ptr))
end

Base.@ccallable function jclp_ps_dual_presolve_apply(
    p_ptr::Ptr{CoinPresolvePodView},
    make_eq_cnt::Cint,
    bnd_records::Ptr{DualActionPod},
)::Cvoid
    ps_dual_presolve_apply!(p_ptr, Int32(make_eq_cnt), bnd_records)
    return nothing
end

Base.@ccallable function jclp_ps_dual_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    bnd_records::Ptr{DualActionPod},
    num_recs::Cint,
)::Cvoid
    ps_dual_postsolve!(p_ptr, bnd_records, Int32(num_recs))
    return nothing
end

# -----------------------------------------------------------------------------
# B7c1: CoinPresolveDoubleton, Tripleton, ImpliedFree
# -----------------------------------------------------------------------------

Base.@ccallable function jclp_ps_doubleton_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    actions_out::Ptr{DoubletonActionPod},
    alloc_dbl_cb::Ptr{Cvoid},
    out_nactions::Ptr{Cint},
    out_nzeros::Ptr{Cint},
    out_nfixed::Ptr{Cint},
    out_nomem::Ptr{Cint},
    out_infeas_col::Ptr{Cint},
    out_infeas_lo::Ptr{Cdouble},
    out_infeas_up::Ptr{Cdouble},
)::Cvoid
    ps_doubleton_presolve!(
        p_ptr, actions_out, alloc_dbl_cb,
        out_nactions, out_nzeros, out_nfixed, out_nomem,
        out_infeas_col, out_infeas_lo, out_infeas_up,
    )
    return nothing
end

Base.@ccallable function jclp_ps_doubleton_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{DoubletonActionPod},
    nactions::Cint,
    index1::Ptr{Cint},
    element1::Ptr{Cdouble},
)::Cvoid
    ps_doubleton_postsolve!(p_ptr, actions, Int32(nactions), index1, element1)
    return nothing
end

Base.@ccallable function jclp_ps_tripleton_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    actions_out::Ptr{TripletonActionPod},
    alloc_dbl_cb::Ptr{Cvoid},
    out_nactions::Ptr{Cint},
    out_nzeros::Ptr{Cint},
    out_nomem::Ptr{Cint},
)::Cvoid
    ps_tripleton_presolve!(
        p_ptr, actions_out, alloc_dbl_cb, out_nactions, out_nzeros, out_nomem,
    )
    return nothing
end

Base.@ccallable function jclp_ps_tripleton_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{TripletonActionPod},
    nactions::Cint,
    index1::Ptr{Cint},
    element1::Ptr{Cdouble},
    index2::Ptr{Cint},
    element2::Ptr{Cdouble},
)::Cvoid
    ps_tripleton_postsolve!(
        p_ptr, actions, Int32(nactions), index1, element1, index2, element2,
    )
    return nothing
end

Base.@ccallable function jclp_ps_check_row(
    mrstrt::Ptr{Cint},
    rowels::Ptr{Cdouble},
    hcol::Ptr{Cint},
    hinrow::Ptr{Cint},
    coeff_factor::Cdouble,
    kill_ratio::Cdouble,
    irowx::Cint,
    irowy::Cint,
    bad_out::Ptr{Cint},
)::Cint
    return Cint(ps_check_row(
        mrstrt, rowels, hcol, hinrow, coeff_factor, kill_ratio,
        Int32(irowx), Int32(irowy), bad_out,
    ))
end

Base.@ccallable function jclp_ps_implied_free_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    fill_level_ptr::Ptr{Cint},
    actions_out::Ptr{ImpliedFreeActionPod},
    infinite_down::Ptr{Cint},
    infinite_up::Ptr{Cint},
    max_down::Ptr{Cdouble},
    max_up::Ptr{Cdouble},
    alloc_dbl_cb::Ptr{Cvoid},
    out_nactions::Ptr{Cint},
    out_unprocessed::Ptr{Cint},
    out_number_free::Ptr{Cint},
    out_infeas_row::Ptr{Cint},
)::Cvoid
    ps_implied_free_presolve!(
        p_ptr, fill_level_ptr, actions_out,
        infinite_down, infinite_up, max_down, max_up,
        alloc_dbl_cb, out_nactions, out_unprocessed, out_number_free, out_infeas_row,
    )
    return nothing
end

Base.@ccallable function jclp_ps_implied_free_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{ImpliedFreeActionPod},
    nactions::Cint,
)::Cvoid
    ps_implied_free_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

# -----------------------------------------------------------------------------
# B7c2: CoinPresolveSubst, CoinPresolveDupcol (dupcol/duprow/gubrow/twoxtwo)
# -----------------------------------------------------------------------------

Base.@ccallable function jclp_ps_subst_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    implied_free::Ptr{Cint},
    which_free::Ptr{Cint},
    number_free::Cint,
    max_look::Cint,
    actions_out::Ptr{SubstActionPod},
    zerocols::Ptr{Cint},
    x_to_y::Ptr{Cint},
    alloc_subst_cb::Ptr{Cvoid},
    out_nactions::Ptr{Cint},
    out_nzerocols::Ptr{Cint},
    out_nomem::Ptr{Cint},
)::Cvoid
    ps_subst_presolve!(
        p_ptr, implied_free, which_free, Int32(number_free), Int32(max_look),
        actions_out, zerocols, x_to_y, alloc_subst_cb,
        out_nactions, out_nzerocols, out_nomem,
    )
    return nothing
end

Base.@ccallable function jclp_ps_subst_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{SubstActionPod},
    nactions::Cint,
)::Cvoid
    ps_subst_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

Base.@ccallable function jclp_ps_coin_init_random_vec(
    work::Ptr{Cdouble},
    n::Cint,
)::Cvoid
    ps_coin_init_random_vec!(work, Int32(n))
    return nothing
end

Base.@ccallable function jclp_ps_dupcol_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    rowmul_buf::Ptr{Cdouble},
    swap_used_buf::Ptr{UInt8},
    actions_out::Ptr{DupcolActionPod},
    fixed_down::Ptr{Cint},
    fixed_up::Ptr{Cint},
    alloc_dbl_cb::Ptr{Cvoid},
    out_nactions::Ptr{Cint},
    out_nfixed_down::Ptr{Cint},
    out_nfixed_up::Ptr{Cint},
)::Cvoid
    ps_dupcol_presolve!(
        p_ptr, rowmul_buf, swap_used_buf, actions_out, fixed_down, fixed_up,
        alloc_dbl_cb, out_nactions, out_nfixed_down, out_nfixed_up,
    )
    return nothing
end

Base.@ccallable function jclp_ps_dupcol_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{DupcolActionPod},
    nactions::Cint,
)::Cvoid
    ps_dupcol_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

Base.@ccallable function jclp_ps_duprow_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    sort_arr::Ptr{Cint},
    workrow::Ptr{Cdouble},
    workcol_buf::Ptr{Cdouble},
    out_nuseless::Ptr{Cint},
    out_infeas_row::Ptr{Cint},
)::Cvoid
    ps_duprow_presolve!(p_ptr, sort_arr, workrow, workcol_buf, out_nuseless, out_infeas_row)
    return nothing
end

Base.@ccallable function jclp_ps_gubrow_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    actions_out::Ptr{GubrowActionPod},
    alloc_gub_cb::Ptr{Cvoid},
)::Cint
    return Cint(ps_gubrow_presolve!(p_ptr, actions_out, alloc_gub_cb))
end

Base.@ccallable function jclp_ps_gubrow_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{GubrowActionPod},
    nactions::Cint,
)::Cvoid
    ps_gubrow_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

Base.@ccallable function jclp_ps_twoxtwo_presolve(
    p_ptr::Ptr{CoinPresolvePodView},
    bound_records::Ptr{TwoxtwoActionPod},
    scratch200::Ptr{Cdouble},
    out_offset::Ptr{Cdouble},
)::Cint
    return Cint(ps_twoxtwo_presolve!(p_ptr, bound_records, scratch200, out_offset))
end

Base.@ccallable function jclp_ps_twoxtwo_postsolve(
    p_ptr::Ptr{CoinPostsolvePodView},
    actions::Ptr{TwoxtwoActionPod},
    nactions::Cint,
)::Cvoid
    ps_twoxtwo_postsolve!(p_ptr, actions, Int32(nactions))
    return nothing
end

# =============================================================================
# Native Pure-Julia SparseLU Bridge (JCLP_BRICK_LU_NATIVE=1)
# Thin PtrView + opaque handle boundary; zero Ptr/unsafe_load inside SparseLU/.
# =============================================================================

Base.@ccallable function jclp_lu_create()::Cint
    return Cint(_lu_handle_create())
end

Base.@ccallable function jclp_lu_copy(src_id::Cint)::Cint
    return Cint(_lu_handle_copy(Int32(src_id)))
end

Base.@ccallable function jclp_lu_assign(dst_id::Cint, src_id::Cint)::Cvoid
    _lu_handle_assign!(Int32(dst_id), Int32(src_id))
    return nothing
end

Base.@ccallable function jclp_lu_destroy(id::Cint)::Cvoid
    _lu_handle_destroy!(Int32(id))
    return nothing
end

Base.@ccallable function jclp_lu_load_and_preprocess(
    id::Cint,
    state::Cint,
    nrows::Cint,
    ncols::Cint,
    nnz::Cint,
    len_area_l::Cint,
    len_area_u::Cint,
    max_pivots::Cint,
    bias_lu::Cint,
    dense_thresh::Cint,
    sparse_thresh::Cint,
    num_trials::Cint,
    do_ft::Cint,
    collect_stats::Cint,
    zero_tol::Cdouble,
    pivot_tol::Cdouble,
    relax_check::Cdouble,
    slack_val::Cdouble,
    area_factor::Cdouble,
    start_col_u::Ptr{Cint},
    len_col_u::Ptr{Cint},
    index_row_u::Ptr{Cint},
    index_col_u::Ptr{Cint},
    element_u::Ptr{Cdouble},
    len_row_u::Ptr{Cint},
)::Cvoid
    lu = _lu_get(Int32(id))
    lu.max_pivots = Int(max_pivots)
    lu.bias_lu = Int(bias_lu)
    lu.dense_threshold = Int(dense_thresh)
    lu.sparse_threshold = Int(sparse_thresh)
    lu.num_trials = Int(num_trials)
    lu.do_forrest_tomlin = do_ft != Cint(0)
    lu.collect_statistics = collect_stats != Cint(0)
    lu.zero_tol = Float64(zero_tol)
    lu.pivot_tol = Float64(pivot_tol)
    lu.relax_check = Float64(relax_check)
    lu.slack_value = Float64(slack_val)
    lu.area_factor = Float64(area_factor)

    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz)
    allocate_areas!(lu, nr, nc, Int(len_area_l), Int(len_area_u); already_scaled = true)

    st = Int(state)
    if st <= 1
        load_triplets!(
            lu, nr, nc, nz,
            PtrView(index_row_u, nz),
            PtrView(index_col_u, nz),
            PtrView(element_u, nz);
            zero_based = true,
        )
        if st == -1 || st == 1
            rc = PtrView(len_row_u, nr)
            cc = PtrView(len_col_u, nc)
            @inbounds for i in 1:nr
                lu.len_row_u[i] = rc[i]
            end
            @inbounds for j in 1:nc
                lu.len_col_u[j] = cc[j]
            end
        end
    else
        rc_view = len_row_u != C_NULL ? PtrView(len_row_u, nr) : nothing
        load_csc!(
            lu, nr, nc, nz,
            PtrView(start_col_u, nc),
            PtrView(len_col_u, nc),
            PtrView(index_row_u, nz),
            PtrView(element_u, nz),
            rc_view;
            zero_based = true,
        )
    end
    preprocess!(lu, st)
    return nothing
end

Base.@ccallable function jclp_lu_factor(
    id::Cint,
    out_permute::Ptr{Cint},
    out_permute_back::Ptr{Cint},
    out_pivot_col::Ptr{Cint},
    out_pivot_col_back::Ptr{Cint},
    out_meta_i32::Ptr{Cint},
    out_area_factor::Ptr{Cdouble},
)::Cint
    lu = _lu_get(Int32(id))
    st = factor!(lu)
    nr = lu.nrows
    nc = lu.ncols

    if st == 0
        if out_permute != C_NULL
            p_out = PtrView(out_permute, nr)
            @inbounds for i in 1:nr
                p_out[i] = lu.permute[i] - Int32(1)
            end
        end
        if out_permute_back != C_NULL
            pb_out = PtrView(out_permute_back, nr)
            @inbounds for i in 1:nr
                pb_out[i] = lu.permute_back[i] - Int32(1)
            end
        end
        if out_pivot_col != C_NULL
            pc_out = PtrView(out_pivot_col, nc)
            @inbounds for i in 1:nc
                v = lu.pivot_col[i]
                pc_out[i] = v > LU_NO_LINK ? (v - Int32(1)) : Int32(-1)
            end
        end
        if out_pivot_col_back != C_NULL
            pcb_out = PtrView(out_pivot_col_back, nr)
            @inbounds for i in 1:nr
                pcb_out[i] = lu.pivot_col_back[i] - Int32(1)
            end
        end
        finalize_basis_permutation!(lu)
    elseif st == -1
        if out_permute != C_NULL
            p_out = PtrView(out_permute, nr)
            @inbounds for i in 1:nr
                v = lu.permute[i]
                p_out[i] = v > LU_NO_LINK ? (v - Int32(1)) : Int32(-1)
            end
        end
        if out_pivot_col != C_NULL
            pc_out = PtrView(out_pivot_col, nc)
            @inbounds for i in 1:nc
                v = lu.pivot_col[i]
                pc_out[i] = v > LU_NO_LINK ? (v - Int32(1)) : Int32(-1)
            end
        end
    end

    if out_meta_i32 != C_NULL
        meta = PtrView(out_meta_i32, 13)
        @inbounds begin
            meta[1]  = Int32(lu.ngood_u)
            meta[2]  = Int32(lu.nu)
            meta[3]  = Int32(lu.nl)
            meta[4]  = Int32(lu.base_l)
            meta[5]  = Int32(lu.ndense)
            meta[6]  = Int32(lu.nslacks)
            meta[7]  = Int32(lu.len_u)
            meta[8]  = Int32(lu.len_l)
            meta[9]  = Int32(lu.len_area_r)
            meta[10] = Int32(lu.total_elements)
            meta[11] = Int32(lu.factor_elements)
            meta[12] = Int32(lu.ncompressions)
            meta[13] = Int32(lu.max_u)
        end
    end
    if out_area_factor != C_NULL
        unsafe_store!(out_area_factor, lu.area_factor)
    end
    return Cint(st)
end

Base.@ccallable function jclp_lu_go_sparse(
    id::Cint,
    sparse_thresh_in::Cint,
    out_sp1::Ptr{Cint},
    out_sp2::Ptr{Cint},
)::Cvoid
    lu = _lu_get(Int32(id))
    lu.sparse_threshold = Int(sparse_thresh_in)
    go_sparse!(lu)
    if out_sp1 != C_NULL
        unsafe_store!(out_sp1, Cint(lu.sparse_threshold))
    end
    if out_sp2 != C_NULL
        unsafe_store!(out_sp2, Cint(lu.sparse_threshold2))
    end
    return nothing
end

Base.@ccallable function jclp_lu_check_sparse(id::Cint)::Cvoid
    check_sparse!(_lu_get(Int32(id)))
    return nothing
end

Base.@ccallable function jclp_lu_reset_stats(id::Cint)::Cvoid
    reset_statistics!(_lu_get(Int32(id)))
    return nothing
end

Base.@ccallable function jclp_lu_get_weights(
    id::Cint,
    weights_ptr::Ptr{Cint},
    nrows::Cint,
)::Cvoid
    get_weights!(_lu_get(Int32(id)), PtrView(weights_ptr, Int(nrows)))
    return nothing
end

Base.@ccallable function jclp_lu_export_for_network(
    id::Cint,
    out_piv_reg::Ptr{Cdouble},
    out_perm_back::Ptr{Cint},
    out_start_col_u::Ptr{Cint},
    out_len_col_u::Ptr{Cint},
    out_idx_row_u::Ptr{Cint},
    out_el_u::Ptr{Cdouble},
)::Cvoid
    lu = _lu_get(Int32(id))
    nr = lu.nrows
    nz = lu.len_u
    pr = PtrView(out_piv_reg, nr)
    pb = PtrView(out_perm_back, nr)
    sc = PtrView(out_start_col_u, nr + 1)
    lc = PtrView(out_len_col_u, nr)
    ir = PtrView(out_idx_row_u, nz)
    eu = PtrView(out_el_u, nz)
    @inbounds for i in 1:nr
        pr[i] = lu.pivot_region[i]
        pb[i] = lu.permute_back[i] - Int32(1)
        sc[i] = lu.start_col_u[i] - Int32(1)
        lc[i] = lu.len_col_u[i]
    end
    @inbounds for k in 1:nz
        ir[k] = lu.index_row_u[k] - Int32(1)
        eu[k] = lu.element_u[k]
    end
    return nothing
end

Base.@ccallable function jclp_lu_btran(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_nnz::Cint,
    w_cap::Cint,
    w_packed::Cint,
    r_el::Ptr{Cdouble},
    r_idx::Ptr{Cint},
    r_nnz::Cint,
    r_cap::Cint,
    r_packed::Cint,
    collect_stats::Cint,
    zero_tol::Cdouble,
)::Cint
    lu = _lu_get(Int32(id))
    lu.collect_statistics = collect_stats != Cint(0)
    lu.zero_tol = Float64(zero_tol)
    wv = PtrView(w_el, Int(w_cap))
    wi = PtrView(w_idx, Int(w_cap))
    rv = PtrView(r_el, Int(r_cap))
    ri = PtrView(r_idx, Int(r_cap))
    wn = Int(w_nnz)
    rn = Int(r_nnz)
    if wn > 0 && w_packed == Cint(0)
        shift_to_1based!(wi, wn)
    end
    shift_to_1based!(ri, rn)
    nnz_out = btran!(lu, wv, wi, wn, w_packed != Cint(0), rv, ri, rn, r_packed != Cint(0))
    shift_to_0based!(ri, nnz_out)
    return Cint(nnz_out)
end

Base.@ccallable function jclp_lu_ftran(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_nnz::Cint,
    w_cap::Cint,
    r_el::Ptr{Cdouble},
    r_idx::Ptr{Cint},
    r_nnz::Cint,
    r_cap::Cint,
    r_packed::Cint,
    no_permute::Cint,
    collect_stats::Cint,
    zero_tol::Cdouble,
)::Cint
    lu = _lu_get(Int32(id))
    lu.collect_statistics = collect_stats != Cint(0)
    lu.zero_tol = Float64(zero_tol)
    wv = PtrView(w_el, Int(w_cap))
    wi = PtrView(w_idx, Int(w_cap))
    rv = PtrView(r_el, Int(r_cap))
    ri = PtrView(r_idx, Int(r_cap))
    np = no_permute != Cint(0)
    wn = Int(w_nnz)
    rn = Int(r_nnz)
    if np
        shift_to_1based!(wi, wn)
    else
        shift_to_1based!(ri, rn)
    end
    nnz_out = ftran!(lu, wv, wi, wn, rv, ri, rn, r_packed != Cint(0); no_permute = np)
    if np
        shift_to_0based!(wi, nnz_out)
    else
        shift_to_0based!(ri, nnz_out)
    end
    return Cint(nnz_out)
end

Base.@ccallable function jclp_lu_ftran_ft(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_cap::Cint,
    r_el::Ptr{Cdouble},
    r_idx::Ptr{Cint},
    r_nnz::Cint,
    r_cap::Cint,
    r_packed::Cint,
    collect_stats::Cint,
    zero_tol::Cdouble,
    out_u_state::Ptr{Cint},
)::Cint
    lu = _lu_get(Int32(id))
    lu.collect_statistics = collect_stats != Cint(0)
    lu.zero_tol = Float64(zero_tol)
    wv = PtrView(w_el, Int(w_cap))
    wi = PtrView(w_idx, Int(w_cap))
    rv = PtrView(r_el, Int(r_cap))
    ri = PtrView(r_idx, Int(r_cap))
    rn = Int(r_nnz)
    shift_to_1based!(ri, rn)
    res = ftran_ft!(lu, wv, wi, rv, ri, rn, r_packed != Cint(0))
    shift_to_0based!(ri, abs(res))
    if out_u_state != C_NULL
        st = PtrView(out_u_state, 2)
        @inbounds begin
            st[1] = lu.start_col_u[lu.max_cols_extra + 1] - Int32(1)
            st[2] = lu.len_col_u[lu.ncols_extra + 1]
        end
    end
    return Cint(res)
end

Base.@ccallable function jclp_lu_ftran_two_ft(
    id::Cint,
    r1_el::Ptr{Cdouble},
    r1_idx::Ptr{Cint},
    r1_cap::Cint,
    r2_el::Ptr{Cdouble},
    r2_idx::Ptr{Cint},
    r2_nnz::Cint,
    r2_cap::Cint,
    r2_packed::Cint,
    r3_el::Ptr{Cdouble},
    r3_idx::Ptr{Cint},
    r3_nnz::Cint,
    r3_cap::Cint,
    r3_packed::Cint,
    no_permute_r3::Cint,
    collect_stats::Cint,
    zero_tol::Cdouble,
    out_meta::Ptr{Cint},
)::Cint
    lu = _lu_get(Int32(id))
    lu.collect_statistics = collect_stats != Cint(0)
    lu.zero_tol = Float64(zero_tol)
    v1 = PtrView(r1_el, Int(r1_cap))
    i1 = PtrView(r1_idx, Int(r1_cap))
    v2 = PtrView(r2_el, Int(r2_cap))
    i2 = PtrView(r2_idx, Int(r2_cap))
    v3 = PtrView(r3_el, Int(r3_cap))
    i3 = PtrView(r3_idx, Int(r3_cap))
    n2 = Int(r2_nnz)
    n3 = Int(r3_nnz)
    shift_to_1based!(i2, n2)
    shift_to_1based!(i3, n3)
    out2, out3 = ftran_two_ft!(
        lu, v1, i1, v2, i2, n2, r2_packed != Cint(0),
        v3, i3, n3, r3_packed != Cint(0);
        no_permute_r3 = no_permute_r3 != Cint(0),
    )
    shift_to_0based!(i2, abs(out2))
    shift_to_0based!(i3, out3)
    st = PtrView(out_meta, 3)
    @inbounds begin
        st[1] = Int32(out3)
        st[2] = lu.start_col_u[lu.max_cols_extra + 1] - Int32(1)
        st[3] = lu.len_col_u[lu.ncols_extra + 1]
    end
    return Cint(out2)
end

Base.@ccallable function jclp_lu_replace_column(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_cap::Cint,
    pivot_row_0based::Cint,
    pivot_check::Cdouble,
    check_before::Cint,
    relax_check::Cdouble,
    zero_tol::Cdouble,
    out_meta_i32::Ptr{Cint},
)::Cint
    lu = _lu_get(Int32(id))
    lu.relax_check = Float64(relax_check)
    lu.zero_tol = Float64(zero_tol)
    ret = replace_column!(
        lu,
        PtrView(w_el, Int(w_cap)),
        PtrView(w_idx, Int(w_cap)),
        Int(pivot_row_0based) + 1,
        Float64(pivot_check),
        check_before != Cint(0),
    )
    if out_meta_i32 != C_NULL
        meta = PtrView(out_meta_i32, 11)
        @inbounds begin
            meta[1]  = Int32(lu.nrows_extra)
            meta[2]  = Int32(lu.ncols_extra)
            meta[3]  = Int32(lu.nu)
            meta[4]  = Int32(lu.ngood_u)
            meta[5]  = Int32(lu.npivots)
            meta[6]  = Int32(lu.nr)
            meta[7]  = Int32(lu.len_r)
            meta[8]  = Int32(lu.len_u)
            meta[9]  = Int32(lu.total_elements)
            meta[10] = Int32(lu.ncompressions)
            meta[11] = Int32(lu.status)
        end
    end
    return Cint(ret)
end

Base.@ccallable function jclp_lu_replace_column_pfi(
    id::Cint,
    r_el::Ptr{Cdouble},
    r_idx::Ptr{Cint},
    r_nnz::Cint,
    r_cap::Cint,
    r_packed::Cint,
    pivot_row_0based::Cint,
    alpha::Cdouble,
    out_npivots::Ptr{Cint},
    out_total_el::Ptr{Cint},
)::Cint
    lu = _lu_get(Int32(id))
    rv = PtrView(r_el, Int(r_cap))
    ri = PtrView(r_idx, Int(r_cap))
    rn = Int(r_nnz)
    shift_to_1based!(ri, rn)
    ret = replace_column_pfi!(lu, rv, ri, rn, r_packed != Cint(0), Int(pivot_row_0based) + 1, Float64(alpha))
    shift_to_0based!(ri, rn)
    if out_npivots != C_NULL
        unsafe_store!(out_npivots, Cint(lu.npivots))
    end
    if out_total_el != C_NULL
        unsafe_store!(out_total_el, Cint(lu.total_elements))
    end
    return Cint(ret)
end

# ==============================================================================
# Phase B — B8: ClpPackedMatrix (@ccallable C ABI -> pure 1-based Julia kernels)
# ==============================================================================

Base.@ccallable function jclp_pm_times(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    has_gaps::Cint,
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    nactive::Cint,
    nnz_cap::Cint,
    scalar::Cdouble,
    x::Ptr{Cdouble},
    y::Ptr{Cdouble},
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
)::Cvoid
    nr = Int(nrows)
    na = Int(nactive)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, na + 1)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    xv = PtrView(x, na)
    yv = PtrView(y, nr)
    gaps = has_gaps != Cint(0)
    if row_scale != C_NULL
        rs = PtrView(row_scale, nr)
        cs = PtrView(col_scale, na)
        if gaps
            clp_mat_times_scaled!(st, PtrView(lengths, na), true, idx, el, na, Float64(scalar), xv, yv, rs, cs)
        else
            clp_mat_times_scaled!(st, nothing, false, idx, el, na, Float64(scalar), xv, yv, rs, cs)
        end
    else
        if gaps
            clp_mat_times_unscaled!(st, PtrView(lengths, na), true, idx, el, na, Float64(scalar), xv, yv)
        else
            clp_mat_times_unscaled!(st, nothing, false, idx, el, na, Float64(scalar), xv, yv)
        end
    end
    return nothing
end

Base.@ccallable function jclp_pm_transpose_times(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    has_gaps::Cint,
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    nactive::Cint,
    nnz_cap::Cint,
    scalar::Cdouble,
    x::Ptr{Cdouble},
    y::Ptr{Cdouble},
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    spare::Ptr{Cdouble},
)::Cvoid
    nr = Int(nrows)
    na = Int(nactive)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, na + 1)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    xv = PtrView(x, nr)
    yv = PtrView(y, na)
    gaps = has_gaps != Cint(0)
    if row_scale != C_NULL
        rs = PtrView(row_scale, nr)
        cs = PtrView(col_scale, na)
        if spare != C_NULL
            sp = PtrView(spare, nr)
            if gaps
                clp_mat_transpose_times_scaled!(st, PtrView(lengths, na), true, idx, el, nr, na, Float64(scalar), xv, yv, rs, cs, sp)
            else
                clp_mat_transpose_times_scaled!(st, nothing, false, idx, el, nr, na, Float64(scalar), xv, yv, rs, cs, sp)
            end
        else
            if gaps
                clp_mat_transpose_times_scaled!(st, PtrView(lengths, na), true, idx, el, nr, na, Float64(scalar), xv, yv, rs, cs, nothing)
            else
                clp_mat_transpose_times_scaled!(st, nothing, false, idx, el, nr, na, Float64(scalar), xv, yv, rs, cs, nothing)
            end
        end
    else
        if gaps
            clp_mat_transpose_times_unscaled!(st, PtrView(lengths, na), true, idx, el, na, Float64(scalar), xv, yv)
        else
            clp_mat_transpose_times_unscaled!(st, nothing, false, idx, el, na, Float64(scalar), xv, yv)
        end
    end
    return nothing
end

Base.@ccallable function jclp_pm_transpose_times_subset(
    starts::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    nwhich::Cint,
    which::Ptr{Cint},
    x::Ptr{Cdouble},
    y::Ptr{Cdouble},
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    spare::Ptr{Cdouble},
)::Cvoid
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    nw = Int(nwhich)
    st = ShiftedPtrView(starts, nc + 1)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    wh = ShiftedPtrView(which, nw)
    xv = PtrView(x, nr)
    yv = PtrView(y, nc)
    if row_scale != C_NULL
        rs = PtrView(row_scale, nr)
        cs = PtrView(col_scale, nc)
        if spare != C_NULL
            clp_mat_transpose_times_subset!(st, idx, el, nr, nw, wh, xv, yv, rs, cs, PtrView(spare, nr))
        else
            clp_mat_transpose_times_subset!(st, idx, el, nr, nw, wh, xv, yv, rs, cs, nothing)
        end
    else
        clp_mat_transpose_times_subset!(st, idx, el, nr, nw, wh, xv, yv, nothing, nothing, nothing)
    end
    return nothing
end

Base.@ccallable function jclp_pm_guts_tt_unscaled(
    starts::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    nactive::Cint,
    nnz_cap::Cint,
    pi::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    status::Ptr{Cuchar},
    zero_tol::Cdouble,
)::Cint
    nr = Int(nrows)
    na = Int(nactive)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, na + 1)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    pv = PtrView(pi, nr)
    oi = ShiftedPtrView(out_idx, na)
    ov = PtrView(out_val, na)
    if status != C_NULL
        nnz = clp_mat_guts_tt_unscaled_status!(st, idx, el, na, pv, oi, ov, PtrView(status, na), Float64(zero_tol))
    else
        nnz = clp_mat_guts_tt_unscaled!(st, idx, el, na, pv, oi, ov, Float64(zero_tol))
    end
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_guts_tt_scaled(
    starts::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    nactive::Cint,
    nnz_cap::Cint,
    pi::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    status::Ptr{Cuchar},
    zero_tol::Cdouble,
)::Cint
    nr = Int(nrows)
    na = Int(nactive)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, na + 1)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    pv = PtrView(pi, nr)
    cs = PtrView(col_scale, na)
    oi = ShiftedPtrView(out_idx, na)
    ov = PtrView(out_val, na)
    if status != C_NULL
        nnz = clp_mat_guts_tt_scaled_status!(st, idx, el, na, pv, cs, oi, ov, PtrView(status, na), Float64(zero_tol))
    else
        nnz = clp_mat_guts_tt_scaled!(st, idx, el, na, pv, cs, oi, ov, Float64(zero_tol))
    end
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_guts_tt_unscaled_dual(
    starts::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    nactive::Cint,
    nnz_cap::Cint,
    pi::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    status::Ptr{Cuchar},
    spare_idx::Ptr{Cint},
    spare_val::Ptr{Cdouble},
    spare_cap::Cint,
    reduced_cost::Ptr{Cdouble},
    io_upper_theta::Ptr{Cdouble},
    acceptable_pivot::Cdouble,
    dual_tol::Cdouble,
    io_nremaining::Ptr{Cint},
    zero_tol::Cdouble,
)::Cint
    nr = Int(nrows)
    na = Int(nactive)
    nz = Int(nnz_cap)
    sc = Int(spare_cap)
    st = ShiftedPtrView(starts, na + 1)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    pv = PtrView(pi, nr)
    oi = ShiftedPtrView(out_idx, na)
    ov = PtrView(out_val, na)
    sv = PtrView(status, na)
    si = ShiftedPtrView(spare_idx, sc)
    spv = PtrView(spare_val, sc)
    rc = PtrView(reduced_cost, na)
    ut0 = unsafe_load(io_upper_theta)
    nrem0 = Int(unsafe_load(io_nremaining))
    nnz, nrem1, ut1 = clp_mat_guts_tt_unscaled_dual!(
        st, idx, el, na, pv, oi, ov, sv, si, spv, rc,
        ut0, Float64(acceptable_pivot), Float64(dual_tol), nrem0, Float64(zero_tol),
    )
    unsafe_store!(io_upper_theta, ut1)
    unsafe_store!(io_nremaining, Cint(nrem1))
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_tt_by_col_unpacked(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    has_gaps::Cint,
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    nactive::Cint,
    nnz_cap::Cint,
    scalar::Cdouble,
    pi::Ptr{Cdouble},
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    zero_tol::Cdouble,
)::Cint
    nr = Int(nrows)
    na = Int(nactive)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, na + 1)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    pv = PtrView(pi, nr)
    oi = ShiftedPtrView(out_idx, na)
    ov = PtrView(out_val, na)
    gaps = has_gaps != Cint(0)
    if row_scale != C_NULL
        rs = PtrView(row_scale, nr)
        cs = PtrView(col_scale, na)
        if gaps
            nnz = clp_mat_tt_by_col_unpacked!(st, PtrView(lengths, na), true, idx, el, na, Float64(scalar), pv, rs, cs, oi, ov, Float64(zero_tol))
        else
            nnz = clp_mat_tt_by_col_unpacked!(st, nothing, false, idx, el, na, Float64(scalar), pv, rs, cs, oi, ov, Float64(zero_tol))
        end
    else
        if gaps
            nnz = clp_mat_tt_by_col_unpacked!(st, PtrView(lengths, na), true, idx, el, na, Float64(scalar), pv, nothing, nothing, oi, ov, Float64(zero_tol))
        else
            nnz = clp_mat_tt_by_col_unpacked!(st, nothing, false, idx, el, na, Float64(scalar), pv, nothing, nothing, oi, ov, Float64(zero_tol))
        end
    end
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_tt_by_col_packed_gaps(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    nactive::Cint,
    nnz_cap::Cint,
    scalar::Cdouble,
    pi_old::Ptr{Cdouble},
    which_row::Ptr{Cint},
    n_in_row::Cint,
    pi_work::Ptr{Cdouble},
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    zero_tol::Cdouble,
)::Cint
    nr = Int(nrows)
    na = Int(nactive)
    nz = Int(nnz_cap)
    nin = Int(n_in_row)
    st = ShiftedPtrView(starts, na + 1)
    lens = PtrView(lengths, na)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    po = PtrView(pi_old, nin)
    wr = ShiftedPtrView(which_row, nin)
    pw = PtrView(pi_work, nr)
    oi = ShiftedPtrView(out_idx, na)
    ov = PtrView(out_val, na)
    if row_scale != C_NULL
        nnz = clp_mat_tt_by_col_packed_gaps!(st, lens, idx, el, na, Float64(scalar), po, wr, nin, pw, PtrView(row_scale, nr), PtrView(col_scale, na), oi, ov, Float64(zero_tol))
    else
        nnz = clp_mat_tt_by_col_packed_gaps!(st, lens, idx, el, na, Float64(scalar), po, wr, nin, pw, nothing, nothing, oi, ov, Float64(zero_tol))
    end
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_tt_by_row_eq1_packed(
    row_starts::Ptr{Cint},
    cols::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    irow_0based::Cint,
    pi0::Cdouble,
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    zero_tol::Cdouble,
    scalar::Cdouble,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    nnz = clp_mat_tt_by_row_eq1_packed!(
        ShiftedPtrView(row_starts, nr + 1),
        ShiftedPtrView(cols, nz),
        PtrView(elements, nz),
        Int(irow_0based) + 1,
        Float64(pi0),
        ShiftedPtrView(out_idx, nc),
        PtrView(out_val, nc),
        Float64(zero_tol),
        Float64(scalar),
    )
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_tt_by_row_eq2_packed(
    row_starts::Ptr{Cint},
    cols::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    irow0_0based::Cint,
    irow1_0based::Cint,
    pi0::Cdouble,
    pi1::Cdouble,
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    marked::Ptr{Cuchar},
    lookup::Ptr{Cint},
    zero_tol::Cdouble,
    scalar::Cdouble,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    nnz = clp_mat_tt_by_row_eq2_packed!(
        ShiftedPtrView(row_starts, nr + 1),
        ShiftedPtrView(cols, nz),
        PtrView(elements, nz),
        Int(irow0_0based) + 1,
        Int(irow1_0based) + 1,
        Float64(pi0),
        Float64(pi1),
        ShiftedPtrView(out_idx, nc),
        PtrView(out_val, nc),
        PtrView(marked, nc),
        PtrView(lookup, nc),
        Float64(zero_tol),
        Float64(scalar),
    )
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_tt_by_row_ge3_packed(
    row_starts::Ptr{Cint},
    cols::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    which_row::Ptr{Cint},
    pi::Ptr{Cdouble},
    n_in_row::Cint,
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    work::Ptr{Cdouble},
    zero_tol::Cdouble,
    scalar::Cdouble,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    nin = Int(n_in_row)
    nnz = clp_mat_tt_by_row_ge3_packed!(
        ShiftedPtrView(row_starts, nr + 1),
        ShiftedPtrView(cols, nz),
        PtrView(elements, nz),
        ShiftedPtrView(which_row, nin),
        PtrView(pi, nin),
        nin,
        ShiftedPtrView(out_idx, nc),
        PtrView(out_val, nc),
        PtrView(work, nc),
        Float64(zero_tol),
        Float64(scalar),
    )
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_tt_by_row_gek_packed(
    row_starts::Ptr{Cint},
    cols::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    which_row::Ptr{Cint},
    pi::Ptr{Cdouble},
    n_in_row::Cint,
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    zero_tol::Cdouble,
    scalar::Cdouble,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    nin = Int(n_in_row)
    nnz = clp_mat_tt_by_row_gek_packed!(
        ShiftedPtrView(row_starts, nr + 1),
        ShiftedPtrView(cols, nz),
        PtrView(elements, nz),
        ShiftedPtrView(which_row, nin),
        PtrView(pi, nin),
        nin,
        nc,
        ShiftedPtrView(out_idx, nc),
        PtrView(out_val, nc),
        Float64(zero_tol),
        Float64(scalar),
    )
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_tt_by_row_unpacked(
    row_starts::Ptr{Cint},
    cols::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    which_row::Ptr{Cint},
    pi::Ptr{Cdouble},
    n_in_row::Cint,
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    marked::Ptr{Cuchar},
    zero_tol::Cdouble,
    scalar::Cdouble,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    nin = Int(n_in_row)
    nnz = clp_mat_tt_by_row_unpacked!(
        ShiftedPtrView(row_starts, nr + 1),
        ShiftedPtrView(cols, nz),
        PtrView(elements, nz),
        ShiftedPtrView(which_row, nin),
        PtrView(pi, nr),
        nin,
        ShiftedPtrView(out_idx, nc),
        PtrView(out_val, nc),
        PtrView(marked, nc),
        Float64(zero_tol),
        Float64(scalar),
    )
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_subset_tt(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    has_gaps::Cint,
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    which::Ptr{Cint},
    n_to_do::Cint,
    pi::Ptr{Cdouble},
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    out_val::Ptr{Cdouble},
)::Cvoid
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    nt = Int(n_to_do)
    st = ShiftedPtrView(starts, nc + 1)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    wh = ShiftedPtrView(which, nt)
    pv = PtrView(pi, nr)
    ov = PtrView(out_val, nt)
    gaps = has_gaps != Cint(0)
    if row_scale != C_NULL
        rs = PtrView(row_scale, nr)
        cs = PtrView(col_scale, nc)
        if gaps
            clp_mat_subset_tt!(st, PtrView(lengths, nc), true, idx, el, wh, nt, pv, rs, cs, ov)
        else
            clp_mat_subset_tt!(st, nothing, false, idx, el, wh, nt, pv, rs, cs, ov)
        end
    else
        if gaps
            clp_mat_subset_tt!(st, PtrView(lengths, nc), true, idx, el, wh, nt, pv, nothing, nothing, ov)
        else
            clp_mat_subset_tt!(st, nothing, false, idx, el, wh, nt, pv, nothing, nothing, ov)
        end
    end
    return nothing
end

Base.@ccallable function jclp_pm_unpack(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    icol_0based::Cint,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    nnz_in::Cint,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, nc + 1)
    lens = PtrView(lengths, nc)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    oi = ShiftedPtrView(out_idx, nr)
    ov = PtrView(out_val, nr)
    if row_scale != C_NULL
        nnz = clp_mat_unpack!(st, lens, idx, el, Int(icol_0based) + 1, PtrView(row_scale, nr), PtrView(col_scale, nc), oi, ov, Int(nnz_in))
    else
        nnz = clp_mat_unpack!(st, lens, idx, el, Int(icol_0based) + 1, nothing, nothing, oi, ov, Int(nnz_in))
    end
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_unpack_packed(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    icol_0based::Cint,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, nc + 1)
    lens = PtrView(lengths, nc)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    oi = ShiftedPtrView(out_idx, nr)
    ov = PtrView(out_val, nr)
    if row_scale != C_NULL
        nnz = clp_mat_unpack_packed!(st, lens, idx, el, Int(icol_0based) + 1, PtrView(row_scale, nr), PtrView(col_scale, nc), oi, ov)
    else
        nnz = clp_mat_unpack_packed!(st, lens, idx, el, Int(icol_0based) + 1, nothing, nothing, oi, ov)
    end
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_add_indexed(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    icol_0based::Cint,
    multiplier::Cdouble,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_val::Ptr{Cdouble},
    nnz_in::Cint,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, nc + 1)
    lens = PtrView(lengths, nc)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    oi = ShiftedPtrView(out_idx, nr)
    ov = PtrView(out_val, nr)
    if row_scale != C_NULL
        nnz = clp_mat_add_indexed!(st, lens, idx, el, Int(icol_0based) + 1, Float64(multiplier), PtrView(row_scale, nr), PtrView(col_scale, nc), oi, ov, Int(nnz_in))
    else
        nnz = clp_mat_add_indexed!(st, lens, idx, el, Int(icol_0based) + 1, Float64(multiplier), nothing, nothing, oi, ov, Int(nnz_in))
    end
    return Cint(nnz)
end

Base.@ccallable function jclp_pm_add_dense(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    icol_0based::Cint,
    multiplier::Cdouble,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    array::Ptr{Cdouble},
)::Cvoid
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, nc + 1)
    lens = PtrView(lengths, nc)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    av = PtrView(array, nr)
    if row_scale != C_NULL
        clp_mat_add_dense!(st, lens, idx, el, Int(icol_0based) + 1, Float64(multiplier), PtrView(row_scale, nr), PtrView(col_scale, nc), av)
    else
        clp_mat_add_dense!(st, lens, idx, el, Int(icol_0based) + 1, Float64(multiplier), nothing, nothing, av)
    end
    return nothing
end

Base.@ccallable function jclp_pm_count_basis(
    lengths::Ptr{Cint},
    ncols::Cint,
    which_col::Ptr{Cint},
    n_col_basic::Cint,
)::Cint
    nc = Int(ncols)
    nb = Int(n_col_basic)
    return Cint(clp_mat_count_basis(PtrView(lengths, nc), ShiftedPtrView(which_col, nb), nb))
end

Base.@ccallable function jclp_pm_fill_basis(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    has_zeros::Cint,
    which_col::Ptr{Cint},
    n_col_basic::Cint,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    index_row_u::Ptr{Cint},
    start_u::Ptr{Cint},
    row_count::Ptr{Cint},
    col_count::Ptr{Cint},
    element_u::Ptr{Cdouble},
    u_cap::Cint,
)::Cvoid
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    nb = Int(n_col_basic)
    uc = Int(u_cap)
    st = ShiftedPtrView(starts, nc + 1)
    lens = PtrView(lengths, nc)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    wc = ShiftedPtrView(which_col, nb)
    iru = ShiftedPtrView(index_row_u, uc)
    su = ShiftedPtrView(start_u, nb + 1)
    rc = PtrView(row_count, nr)
    cc = PtrView(col_count, nb)
    eu = PtrView(element_u, uc)
    hz = has_zeros != Cint(0)
    if row_scale != C_NULL
        clp_mat_fill_basis!(st, lens, idx, el, hz, wc, nb, PtrView(row_scale, nr), PtrView(col_scale, nc), iru, su, rc, cc, eu)
    else
        clp_mat_fill_basis!(st, lens, idx, el, hz, wc, nb, nothing, nothing, iru, su, rc, cc, eu)
    end
    return nothing
end

Base.@ccallable function jclp_pm_range_of_elements(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    elements::Ptr{Cdouble},
    ncols::Cint,
    nnz_cap::Cint,
    out_sn::Ptr{Cdouble},
    out_ln::Ptr{Cdouble},
    out_sp::Ptr{Cdouble},
    out_lp::Ptr{Cdouble},
)::Cvoid
    nc = Int(ncols)
    nz = Int(nnz_cap)
    sn, ln, sp, lp = clp_mat_range_of_elements(
        ShiftedPtrView(starts, nc + 1),
        PtrView(lengths, nc),
        PtrView(elements, nz),
        nc,
    )
    unsafe_store!(out_sn, sn)
    unsafe_store!(out_ln, ln)
    unsafe_store!(out_sp, sp)
    unsafe_store!(out_lp, lp)
    return nothing
end

Base.@ccallable function jclp_pm_scale_row_copy(
    row_starts::Ptr{Cint},
    cols::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
)::Cvoid
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    clp_mat_scale_row_copy!(
        ShiftedPtrView(row_starts, nr + 1),
        ShiftedPtrView(cols, nz),
        PtrView(elements, nz),
        nr,
        PtrView(row_scale, nr),
        PtrView(col_scale, nc),
    )
    return nothing
end

Base.@ccallable function jclp_pm_really_scale(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    nnz_cap::Cint,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
)::Cvoid
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz_cap)
    clp_mat_really_scale!(
        ShiftedPtrView(starts, nc + 1),
        PtrView(lengths, nc),
        ShiftedPtrView(indices, nz),
        PtrView(elements, nz),
        nc,
        PtrView(row_scale, nr),
        PtrView(col_scale, nc),
    )
    return nothing
end

Base.@ccallable function jclp_pm_partial_pricing(
    starts::Ptr{Cint},
    lengths::Ptr{Cint},
    indices::Ptr{Cint},
    elements::Ptr{Cdouble},
    nrows::Cint,
    nactive::Cint,
    nnz_cap::Cint,
    start_frac::Cdouble,
    end_frac::Cdouble,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    duals::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    reduced_cost::Ptr{Cdouble},
    status::Ptr{Cuchar},
    seq_out_0based::Cint,
    io_best_seq_0based::Ptr{Cint},
    best_dj::Cdouble,
    io_n_wanted::Ptr{Cint},
    orig_wanted::Cint,
    min_scan::Cint,
    min_good::Cint,
    tol::Cdouble,
    out_saved_dj::Ptr{Cdouble},
)::Cvoid
    nr = Int(nrows)
    na = Int(nactive)
    nz = Int(nnz_cap)
    st = ShiftedPtrView(starts, na + 1)
    lens = PtrView(lengths, na)
    idx = ShiftedPtrView(indices, nz)
    el = PtrView(elements, nz)
    dv = PtrView(duals, nr)
    cv = PtrView(cost, na)
    rc = PtrView(reduced_cost, na)
    sv = PtrView(status, na)
    best0 = Int(unsafe_load(io_best_seq_0based))
    best1 = best0 >= 0 ? (best0 + 1) : 0
    seq_out1 = Int(seq_out_0based) >= 0 ? (Int(seq_out_0based) + 1) : 0
    nw0 = Int(unsafe_load(io_n_wanted))
    if row_scale != C_NULL
        b_out, nw_out, sdj = clp_mat_partial_pricing!(
            st, lens, idx, el, na, Float64(start_frac), Float64(end_frac),
            PtrView(row_scale, nr), PtrView(col_scale, na),
            dv, cv, rc, sv, seq_out1, best1, Float64(best_dj),
            nw0, Int(orig_wanted), Int(min_scan), Int(min_good), Float64(tol),
        )
    else
        b_out, nw_out, sdj = clp_mat_partial_pricing!(
            st, lens, idx, el, na, Float64(start_frac), Float64(end_frac),
            nothing, nothing,
            dv, cv, rc, sv, seq_out1, best1, Float64(best_dj),
            nw0, Int(orig_wanted), Int(min_scan), Int(min_good), Float64(tol),
        )
    end
    unsafe_store!(io_best_seq_0based, Cint(b_out > 0 ? (b_out - 1) : -1))
    unsafe_store!(io_n_wanted, Cint(nw_out))
    unsafe_store!(out_saved_dj, sdj)
    return nothing
end

# ==============================================================================
# Phase B — B9a: ClpDualRowDantzig & ClpDualRowSteepest (@ccallable C ABI)
# ==============================================================================

Base.@ccallable function jclp_drd_pivot_row(
    pivot_var::Ptr{Cint},
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    status::Ptr{Cuchar},
    nrows::Cint,
    ncols::Cint,
    primal_tol::Cdouble,
    largest_primal_err::Cdouble,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    r = clp_drd_pivot_row(
        ShiftedPtrView(pivot_var, nr),
        PtrView(solution, nt),
        PtrView(lower, nt),
        PtrView(upper, nt),
        PtrView(status, nt),
        nr, nc, Float64(primal_tol), Float64(largest_primal_err),
    )
    return Cint(r > 0 ? (r - 1) : -1)
end

Base.@ccallable function jclp_drd_extract_alpha(
    work::Ptr{Cdouble},
    which::Ptr{Cint},
    nnz::Cint,
    cap::Cint,
    packed::Cint,
    pivot_row_0based::Cint,
)::Cdouble
    c = Int(cap)
    return clp_drd_extract_alpha(
        PtrView(work, c),
        ShiftedPtrView(which, c),
        Int(nnz),
        packed != Cint(0),
        Int(pivot_row_0based) + 1,
    )
end

Base.@ccallable function jclp_drd_update_primal_solution(
    work::Ptr{Cdouble},
    which::Ptr{Cint},
    nnz::Cint,
    cap::Cint,
    packed::Cint,
    pivot_var::Ptr{Cint},
    solution::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    nrows::Cint,
    ntotal::Cint,
    primal_ratio::Cdouble,
)::Cdouble
    c = Int(cap)
    nr = Int(nrows)
    nt = Int(ntotal)
    return clp_drd_update_primal_solution!(
        PtrView(work, c),
        ShiftedPtrView(which, c),
        Int(nnz),
        packed != Cint(0),
        ShiftedPtrView(pivot_var, nr),
        PtrView(solution, nt),
        PtrView(cost, nt),
        Float64(primal_ratio),
    )
end

Base.@ccallable function jclp_drs_pivot_row(
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    io_nnz::Ptr{Cint},
    weights::Ptr{Cdouble},
    pivot_var::Ptr{Cint},
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    status::Ptr{Cuchar},
    nrows::Cint,
    ntotal::Cint,
    last_pivot_row_0based::Cint,
    mode::Cint,
    n_factor_elems::Cint,
    primal_tol::Cdouble,
    largest_primal_err::Cdouble,
    largest_dual_err::Cdouble,
    n_iters::Cint,
    last_bad_iter::Cint,
    rand_u::Cdouble,
    out_tol_changed::Ptr{Cint},
)::Cint
    nr = Int(nrows)
    nt = Int(ntotal)
    nnz0 = Int(unsafe_load(io_nnz))
    lpr = Int(last_pivot_row_0based)
    lpr1 = lpr >= 0 ? (lpr + 1) : 0
    chosen, nnz1, tc = clp_drs_pivot_row!(
        PtrView(infeas, nr),
        ShiftedPtrView(infeas_idx, nr),
        nnz0,
        PtrView(weights, nr),
        ShiftedPtrView(pivot_var, nr),
        PtrView(solution, nt),
        PtrView(lower, nt),
        PtrView(upper, nt),
        PtrView(status, nt),
        nr, lpr1, Int(mode), Int(n_factor_elems),
        Float64(primal_tol), Float64(largest_primal_err), Float64(largest_dual_err),
        Int(n_iters), Int(last_bad_iter), Float64(rand_u),
    )
    unsafe_store!(io_nnz, Cint(nnz1))
    unsafe_store!(out_tol_changed, Cint(tc ? 1 : 0))
    return Cint(chosen > 0 ? (chosen - 1) : -1)
end

Base.@ccallable function jclp_drs_compact_infeasible(
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    nnz::Cint,
    nrows::Cint,
)::Cint
    nr = Int(nrows)
    return Cint(clp_drs_compact_infeasible!(PtrView(infeas, nr), ShiftedPtrView(infeas_idx, nr), Int(nnz)))
end

Base.@ccallable function jclp_drs_prep_ftran(
    in_val::Ptr{Cdouble},
    in_idx::Ptr{Cint},
    nnz::Cint,
    in_cap::Cint,
    permute::Ptr{Cint},
    spare_val::Ptr{Cdouble},
    spare_idx::Ptr{Cint},
    spare_cap::Cint,
    nrows::Cint,
)::Cdouble
    ic = Int(in_cap)
    sc = Int(spare_cap)
    nr = Int(nrows)
    iv = PtrView(in_val, ic)
    ii = ShiftedPtrView(in_idx, ic)
    sv = PtrView(spare_val, sc)
    si = ShiftedPtrView(spare_idx, sc)
    if permute != C_NULL
        return clp_drs_prep_ftran!(iv, ii, Int(nnz), ShiftedPtrView(permute, nr), sv, si)
    else
        return clp_drs_prep_ftran!(iv, ii, Int(nnz), nothing, sv, si)
    end
end

Base.@ccallable function jclp_drs_apply_weight_update(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    nnz::Cint,
    upd_cap::Cint,
    spare_val::Ptr{Cdouble},
    spare_cap::Cint,
    pivot_col::Ptr{Cint},
    weights::Ptr{Cdouble},
    alt_val::Ptr{Cdouble},
    alt_idx::Ptr{Cint},
    alt_cap::Cint,
    nrows::Cint,
    pivot_row_0based::Cint,
    model_alpha::Cdouble,
    norm_in::Cdouble,
    is_network::Cint,
    out_n_save::Ptr{Cint},
)::Cdouble
    uc = Int(upd_cap)
    sc = Int(spare_cap)
    ac = Int(alt_cap)
    nr = Int(nrows)
    uv = PtrView(upd_val, uc)
    ui = ShiftedPtrView(upd_idx, uc)
    sv = PtrView(spare_val, sc)
    wv = PtrView(weights, nr)
    av = PtrView(alt_val, ac)
    ai = ShiftedPtrView(alt_idx, ac)
    pr1 = Int(pivot_row_0based) + 1
    if pivot_col != C_NULL
        alpha, nsave = clp_drs_apply_weight_update!(
            uv, ui, Int(nnz), sv, ShiftedPtrView(pivot_col, nr), wv, av, ai,
            pr1, Float64(model_alpha), Float64(norm_in), is_network != Cint(0),
        )
    else
        alpha, nsave = clp_drs_apply_weight_update!(
            uv, ui, Int(nnz), sv, nothing, wv, av, ai,
            pr1, Float64(model_alpha), Float64(norm_in), is_network != Cint(0),
        )
    end
    unsafe_store!(out_n_save, Cint(nsave))
    return alpha
end

Base.@ccallable function jclp_drs_update_primal_solution(
    work::Ptr{Cdouble},
    which::Ptr{Cint},
    nnz::Cint,
    work_cap::Cint,
    packed::Cint,
    pivot_var::Ptr{Cint},
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    io_infeas_nnz::Ptr{Cint},
    solution::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    nrows::Cint,
    ntotal::Cint,
    primal_ratio::Cdouble,
    primal_tol::Cdouble,
    pivot_row_0based::Cint,
)::Cdouble
    wc = Int(work_cap)
    nr = Int(nrows)
    nt = Int(ntotal)
    innz0 = Int(unsafe_load(io_infeas_nnz))
    chg, innz1 = clp_drs_update_primal_solution!(
        PtrView(work, wc),
        ShiftedPtrView(which, wc),
        Int(nnz),
        packed != Cint(0),
        ShiftedPtrView(pivot_var, nr),
        PtrView(infeas, nr),
        ShiftedPtrView(infeas_idx, nr),
        innz0,
        PtrView(solution, nt),
        PtrView(cost, nt),
        PtrView(lower, nt),
        PtrView(upper, nt),
        Float64(primal_ratio),
        Float64(primal_tol),
        Int(pivot_row_0based) + 1,
    )
    unsafe_store!(io_infeas_nnz, Cint(innz1))
    return chg
end

Base.@ccallable function jclp_drs_unroll_weights(
    weights::Ptr{Cdouble},
    saved_val::Ptr{Cdouble},
    saved_idx::Ptr{Cint},
    nnz::Cint,
    saved_cap::Cint,
    packed::Cint,
    nrows::Cint,
)::Cvoid
    sc = Int(saved_cap)
    nr = Int(nrows)
    clp_drs_unroll_weights!(
        PtrView(weights, nr),
        PtrView(saved_val, sc),
        ShiftedPtrView(saved_idx, sc),
        Int(nnz),
        packed != Cint(0),
    )
    return nothing
end

Base.@ccallable function jclp_drs_looks_optimal(
    pivot_var::Ptr{Cint},
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    nrows::Cint,
    ntotal::Cint,
    primal_tol::Cdouble,
    largest_primal_err::Cdouble,
)::Cint
    nr = Int(nrows)
    nt = Int(ntotal)
    ok = clp_drs_looks_optimal(
        ShiftedPtrView(pivot_var, nr),
        PtrView(solution, nt),
        PtrView(lower, nt),
        PtrView(upper, nt),
        nr, Float64(primal_tol), Float64(largest_primal_err),
    )
    return Cint(ok ? 1 : 0)
end

Base.@ccallable function jclp_drs_recompute_infeasibilities(
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    pivot_var::Ptr{Cint},
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    nrows::Cint,
    ntotal::Cint,
    primal_tol::Cdouble,
)::Cint
    nr = Int(nrows)
    nt = Int(ntotal)
    nnz = clp_drs_recompute_infeasibilities!(
        PtrView(infeas, nr),
        ShiftedPtrView(infeas_idx, nr),
        ShiftedPtrView(pivot_var, nr),
        PtrView(solution, nt),
        PtrView(lower, nt),
        PtrView(upper, nt),
        nr, Float64(primal_tol),
    )
    return Cint(nnz)
end

Base.@ccallable function jclp_drs_restore_weights(
    weights::Ptr{Cdouble},
    saved_val::Ptr{Cdouble},
    saved_idx::Ptr{Cint},
    back::Ptr{Cint},
    pivot_var::Ptr{Cint},
    nrows::Cint,
    ntotal::Cint,
    first_row_0based::Cint,
    last_row_0based_excl::Cint,
)::Cvoid
    nr = Int(nrows)
    nt = Int(ntotal)
    clp_drs_restore_weights!(
        PtrView(weights, nr),
        PtrView(saved_val, nr),
        ShiftedPtrView(saved_idx, nr),
        PtrView(back, nt),
        ShiftedPtrView(pivot_var, nr),
        nr, nt,
        Int(first_row_0based) + 1,
        Int(last_row_0based_excl),
    )
    return nothing
end

# ==============================================================================
# Phase B — B9b: ClpPrimalColumnDantzig & ClpPrimalColumnSteepest (@ccallable)
# ==============================================================================

Base.@ccallable function jclp_pcd_update_djs(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    rc::Ptr{Cdouble},
    rc_len::Cint,
)::Cvoid
    uc = Int(upd_cap)
    clp_pcd_update_djs!(PtrView(upd_val, uc), ShiftedPtrView(upd_idx, uc), Int(n_upd), PtrView(rc, Int(rc_len)))
    return nothing
end

Base.@ccallable function jclp_pcd_price(
    rc::Ptr{Cdouble},
    status::Ptr{Cuchar},
    nrows::Cint,
    ncols::Cint,
    dual_tol::Cdouble,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    r = clp_pcd_price(PtrView(rc, nt), PtrView(status, nt), nr, nc, Float64(dual_tol))
    return Cint(r > 0 ? (r - 1) : -1)
end

Base.@ccallable function jclp_pcs_price_scan(
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    nnz::Cint,
    weights::Ptr{Cdouble},
    status::Ptr{Cuchar},
    ntotal::Cint,
    mode::Cint,
    switch_type::Cint,
    n_wanted::Cint,
    tol_sq::Cdouble,
    rand_u::Cdouble,
    out_best_dj::Ptr{Cdouble},
)::Cint
    nt = Int(ntotal)
    best_seq, best_dj = clp_pcs_price_scan(
        PtrView(infeas, nt),
        ShiftedPtrView(infeas_idx, nt),
        Int(nnz),
        PtrView(weights, nt),
        PtrView(status, nt),
        Int(mode),
        Int(switch_type),
        Int(n_wanted),
        Float64(tol_sq),
        Float64(rand_u),
    )
    if out_best_dj != C_NULL
        unsafe_store!(out_best_dj, best_dj)
    end
    return Cint(best_seq > 0 ? (best_seq - 1) : -1)
end

Base.@ccallable function jclp_pcs_just_djs_section(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    rc::Ptr{Cdouble},
    rc_len::Cint,
    status::Ptr{Cuchar},
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    infeas_nnz::Cint,
    ntotal::Cint,
    add_seq::Cint,
    slack_mult::Cdouble,
    tol::Cdouble,
)::Cint
    uc = Int(upd_cap)
    nt = Int(ntotal)
    nnz1 = clp_pcs_just_djs_section!(
        PtrView(upd_val, uc),
        ShiftedPtrView(upd_idx, uc),
        Int(n_upd),
        PtrView(rc, Int(rc_len)),
        PtrView(status, nt),
        PtrView(infeas, nt),
        ShiftedPtrView(infeas_idx, nt),
        Int(infeas_nnz),
        Int(add_seq),
        Float64(slack_mult),
        Float64(tol),
    )
    return Cint(nnz1)
end

Base.@ccallable function jclp_pcs_devex_section(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    rc::Ptr{Cdouble},
    rc_len::Cint,
    status::Ptr{Cuchar},
    weights::Ptr{Cdouble},
    ref_bits::Ptr{Cuint},
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    infeas_nnz::Cint,
    ntotal::Cint,
    add_seq::Cint,
    scale_factor::Cdouble,
    devex::Cdouble,
    slack_mult::Cdouble,
    tol::Cdouble,
)::Cint
    uc = Int(upd_cap)
    nt = Int(ntotal)
    nwords = (nt + 31) >> 5
    nnz1 = clp_pcs_devex_section!(
        PtrView(upd_val, uc),
        ShiftedPtrView(upd_idx, uc),
        Int(n_upd),
        PtrView(rc, Int(rc_len)),
        PtrView(status, nt),
        PtrView(weights, nt),
        PtrView(ref_bits, nwords),
        PtrView(infeas, nt),
        ShiftedPtrView(infeas_idx, nt),
        Int(infeas_nnz),
        Int(add_seq),
        Float64(scale_factor),
        Float64(devex),
        Float64(slack_mult),
        Float64(tol),
    )
    return Cint(nnz1)
end

Base.@ccallable function jclp_pcs_steepest_rows(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    other::Ptr{Cdouble},
    rc::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    status::Ptr{Cuchar},
    weights::Ptr{Cdouble},
    ref_bits::Ptr{Cuint},
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    infeas_nnz::Cint,
    scale_factor::Cdouble,
    devex::Cdouble,
    mode::Cint,
    ref_in::Cdouble,
    tol::Cdouble,
)::Cint
    uc = Int(upd_cap)
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    uv = PtrView(upd_val, uc)
    ui = ShiftedPtrView(upd_idx, uc)
    ov = PtrView(other, nr)
    rcv = PtrView(rc, nr)
    sv = PtrView(status, nt)
    wv = PtrView(weights, nt)
    infv = PtrView(infeas, nt)
    infi = ShiftedPtrView(infeas_idx, nt)
    if ref_bits != C_NULL
        nwords = (nt + 31) >> 5
        nnz1 = clp_pcs_steepest_rows!(uv, ui, Int(n_upd), ov, rcv, sv, wv, PtrView(ref_bits, nwords), infv, infi, Int(infeas_nnz), nc, Float64(scale_factor), Float64(devex), Int(mode), Float64(ref_in), Float64(tol))
    else
        nnz1 = clp_pcs_steepest_rows!(uv, ui, Int(n_upd), ov, rcv, sv, wv, nothing, infv, infi, Int(infeas_nnz), nc, Float64(scale_factor), Float64(devex), Int(mode), Float64(ref_in), Float64(tol))
    end
    return Cint(nnz1)
end

Base.@ccallable function jclp_pcs_steepest_cols_rc1(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    infeas_nnz::Cint,
    ntotal::Cint,
)::Cint
    uc = Int(upd_cap)
    nt = Int(ntotal)
    return Cint(clp_pcs_steepest_cols_rc1!(
        PtrView(upd_val, uc), ShiftedPtrView(upd_idx, uc), Int(n_upd),
        PtrView(infeas, nt), ShiftedPtrView(infeas_idx, nt), Int(infeas_nnz),
    ))
end

Base.@ccallable function jclp_pcs_steepest_cols_rc0(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    rc::Ptr{Cdouble},
    ncols::Cint,
    status::Ptr{Cuchar},
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    infeas_nnz::Cint,
    ntotal::Cint,
    tol::Cdouble,
)::Cint
    uc = Int(upd_cap)
    nt = Int(ntotal)
    return Cint(clp_pcs_steepest_cols_rc0!(
        PtrView(upd_val, uc), ShiftedPtrView(upd_idx, uc), Int(n_upd),
        PtrView(rc, Int(ncols)), PtrView(status, nt),
        PtrView(infeas, nt), ShiftedPtrView(infeas_idx, nt), Int(infeas_nnz),
        Float64(tol),
    ))
end

Base.@ccallable function jclp_pcs_just_devex_section(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    weights::Ptr{Cdouble},
    ref_bits::Ptr{Cuint},
    ntotal::Cint,
    add_seq::Cint,
    negate_piv::Cint,
    devex::Cdouble,
)::Cvoid
    uc = Int(upd_cap)
    nt = Int(ntotal)
    nwords = (nt + 31) >> 5
    clp_pcs_just_devex_section!(
        PtrView(upd_val, uc), ShiftedPtrView(upd_idx, uc), Int(n_upd),
        PtrView(weights, nt), PtrView(ref_bits, nwords),
        Int(add_seq), negate_piv != Cint(0), Float64(devex),
    )
    return nothing
end

Base.@ccallable function jclp_pcs_just_steepest_rows(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    other::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    weights::Ptr{Cdouble},
    ref_bits::Ptr{Cuint},
    devex::Cdouble,
    mode::Cint,
    ref_in::Cdouble,
)::Cvoid
    uc = Int(upd_cap)
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    uv = PtrView(upd_val, uc)
    ui = ShiftedPtrView(upd_idx, uc)
    ov = PtrView(other, nr)
    wv = PtrView(weights, nt)
    if ref_bits != C_NULL
        nwords = (nt + 31) >> 5
        clp_pcs_just_steepest_rows!(uv, ui, Int(n_upd), ov, wv, PtrView(ref_bits, nwords), nc, Float64(devex), Int(mode), Float64(ref_in))
    else
        clp_pcs_just_steepest_rows!(uv, ui, Int(n_upd), ov, wv, nothing, nc, Float64(devex), Int(mode), Float64(ref_in))
    end
    return nothing
end

Base.@ccallable function jclp_pcs_just_steepest_cols(
    upd_val::Ptr{Cdouble},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    upd_cap::Cint,
    upd2_val::Ptr{Cdouble},
    ntotal::Cint,
    weights::Ptr{Cdouble},
    ref_bits::Ptr{Cuint},
    devex::Cdouble,
    mode::Cint,
    ref_in::Cdouble,
)::Cvoid
    uc = Int(upd_cap)
    nt = Int(ntotal)
    uv = PtrView(upd_val, uc)
    ui = ShiftedPtrView(upd_idx, uc)
    u2v = PtrView(upd2_val, uc)
    wv = PtrView(weights, nt)
    if ref_bits != C_NULL
        nwords = (nt + 31) >> 5
        clp_pcs_just_steepest_cols!(uv, ui, Int(n_upd), u2v, wv, PtrView(ref_bits, nwords), Float64(devex), Int(mode), Float64(ref_in))
    else
        clp_pcs_just_steepest_cols!(uv, ui, Int(n_upd), u2v, wv, nothing, Float64(devex), Int(mode), Float64(ref_in))
    end
    return nothing
end

Base.@ccallable function jclp_pcs_redo_infeasibilities(
    infeas::Ptr{Cdouble},
    infeas_idx::Ptr{Cint},
    rc::Ptr{Cdouble},
    status::Ptr{Cuchar},
    ntotal::Cint,
    tol::Cdouble,
)::Cint
    nt = Int(ntotal)
    return Cint(clp_pcs_redo_infeasibilities!(
        PtrView(infeas, nt), ShiftedPtrView(infeas_idx, nt),
        PtrView(rc, nt), PtrView(status, nt), nt, Float64(tol),
    ))
end

Base.@ccallable function jclp_pcs_unroll_weights(
    weights::Ptr{Cdouble},
    saved_val::Ptr{Cdouble},
    saved_idx::Ptr{Cint},
    nnz::Cint,
    ntotal::Cint,
    saved_cap::Cint,
)::Cvoid
    nt = Int(ntotal)
    sc = Int(saved_cap)
    clp_pcs_unroll_weights!(PtrView(weights, nt), PtrView(saved_val, sc), ShiftedPtrView(saved_idx, sc), Int(nnz))
    return nothing
end

Base.@ccallable function jclp_pcs_looks_optimal_simple(
    rc::Ptr{Cdouble},
    status::Ptr{Cuchar},
    ntotal::Cint,
    tol::Cdouble,
)::Cint
    nt = Int(ntotal)
    ok = clp_pcs_looks_optimal_simple(PtrView(rc, nt), PtrView(status, nt), nt, Float64(tol))
    return Cint(ok ? 1 : 0)
end

# ==============================================================================
# Phase B — B9c: ClpNonLinearCost (@ccallable C ABI -> pure 1-based Julia)
# ==============================================================================

Base.@ccallable function jclp_nlc_feasible_bounds(
    nlc_status::Ptr{Cuchar},
    bound::Ptr{Cdouble},
    cost2::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    ntotal::Cint,
)::Cvoid
    nt = Int(ntotal)
    clp_nlc_feasible_bounds!(
        PtrView(nlc_status, nt), PtrView(bound, nt), PtrView(cost2, nt),
        PtrView(lower, nt), PtrView(upper, nt), PtrView(cost, nt), nt,
    )
    return nothing
end

Base.@ccallable function jclp_nlc_go_back_all_m2(
    nlc_status::Ptr{Cuchar},
    pivot_var::Ptr{Cint},
    upd_idx::Ptr{Cint},
    n_upd::Cint,
    nrows::Cint,
    ntotal::Cint,
)::Cvoid
    nr = Int(nrows)
    nt = Int(ntotal)
    nu = Int(n_upd)
    clp_nlc_go_back_all_m2!(
        PtrView(nlc_status, nt), ShiftedPtrView(pivot_var, nr), ShiftedPtrView(upd_idx, nu), nu,
    )
    return nothing
end

Base.@ccallable function jclp_nlc_check_infeas_subset_m2(
    nlc_status::Ptr{Cuchar},
    bound::Ptr{Cdouble},
    cost2::Ptr{Cdouble},
    pivot_var::Ptr{Cint},
    row_idx::Ptr{Cint},
    n_in_arr::Cint,
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    nrows::Cint,
    ntotal::Cint,
    primal_tol::Cdouble,
    infeas_weight::Cdouble,
    n_infeas::Cint,
)::Cint
    nr = Int(nrows)
    nt = Int(ntotal)
    na = Int(n_in_arr)
    return Cint(clp_nlc_check_infeas_subset_m2!(
        PtrView(nlc_status, nt), PtrView(bound, nt), PtrView(cost2, nt),
        ShiftedPtrView(pivot_var, nr), ShiftedPtrView(row_idx, na), na,
        PtrView(solution, nt), PtrView(lower, nt), PtrView(upper, nt), PtrView(cost, nt),
        Float64(primal_tol), Float64(infeas_weight), Int(n_infeas),
    ))
end

Base.@ccallable function jclp_nlc_check_changed_m2(
    nlc_status::Ptr{Cuchar},
    bound::Ptr{Cdouble},
    cost2::Ptr{Cdouble},
    pivot_var::Ptr{Cint},
    upd_idx::Ptr{Cint},
    upd_work::Ptr{Cdouble},
    upd_cap::Cint,
    n_in_arr::Cint,
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    nrows::Cint,
    ntotal::Cint,
    primal_tol::Cdouble,
    infeas_weight::Cdouble,
    io_n_infeas::Ptr{Cint},
)::Cint
    nr = Int(nrows)
    nt = Int(ntotal)
    uc = Int(upd_cap)
    ninf0 = Int(unsafe_load(io_n_infeas))
    nchg, ninf1 = clp_nlc_check_changed_m2!(
        PtrView(nlc_status, nt), PtrView(bound, nt), PtrView(cost2, nt),
        ShiftedPtrView(pivot_var, nr), ShiftedPtrView(upd_idx, uc), PtrView(upd_work, uc),
        Int(n_in_arr),
        PtrView(solution, nt), PtrView(lower, nt), PtrView(upper, nt), PtrView(cost, nt),
        Float64(primal_tol), Float64(infeas_weight), ninf0,
    )
    unsafe_store!(io_n_infeas, Cint(ninf1))
    return Cint(nchg)
end

Base.@ccallable function jclp_nlc_set_one_m2(
    nlc_status::Ptr{Cuchar},
    bound::Ptr{Cdouble},
    cost2::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    model_status::Ptr{Cuchar},
    ntotal::Cint,
    iseq_0based::Cint,
    val::Cdouble,
    primal_tol::Cdouble,
    infeas_weight::Cdouble,
    io_n_infeas::Ptr{Cint},
)::Cdouble
    nt = Int(ntotal)
    ninf0 = Int(unsafe_load(io_n_infeas))
    diff, ninf1 = clp_nlc_set_one_m2!(
        PtrView(nlc_status, nt), PtrView(bound, nt), PtrView(cost2, nt),
        PtrView(lower, nt), PtrView(upper, nt), PtrView(cost, nt), PtrView(model_status, nt),
        Int(iseq_0based) + 1, Float64(val), Float64(primal_tol), Float64(infeas_weight), ninf0,
    )
    unsafe_store!(io_n_infeas, Cint(ninf1))
    return diff
end

Base.@ccallable function jclp_nlc_nearest_m2(
    nlc_status::Ptr{Cuchar},
    bound::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    ntotal::Cint,
    iseq_0based::Cint,
    sol_val::Cdouble,
)::Cdouble
    nt = Int(ntotal)
    return clp_nlc_nearest_m2(
        PtrView(nlc_status, nt), PtrView(bound, nt), PtrView(lower, nt), PtrView(upper, nt),
        Int(iseq_0based) + 1, Float64(sol_val),
    )
end

Base.@ccallable function jclp_nlc_refresh_one(
    nlc_status::Ptr{Cuchar},
    bound::Ptr{Cdouble},
    cost2::Ptr{Cdouble},
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    ntotal::Cint,
    iseq_0based::Cint,
    primal_tol::Cdouble,
    infeas_cost::Cdouble,
)::Cvoid
    nt = Int(ntotal)
    clp_nlc_refresh_one!(
        PtrView(nlc_status, nt), PtrView(bound, nt), PtrView(cost2, nt),
        PtrView(solution, nt), PtrView(lower, nt), PtrView(upper, nt), PtrView(cost, nt),
        Int(iseq_0based) + 1, Float64(primal_tol), Float64(infeas_cost),
    )
    return nothing
end

# ==============================================================================
# B10: ClpFactorization (direct pure-Julia SparseLU integration)
# ==============================================================================

Base.@ccallable function jclp_clpf_create()::Cint
    return Cint(_cf_handle_create())
end

Base.@ccallable function jclp_clpf_copy(src_id::Cint)::Cint
    return Cint(_cf_handle_copy(Int32(src_id)))
end

Base.@ccallable function jclp_clpf_assign(dst_id::Cint, src_id::Cint)::Cvoid
    _cf_handle_assign!(Int32(dst_id), Int32(src_id))
    return nothing
end

Base.@ccallable function jclp_clpf_destroy(id::Cint)::Cvoid
    _cf_handle_destroy!(Int32(id))
    return nothing
end

# --- Scalar getters / setters (O(1) concrete pool lookup) ---

Base.@ccallable function jclp_clpf_get_int(id::Cint, field::Cint)::Cint
    cf = _cf_get(Int32(id))
    lu = cf.lu
    f = Int(field)
    if f == 0;  return Cint(lu.total_elements)
    elseif f == 1;  return Cint(lu.max_pivots)
    elseif f == 2;  return Cint(lu.npivots)
    elseif f == 3;  return Cint(lu.sparse_threshold)
    elseif f == 4;  return Cint(lu.status)
    elseif f == 5;  return Cint(lu.ndense)
    elseif f == 6;  return Cint(lu.len_u)
    elseif f == 7;  return Cint(lu.len_l)
    elseif f == 8;  return Cint(lu.len_r)
    elseif f == 9;  return Cint(cf.message_level)
    elseif f == 10; return Cint(lu.nrows)
    elseif f == 11; return Cint(lu.dense_threshold)
    elseif f == 12; return Cint(cf.persistence_flag)
    elseif f == 13; return Cint(lu.max_rows_extra)
    elseif f == 14; return Cint(lu.max_cols_extra)
    elseif f == 15; return Cint(lu.bias_lu)
    elseif f == 16; return Cint(lu.do_forrest_tomlin ? 1 : 0)
    else;           return Cint(0)
    end
end

Base.@ccallable function jclp_clpf_set_int(id::Cint, field::Cint, val::Cint)::Cvoid
    cf = _cf_get(Int32(id))
    lu = cf.lu
    f = Int(field)
    v = Int(val)
    if f == 1;      lu.max_pivots = v
    elseif f == 2;  lu.npivots = v
    elseif f == 3;  lu.sparse_threshold = v
    elseif f == 4;  lu.status = v
    elseif f == 9;  cf.message_level = v
    elseif f == 10; lu.nrows = v
    elseif f == 11; lu.dense_threshold = v
    elseif f == 12; cf.persistence_flag = v
    elseif f == 15; lu.bias_lu = v
    elseif f == 16; lu.do_forrest_tomlin = (v != 0)
    elseif f == 17; cf.do_statistics = (v != 0)
    end
    return nothing
end

Base.@ccallable function jclp_clpf_get_double(id::Cint, field::Cint)::Cdouble
    cf = _cf_get(Int32(id))
    lu = cf.lu
    f = Int(field)
    if f == 0;      return Cdouble(lu.area_factor)
    elseif f == 1;  return Cdouble(lu.zero_tol)
    elseif f == 2;  return Cdouble(lu.pivot_tol)
    elseif f == 3;  return Cdouble(clp_fact_adjusted_area_factor(cf))
    elseif f == 4;  return Cdouble(lu.relax_check)
    else;           return Cdouble(0.0)
    end
end

Base.@ccallable function jclp_clpf_set_double(id::Cint, field::Cint, val::Cdouble)::Cvoid
    cf = _cf_get(Int32(id))
    lu = cf.lu
    f = Int(field)
    v = Float64(val)
    if f == 0;      lu.area_factor = v
    elseif f == 1;  lu.zero_tol = v
    elseif f == 2;  lu.pivot_tol = v
    elseif f == 4;  lu.relax_check = v
    end
    return nothing
end

Base.@ccallable function jclp_clpf_clear_arrays(id::Cint)::Cvoid
    clp_fact_clear_arrays!(_cf_get(Int32(id)))
    return nothing
end

Base.@ccallable function jclp_clpf_almost_destructor(id::Cint)::Cvoid
    clp_fact_almost_destructor!(_cf_get(Int32(id)))
    return nothing
end

Base.@ccallable function jclp_clpf_set_default_values(id::Cint)::Cvoid
    clp_fact_set_default_values!(_cf_get(Int32(id)))
    return nothing
end

Base.@ccallable function jclp_clpf_adjust_area_factor(id::Cint)::Cvoid
    clp_fact_adjust_area_factor!(_cf_get(Int32(id)))
    return nothing
end

Base.@ccallable function jclp_clpf_time_to_refactorize(id::Cint)::Cint
    return Cint(clp_fact_time_to_refactorize!(_cf_get(Int32(id))) ? 1 : 0)
end

Base.@ccallable function jclp_clpf_safer_tolerances(
    id::Cint,
    zero_val::Cdouble,
    pivot_val::Cdouble,
)::Cvoid
    clp_fact_safer_tolerances!(_cf_get(Int32(id)), Float64(zero_val), Float64(pivot_val))
    return nothing
end

Base.@ccallable function jclp_clpf_go_sparse(id::Cint)::Cvoid
    clp_fact_go_sparse!(_cf_get(Int32(id)))
    return nothing
end

Base.@ccallable function jclp_clpf_clean_up(id::Cint)::Cvoid
    clp_fact_clean_up!(_cf_get(Int32(id)))
    return nothing
end

Base.@ccallable function jclp_clpf_get_weights(
    id::Cint,
    weights_ptr::Ptr{Cint},
    nrows::Cint,
)::Cvoid
    clp_fact_get_weights!(_cf_get(Int32(id)), PtrView(weights_ptr, Int(nrows)))
    return nothing
end

Base.@ccallable function jclp_clpf_empty_rows(
    id::Cint,
    n_to_empty::Cint,
    which_0based::Ptr{Cint},
)::Cvoid
    n = Int(n_to_empty)
    if n > 0
        clp_fact_empty_rows!(_cf_get(Int32(id)), n, ShiftedPtrView(which_0based, n))
    end
    return nothing
end

# --- Direct zero-copy factorize from ClpPackedMatrix ---

@inline function _copy_lu_permutes_0based!(
    lu::SparseLU,
    out_permute0::Ptr{Cint},
    out_pivot_col0::Ptr{Cint},
    st::Int,
)::Nothing
    nr = lu.nrows
    nc = lu.ncols
    if st == 0
        if out_permute0 != C_NULL
            p_out = PtrView(out_permute0, nr)
            @inbounds for i in 1:nr
                p_out[i] = lu.permute[i] - Int32(1)
            end
        end
        if out_pivot_col0 != C_NULL
            pc_out = PtrView(out_pivot_col0, nr)
            @inbounds for i in 1:nr
                pc_out[i] = lu.pivot_col[i] - Int32(1)
            end
        end
    elseif st == -1
        if out_pivot_col0 != C_NULL
            pc_out = PtrView(out_pivot_col0, nc)
            @inbounds for i in 1:nc
                v = lu.pivot_col[i]
                pc_out[i] = v > LU_NO_LINK ? (v - Int32(1)) : Int32(-1)
            end
        end
    end
    return nothing
end

Base.@ccallable function jclp_clpf_factorize_packed(
    id::Cint,
    nrows::Cint,
    ncols_model::Cint,
    n_row_basic::Cint,
    n_basic::Cint,
    pivot_temp_0::Ptr{Cint},
    mat_starts_0::Ptr{Cint},
    mat_lengths::Ptr{Cint},
    mat_indices_0::Ptr{Cint},
    mat_elements::Ptr{Cdouble},
    mat_nnz_cap::Cint,
    has_zeros::Cint,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    number_iterations::Cint,
    n_times_round::Cint,
    pivot_variable_0::Ptr{Cint},
    use_number_rows::Cint,
    out_permute0::Ptr{Cint},
    out_pivot_col0::Ptr{Cint},
)::Cint
    cf = _cf_get(Int32(id))
    nr = Int(nrows)
    nc_m = Int(ncols_model)
    nrb = Int(n_row_basic)
    nb = Int(n_basic)
    nnz_c = Int(mat_nnz_cap)
    hz = has_zeros != Cint(0)
    nit = Int(number_iterations)
    ntr = Int(n_times_round)
    unr = Int(use_number_rows)

    pt = ShiftedPtrView(pivot_temp_0, max(nb, unr))
    ms = ShiftedPtrView(mat_starts_0, nc_m + 1)
    ml = PtrView(mat_lengths, nc_m)
    mi = ShiftedPtrView(mat_indices_0, nnz_c)
    me = PtrView(mat_elements, nnz_c)
    pv = ShiftedPtrView(pivot_variable_0, max(nr, unr))

    if row_scale != C_NULL && col_scale != C_NULL
        st = clp_fact_factorize_packed!(
            cf, nr, nc_m, nrb, nb, pt, ms, ml, mi, me, hz,
            PtrView(row_scale, nr), PtrView(col_scale, nc_m),
            nit, ntr, pv, unr,
        )
    else
        st = clp_fact_factorize_packed!(
            cf, nr, nc_m, nrb, nb, pt, ms, ml, mi, me, hz,
            nothing, nothing,
            nit, ntr, pv, unr,
        )
    end
    _copy_lu_permutes_0based!(cf.lu, out_permute0, out_pivot_col0, st)
    return Cint(st)
end

Base.@ccallable function jclp_clpf_factorize_csc(
    id::Cint,
    nrows::Cint,
    ncols::Cint,
    nnz::Cint,
    area_l::Cint,
    area_u::Cint,
    start_col_0::Ptr{Cint},
    len_col::Ptr{Cint},
    idx_row_0::Ptr{Cint},
    el_u::Ptr{Cdouble},
    len_row::Ptr{Cint},
    state::Cint,
    out_pivot_col0::Ptr{Cint},
)::Cint
    cf = _cf_get(Int32(id))
    nr = Int(nrows)
    nc = Int(ncols)
    nz = Int(nnz)
    st = clp_fact_factorize_csc!(
        cf, nr, nc, nz, Int(area_l), Int(area_u),
        ShiftedPtrView(start_col_0, nc + 1),
        PtrView(len_col, nc),
        ShiftedPtrView(idx_row_0, nz),
        PtrView(el_u, nz),
        PtrView(len_row, nr),
        Int(state),
    )
    if st == -1 && out_pivot_col0 != C_NULL
        _copy_lu_permutes_0based!(cf.lu, C_NULL, out_pivot_col0, st)
    end
    return Cint(st)
end

Base.@ccallable function jclp_clpf_reorder_and_finalize(
    id::Cint,
    ncols_model::Cint,
    n_row_basic::Cint,
    use_number_rows::Cint,
    pivot_temp_0::Ptr{Cint},
    pivot_variable_0::Ptr{Cint},
    do_check_sparse::Cint,
    out_permute0::Ptr{Cint},
    out_pivot_col0::Ptr{Cint},
)::Cvoid
    cf = _cf_get(Int32(id))
    unr = Int(use_number_rows)
    clp_fact_reorder_and_finalize!(
        cf, Int(ncols_model), Int(n_row_basic), unr,
        ShiftedPtrView(pivot_temp_0, unr),
        ShiftedPtrView(pivot_variable_0, max(cf.lu.nrows, unr)),
        do_check_sparse != Cint(0),
    )
    _copy_lu_permutes_0based!(cf.lu, out_permute0, out_pivot_col0, 0)
    return nothing
end

Base.@ccallable function jclp_clpf_export_for_network(
    id::Cint,
    out_piv_reg::Ptr{Cdouble},
    out_perm_back::Ptr{Cint},
    out_start_col_u::Ptr{Cint},
    out_len_col_u::Ptr{Cint},
    out_idx_row_u::Ptr{Cint},
    out_el_u::Ptr{Cdouble},
)::Cvoid
    lu = _cf_get(Int32(id)).lu
    nr = lu.nrows
    nz = lu.len_u
    pr = PtrView(out_piv_reg, nr)
    pb = PtrView(out_perm_back, nr)
    sc = PtrView(out_start_col_u, nr + 1)
    lc = PtrView(out_len_col_u, nr)
    ir = PtrView(out_idx_row_u, nz)
    eu = PtrView(out_el_u, nz)
    @inbounds for i in 1:nr
        pr[i] = lu.pivot_region[i]
        pb[i] = lu.permute_back[i] - Int32(1)
        sc[i] = lu.start_col_u[i] - Int32(1)
        lc[i] = lu.len_col_u[i]
    end
    @inbounds for k in 1:nz
        ir[k] = lu.index_row_u[k] - Int32(1)
        eu[k] = lu.element_u[k]
    end
    return nothing
end

# --- Hot simplex operations ---

Base.@ccallable function jclp_clpf_replace_column(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_cap::Cint,
    tab_el::Ptr{Cdouble},
    tab_idx::Ptr{Cint},
    tab_nnz::Cint,
    tab_cap::Cint,
    tab_packed::Cint,
    pivot_row_0::Cint,
    pivot_check::Cdouble,
    check_before::Cint,
    n_new::Cint,
    n_old::Cint,
    io_permute0::Ptr{Cint},
    io_pivot_col0::Ptr{Cint},
)::Cint
    cf = _cf_get(Int32(id))
    lu = cf.lu
    pr = Int(pivot_row_0) + 1
    if lu.do_forrest_tomlin
        ret = clp_fact_replace_column!(
            cf,
            PtrView(w_el, Int(w_cap)),
            PtrView(w_idx, Int(w_cap)),
            PtrView(w_el, 0),
            PtrView(w_idx, 0),
            0, false,
            pr, Float64(pivot_check), check_before != Cint(0),
            Int(n_new), Int(n_old),
        )
        if ret == 0
            # Keep C++ 0-based permute/pivotColumn mirrors synchronized in O(1)
            new_row = lu.nrows_extra
            @inbounds begin
                if io_permute0 != C_NULL
                    p0 = PtrView(io_permute0, new_row + 1)
                    p0[new_row] = lu.permute[new_row] - Int32(1)
                    p0[new_row + 1] = Int32(-1)
                end
                if io_pivot_col0 != C_NULL
                    pc0 = PtrView(io_pivot_col0, max(pr, new_row))
                    pc0[pr] = lu.pivot_col[pr] - Int32(1)
                end
            end
        end
    else
        tn = Int(tab_nnz)
        ti = PtrView(tab_idx, Int(tab_cap))
        shift_to_1based!(ti, tn)
        ret = clp_fact_replace_column!(
            cf,
            PtrView(w_el, Int(w_cap)),
            PtrView(w_idx, Int(w_cap)),
            PtrView(tab_el, Int(tab_cap)),
            ti, tn, tab_packed != Cint(0),
            pr, Float64(pivot_check), check_before != Cint(0),
            Int(n_new), Int(n_old),
        )
        shift_to_0based!(ti, tn)
        if ret == 0 && io_pivot_col0 != C_NULL
            p_slot = lu.nrows + lu.npivots
            pc0 = PtrView(io_pivot_col0, p_slot)
            @inbounds pc0[p_slot] = lu.pivot_col[p_slot] - Int32(1)
        end
    end
    return Cint(ret)
end

Base.@ccallable function jclp_clpf_update_column_ft(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_cap::Cint,
    r_el::Ptr{Cdouble},
    r_idx::Ptr{Cint},
    r_nnz::Cint,
    r_cap::Cint,
    r_packed::Cint,
)::Cint
    cf = _cf_get(Int32(id))
    wv = PtrView(w_el, Int(w_cap))
    wi = PtrView(w_idx, Int(w_cap))
    rv = PtrView(r_el, Int(r_cap))
    ri = PtrView(r_idx, Int(r_cap))
    rn = Int(r_nnz)
    shift_to_1based!(ri, rn)
    res = clp_fact_update_column_ft!(cf, wv, wi, rv, ri, rn, r_packed != Cint(0))
    shift_to_0based!(ri, abs(res))
    return Cint(res)
end

Base.@ccallable function jclp_clpf_update_column(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_nnz::Cint,
    w_cap::Cint,
    r_el::Ptr{Cdouble},
    r_idx::Ptr{Cint},
    r_nnz::Cint,
    r_cap::Cint,
    r_packed::Cint,
    no_permute::Cint,
)::Cint
    cf = _cf_get(Int32(id))
    wv = PtrView(w_el, Int(w_cap))
    wi = PtrView(w_idx, Int(w_cap))
    rv = PtrView(r_el, Int(r_cap))
    ri = PtrView(r_idx, Int(r_cap))
    np = no_permute != Cint(0)
    wn = Int(w_nnz)
    rn = Int(r_nnz)
    if np
        shift_to_1based!(wi, wn)
    else
        shift_to_1based!(ri, rn)
    end
    nnz_out = clp_fact_update_column!(cf, wv, wi, wn, rv, ri, rn, r_packed != Cint(0), np)
    if np
        shift_to_0based!(wi, nnz_out)
    else
        shift_to_0based!(ri, nnz_out)
    end
    return Cint(nnz_out)
end

Base.@ccallable function jclp_clpf_update_column_debug(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_nnz::Cint,
    w_cap::Cint,
    r_el::Ptr{Cdouble},
    r_idx::Ptr{Cint},
    r_nnz::Cint,
    r_cap::Cint,
    r_packed::Cint,
    no_permute::Cint,
)::Cint
    cf = _cf_get(Int32(id))
    wv = PtrView(w_el, Int(w_cap))
    wi = PtrView(w_idx, Int(w_cap))
    rv = PtrView(r_el, Int(r_cap))
    ri = PtrView(r_idx, Int(r_cap))
    np = no_permute != Cint(0)
    wn = Int(w_nnz)
    rn = Int(r_nnz)
    if np
        shift_to_1based!(wi, wn)
    else
        shift_to_1based!(ri, rn)
    end
    nnz_out = clp_fact_update_column_for_debug!(cf, wv, wi, wn, rv, ri, rn, r_packed != Cint(0), np)
    if np
        shift_to_0based!(wi, nnz_out)
    else
        shift_to_0based!(ri, nnz_out)
    end
    return Cint(nnz_out)
end

Base.@ccallable function jclp_clpf_update_two_columns_ft(
    id::Cint,
    r1_el::Ptr{Cdouble},
    r1_idx::Ptr{Cint},
    r1_cap::Cint,
    r2_el::Ptr{Cdouble},
    r2_idx::Ptr{Cint},
    r2_nnz::Cint,
    r2_cap::Cint,
    r2_packed::Cint,
    r3_el::Ptr{Cdouble},
    r3_idx::Ptr{Cint},
    r3_nnz::Cint,
    r3_cap::Cint,
    r3_packed::Cint,
    no_permute_r3::Cint,
    out_r3_nnz::Ptr{Cint},
)::Cint
    cf = _cf_get(Int32(id))
    v1 = PtrView(r1_el, Int(r1_cap))
    i1 = PtrView(r1_idx, Int(r1_cap))
    v2 = PtrView(r2_el, Int(r2_cap))
    i2 = PtrView(r2_idx, Int(r2_cap))
    v3 = PtrView(r3_el, Int(r3_cap))
    i3 = PtrView(r3_idx, Int(r3_cap))
    n2 = Int(r2_nnz)
    n3 = Int(r3_nnz)
    shift_to_1based!(i2, n2)
    shift_to_1based!(i3, n3)
    out2, out3 = clp_fact_update_two_columns_ft!(
        cf, v1, i1,
        v2, i2, n2, r2_packed != Cint(0),
        v3, i3, n3, r3_packed != Cint(0),
        no_permute_r3 != Cint(0),
    )
    shift_to_0based!(i2, abs(out2))
    shift_to_0based!(i3, out3)
    if out_r3_nnz != C_NULL
        unsafe_store!(out_r3_nnz, Cint(out3))
    end
    return Cint(out2)
end

Base.@ccallable function jclp_clpf_update_column_transpose(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_nnz::Cint,
    w_cap::Cint,
    w_packed::Cint,
    r_el::Ptr{Cdouble},
    r_idx::Ptr{Cint},
    r_nnz::Cint,
    r_cap::Cint,
    r_packed::Cint,
)::Cint
    cf = _cf_get(Int32(id))
    wv = PtrView(w_el, Int(w_cap))
    wi = PtrView(w_idx, Int(w_cap))
    rv = PtrView(r_el, Int(r_cap))
    ri = PtrView(r_idx, Int(r_cap))
    wn = Int(w_nnz)
    rn = Int(r_nnz)
    if wn > 0 && w_packed == Cint(0)
        shift_to_1based!(wi, wn)
    end
    shift_to_1based!(ri, rn)
    nnz_out = clp_fact_update_column_transpose!(cf, wv, wi, wn, w_packed != Cint(0), rv, ri, rn, r_packed != Cint(0))
    shift_to_0based!(ri, nnz_out)
    return Cint(nnz_out)
end

Base.@ccallable function jclp_clpf_update_two_columns_transpose(
    id::Cint,
    w_el::Ptr{Cdouble},
    w_idx::Ptr{Cint},
    w_nnz::Cint,
    w_cap::Cint,
    w_packed::Cint,
    r2_el::Ptr{Cdouble},
    r2_idx::Ptr{Cint},
    r2_nnz::Cint,
    r2_cap::Cint,
    r2_packed::Cint,
    r3_el::Ptr{Cdouble},
    r3_idx::Ptr{Cint},
    r3_nnz::Cint,
    r3_cap::Cint,
    r3_packed::Cint,
    out_nnz2::Ptr{Cint},
    out_nnz3::Ptr{Cint},
)::Cvoid
    cf = _cf_get(Int32(id))
    wv = PtrView(w_el, Int(w_cap))
    wi = PtrView(w_idx, Int(w_cap))
    v2 = PtrView(r2_el, Int(r2_cap))
    i2 = PtrView(r2_idx, Int(r2_cap))
    v3 = PtrView(r3_el, Int(r3_cap))
    i3 = PtrView(r3_idx, Int(r3_cap))
    wn = Int(w_nnz)
    n2 = Int(r2_nnz)
    n3 = Int(r3_nnz)
    if wn > 0 && w_packed == Cint(0)
        shift_to_1based!(wi, wn)
    end
    shift_to_1based!(i2, n2)
    shift_to_1based!(i3, n3)
    out2, out3 = clp_fact_update_two_columns_transpose!(
        cf, wv, wi, wn, w_packed != Cint(0),
        v2, i2, n2, r2_packed != Cint(0),
        v3, i3, n3, r3_packed != Cint(0),
    )
    shift_to_0based!(i2, out2)
    shift_to_0based!(i3, out3)
    if out_nnz2 != C_NULL
        unsafe_store!(out_nnz2, Cint(out2))
    end
    if out_nnz3 != C_NULL
        unsafe_store!(out_nnz3, Cint(out3))
    end
    return nothing
end

# ==============================================================================
# Phase B — B11a-1: ClpSimplexDual hot kernels (@ccallable C ABI)
# ==============================================================================

Base.@ccallable function jclp_dual_column0(
    row_val::Ptr{Cdouble},
    row_idx::Ptr{Cint},
    row_nnz::Cint,
    row_cap::Cint,
    col_val::Ptr{Cdouble},
    col_idx::Ptr{Cint},
    col_nnz::Cint,
    col_cap::Cint,
    spare_val::Ptr{Cdouble},
    spare_idx::Ptr{Cint},
    spare_cap::Cint,
    dj::Ptr{Cdouble},
    status::Ptr{Cuchar},
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    nrows::Cint,
    ncols::Cint,
    more_special_options::Cint,
    seq_out_0based::Cint,
    dual_tol::Cdouble,
    dual_bound::Cdouble,
    acceptable_pivot::Cdouble,
    out_upper_return::Ptr{Cdouble},
    out_bad_free::Ptr{Cdouble},
    io_seq_in_0based::Ptr{Cint},
    io_theta::Ptr{Cdouble},
    io_alpha::Ptr{Cdouble},
    io_number_fake::Ptr{Cint},
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    rc = Int(row_cap)
    cc = Int(col_cap)
    sc = Int(spare_cap)
    seq_out1 = Int(seq_out_0based) >= 0 ? (Int(seq_out_0based) + 1) : 0
    seq_in0 = Int(unsafe_load(io_seq_in_0based))
    seq_in1 = seq_in0 >= 0 ? (seq_in0 + 1) : 0

    n_rem, upper_ret, bad_free, seq_in_out, th_out, al_out, n_fake_out = clp_dual_column0!(
        PtrView(row_val, rc),
        ShiftedPtrView(row_idx, rc),
        Int(row_nnz),
        PtrView(col_val, cc),
        ShiftedPtrView(col_idx, cc),
        Int(col_nnz),
        PtrView(spare_val, sc),
        ShiftedPtrView(spare_idx, sc),
        PtrView(dj, nt),
        PtrView(status, nt),
        PtrView(solution, nt),
        PtrView(lower, nt),
        PtrView(upper, nt),
        nc,
        Int(more_special_options),
        seq_out1,
        Float64(dual_tol),
        Float64(dual_bound),
        Float64(acceptable_pivot),
        seq_in1,
        Float64(unsafe_load(io_theta)),
        Float64(unsafe_load(io_alpha)),
        Int(unsafe_load(io_number_fake)),
    )
    unsafe_store!(out_upper_return, upper_ret)
    unsafe_store!(out_bad_free, bad_free)
    unsafe_store!(io_seq_in_0based, Cint(seq_in_out > 0 ? (seq_in_out - 1) : -1))
    unsafe_store!(io_theta, th_out)
    unsafe_store!(io_alpha, al_out)
    unsafe_store!(io_number_fake, Cint(n_fake_out))
    return Cint(n_rem)
end

Base.@ccallable function jclp_dual_column(
    row_val::Ptr{Cdouble},
    row_idx::Ptr{Cint},
    row_nnz::Cint,
    row_cap::Cint,
    col_val::Ptr{Cdouble},
    col_idx::Ptr{Cint},
    col_nnz::Cint,
    col_cap::Cint,
    spare_val::Ptr{Cdouble},
    spare_idx::Ptr{Cint},
    spare_cap::Cint,
    spare2_val::Ptr{Cdouble},
    spare2_idx::Ptr{Cint},
    spare2_cap::Cint,
    dj::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    status::Ptr{Cuchar},
    nrows::Cint,
    ncols::Cint,
    dual_tol::Cdouble,
    dual_bound::Cdouble,
    dual_out::Cdouble,
    lower_out::Cdouble,
    upper_out::Cdouble,
    large_value::Cdouble,
    objective_value::Cdouble,
    special_options::Cint,
    more_special_options::Cint,
    factor_pivots::Cint,
    seq_out_0based::Cint,
    acceptable_pivot::Cdouble,
    acceptable_pivot_member::Cdouble,
    spare_int0::Cint,
    spare_int1_0based::Cint,
    spare_nnz_in::Cint,
    spare_dbl0::Cdouble,
    spare_dbl2::Cdouble,
    spare_dbl3::Cdouble,
    io_seq_in_0based::Ptr{Cint},
    out_dir_in::Ptr{Cint},
    io_theta::Ptr{Cdouble},
    out_alpha::Ptr{Cdouble},
    out_lower_in::Ptr{Cdouble},
    out_upper_in::Ptr{Cdouble},
    out_value_in::Ptr{Cdouble},
    out_dual_in::Ptr{Cdouble},
    out_acc_piv_mem::Ptr{Cdouble},
    io_number_fake::Ptr{Cint},
    io_number_changed::Ptr{Cint},
)::Cdouble
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    rc = Int(row_cap)
    cc = Int(col_cap)
    sc = Int(spare_cap)
    s2c = Int(spare2_cap)
    seq_out1 = Int(seq_out_0based) >= 0 ? (Int(seq_out_0based) + 1) : 0
    sp1 = Int(spare_int1_0based) >= 0 ? (Int(spare_int1_0based) + 1) : 0
    seq_in0 = Int(unsafe_load(io_seq_in_0based))
    seq_in1 = seq_in0 >= 0 ? (seq_in0 + 1) : 0

    best_possible, seq_in_out, dir_in, th_out, al_out,
    lo_in, up_in, val_in, dj_in, acc_mem, n_fake_out, n_chg_out = clp_dual_column!(
        PtrView(row_val, rc),
        ShiftedPtrView(row_idx, rc),
        Int(row_nnz),
        PtrView(col_val, cc),
        ShiftedPtrView(col_idx, cc),
        Int(col_nnz),
        PtrView(spare_val, sc),
        ShiftedPtrView(spare_idx, sc),
        PtrView(spare2_val, s2c),
        ShiftedPtrView(spare2_idx, s2c),
        PtrView(dj, nt),
        PtrView(cost, nt),
        PtrView(solution, nt),
        PtrView(lower, nt),
        PtrView(upper, nt),
        PtrView(status, nt),
        nr,
        nc,
        Float64(dual_tol),
        Float64(dual_bound),
        Float64(dual_out),
        Float64(lower_out),
        Float64(upper_out),
        Float64(large_value),
        Float64(objective_value),
        Int(special_options),
        Int(more_special_options),
        Int(factor_pivots),
        seq_out1,
        Float64(acceptable_pivot),
        Float64(acceptable_pivot_member),
        Int(spare_int0),
        sp1,
        Int(spare_nnz_in),
        Float64(spare_dbl0),
        Float64(spare_dbl2),
        Float64(spare_dbl3),
        seq_in1,
        Float64(unsafe_load(io_theta)),
        Int(unsafe_load(io_number_fake)),
        Int(unsafe_load(io_number_changed)),
    )

    unsafe_store!(io_seq_in_0based, Cint(seq_in_out > 0 ? (seq_in_out - 1) : -1))
    unsafe_store!(out_dir_in, Cint(dir_in))
    unsafe_store!(io_theta, th_out)
    unsafe_store!(out_alpha, al_out)
    unsafe_store!(out_lower_in, lo_in)
    unsafe_store!(out_upper_in, up_in)
    unsafe_store!(out_value_in, val_in)
    unsafe_store!(out_dual_in, dj_in)
    unsafe_store!(out_acc_piv_mem, acc_mem)
    unsafe_store!(io_number_fake, Cint(n_fake_out))
    unsafe_store!(io_number_changed, Cint(n_chg_out))
    return Cdouble(best_possible)
end

Base.@ccallable function jclp_dual_update_duals_in_dual(
    row_val::Ptr{Cdouble},
    row_idx::Ptr{Cint},
    row_nnz::Cint,
    row_cap::Cint,
    col_val::Ptr{Cdouble},
    col_idx::Ptr{Cint},
    col_nnz::Cint,
    col_cap::Cint,
    out_val::Ptr{Cdouble},
    out_idx::Ptr{Cint},
    out_cap::Cint,
    mat_starts::Ptr{Cint},
    mat_lengths::Ptr{Cint},
    mat_indices::Ptr{Cint},
    mat_elements::Ptr{Cdouble},
    mat_nnz_cap::Cint,
    row_scale::Ptr{Cdouble},
    col_scale::Ptr{Cdouble},
    dj::Ptr{Cdouble},
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    cost::Ptr{Cdouble},
    status::Ptr{Cuchar},
    nrows::Cint,
    ncols::Cint,
    theta::Cdouble,
    dual_tol::Cdouble,
    largest_dual_err::Cdouble,
    dual_bound::Cdouble,
    more_special_options::Cint,
    io_number_fake::Ptr{Cint},
    out_obj_change::Ptr{Cdouble},
    out_row_nnz::Ptr{Cint},
    out_col_nnz::Ptr{Cint},
    out_out_nnz::Ptr{Cint},
    full_recompute::Cint,
)::Cint
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    nz = Int(mat_nnz_cap)
    rc = Int(row_cap)
    cc = Int(col_cap)
    oc = Int(out_cap)

    rv = PtrView(row_val, rc)
    ri = ShiftedPtrView(row_idx, rc)
    cv = PtrView(col_val, cc)
    ci = ShiftedPtrView(col_idx, cc)
    ov = PtrView(out_val, oc)
    oi = ShiftedPtrView(out_idx, oc)
    ms = ShiftedPtrView(mat_starts, nc + 1)
    ml = PtrView(mat_lengths, nc)
    mi = ShiftedPtrView(mat_indices, nz)
    me = PtrView(mat_elements, nz)
    djv = PtrView(dj, nt)
    solv = PtrView(solution, nt)
    lov = PtrView(lower, nt)
    upv = PtrView(upper, nt)
    cov = PtrView(cost, nt)
    stv = PtrView(status, nt)
    nfake0 = Int(unsafe_load(io_number_fake))
    full_rec = full_recompute != Cint(0)

    if row_scale != C_NULL
        tot_inf, r_nnz, c_nnz, o_nnz, chg_obj, nfake1 = clp_update_duals_in_dual!(
            rv, ri, Int(row_nnz),
            cv, ci, Int(col_nnz),
            ov, oi, 0,
            ms, ml, mi, me,
            PtrView(row_scale, nr), PtrView(col_scale, nc),
            djv, solv, lov, upv, cov, stv,
            nr, nc,
            Float64(theta), Float64(dual_tol), Float64(largest_dual_err), Float64(dual_bound),
            Int(more_special_options), nfake0, full_rec,
        )
    else
        tot_inf, r_nnz, c_nnz, o_nnz, chg_obj, nfake1 = clp_update_duals_in_dual!(
            rv, ri, Int(row_nnz),
            cv, ci, Int(col_nnz),
            ov, oi, 0,
            ms, ml, mi, me,
            nothing, nothing,
            djv, solv, lov, upv, cov, stv,
            nr, nc,
            Float64(theta), Float64(dual_tol), Float64(largest_dual_err), Float64(dual_bound),
            Int(more_special_options), nfake0, full_rec,
        )
    end
    unsafe_store!(io_number_fake, Cint(nfake1))
    unsafe_store!(out_obj_change, chg_obj)
    unsafe_store!(out_row_nnz, Cint(r_nnz))
    unsafe_store!(out_col_nnz, Cint(c_nnz))
    unsafe_store!(out_out_nnz, Cint(o_nnz))
    return Cint(tot_inf)
end

Base.@ccallable function jclp_dual_update_duals_in_values_pass(
    row_val::Ptr{Cdouble},
    row_idx::Ptr{Cint},
    row_nnz::Cint,
    row_cap::Cint,
    col_val::Ptr{Cdouble},
    col_idx::Ptr{Cint},
    col_nnz::Cint,
    col_cap::Cint,
    dj::Ptr{Cdouble},
    status::Ptr{Cuchar},
    nrows::Cint,
    ncols::Cint,
    theta::Cdouble,
    dual_tol::Cdouble,
)::Cvoid
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    rc = Int(row_cap)
    cc = Int(col_cap)
    clp_update_duals_in_values_pass!(
        PtrView(row_val, rc),
        ShiftedPtrView(row_idx, rc),
        Int(row_nnz),
        PtrView(col_val, cc),
        ShiftedPtrView(col_idx, cc),
        Int(col_nnz),
        PtrView(dj, nt),
        PtrView(status, nt),
        nc,
        Float64(theta),
        Float64(dual_tol),
    )
    return nothing
end

Base.@ccallable function jclp_dual_flip_bounds(
    row_idx::Ptr{Cint},
    row_nnz::Cint,
    row_cap::Cint,
    col_idx::Ptr{Cint},
    col_nnz::Cint,
    col_cap::Cint,
    solution::Ptr{Cdouble},
    lower::Ptr{Cdouble},
    upper::Ptr{Cdouble},
    status::Ptr{Cuchar},
    nrows::Cint,
    ncols::Cint,
)::Cvoid
    nr = Int(nrows)
    nc = Int(ncols)
    nt = nr + nc
    clp_flip_bounds!(
        ShiftedPtrView(row_idx, Int(row_cap)),
        Int(row_nnz),
        ShiftedPtrView(col_idx, Int(col_cap)),
        Int(col_nnz),
        PtrView(solution, nt),
        PtrView(lower, nt),
        PtrView(upper, nt),
        PtrView(status, nt),
        nc,
    )
    return nothing
end

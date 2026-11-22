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

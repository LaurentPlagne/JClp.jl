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

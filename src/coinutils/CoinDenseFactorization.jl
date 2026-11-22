# Faithful translation of CoinUtils/src/CoinDenseFactorization.{hpp,cpp}
# Supports both solveMode % 10 == 0 (built-in LU) and solveMode % 10 != 0 (LAPACK dgetrf_/dgetrs_
# via Apple Accelerate on macOS, matching COIN_FACTORIZATION_DENSE_CODE == 1 bit-for-bit).

const ACCELERATE_LIB = "/System/Library/Frameworks/Accelerate.framework/Accelerate"

@inline function _lapack_dgetrf!(m::Int32, n::Int32, A::Ptr{Float64}, lda::Int32, ipiv::Ptr{Int32})::Int32
    m_ref = Ref{Int32}(m)
    n_ref = Ref{Int32}(n)
    lda_ref = Ref{Int32}(lda)
    info_ref = Ref{Int32}(Int32(0))
    ccall(
        (:dgetrf_, ACCELERATE_LIB),
        Cvoid,
        (Ref{Int32}, Ref{Int32}, Ptr{Float64}, Ref{Int32}, Ptr{Int32}, Ref{Int32}),
        m_ref, n_ref, A, lda_ref, ipiv, info_ref,
    )
    return info_ref[]
end

@inline function _lapack_dgetrs!(trans::UInt8, n::Int32, nrhs::Int32, A::Ptr{Float64}, lda::Int32, ipiv::Ptr{Int32}, B::Ptr{Float64}, ldb::Int32)::Int32
    trans_ref = Ref{UInt8}(trans)
    n_ref = Ref{Int32}(n)
    nrhs_ref = Ref{Int32}(nrhs)
    lda_ref = Ref{Int32}(lda)
    ldb_ref = Ref{Int32}(ldb)
    info_ref = Ref{Int32}(Int32(0))
    ccall(
        (:dgetrs_, ACCELERATE_LIB),
        Cvoid,
        (Ref{UInt8}, Ref{Int32}, Ref{Int32}, Ptr{Float64}, Ref{Int32}, Ptr{Int32}, Ptr{Float64}, Ref{Int32}, Ref{Int32}, Cint),
        trans_ref, n_ref, nrhs_ref, A, lda_ref, ipiv, B, ldb_ref, info_ref, Cint(1),
    )
    return info_ref[]
end

# =============================================================================
# 1. Raw-pointer kernels (0-based indices) for C++ bridge (`JCLP_BRICK_B4`)
# =============================================================================

# CoinDenseFactorization::factor() (lines 190-286)
# Updates out_status and io_solve_mode; returns numberGoodU.
function cdf_factor_ptr!(
    n_rows::Int32,
    n_cols::Int32,
    elements::Ptr{Float64},
    pivot_row::Ptr{Int32},
    work_area::Ptr{Float64},
    zero_tol::Float64,
    io_solve_mode::Ptr{Int32},
    out_status::Ptr{Int32},
)::Int32
    nr = Int(n_rows)
    nc = Int(n_cols)
    sm = unsafe_load(io_solve_mode)
    if n_rows == n_cols && (sm % Int32(10)) != Int32(0)
        info = _lapack_dgetrf!(n_rows, n_rows, elements, n_rows, pivot_row)
        if info == Int32(0)
            unsafe_store!(io_solve_mode, Int32(1) + Int32(10) * div(sm, Int32(10)))
            coin_zero_n!(work_area, 2 * nr)
            unsafe_store!(out_status, Int32(0))
            return n_rows
        else
            unsafe_store!(io_solve_mode, Int32(10) * div(sm, Int32(10)))
        end
    end
    status = Int32(0)
    j = 0
    while j < nr
        unsafe_store!(pivot_row, Int32(j), j + nr + 1)
        j += 1
    end
    n_good_u = Int32(0)
    i = 0
    while i < nc
        col_off = i * nr
        i_row = -1
        largest = zero_tol
        r = i
        while r < nr
            v = abs(unsafe_load(elements, col_off + r + 1))
            if v > largest
                largest = v
                i_row = r
            end
            r += 1
        end
        if i_row >= 0
            if i_row != i
                k = 0
                while k <= i
                    k_off = k * nr
                    v = unsafe_load(elements, k_off + i + 1)
                    unsafe_store!(elements, unsafe_load(elements, k_off + i_row + 1), k_off + i + 1)
                    unsafe_store!(elements, v, k_off + i_row + 1)
                    k += 1
                end
                ip = unsafe_load(pivot_row, i + nr + 1)
                unsafe_store!(pivot_row, unsafe_load(pivot_row, i_row + nr + 1), i + nr + 1)
                unsafe_store!(pivot_row, ip, i_row + nr + 1)
            end
            piv_val = 1.0 / unsafe_load(elements, col_off + i + 1)
            unsafe_store!(elements, piv_val, col_off + i + 1)
            r = i + 1
            while r < nr
                unsafe_store!(elements, unsafe_load(elements, col_off + r + 1) * piv_val, col_off + r + 1)
                r += 1
            end
            k = i + 1
            while k < nc
                k_off = k * nr
                if i_row != i
                    v = unsafe_load(elements, k_off + i + 1)
                    unsafe_store!(elements, unsafe_load(elements, k_off + i_row + 1), k_off + i + 1)
                    unsafe_store!(elements, v, k_off + i_row + 1)
                end
                val_ki = unsafe_load(elements, k_off + i + 1)
                r = i + 1
                while r < nr
                    cur = unsafe_load(elements, k_off + r + 1)
                    unsafe_store!(elements, cur - val_ki * unsafe_load(elements, col_off + r + 1), k_off + r + 1)
                    r += 1
                end
                k += 1
            end
        else
            status = Int32(-1)
            break
        end
        n_good_u += Int32(1)
        i += 1
    end
    j = 0
    while j < nr
        k = Int(unsafe_load(pivot_row, j + nr + 1))
        unsafe_store!(pivot_row, Int32(j), k + 1)
        j += 1
    end
    unsafe_store!(out_status, status)
    return n_good_u
end

# CoinDenseFactorization::replaceColumn() (lines 356-426)
# Returns 0=OK, 2=singular, 3=no room (caller increments numberPivots_ on return == 0).
function cdf_replace_column_ptr!(
    n_rows::Int32,
    n_cols::Int32,
    n_pivots::Int32,
    max_pivots::Int32,
    solve_mode::Int32,
    elements_base::Ptr{Float64},
    pivot_row_arr::Ptr{Int32},
    zero_tol::Float64,
    region::Ptr{Float64},
    region_index::Ptr{Int32},
    n_nonzero::Int32,
    packed_mode::Bool,
    pivot_row_idx::Int32,
    pivot_check::Float64,
)::Int32
    if n_pivots == max_pivots
        return Int32(3)
    end
    nr = Int(n_rows)
    eta_off = nr * (Int(n_cols) + Int(n_pivots))
    coin_zero_n!(elements_base + eta_off * sizeof(Float64), nr)
    if abs(pivot_check) < zero_tol
        return Int32(2)
    end
    piv_val = 1.0 / pivot_check
    nnz = Int(n_nonzero)
    if (solve_mode % Int32(10)) == Int32(0)
        if packed_mode
            i = 1
            while i <= nnz
                r_in = Int(unsafe_load(region_index, i))
                val = unsafe_load(region, i)
                r_perm = Int(unsafe_load(pivot_row_arr, r_in + 1))
                unsafe_store!(elements_base, val, eta_off + r_perm + 1)
                i += 1
            end
        else
            i = 1
            while i <= nnz
                r_in = Int(unsafe_load(region_index, i))
                val = unsafe_load(region, r_in + 1)
                r_perm = Int(unsafe_load(pivot_row_arr, r_in + 1))
                unsafe_store!(elements_base, val, eta_off + r_perm + 1)
                i += 1
            end
        end
        real_piv = Int(unsafe_load(pivot_row_arr, Int(pivot_row_idx) + 1))
        unsafe_store!(elements_base, piv_val, eta_off + real_piv + 1)
        unsafe_store!(pivot_row_arr, Int32(real_piv), 2 * nr + Int(n_pivots) + 1)
    else
        # LAPACK mode (no dense permutation)
        if packed_mode
            i = 1
            while i <= nnz
                r_in = Int(unsafe_load(region_index, i))
                unsafe_store!(elements_base, unsafe_load(region, i), eta_off + r_in + 1)
                i += 1
            end
        else
            i = 1
            while i <= nnz
                r_in = Int(unsafe_load(region_index, i))
                unsafe_store!(elements_base, unsafe_load(region, r_in + 1), eta_off + r_in + 1)
                i += 1
            end
        end
        unsafe_store!(elements_base, piv_val, eta_off + Int(pivot_row_idx) + 1)
        unsafe_store!(pivot_row_arr, pivot_row_idx, 2 * nr + Int(n_pivots) + 1)
    end
    return Int32(0)
end

# CoinDenseFactorization::updateColumn() (lines 430-615) — returns new n_nonzero
function cdf_update_column_ptr!(
    n_rows::Int32,
    n_pivots::Int32,
    solve_mode::Int32,
    elements_base::Ptr{Float64},
    pivot_row_arr::Ptr{Int32},
    zero_tol::Float64,
    region1::Ptr{Float64},
    region2::Ptr{Float64},
    region_index::Ptr{Int32},
    n_nonzero_in::Int32,
    packed_mode::Bool,
    no_permute::Bool,
)::Int32
    nr = Int(n_rows)
    nnz_in = Int(n_nonzero_in)
    use_lapack = (solve_mode % Int32(10)) != Int32(0)
    reg = region1
    if !use_lapack
        if !packed_mode
            if !no_permute
                j = 0
                while j < nr
                    ir = Int(unsafe_load(pivot_row_arr, j + nr + 1))
                    unsafe_store!(reg, unsafe_load(region2, ir + 1), j + 1)
                    unsafe_store!(region2, 0.0, ir + 1)
                    j += 1
                end
            else
                reg = region2
            end
        else
            j = 1
            while j <= nnz_in
                jr = Int(unsafe_load(region_index, j))
                ir = Int(unsafe_load(pivot_row_arr, jr + 1))
                unsafe_store!(reg, unsafe_load(region2, j), ir + 1)
                unsafe_store!(region2, 0.0, j)
                j += 1
            end
        end
        # Base L forward solve
        i = 0
        while i < nr
            col_off = i * nr
            val = unsafe_load(reg, i + 1)
            j = i + 1
            while j < nr
                unsafe_store!(reg, unsafe_load(reg, j + 1) - val * unsafe_load(elements_base, col_off + j + 1), j + 1)
                j += 1
            end
            i += 1
        end
        # Base U backward solve
        i = nr - 1
        while i >= 0
            col_off = i * nr
            val = unsafe_load(reg, i + 1) * unsafe_load(elements_base, col_off + i + 1)
            unsafe_store!(reg, val, i + 1)
            j = 0
            while j < i
                unsafe_store!(reg, unsafe_load(reg, j + 1) - val * unsafe_load(elements_base, col_off + j + 1), j + 1)
                j += 1
            end
            i -= 1
        end
    else
        # LAPACK mode
        if !packed_mode
            if !no_permute
                j = 0
                while j < nr
                    unsafe_store!(reg, unsafe_load(region2, j + 1), j + 1)
                    unsafe_store!(region2, 0.0, j + 1)
                    j += 1
                end
            else
                reg = region2
            end
        else
            j = 1
            while j <= nnz_in
                jr = Int(unsafe_load(region_index, j))
                unsafe_store!(reg, unsafe_load(region2, j), jr + 1)
                unsafe_store!(region2, 0.0, j)
                j += 1
            end
        end
        _lapack_dgetrs!(UInt8('N'), n_rows, Int32(1), elements_base, n_rows, pivot_row_arr, reg, n_rows)
    end

    # PFI eta updates
    np = Int(n_pivots)
    p = 0
    while p < np
        eta_off = nr * (nr + p)
        ip = Int(unsafe_load(pivot_row_arr, p + 2 * nr + 1))
        val = unsafe_load(reg, ip + 1) * unsafe_load(elements_base, eta_off + ip + 1)
        j = 0
        while j < nr
            unsafe_store!(reg, unsafe_load(reg, j + 1) - val * unsafe_load(elements_base, eta_off + j + 1), j + 1)
            j += 1
        end
        unsafe_store!(reg, val, ip + 1)
        p += 1
    end

    # Permute back and extract nonzeros
    nnz_out = 0
    if !no_permute
        if !use_lapack
            if !packed_mode
                j = 0
                while j < nr
                    ir = Int(unsafe_load(pivot_row_arr, j + 1))
                    val = unsafe_load(reg, ir + 1)
                    unsafe_store!(reg, 0.0, ir + 1)
                    if abs(val) > zero_tol
                        unsafe_store!(region2, val, j + 1)
                        nnz_out += 1
                        unsafe_store!(region_index, Int32(j), nnz_out)
                    end
                    j += 1
                end
            else
                j = 0
                while j < nr
                    ir = Int(unsafe_load(pivot_row_arr, j + 1))
                    val = unsafe_load(reg, ir + 1)
                    unsafe_store!(reg, 0.0, ir + 1)
                    if abs(val) > zero_tol
                        nnz_out += 1
                        unsafe_store!(region2, val, nnz_out)
                        unsafe_store!(region_index, Int32(j), nnz_out)
                    end
                    j += 1
                end
            end
        else
            if !packed_mode
                j = 0
                while j < nr
                    val = unsafe_load(reg, j + 1)
                    unsafe_store!(reg, 0.0, j + 1)
                    if abs(val) > zero_tol
                        unsafe_store!(region2, val, j + 1)
                        nnz_out += 1
                        unsafe_store!(region_index, Int32(j), nnz_out)
                    end
                    j += 1
                end
            else
                j = 0
                while j < nr
                    val = unsafe_load(reg, j + 1)
                    unsafe_store!(reg, 0.0, j + 1)
                    if abs(val) > zero_tol
                        nnz_out += 1
                        unsafe_store!(region2, val, nnz_out)
                        unsafe_store!(region_index, Int32(j), nnz_out)
                    end
                    j += 1
                end
            end
        end
    else
        j = 0
        while j < nr
            val = unsafe_load(reg, j + 1)
            if abs(val) > zero_tol
                nnz_out += 1
                unsafe_store!(region_index, Int32(j), nnz_out)
            else
                unsafe_store!(reg, 0.0, j + 1)
            end
            j += 1
        end
    end
    return Int32(nnz_out)
end

# CoinDenseFactorization::updateColumnTranspose() (lines 758-909) — returns new n_nonzero
function cdf_update_column_transpose_ptr!(
    n_rows::Int32,
    n_pivots::Int32,
    solve_mode::Int32,
    elements_base::Ptr{Float64},
    pivot_row_arr::Ptr{Int32},
    zero_tol::Float64,
    region1::Ptr{Float64},
    region2::Ptr{Float64},
    region_index::Ptr{Int32},
    n_nonzero_in::Int32,
    packed_mode::Bool,
)::Int32
    nr = Int(n_rows)
    nnz_in = Int(n_nonzero_in)
    use_lapack = (solve_mode % Int32(10)) != Int32(0)
    if !use_lapack
        if !packed_mode
            j = 0
            while j < nr
                ir = Int(unsafe_load(pivot_row_arr, j + 1))
                unsafe_store!(region1, unsafe_load(region2, j + 1), ir + 1)
                unsafe_store!(region2, 0.0, j + 1)
                j += 1
            end
        else
            j = 1
            while j <= nnz_in
                jr = Int(unsafe_load(region_index, j))
                ir = Int(unsafe_load(pivot_row_arr, jr + 1))
                unsafe_store!(region1, unsafe_load(region2, j), ir + 1)
                unsafe_store!(region2, 0.0, j)
                j += 1
            end
        end
    else
        if !packed_mode
            j = 0
            while j < nr
                unsafe_store!(region1, unsafe_load(region2, j + 1), j + 1)
                unsafe_store!(region2, 0.0, j + 1)
                j += 1
            end
        else
            j = 1
            while j <= nnz_in
                jr = Int(unsafe_load(region_index, j))
                unsafe_store!(region1, unsafe_load(region2, j), jr + 1)
                unsafe_store!(region2, 0.0, j)
                j += 1
            end
        end
    end

    # PFI eta transpose updates (reverse order)
    p = Int(n_pivots) - 1
    while p >= 0
        eta_off = nr * (nr + p)
        ip = Int(unsafe_load(pivot_row_arr, p + 2 * nr + 1))
        val = unsafe_load(region1, ip + 1)
        j = 0
        while j < ip
            val -= unsafe_load(region1, j + 1) * unsafe_load(elements_base, eta_off + j + 1)
            j += 1
        end
        j = ip + 1
        while j < nr
            val -= unsafe_load(region1, j + 1) * unsafe_load(elements_base, eta_off + j + 1)
            j += 1
        end
        unsafe_store!(region1, val * unsafe_load(elements_base, eta_off + ip + 1), ip + 1)
        p -= 1
    end

    if !use_lapack
        # Base U^T forward solve
        i = 0
        while i < nr
            col_off = i * nr
            val = unsafe_load(region1, i + 1)
            j = 0
            while j < i
                val -= unsafe_load(region1, j + 1) * unsafe_load(elements_base, col_off + j + 1)
                j += 1
            end
            unsafe_store!(region1, val * unsafe_load(elements_base, col_off + i + 1), i + 1)
            i += 1
        end
        # Base L^T backward solve
        i = nr - 1
        while i >= 0
            col_off = i * nr
            val = unsafe_load(region1, i + 1)
            j = i + 1
            while j < nr
                val -= unsafe_load(region1, j + 1) * unsafe_load(elements_base, col_off + j + 1)
                j += 1
            end
            unsafe_store!(region1, val, i + 1)
            i -= 1
        end
    else
        _lapack_dgetrs!(UInt8('T'), n_rows, Int32(1), elements_base, n_rows, pivot_row_arr, region1, n_rows)
    end

    # Permute back and extract nonzeros
    nnz_out = 0
    if !use_lapack
        if !packed_mode
            j = 0
            while j < nr
                ir = Int(unsafe_load(pivot_row_arr, j + nr + 1))
                val = unsafe_load(region1, j + 1)
                unsafe_store!(region1, 0.0, j + 1)
                if abs(val) > zero_tol
                    unsafe_store!(region2, val, ir + 1)
                    nnz_out += 1
                    unsafe_store!(region_index, Int32(ir), nnz_out)
                end
                j += 1
            end
        else
            j = 0
            while j < nr
                ir = Int(unsafe_load(pivot_row_arr, j + nr + 1))
                val = unsafe_load(region1, j + 1)
                unsafe_store!(region1, 0.0, j + 1)
                if abs(val) > zero_tol
                    nnz_out += 1
                    unsafe_store!(region2, val, nnz_out)
                    unsafe_store!(region_index, Int32(ir), nnz_out)
                end
                j += 1
            end
        end
    else
        if !packed_mode
            j = 0
            while j < nr
                val = unsafe_load(region1, j + 1)
                unsafe_store!(region1, 0.0, j + 1)
                if abs(val) > zero_tol
                    unsafe_store!(region2, val, j + 1)
                    nnz_out += 1
                    unsafe_store!(region_index, Int32(j), nnz_out)
                end
                j += 1
            end
        else
            j = 0
            while j < nr
                val = unsafe_load(region1, j + 1)
                unsafe_store!(region1, 0.0, j + 1)
                if abs(val) > zero_tol
                    nnz_out += 1
                    unsafe_store!(region2, val, nnz_out)
                    unsafe_store!(region_index, Int32(j), nnz_out)
                end
                j += 1
            end
        end
    end
    return Int32(nnz_out)
end

# =============================================================================
# 2. Standalone Julia `CoinDenseFactorization`
# =============================================================================

mutable struct CoinDenseFactorization
    pivot_tolerance::Float64
    zero_tolerance::Float64
    number_rows::Int32
    number_columns::Int32
    number_good_u::Int32
    maximum_pivots::Int32
    number_pivots::Int32
    solve_mode::Int32
    status::Int32
    elements::Vector{Float64}
    pivot_row::Vector{Int32}
    work_area::Vector{Float64}
end

function CoinDenseFactorization(max_pivots::Integer = 200; solve_mode::Integer = 0)
    return CoinDenseFactorization(
        0.1, 1e-13,
        Int32(0), Int32(0), Int32(0), Int32(max_pivots), Int32(0), Int32(solve_mode), Int32(-1),
        Float64[], Int32[], Float64[],
    )
end

function get_areas!(df::CoinDenseFactorization, nr::Integer, nc::Integer)::Nothing
    df.number_rows = Int32(nr)
    df.number_columns = Int32(nc)
    sz = Int(nr) * (Int(nr) + max(Int(df.maximum_pivots), (Int(nr) + 1) >> 1))
    if length(df.elements) < sz
        resize!(df.elements, sz)
    end
    pr_sz = 2 * Int(nr) + Int(df.maximum_pivots)
    if length(df.pivot_row) < pr_sz
        resize!(df.pivot_row, pr_sz)
    end
    wa_sz = 2 * Int(nr)
    if length(df.work_area) < wa_sz
        resize!(df.work_area, wa_sz)
    end
    return nothing
end

function factor_dense!(df::CoinDenseFactorization, B::Matrix{Float64})::Int32
    nr, nc = size(B)
    get_areas!(df, nr, nc)
    @inbounds for j in 1:nc, i in 1:nr
        df.elements[(j - 1) * nr + i] = B[i, j]
    end
    df.number_pivots = Int32(0)
    sm_ref = Ref{Int32}(df.solve_mode)
    st_ref = Ref{Int32}(Int32(0))
    df.number_good_u = GC.@preserve df sm_ref st_ref cdf_factor_ptr!(
        df.number_rows, df.number_columns, pointer(df.elements), pointer(df.pivot_row),
        pointer(df.work_area), df.zero_tolerance,
        Base.unsafe_convert(Ptr{Int32}, sm_ref), Base.unsafe_convert(Ptr{Int32}, st_ref),
    )
    df.solve_mode = sm_ref[]
    df.status = st_ref[]
    return df.status
end

function replace_column!(
    df::CoinDenseFactorization,
    region::CoinIndexedVector,
    pivot_row0::Int32,
    pivot_check::Float64,
)::Int32
    rc = GC.@preserve df region cdf_replace_column_ptr!(
        df.number_rows, df.number_columns, df.number_pivots, df.maximum_pivots, df.solve_mode,
        pointer(df.elements), pointer(df.pivot_row), df.zero_tolerance,
        pointer(region.elements), pointer(region.indices), region.n_elements,
        region.packed_mode, pivot_row0, pivot_check,
    )
    if rc == Int32(0)
        df.number_pivots += Int32(1)
    end
    return rc
end

function update_column!(
    df::CoinDenseFactorization,
    region1::CoinIndexedVector,
    region2::CoinIndexedVector,
    no_permute::Bool = false,
)::Int32
    nnz = GC.@preserve df region1 region2 cdf_update_column_ptr!(
        df.number_rows, df.number_pivots, df.solve_mode, pointer(df.elements), pointer(df.pivot_row),
        df.zero_tolerance, pointer(region1.elements), pointer(region2.elements),
        pointer(region2.indices), region2.n_elements, region2.packed_mode, no_permute,
    )
    set_num_elements!(region2, nnz)
    return Int32(0)
end

function update_column_transpose!(
    df::CoinDenseFactorization,
    region1::CoinIndexedVector,
    region2::CoinIndexedVector,
)::Int32
    nnz = GC.@preserve df region1 region2 cdf_update_column_transpose_ptr!(
        df.number_rows, df.number_pivots, df.solve_mode, pointer(df.elements), pointer(df.pivot_row),
        df.zero_tolerance, pointer(region1.elements), pointer(region2.elements),
        pointer(region2.indices), region2.n_elements, region2.packed_mode,
    )
    set_num_elements!(region2, nnz)
    return Int32(0)
end

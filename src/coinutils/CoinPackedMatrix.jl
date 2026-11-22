# Faithful translation of CoinUtils/src/CoinPackedMatrix.{hpp,cpp}
# Provides:
#   1. Zero-allocation raw-pointer kernels (0-based indices) for C++ bridge (`JCLP_BRICK_B3`)
#   2. Standalone Julia `CoinPackedMatrix` struct (incl. reverse_ordering! / reverse_ordered_copy).

# =============================================================================
# 1. Raw-pointer kernels (0-based indices matching CoinPackedMatrix)
# =============================================================================

@inline function cpm_count_ortho_length_ptr!(
    major_dim::Int32,
    minor_dim::Int32,
    start::Ptr{Int32},
    length_arr::Ptr{Int32},
    index::Ptr{Int32},
    ortho_length::Ptr{Int32},
)::Nothing
    coin_zero_n!(ortho_length, Int(minor_dim))
    md = Int(major_dim)
    i = 1
    while i <= md
        s = Int(unsafe_load(start, i))
        l = Int(unsafe_load(length_arr, i))
        j = s + 1
        j_end = s + l
        while j <= j_end
            ind = Int(unsafe_load(index, j)) + 1
            unsafe_store!(ortho_length, unsafe_load(ortho_length, ind) + Int32(1), ind)
            j += 1
        end
        i += 1
    end
    return nothing
end

# Scatter body of CoinPackedMatrix::reverseOrderedCopyOf (lines 958-1005)
# Assumes dst_start[0..dst_major_dim] and dst_length[0..dst_major_dim-1] are initialized.
@inline function cpm_reverse_scatter_ptr!(
    src_major_dim::Int32,
    src_start::Ptr{Int32},
    src_length::Ptr{Int32},
    src_index::Ptr{Int32},
    src_element::Ptr{Float64},
    dst_major_dim::Int32,
    dst_start::Ptr{Int32},
    dst_length::Ptr{Int32},
    dst_index::Ptr{Int32},
    dst_element::Ptr{Float64},
)::Nothing
    smd = Int(src_major_dim)
    first = Int32(0)
    i = 0
    while i < smd
        last = first + unsafe_load(src_length, i + 1)
        j = first
        first = unsafe_load(src_start, i + 2)
        while j != last
            ind = Int(unsafe_load(src_index, Int(j) + 1))
            put = unsafe_load(dst_start, ind + 1)
            unsafe_store!(dst_start, put + Int32(1), ind + 1)
            unsafe_store!(dst_element, unsafe_load(src_element, Int(j) + 1), Int(put) + 1)
            unsafe_store!(dst_index, Int32(i), Int(put) + 1)
            j += Int32(1)
        end
        i += 1
    end
    dmd = Int(dst_major_dim)
    k = 1
    while k <= dmd
        unsafe_store!(dst_start, unsafe_load(dst_start, k) - unsafe_load(dst_length, k), k)
        k += 1
    end
    return nothing
end

# CoinPackedMatrix::removeGaps (lines 575-642)
# Returns updated size_.
@inline function cpm_remove_gaps_ptr!(
    major_dim::Int32,
    size_in::Int32,
    start::Ptr{Int32},
    length_arr::Ptr{Int32},
    index::Ptr{Int32},
    element::Ptr{Float64},
    remove_value::Float64,
)::Int32
    md = Int(major_dim)
    if remove_value < 0.0
        if size_in < unsafe_load(start, md + 1)
            size = Int32(0)
            i = 1
            while i < md + 1
                si = unsafe_load(start, i + 1)
                size += unsafe_load(length_arr, i)
                if si > size
                    break
                end
                i += 1
            end
            while i < md
                si = unsafe_load(start, i + 1)
                li = unsafe_load(length_arr, i + 1)
                unsafe_store!(start, size, i + 1)
                j = si
                j_end = si + li
                while j < j_end
                    unsafe_store!(index, unsafe_load(index, Int(j) + 1), Int(size) + 1)
                    unsafe_store!(element, unsafe_load(element, Int(j) + 1), Int(size) + 1)
                    size += Int32(1)
                    j += Int32(1)
                end
                i += 1
            end
            unsafe_store!(start, size, md + 1)
        end
        return size_in
    else
        put = Int32(0)
        st = Int32(0)
        i = 0
        while i < md
            si = st
            st = unsafe_load(start, i + 2)
            li = unsafe_load(length_arr, i + 1)
            j = si
            j_end = si + li
            while j < j_end
                val = unsafe_load(element, Int(j) + 1)
                if abs(val) > remove_value
                    unsafe_store!(index, unsafe_load(index, Int(j) + 1), Int(put) + 1)
                    unsafe_store!(element, val, Int(put) + 1)
                    put += Int32(1)
                end
                j += Int32(1)
            end
            unsafe_store!(length_arr, put - unsafe_load(start, i + 1), i + 1)
            unsafe_store!(start, put, i + 2)
            i += 1
        end
        return put
    end
end

# CoinPackedMatrix::timesMajor (lines 1764-1775) — exact reverse major loop order
@inline function cpm_times_major_ptr!(
    major_dim::Int32,
    minor_dim::Int32,
    start::Ptr{Int32},
    length_arr::Ptr{Int32},
    index::Ptr{Int32},
    element::Ptr{Float64},
    x::Ptr{Float64},
    y::Ptr{Float64},
)::Nothing
    coin_zero_n!(y, Int(minor_dim))
    i = Int(major_dim) - 1
    while i >= 0
        xi = unsafe_load(x, i + 1)
        if xi != 0.0
            first = Int(unsafe_load(start, i + 1))
            last = first + Int(unsafe_load(length_arr, i + 1))
            j = first + 1
            while j <= last
                row = Int(unsafe_load(index, j)) + 1
                unsafe_store!(y, unsafe_load(y, row) + xi * unsafe_load(element, j), row)
                j += 1
            end
        end
        i -= 1
    end
    return nothing
end

# CoinPackedMatrix::timesMinor (lines 1795-1805) — exact reverse major loop order
@inline function cpm_times_minor_ptr!(
    major_dim::Int32,
    start::Ptr{Int32},
    length_arr::Ptr{Int32},
    index::Ptr{Int32},
    element::Ptr{Float64},
    x::Ptr{Float64},
    y::Ptr{Float64},
)::Nothing
    coin_zero_n!(y, Int(major_dim))
    i = Int(major_dim) - 1
    while i >= 0
        yi = 0.0
        first = Int(unsafe_load(start, i + 1))
        last = first + Int(unsafe_load(length_arr, i + 1))
        j = first + 1
        while j <= last
            yi += unsafe_load(x, Int(unsafe_load(index, j)) + 1) * unsafe_load(element, j)
            j += 1
        end
        unsafe_store!(y, yi, i + 1)
        i -= 1
    end
    return nothing
end

# CoinPackedMatrix::eliminateDuplicates (lines 535-572), given caller-owned mark buffer
@inline function cpm_eliminate_duplicates_ptr!(
    major_dim::Int32,
    minor_dim::Int32,
    start::Ptr{Int32},
    length_arr::Ptr{Int32},
    index::Ptr{Int32},
    element::Ptr{Float64},
    threshold::Float64,
    mark::Ptr{Int32},
)::Int32
    coin_fill_n!(mark, Int(minor_dim), Int32(-1))
    num_elim = Int32(0)
    md = Int(major_dim)
    i = 1
    while i <= md
        k = unsafe_load(start, i)
        end_j = k + unsafe_load(length_arr, i)
        j = k
        while j < end_j
            ind = Int(unsafe_load(index, Int(j) + 1)) + 1
            m = unsafe_load(mark, ind)
            if m == Int32(-1)
                unsafe_store!(mark, j, ind)
            else
                unsafe_store!(element, unsafe_load(element, Int(m) + 1) + unsafe_load(element, Int(j) + 1), Int(m) + 1)
                unsafe_store!(element, 0.0, Int(j) + 1)
            end
            j += Int32(1)
        end
        j = k
        while j < end_j
            ind = Int(unsafe_load(index, Int(j) + 1)) + 1
            unsafe_store!(mark, Int32(-1), ind)
            val = unsafe_load(element, Int(j) + 1)
            if abs(val) >= threshold
                unsafe_store!(element, val, Int(k) + 1)
                unsafe_store!(index, unsafe_load(index, Int(j) + 1), Int(k) + 1)
                k += Int32(1)
            end
            j += Int32(1)
        end
        num_elim += end_j - k
        unsafe_store!(length_arr, k - unsafe_load(start, i), i)
        i += 1
    end
    return num_elim
end

# Core loop of CoinPackedMatrix::cleanMatrix (lines 661-699), given caller-owned mark buffer
# Returns numberEliminated; writes compacted start_, length_, index_, element_ and sorts each major vector.
@inline function cpm_clean_matrix_core_ptr!(
    major_dim::Int32,
    minor_dim::Int32,
    start::Ptr{Int32},
    length_arr::Ptr{Int32},
    index::Ptr{Int32},
    element::Ptr{Float64},
    threshold::Float64,
    mark::Ptr{Int32},
)::Int32
    coin_fill_n!(mark, Int(minor_dim), Int32(-1))
    num_elim = Int32(0)
    n = Int32(0)
    md = Int(major_dim)
    i = 1
    while i <= md
        k = unsafe_load(start, i)
        unsafe_store!(start, n, i)
        end_j = k + unsafe_load(length_arr, i)
        j = k
        while j < end_j
            ind = Int(unsafe_load(index, Int(j) + 1)) + 1
            m = unsafe_load(mark, ind)
            if m == Int32(-1)
                unsafe_store!(mark, j, ind)
            else
                unsafe_store!(element, unsafe_load(element, Int(m) + 1) + unsafe_load(element, Int(j) + 1), Int(m) + 1)
                unsafe_store!(element, 0.0, Int(j) + 1)
            end
            j += Int32(1)
        end
        j = k
        while j < end_j
            ind = Int(unsafe_load(index, Int(j) + 1)) + 1
            unsafe_store!(mark, Int32(-1), ind)
            val = unsafe_load(element, Int(j) + 1)
            if abs(val) >= threshold
                unsafe_store!(element, val, Int(n) + 1)
                unsafe_store!(index, unsafe_load(index, Int(j) + 1), Int(n) + 1)
                n += Int32(1)
                k += Int32(1)
            end
            j += Int32(1)
        end
        num_elim += end_j - k
        si = unsafe_load(start, i)
        len_i = n - si
        unsafe_store!(length_arr, len_i, i)
        if len_i > Int32(1)
            coin_sort_2_std!(index + Int(si) * sizeof(Int32), element + Int(si) * sizeof(Float64), Int(len_i))
        end
        i += 1
    end
    unsafe_store!(start, n, md + 1)
    return num_elim
end

# =============================================================================
# 2. Standalone Julia `CoinPackedMatrix` (0-based indices & starts)
# =============================================================================

mutable struct CoinPackedMatrix
    col_ordered::Bool
    extra_gap::Float64
    extra_major::Float64
    element::Vector{Float64}
    index::Vector{Int32}
    start::Vector{Int32}
    length::Vector{Int32}
    major_dim::Int32
    minor_dim::Int32
    size::Int32
end

function CoinPackedMatrix(col_ordered::Bool = true, extra_gap::Float64 = 0.0, extra_major::Float64 = 0.0)
    return CoinPackedMatrix(
        col_ordered, extra_gap, extra_major,
        Float64[], Int32[], Int32[0], Int32[],
        Int32(0), Int32(0), Int32(0),
    )
end

function CoinPackedMatrix(
    col_ordered::Bool,
    minor_dim::Integer,
    major_dim::Integer,
    element::Vector{Float64},
    index::Vector{Int32},
    start::Vector{Int32},
    length_arr::Vector{Int32};
    extra_gap::Float64 = 0.0,
    extra_major::Float64 = 0.0,
)
    sz = Int32(sum(Int, length_arr))
    return CoinPackedMatrix(
        col_ordered, extra_gap, extra_major,
        copy(element), copy(index), copy(start), copy(length_arr),
        Int32(major_dim), Int32(minor_dim), sz,
    )
end

@inline is_col_ordered(m::CoinPackedMatrix)::Bool = m.col_ordered
@inline get_num_rows(m::CoinPackedMatrix)::Int32 = m.col_ordered ? m.minor_dim : m.major_dim
@inline get_num_cols(m::CoinPackedMatrix)::Int32 = m.col_ordered ? m.major_dim : m.minor_dim
@inline get_num_elements(m::CoinPackedMatrix)::Int32 = m.size

function count_ortho_length!(ortho::Vector{Int32}, m::CoinPackedMatrix)::Nothing
    resize!(ortho, Int(m.minor_dim))
    GC.@preserve m ortho cpm_count_ortho_length_ptr!(
        m.major_dim, m.minor_dim,
        pointer(m.start), pointer(m.length), pointer(m.index), pointer(ortho),
    )
    return nothing
end

function reverse_ordered_copy_of!(dst::CoinPackedMatrix, rhs::CoinPackedMatrix)::Nothing
    if dst === rhs
        reverse_ordering!(dst)
        return nothing
    end
    dst.col_ordered = !rhs.col_ordered
    dst.major_dim = rhs.minor_dim
    dst.minor_dim = rhs.major_dim
    dst.size = rhs.size
    md = Int(dst.major_dim)
    resize!(dst.start, md + 1)
    resize!(dst.length, md)
    if dst.size == Int32(0)
        fill!(dst.start, Int32(0))
        fill!(dst.length, Int32(0))
        Base.empty!(dst.index)
        Base.empty!(dst.element)
        return nothing
    end
    count_ortho_length!(dst.length, rhs)
    dst.start[1] = Int32(0)
    for i in 1:md
        @inbounds dst.start[i + 1] = dst.start[i] + dst.length[i]
    end
    nnz = Int(dst.start[md + 1])
    resize!(dst.index, nnz)
    resize!(dst.element, nnz)
    GC.@preserve rhs dst cpm_reverse_scatter_ptr!(
        rhs.major_dim, pointer(rhs.start), pointer(rhs.length), pointer(rhs.index), pointer(rhs.element),
        dst.major_dim, pointer(dst.start), pointer(dst.length), pointer(dst.index), pointer(dst.element),
    )
    return nothing
end

function reverse_ordering!(m::CoinPackedMatrix)::Nothing
    tmp = CoinPackedMatrix(!m.col_ordered, m.extra_gap, m.extra_major)
    reverse_ordered_copy_of!(tmp, m)
    m.col_ordered = tmp.col_ordered
    m.element = tmp.element
    m.index = tmp.index
    m.start = tmp.start
    m.length = tmp.length
    m.major_dim = tmp.major_dim
    m.minor_dim = tmp.minor_dim
    m.size = tmp.size
    return nothing
end

function remove_gaps!(m::CoinPackedMatrix, remove_value::Float64 = -1.0)::Nothing
    m.size = GC.@preserve m cpm_remove_gaps_ptr!(
        m.major_dim, m.size, pointer(m.start), pointer(m.length),
        pointer(m.index), pointer(m.element), remove_value,
    )
    return nothing
end

function eliminate_duplicates!(m::CoinPackedMatrix, threshold::Float64)::Int32
    mark = Vector{Int32}(undef, Int(m.minor_dim))
    elim = GC.@preserve m mark cpm_eliminate_duplicates_ptr!(
        m.major_dim, m.minor_dim, pointer(m.start), pointer(m.length),
        pointer(m.index), pointer(m.element), threshold, pointer(mark),
    )
    m.size -= elim
    return elim
end

function clean_matrix!(m::CoinPackedMatrix, threshold::Float64 = 1e-20)::Int32
    if m.major_dim == Int32(0)
        m.extra_gap = 0.0
        m.extra_major = 0.0
        return Int32(0)
    end
    mark = Vector{Int32}(undef, Int(m.minor_dim))
    elim = GC.@preserve m mark cpm_clean_matrix_core_ptr!(
        m.major_dim, m.minor_dim, pointer(m.start), pointer(m.length),
        pointer(m.index), pointer(m.element), threshold, pointer(mark),
    )
    m.size -= elim
    m.extra_gap = 0.0
    m.extra_major = 0.0
    resize!(m.index, Int(m.size))
    resize!(m.element, Int(m.size))
    return elim
end

function times!(y::Vector{Float64}, m::CoinPackedMatrix, x::Vector{Float64})::Nothing
    if m.col_ordered
        GC.@preserve m x y cpm_times_major_ptr!(
            m.major_dim, m.minor_dim, pointer(m.start), pointer(m.length),
            pointer(m.index), pointer(m.element), pointer(x), pointer(y),
        )
    else
        GC.@preserve m x y cpm_times_minor_ptr!(
            m.major_dim, pointer(m.start), pointer(m.length),
            pointer(m.index), pointer(m.element), pointer(x), pointer(y),
        )
    end
    return nothing
end

function transpose_times!(y::Vector{Float64}, m::CoinPackedMatrix, x::Vector{Float64})::Nothing
    if m.col_ordered
        GC.@preserve m x y cpm_times_minor_ptr!(
            m.major_dim, pointer(m.start), pointer(m.length),
            pointer(m.index), pointer(m.element), pointer(x), pointer(y),
        )
    else
        GC.@preserve m x y cpm_times_major_ptr!(
            m.major_dim, m.minor_dim, pointer(m.start), pointer(m.length),
            pointer(m.index), pointer(m.element), pointer(x), pointer(y),
        )
    end
    return nothing
end

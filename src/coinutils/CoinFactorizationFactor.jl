# Faithful translation of CoinUtils Markowitz Sparse LU Factorization (B5c):
#   - CoinFactorization1.cpp: preProcess, factor (singular), pivotRowSingleton,
#     pivotColumnSingleton, pivotOneOtherRow, getColumnSpace, getRowSpace, cleanup
#   - CoinFactorization2.cpp: factorSparse, factorSparseSmall, factorSparseLarge,
#     factorDense, separateLinks
#   - CoinFactorization.hpp:  addLink, deleteLink, pivot<T>
#   - CoinFactorization4.cpp: goSparse, checkSparse
# Zero-allocation, hardware arm64 FMA (`_fmsub`), `--trim=safe` compatible.

const CF_LARGELIMIT  = Int32(65530)
const CF_SMALL_SET   = Int32(65531)
const CF_SMALL_UNSET = Int32(65532)
const CF_LARGE_SET   = typemax(Int32) - Int32(10)
const CF_LARGE_UNSET = CF_LARGE_SET + Int32(1)
const CF_BITS_PER_INT  = Int32(32)
const CF_SHIFT_PER_INT = Int32(5)
const CF_MASK_PER_INT  = Int32(0x1f)
const CF_COMPRESS_SENTINEL = typemax(Int32) >> 1

# Compile-time byte offsets into CoinFactorizationFactorStatePod for zero-alloc write-back
const _FS_OFF_workArea2          = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :workArea2))
const _FS_OFF_areaFactor         = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :areaFactor))
const _FS_OFF_numberGoodU        = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :numberGoodU))
const _FS_OFF_numberGoodL        = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :numberGoodL))
const _FS_OFF_numberSlacks       = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :numberSlacks))
const _FS_OFF_numberU            = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :numberU))
const _FS_OFF_maximumU           = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :maximumU))
const _FS_OFF_numberL            = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :numberL))
const _FS_OFF_baseL              = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :baseL))
const _FS_OFF_lengthL            = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :lengthL))
const _FS_OFF_numberR            = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :numberR))
const _FS_OFF_lengthR            = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :lengthR))
const _FS_OFF_lengthAreaR        = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :lengthAreaR))
const _FS_OFF_totalElements      = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :totalElements))
const _FS_OFF_factorElements     = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :factorElements))
const _FS_OFF_numberDense        = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :numberDense))
const _FS_OFF_denseThreshold     = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :denseThreshold))
const _FS_OFF_numberCompressions = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :numberCompressions))
const _FS_OFF_status             = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :status))
const _FS_OFF_workArea2Size      = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :workArea2Size))
const _FS_OFF_sparseThreshold    = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :sparseThreshold))
const _FS_OFF_sparseThreshold2   = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :sparseThreshold2))
const _FS_OFF_useExtraR          = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :useExtraR))
const _FS_OFF_freeDense          = fieldoffset(CoinFactorizationFactorStatePod, Base.fieldindex(CoinFactorizationFactorStatePod, :freeDense))

@inline function _fs_store_i32!(st_ptr::Ptr{CoinFactorizationFactorStatePod}, off::UInt, val::Int32)::Nothing
    unsafe_store!(Ptr{Int32}(UInt(st_ptr) + off), val)
    return nothing
end

@inline function _fs_store_f64!(st_ptr::Ptr{CoinFactorizationFactorStatePod}, off::UInt, val::Float64)::Nothing
    unsafe_store!(Ptr{Float64}(UInt(st_ptr) + off), val)
    return nothing
end

@inline function _fs_store_u32ptr!(st_ptr::Ptr{CoinFactorizationFactorStatePod}, off::UInt, val::Ptr{UInt32})::Nothing
    unsafe_store!(Ptr{Ptr{UInt32}}(UInt(st_ptr) + off), val)
    return nothing
end

struct _CFTimeval
    tv_sec::Int64
    tv_usec::Int32
    _pad::Int32
end

@inline function _cf_wallclock_time()::Float64
    tv = Ref{_CFTimeval}(_CFTimeval(Int64(0), Int32(0), Int32(0)))
    ccall(:gettimeofday, Cint, (Ref{_CFTimeval}, Ptr{Cvoid}), tv, C_NULL)
    t = tv[]
    return Float64(t.tv_sec) + Float64(t.tv_usec) / 1000000.0
end

# =============================================================================
# Linked list helpers (CoinFactorization.hpp:804-842)
# =============================================================================

@inline function _cf_add_link!(
    first_count::Ptr{Int32},
    next_count::Ptr{Int32},
    last_count::Ptr{Int32},
    index::Int32,
    count::Int32,
)::Nothing
    cnt1 = Int(count) + 1
    idx1 = Int(index) + 1
    next = unsafe_load(first_count, cnt1)
    unsafe_store!(last_count, Int32(-2) - count, idx1)
    if next < Int32(0)
        unsafe_store!(first_count, index, cnt1)
        unsafe_store!(next_count, Int32(-1), idx1)
    else
        unsafe_store!(first_count, index, cnt1)
        unsafe_store!(next_count, next, idx1)
        unsafe_store!(last_count, index, Int(next) + 1)
    end
    return nothing
end

@inline function _cf_delete_link!(
    first_count::Ptr{Int32},
    next_count::Ptr{Int32},
    last_count::Ptr{Int32},
    index::Int32,
)::Nothing
    idx1 = Int(index) + 1
    next = unsafe_load(next_count, idx1)
    last = unsafe_load(last_count, idx1)
    if last >= Int32(0)
        unsafe_store!(next_count, next, Int(last) + 1)
    else
        cnt = -last - Int32(2)
        unsafe_store!(first_count, next, Int(cnt) + 1)
    end
    if next >= Int32(0)
        unsafe_store!(last_count, last, Int(next) + 1)
    end
    unsafe_store!(next_count, Int32(-2), idx1)
    unsafe_store!(last_count, Int32(-2), idx1)
    return nothing
end

# =============================================================================
# separateLinks (CoinFactorization2.cpp:780-829)
# =============================================================================

function cf_separate_links!(
    first_count::Ptr{Int32},
    next_count::Ptr{Int32},
    last_count::Ptr{Int32},
    number_rows::Int32,
    count::Int32,
    rows_first::Bool,
)::Nothing
    next = unsafe_load(first_count, Int(count) + 1)
    first_row = Int32(-1)
    first_col = Int32(-1)
    last_row = Int32(-1)
    last_col = Int32(-1)
    while next >= Int32(0)
        next2 = unsafe_load(next_count, Int(next) + 1)
        if next >= number_rows
            unsafe_store!(next_count, Int32(-1), Int(next) + 1)
            if first_col >= Int32(0)
                unsafe_store!(last_count, last_col, Int(next) + 1)
                unsafe_store!(next_count, next, Int(last_col) + 1)
            else
                unsafe_store!(last_count, Int32(-2) - count, Int(next) + 1)
                first_col = next
            end
            last_col = next
        else
            if first_row >= Int32(0)
                unsafe_store!(last_count, last_row, Int(next) + 1)
                unsafe_store!(next_count, next, Int(last_row) + 1)
            else
                unsafe_store!(last_count, Int32(-2) - count, Int(next) + 1)
                first_row = next
            end
            last_row = next
        end
        next = next2
    end
    if rows_first && first_row >= Int32(0)
        unsafe_store!(first_count, first_row, Int(count) + 1)
        unsafe_store!(next_count, first_col, Int(last_row) + 1)
        if first_col >= Int32(0)
            unsafe_store!(last_count, last_row, Int(first_col) + 1)
        end
    elseif first_row < Int32(0)
        unsafe_store!(first_count, first_col, Int(count) + 1)
    elseif first_col >= Int32(0)
        unsafe_store!(first_count, first_col, Int(count) + 1)
        unsafe_store!(next_count, first_row, Int(last_col) + 1)
        if first_row >= Int32(0)
            unsafe_store!(last_count, last_col, Int(first_row) + 1)
        end
    end
    return nothing
end

# =============================================================================
# getColumnSpace (CoinFactorization1.cpp:1432-1549)
# =============================================================================

function cf_get_column_space!(
    st::CoinFactorizationFactorStatePod,
    i_column::Int32,
    extra_needed::Int32,
    n_comp::Int32,
    status::Int32,
)::Tuple{Bool,Int32,Int32}
    nic = st.numberInColumn
    nicp = st.numberInColumnPlus
    next_col = st.nextColumn
    last_col = st.lastColumn
    start_col_u = st.startColumnU
    el_u = st.elementU
    idx_row_u = st.indexRowU
    mce = st.maximumColumnsExtra
    mce1 = Int(mce) + 1
    lau = st.lengthAreaU

    ic1 = Int(i_column) + 1
    number = unsafe_load(nicp, ic1) + unsafe_load(nic, ic1)
    space = lau - unsafe_load(start_col_u, mce1)

    if space < extra_needed + number + Int32(4)
        # Compression
        col = unsafe_load(next_col, mce1)
        put = Int32(0)
        while col != mce
            c1 = Int(col) + 1
            sc = unsafe_load(start_col_u, c1)
            nc_val = unsafe_load(nic, c1)
            if sc >= Int32(0)
                ncp_val = unsafe_load(nicp, c1)
                g = Int(sc - ncp_val)
                ge = Int(sc + nc_val)
                unsafe_store!(start_col_u, put + ncp_val, c1)
            else
                g = Int(-sc)
                ge = g + Int(nc_val)
                unsafe_store!(start_col_u, -put, c1)
            end
            i = g
            while i < ge
                p1 = Int(put) + 1
                unsafe_store!(idx_row_u, unsafe_load(idx_row_u, i + 1), p1)
                unsafe_store!(el_u, unsafe_load(el_u, i + 1), p1)
                put += Int32(1)
                i += 1
            end
            col = unsafe_load(next_col, c1)
        end
        n_comp += Int32(1)
        unsafe_store!(start_col_u, put, mce1)
        space = lau - put
        if extra_needed == CF_COMPRESS_SENTINEL
            return true, n_comp, status
        end
        if space < extra_needed + number + Int32(2)
            return false, n_comp, Int32(-99)
        end
    end

    put = unsafe_load(start_col_u, mce1)
    nxt = unsafe_load(next_col, ic1)
    lst = unsafe_load(last_col, ic1)

    if extra_needed != Int32(0) || nxt != mce
        # Remove from list
        unsafe_store!(next_col, nxt, Int(lst) + 1)
        unsafe_store!(last_col, lst, Int(nxt) + 1)
        # Insert at end
        lst2 = unsafe_load(last_col, mce1)
        unsafe_store!(next_col, i_column, Int(lst2) + 1)
        unsafe_store!(last_col, i_column, mce1)
        unsafe_store!(last_col, lst2, ic1)
        unsafe_store!(next_col, mce, ic1)
        # Move
        ncp_ic = unsafe_load(nicp, ic1)
        g = Int(unsafe_load(start_col_u, ic1) - ncp_ic)
        unsafe_store!(start_col_u, put + ncp_ic, ic1)
        num_i = Int(number)
        put_i = Int(put)
        if num_i < 50
            i = 0
            if (num_i & 1) != 0
                unsafe_store!(el_u, unsafe_load(el_u, g + 1), put_i + 1)
                unsafe_store!(idx_row_u, unsafe_load(idx_row_u, g + 1), put_i + 1)
                i = 1
            end
            while i < num_i
                v0 = unsafe_load(el_u, g + i + 1)
                v1 = unsafe_load(el_u, g + i + 2)
                r0 = unsafe_load(idx_row_u, g + i + 1)
                r1 = unsafe_load(idx_row_u, g + i + 2)
                unsafe_store!(el_u, v0, put_i + i + 1)
                unsafe_store!(el_u, v1, put_i + i + 2)
                unsafe_store!(idx_row_u, r0, put_i + i + 1)
                unsafe_store!(idx_row_u, r1, put_i + i + 2)
                i += 2
            end
        else
            coin_memcpy_n!(idx_row_u + g * sizeof(Int32), num_i, idx_row_u + put_i * sizeof(Int32))
            coin_memcpy_n!(el_u + g * sizeof(Float64), num_i, el_u + put_i * sizeof(Float64))
        end
        put += number
        new_end = put + extra_needed + Int32(2)
        unsafe_store!(start_col_u, new_end, mce1)
        if new_end > lau
            return false, n_comp, status
        end
    else
        lst1 = Int(lst) + 1
        unsafe_store!(start_col_u, unsafe_load(start_col_u, lst1) + unsafe_load(nic, lst1), mce1)
    end
    return true, n_comp, status
end

# =============================================================================
# getRowSpace (CoinFactorization1.cpp:1554-1621)
# =============================================================================

function cf_get_row_space!(
    st::CoinFactorizationFactorStatePod,
    i_row::Int32,
    extra_needed::Int32,
    n_comp::Int32,
    status::Int32,
)::Tuple{Bool,Int32,Int32}
    nir = st.numberInRow
    start_row_u = st.startRowU
    next_row = st.nextRow
    last_row = st.lastRow
    idx_col_u = st.indexColumnU
    mre = st.maximumRowsExtra
    mre1 = Int(mre) + 1
    lau = st.lengthAreaU

    ir1 = Int(i_row) + 1
    number = unsafe_load(nir, ir1)
    space = lau - unsafe_load(start_row_u, mre1)

    if space < extra_needed + number + Int32(2)
        # Compression
        r = unsafe_load(next_row, mre1)
        put = Int32(0)
        while r != mre
            r1 = Int(r) + 1
            g = Int(unsafe_load(start_row_u, r1))
            ge = g + Int(unsafe_load(nir, r1))
            unsafe_store!(start_row_u, put, r1)
            i = g
            while i < ge
                unsafe_store!(idx_col_u, unsafe_load(idx_col_u, i + 1), Int(put) + 1)
                put += Int32(1)
                i += 1
            end
            r = unsafe_load(next_row, r1)
        end
        n_comp += Int32(1)
        unsafe_store!(start_row_u, put, mre1)
        space = lau - put
        if space < extra_needed + number + Int32(2)
            return false, n_comp, Int32(-99)
        end
    end

    put = unsafe_load(start_row_u, mre1)
    nxt = unsafe_load(next_row, ir1)
    lst = unsafe_load(last_row, ir1)
    # Remove from list
    unsafe_store!(next_row, nxt, Int(lst) + 1)
    unsafe_store!(last_row, lst, Int(nxt) + 1)
    # Insert at end
    lst2 = unsafe_load(last_row, mre1)
    unsafe_store!(next_row, i_row, Int(lst2) + 1)
    unsafe_store!(last_row, i_row, mre1)
    unsafe_store!(last_row, lst2, ir1)
    unsafe_store!(next_row, mre, ir1)
    # Move
    g = Int(unsafe_load(start_row_u, ir1))
    unsafe_store!(start_row_u, put, ir1)
    rem = Int(number)
    put_i = Int(put)
    while rem > 0
        rem -= 1
        unsafe_store!(idx_col_u, unsafe_load(idx_col_u, g + 1), put_i + 1)
        put_i += 1
        g += 1
    end
    unsafe_store!(start_row_u, Int32(put_i) + extra_needed + Int32(4), mre1)
    return true, n_comp, status
end

# =============================================================================
# preProcess (CoinFactorization1.cpp:731-1011)
# =============================================================================

function cf_pre_process_core!(
    st::CoinFactorizationFactorStatePod,
    state::Int32,
    total_elements::Int32,
)::Int32
    idx_row = st.indexRowU
    idx_col = st.indexColumnU
    el = st.elementU
    num_el = st.lengthU
    nir = st.numberInRow
    nic = st.numberInColumn
    nicp = st.numberInColumnPlus
    start_row = st.startRowU
    start_col = st.startColumnU
    nr = Int(st.numberRows)
    nc = Int(st.numberColumns)
    mre = st.maximumRowsExtra
    mce = st.maximumColumnsExtra

    if state < Int32(4)
        total_elements = num_el
    end

    if state == Int32(0)
        # case 0: counts
        coin_zero_n!(nir, nr + 1)
        coin_zero_n!(nic, Int(mce) + 1)
        i = 0
        nel = Int(num_el)
        while i < nel
            r1 = Int(unsafe_load(idx_row, i + 1)) + 1
            c1 = Int(unsafe_load(idx_col, i + 1)) + 1
            unsafe_store!(nir, unsafe_load(nir, r1) + Int32(1), r1)
            unsafe_store!(nic, unsafe_load(nic, c1) + Int32(1), c1)
            i += 1
        end
    end

    if state == Int32(0) || state == Int32(-1) || state == Int32(1)
        # case -1 / 1: sort
        pos = Int32(0)
        ic = 0
        while ic < nc
            pos += unsafe_load(nic, ic + 1)
            unsafe_store!(start_col, pos, ic + 1)
            ic += 1
        end
        k = Int(num_el) - 1
        while k >= 0
            col = unsafe_load(idx_col, k + 1)
            if col >= Int32(0)
                val = unsafe_load(el, k + 1)
                row = unsafe_load(idx_row, k + 1)
                unsafe_store!(idx_col, Int32(-1), k + 1)
                while true
                    c1 = Int(col) + 1
                    look = unsafe_load(start_col, c1) - Int32(1)
                    unsafe_store!(start_col, look, c1)
                    l1 = Int(look) + 1
                    val_save = unsafe_load(el, l1)
                    col_save = unsafe_load(idx_col, l1)
                    row_save = unsafe_load(idx_row, l1)
                    unsafe_store!(el, val, l1)
                    unsafe_store!(idx_row, row, l1)
                    unsafe_store!(idx_col, Int32(-1), l1)
                    if col_save >= Int32(0)
                        col = col_save
                        val = val_save
                        row = row_save
                    else
                        break
                    end
                end
            end
            k -= 1
        end
    end

    if state == Int32(0) || state == Int32(-1) || state == Int32(1) || state == Int32(2)
        # case 2: move largest in column to beginning and do row part
        pos = Int32(0)
        ir = 0
        while ir < nr
            unsafe_store!(start_row, pos, ir + 1)
            pos += unsafe_load(nir, ir + 1)
            ir += 1
        end
        coin_zero_n!(nir, nr)
        ic = 0
        while ic < nc
            num = Int(unsafe_load(nic, ic + 1))
            if num != 0
                first = Int(unsafe_load(start_col, ic + 1))
                largest = first
                ir_save = unsafe_load(idx_row, first + 1)
                val_save = unsafe_load(el, first + 1)
                val_largest = abs(val_save)
                rs1 = Int(ir_save) + 1
                look = unsafe_load(nir, rs1)
                unsafe_store!(nir, look + Int32(1), rs1)
                unsafe_store!(idx_col, Int32(ic), Int(unsafe_load(start_row, rs1) + look) + 1)
                k = first + 1
                k_end = first + num
                while k < k_end
                    r = unsafe_load(idx_row, k + 1)
                    r1 = Int(r) + 1
                    lk = unsafe_load(nir, r1)
                    unsafe_store!(nir, lk + Int32(1), r1)
                    unsafe_store!(idx_col, Int32(ic), Int(unsafe_load(start_row, r1) + lk) + 1)
                    v = unsafe_load(el, k + 1)
                    va = abs(v)
                    if va > val_largest
                        val_largest = va
                        largest = k
                    end
                    k += 1
                end
                unsafe_store!(idx_row, unsafe_load(idx_row, largest + 1), first + 1)
                unsafe_store!(el, unsafe_load(el, largest + 1), first + 1)
                unsafe_store!(idx_row, ir_save, largest + 1)
                unsafe_store!(el, val_save, largest + 1)
            end
            ic += 1
        end
    end

    if state == Int32(0) || state == Int32(-1) || state == Int32(1) || state == Int32(2) || state == Int32(3)
        # case 3: links and initialize pivots
        last_row = st.lastRow
        next_row = st.nextRow
        last_col = st.lastColumn
        next_col = st.nextColumn
        fc = st.firstCount
        ncnt = st.nextCount
        lcnt = st.lastCount

        coin_fill_n!(fc, Int(st.biggerDimension) + 2, Int32(-1))
        coin_fill_n!(st.pivotColumn, nc, Int32(-1))
        coin_zero_n!(nicp, Int(mce) + 1)

        ir = 0
        while ir < nr
            unsafe_store!(last_row, Int32(ir - 1), ir + 1)
            unsafe_store!(next_row, Int32(ir + 1), ir + 1)
            _cf_add_link!(fc, ncnt, lcnt, Int32(ir), unsafe_load(nir, ir + 1))
            ir += 1
        end
        mre1 = Int(mre) + 1
        unsafe_store!(last_row, Int32(nr - 1), mre1)
        unsafe_store!(next_row, Int32(0), mre1)
        if nr > 0
            unsafe_store!(last_row, mre, 1)
            unsafe_store!(next_row, mre, nr)
        end
        unsafe_store!(start_row, num_el, mre1)

        ic = 0
        while ic < nc
            unsafe_store!(last_col, Int32(ic - 1), ic + 1)
            unsafe_store!(next_col, Int32(ic + 1), ic + 1)
            _cf_add_link!(fc, ncnt, lcnt, Int32(ic + nr), unsafe_load(nic, ic + 1))
            ic += 1
        end
        mce1 = Int(mce) + 1
        unsafe_store!(last_col, Int32(nc - 1), mce1)
        unsafe_store!(next_col, Int32(0), mce1)
        if nc > 0
            unsafe_store!(last_col, mce, 1)
            unsafe_store!(next_col, mce, nc)
        else
            unsafe_store!(last_col, mce, 1)
        end
        unsafe_store!(start_col, num_el, mce1)
    elseif state == Int32(4)
        # case 4: move largest in column to beginning (after slacks removed)
        piv_reg = st.pivotRegion
        ir = 0
        while ir < nr
            if unsafe_load(nir, ir + 1) >= Int32(0)
                unsafe_store!(nir, Int32(0), ir + 1)
            end
            ir += 1
        end
        ic = 0
        while ic < nc
            num = Int(unsafe_load(nic, ic + 1))
            if num != 0
                orig_first = Int(unsafe_load(start_col, ic + 1))
                first = orig_first
                largest = -1
                val_largest = -1.0
                n_other = 0
                k = first
                k_end = first + num
                while k < k_end
                    r = unsafe_load(idx_row, k + 1)
                    v = unsafe_load(el, k + 1)
                    r1 = Int(r) + 1
                    cur_nir = unsafe_load(nir, r1)
                    if cur_nir >= Int32(0)
                        unsafe_store!(nir, cur_nir + Int32(1), r1)
                        va = abs(v)
                        if va > val_largest
                            val_largest = va
                            largest = n_other
                        end
                        unsafe_store!(start_row, r, n_other + 1)
                        unsafe_store!(piv_reg, v, n_other + 1)
                        n_other += 1
                    else
                        unsafe_store!(idx_row, r, first + 1)
                        unsafe_store!(el, v, first + 1)
                        first += 1
                    end
                    k += 1
                end
                unsafe_store!(nicp, Int32(first - orig_first), ic + 1)
                unsafe_store!(start_col, Int32(first), ic + 1)
                if largest >= 0
                    unsafe_store!(idx_row, unsafe_load(start_row, largest + 1), first + 1)
                    unsafe_store!(el, unsafe_load(piv_reg, largest + 1), first + 1)
                    first += 1
                end
                k = 0
                while k < n_other
                    if k != largest
                        unsafe_store!(idx_row, unsafe_load(start_row, k + 1), first + 1)
                        unsafe_store!(el, unsafe_load(piv_reg, k + 1), first + 1)
                        first += 1
                    end
                    k += 1
                end
                unsafe_store!(nic, Int32(first) - unsafe_load(start_col, ic + 1), ic + 1)
            end
            ic += 1
        end
        # Row part
        pos = Int32(0)
        ir = 0
        while ir < nr
            unsafe_store!(start_row, pos, ir + 1)
            n = unsafe_load(nir, ir + 1)
            if n > Int32(0)
                unsafe_store!(nir, Int32(0), ir + 1)
                pos += n
            end
            ir += 1
        end
        ic = 0
        while ic < nc
            num = Int(unsafe_load(nic, ic + 1))
            if num != 0
                first = Int(unsafe_load(start_col, ic + 1))
                k = first
                k_end = first + num
                while k < k_end
                    r1 = Int(unsafe_load(idx_row, k + 1)) + 1
                    lk = unsafe_load(nir, r1)
                    unsafe_store!(nir, lk + Int32(1), r1)
                    unsafe_store!(idx_col, Int32(ic), Int(unsafe_load(start_row, r1) + lk) + 1)
                    k += 1
                end
            end
            ic += 1
        end
        # Modified 3
        last_col = st.lastColumn
        next_col = st.nextColumn
        fc = st.firstCount
        ncnt = st.nextCount
        lcnt = st.lastCount
        slack_val = st.slackValue

        num_good = 0
        unsafe_store!(st.startColumnL, Int32(0), 1)
        ir = 0
        while ir < nr
            if unsafe_load(nir, ir + 1) < Int32(0)
                unsafe_store!(nir, Int32(0), ir + 1)
                unsafe_store!(piv_reg, slack_val, num_good + 1)
                num_good += 1
            end
            ir += 1
        end
        ic = 0
        while ic < nc
            unsafe_store!(last_col, Int32(ic - 1), ic + 1)
            unsafe_store!(next_col, Int32(ic + 1), ic + 1)
            num = unsafe_load(nic, ic + 1)
            _cf_delete_link!(fc, ncnt, lcnt, Int32(ic + nr))
            _cf_add_link!(fc, ncnt, lcnt, Int32(ic + nr), num)
            ic += 1
        end
        mce1 = Int(mce) + 1
        unsafe_store!(last_col, Int32(nc - 1), mce1)
        unsafe_store!(next_col, Int32(0), mce1)
        unsafe_store!(last_col, mce, 1)
        if nc > 0
            unsafe_store!(next_col, mce, nc)
        end
        unsafe_store!(start_col, num_el, mce1)
    end
    return total_elements
end

function cf_pre_process!(st_ptr::Ptr{CoinFactorizationFactorStatePod}, state::Int32)::Nothing
    st = unsafe_load(st_ptr)
    tot = cf_pre_process_core!(st, state, st.totalElements)
    _fs_store_i32!(st_ptr, _FS_OFF_totalElements, tot)
    return nothing
end

# =============================================================================
# pivotRowSingleton (CoinFactorization1.cpp:1202-1306)
# =============================================================================

function cf_pivot_row_singleton!(
    st::CoinFactorizationFactorStatePod,
    pivot_row::Int32,
    pivot_column::Int32,
    length_l::Int32,
    number_good_l::Int32,
    number_good_u::Int32,
)::Tuple{Bool,Int32,Int32}
    start_col_u = st.startColumnU
    nir = st.numberInRow
    nic = st.numberInColumn
    idx_row_u = st.indexRowU
    start_row_u = st.startRowU
    next_row = st.nextRow
    last_row = st.lastRow
    fc = st.firstCount
    ncnt = st.nextCount
    lcnt = st.lastCount

    pc1 = Int(pivot_column) + 1
    sc = Int(unsafe_load(start_col_u, pc1))
    num_do_col = Int(unsafe_load(nic, pc1)) - 1
    end_col = sc + num_do_col + 1

    piv_row_pos = sc
    while unsafe_load(idx_row_u, piv_row_pos + 1) != pivot_row
        piv_row_pos += 1
    end

    l = Int(length_l)
    if l + num_do_col > Int(st.lengthAreaL)
        return false, length_l, number_good_l
    end

    start_col_l = st.startColumnL
    el_l = st.elementL
    idx_row_l = st.indexRowL
    unsafe_store!(start_col_l, Int32(l), Int(number_good_l) + 1)
    number_good_l += Int32(1)
    unsafe_store!(start_col_l, Int32(l + num_do_col), Int(number_good_l) + 1)
    length_l += Int32(num_do_col)

    el_u = st.elementU
    piv_mult = 1.0 / unsafe_load(el_u, piv_row_pos + 1)
    unsafe_store!(st.pivotRegion, piv_mult, Int(number_good_u) + 1)

    idx_col_u = st.indexColumnU
    i = sc
    while i < piv_row_pos
        ir = unsafe_load(idx_row_u, i + 1)
        unsafe_store!(idx_row_l, ir, l + 1)
        unsafe_store!(el_l, unsafe_load(el_u, i + 1) * piv_mult, l + 1)
        l += 1
        ir1 = Int(ir) + 1
        s = Int(unsafe_load(start_row_u, ir1))
        inir = unsafe_load(nir, ir1)
        e = s + Int(inir)
        w = s
        while unsafe_load(idx_col_u, w + 1) != pivot_column
            w += 1
        end
        unsafe_store!(idx_col_u, unsafe_load(idx_col_u, e), w + 1)
        inir -= Int32(1)
        unsafe_store!(nir, inir, ir1)
        _cf_delete_link!(fc, ncnt, lcnt, ir)
        _cf_add_link!(fc, ncnt, lcnt, ir, inir)
        i += 1
    end
    i = piv_row_pos + 1
    while i < end_col
        ir = unsafe_load(idx_row_u, i + 1)
        unsafe_store!(idx_row_l, ir, l + 1)
        unsafe_store!(el_l, unsafe_load(el_u, i + 1) * piv_mult, l + 1)
        l += 1
        ir1 = Int(ir) + 1
        s = Int(unsafe_load(start_row_u, ir1))
        inir = unsafe_load(nir, ir1)
        e = s + Int(inir)
        w = s
        while unsafe_load(idx_col_u, w + 1) != pivot_column
            w += 1
        end
        unsafe_store!(idx_col_u, unsafe_load(idx_col_u, e), w + 1)
        inir -= Int32(1)
        unsafe_store!(nir, inir, ir1)
        _cf_delete_link!(fc, ncnt, lcnt, ir)
        _cf_add_link!(fc, ncnt, lcnt, ir, inir)
        i += 1
    end

    unsafe_store!(nic, Int32(0), pc1)
    pr1 = Int(pivot_row) + 1
    unsafe_store!(nir, Int32(0), pr1)
    _cf_delete_link!(fc, ncnt, lcnt, pivot_row)
    _cf_delete_link!(fc, ncnt, lcnt, pivot_column + st.numberRows)

    nxt = unsafe_load(next_row, pr1)
    lst = unsafe_load(last_row, pr1)
    unsafe_store!(next_row, nxt, Int(lst) + 1)
    unsafe_store!(last_row, lst, Int(nxt) + 1)
    unsafe_store!(last_row, Int32(-2), pr1)
    unsafe_store!(next_row, number_good_u, pr1)
    return true, length_l, number_good_l
end

# =============================================================================
# pivotColumnSingleton (CoinFactorization1.cpp:1309-1427)
# =============================================================================

function cf_pivot_column_singleton!(
    st::CoinFactorizationFactorStatePod,
    pivot_row::Int32,
    pivot_column::Int32,
    length_l::Int32,
    number_good_l::Int32,
    number_good_u::Int32,
)::Tuple{Bool,Int32}
    nir = st.numberInRow
    nic = st.numberInColumn
    nicp = st.numberInColumnPlus
    start_col_u = st.startColumnU
    start_row_u = st.startRowU
    idx_col_u = st.indexColumnU
    idx_row_u = st.indexRowU
    save_col = st.saveColumn
    next_row = st.nextRow
    last_row = st.lastRow
    el_u = st.elementU
    fc = st.firstCount
    ncnt = st.nextCount
    lcnt = st.lastCount
    nr = st.numberRows

    pr1 = Int(pivot_row) + 1
    pc1 = Int(pivot_column) + 1
    num_do_row = Int(unsafe_load(nir, pr1)) - 1
    sc = Int(unsafe_load(start_col_u, pc1))
    sr = Int(unsafe_load(start_row_u, pr1))
    er = sr + num_do_row + 1

    put = 0
    i = sr
    while i < er
        ic = unsafe_load(idx_col_u, i + 1)
        if ic != pivot_column
            put += 1
            unsafe_store!(save_col, ic, put)
        end
        i += 1
    end

    nxt = unsafe_load(next_row, pr1)
    lst = unsafe_load(last_row, pr1)
    unsafe_store!(next_row, nxt, Int(lst) + 1)
    unsafe_store!(last_row, lst, Int(nxt) + 1)
    unsafe_store!(next_row, number_good_u, pr1)
    unsafe_store!(last_row, Int32(-2), pr1)

    unsafe_store!(st.pivotRegion, 1.0 / unsafe_load(el_u, sc + 1), Int(number_good_u) + 1)
    unsafe_store!(nic, Int32(0), pc1)

    i = 0
    while i < num_do_row
        ic = unsafe_load(save_col, i + 1)
        ic1 = Int(ic) + 1
        cur_nic = unsafe_load(nic, ic1)
        if cur_nic != Int32(0)
            num = cur_nic - Int32(1)
            _cf_delete_link!(fc, ncnt, lcnt, ic + nr)
            _cf_add_link!(fc, ncnt, lcnt, ic + nr, num)
            if num != Int32(0)
                s = Int(unsafe_load(start_col_u, ic1))
                piv = s
                while unsafe_load(idx_row_u, piv + 1) != pivot_row
                    piv += 1
                end
                if piv != s
                    val = unsafe_load(el_u, s + 1)
                    ir_s = unsafe_load(idx_row_u, s + 1)
                    unsafe_store!(el_u, unsafe_load(el_u, piv + 1), s + 1)
                    unsafe_store!(idx_row_u, unsafe_load(idx_row_u, piv + 1), s + 1)
                    unsafe_store!(el_u, unsafe_load(el_u, s + 2), piv + 1)
                    unsafe_store!(idx_row_u, unsafe_load(idx_row_u, s + 2), piv + 1)
                    unsafe_store!(el_u, val, s + 2)
                    unsafe_store!(idx_row_u, ir_s, s + 2)
                else
                    ir_save = unsafe_load(idx_row_u, s + 2)
                    val_save = unsafe_load(el_u, s + 2)
                    val_largest = abs(val_save)
                    e = s + Int(cur_nic)
                    largest = s + 1
                    k = s + 2
                    while k < e
                        va = abs(unsafe_load(el_u, k + 1))
                        if va > val_largest
                            val_largest = va
                            largest = k
                        end
                        k += 1
                    end
                    unsafe_store!(idx_row_u, unsafe_load(idx_row_u, largest + 1), s + 2)
                    unsafe_store!(el_u, unsafe_load(el_u, largest + 1), s + 2)
                    unsafe_store!(idx_row_u, ir_save, largest + 1)
                    unsafe_store!(el_u, val_save, largest + 1)
                end
            end
            unsafe_store!(nic, num, ic1)
            unsafe_store!(nicp, unsafe_load(nicp, ic1) + Int32(1), ic1)
            unsafe_store!(start_col_u, unsafe_load(start_col_u, ic1) + Int32(1), ic1)
        end
        i += 1
    end

    _cf_delete_link!(fc, ncnt, lcnt, pivot_row)
    _cf_delete_link!(fc, ncnt, lcnt, pivot_column + nr)
    unsafe_store!(nir, Int32(0), pr1)

    start_col_l = st.startColumnL
    unsafe_store!(start_col_l, length_l, Int(number_good_l) + 1)
    number_good_l += Int32(1)
    unsafe_store!(start_col_l, length_l, Int(number_good_l) + 1)
    return true, number_good_l
end

# =============================================================================
# pivotOneOtherRow (CoinFactorization1.cpp:2190-2502)
# =============================================================================

function cf_pivot_one_other_row!(
    st::CoinFactorizationFactorStatePod,
    pivot_row::Int32,
    pivot_column::Int32,
    length_l::Int32,
    number_good_l::Int32,
    number_good_u::Int32,
    total_elements::Int32,
    n_comp::Int32,
    status::Int32,
)::Tuple{Bool,Int32,Int32,Int32,Int32,Int32}
    nir = st.numberInRow
    nic = st.numberInColumn
    nicp = st.numberInColumnPlus
    start_row_u = st.startRowU
    start_col_u = st.startColumnU
    next_row = st.nextRow
    last_row = st.lastRow
    el_u = st.elementU
    idx_row_u = st.indexRowU
    idx_col_u = st.indexColumnU
    save_col = st.saveColumn
    next_col = st.nextColumn
    fc = st.firstCount
    ncnt = st.nextCount
    lcnt = st.lastCount
    nr = st.numberRows
    tol = st.zeroTolerance

    pr1 = Int(pivot_row) + 1
    pc1 = Int(pivot_column) + 1
    num_in_piv_row = Int(unsafe_load(nir, pr1)) - 1
    sc_piv = Int(unsafe_load(start_col_u, pc1))
    sr_piv = Int(unsafe_load(start_row_u, pr1))
    er_piv = sr_piv + num_in_piv_row + 1

    nxt = unsafe_load(next_row, pr1)
    lst = unsafe_load(last_row, pr1)
    unsafe_store!(next_row, nxt, Int(lst) + 1)
    unsafe_store!(last_row, lst, Int(nxt) + 1)
    unsafe_store!(next_row, number_good_u, pr1)
    unsafe_store!(last_row, Int32(-2), pr1)
    unsafe_store!(nir, Int32(0), pr1)

    l = Int(length_l)
    if l + 1 > Int(st.lengthAreaL)
        return false, length_l, number_good_l, total_elements, n_comp, status
    end
    start_col_l = st.startColumnL
    el_l = st.elementL
    idx_row_l = st.indexRowL
    unsafe_store!(start_col_l, Int32(l), Int(number_good_l) + 1)
    number_good_l += Int32(1)
    unsafe_store!(start_col_l, Int32(l + 1), Int(number_good_l) + 1)
    length_l += Int32(1)

    if unsafe_load(idx_row_u, sc_piv + 1) == pivot_row
        piv_el = unsafe_load(el_u, sc_piv + 1)
        other_mult = unsafe_load(el_u, sc_piv + 2)
        other_row = unsafe_load(idx_row_u, sc_piv + 2)
    else
        piv_el = unsafe_load(el_u, sc_piv + 2)
        other_mult = unsafe_load(el_u, sc_piv + 1)
        other_row = unsafe_load(idx_row_u, sc_piv + 1)
    end
    or1 = Int(other_row) + 1
    num_save = unsafe_load(nir, or1)
    piv_mult = 1.0 / piv_el
    unsafe_store!(st.pivotRegion, piv_mult, Int(number_good_u) + 1)
    unsafe_store!(nic, Int32(0), pc1)
    other_mult = other_mult * piv_mult
    unsafe_store!(idx_row_l, other_row, l + 1)
    unsafe_store!(el_l, other_mult, l + 1)

    # Remove pivotColumn from otherRow's row list
    s_other = Int(unsafe_load(start_row_u, or1))
    e_other = s_other + Int(num_save)
    w = s_other
    while unsafe_load(idx_col_u, w + 1) != pivot_column
        w += 1
    end
    e_other -= 1
    unsafe_store!(idx_col_u, unsafe_load(idx_col_u, e_other + 1), w + 1)

    num_added = Int32(0)
    num_deleted = Int32(0)

    j = sr_piv
    while j < er_piv
        ic = unsafe_load(idx_col_u, j + 1)
        if ic != pivot_column
            ic1 = Int(ic) + 1
            sc = Int(unsafe_load(start_col_u, ic1))
            ec = sc + Int(unsafe_load(nic, ic1))
            ir = unsafe_load(idx_row_u, sc + 1)
            val = unsafe_load(el_u, sc + 1)
            found_other = false

            put = sc + 1
            pos_largest = -1
            this_piv_val = 0.0
            other_el = 0.0
            largest = 0.0

            next_val = put < ec ? unsafe_load(el_u, put + 1) : 0.0
            next_ir = put < ec ? unsafe_load(idx_row_u, put + 1) : Int32(0)

            if ir != pivot_row
                if ir != other_row
                    largest = abs(val)
                    unsafe_store!(el_u, val, put + 1)
                    unsafe_store!(idx_row_u, ir, put + 1)
                    pos_largest = put
                    put += 1
                    i = sc + 1
                    while i < ec
                        ir = next_ir
                        val = next_val
                        if i + 1 < ec
                            next_ir = unsafe_load(idx_row_u, i + 2)
                            next_val = unsafe_load(el_u, i + 2)
                        end
                        if ir != pivot_row
                            if ir != other_row
                                unsafe_store!(idx_row_u, ir, put + 1)
                                unsafe_store!(el_u, val, put + 1)
                                put += 1
                            else
                                other_el = val
                                found_other = true
                            end
                        else
                            this_piv_val = val
                        end
                        i += 1
                    end
                else
                    other_el = val
                    found_other = true
                    largest = 0.0
                    i = sc + 1
                    while i < ec
                        ir = next_ir
                        val = next_val
                        if i + 1 < ec
                            next_ir = unsafe_load(idx_row_u, i + 2)
                            next_val = unsafe_load(el_u, i + 2)
                        end
                        if ir != pivot_row
                            unsafe_store!(idx_row_u, ir, put + 1)
                            unsafe_store!(el_u, val, put + 1)
                            av = abs(val)
                            if av > largest
                                largest = av
                                pos_largest = put
                            end
                            put += 1
                        else
                            this_piv_val = val
                        end
                        i += 1
                    end
                end
            else
                largest = 0.0
                this_piv_val = val
                i = sc + 1
                while i < ec
                    ir = next_ir
                    val = next_val
                    if i + 1 < ec
                        next_ir = unsafe_load(idx_row_u, i + 2)
                        next_val = unsafe_load(el_u, i + 2)
                    end
                    if ir != other_row
                        unsafe_store!(idx_row_u, ir, put + 1)
                        unsafe_store!(el_u, val, put + 1)
                        av = abs(val)
                        if av > largest
                            largest = av
                            pos_largest = put
                        end
                        put += 1
                    else
                        other_el = val
                        found_other = true
                    end
                    i += 1
                end
            end

            # Slot in pivot
            unsafe_store!(el_u, this_piv_val, sc + 1)
            unsafe_store!(idx_row_u, pivot_row, sc + 1)
            sc += 1
            unsafe_store!(nic, Int32(put - sc), ic1)
            unsafe_store!(nicp, unsafe_load(nicp, ic1) + Int32(1), ic1)
            unsafe_store!(start_col_u, Int32(sc), ic1)

            # Hardware arm64 fmsub matching clang++ -O3
            other_el = _fmsub(this_piv_val, other_mult, other_el)
            av = abs(other_el)

            if av > tol
                if !found_other
                    num_added += Int32(1)
                    unsafe_store!(save_col, ic, Int(num_added))
                    nxt_c1 = Int(unsafe_load(next_col, ic1)) + 1
                    sp = Int(unsafe_load(start_col_u, nxt_c1)) - put - Int(unsafe_load(nicp, nxt_c1))
                    if sp <= 0
                        num = unsafe_load(nic, ic1)
                        ok, n_comp, status = cf_get_column_space!(st, ic, num + Int32(1), n_comp, status)
                        if !ok
                            return false, length_l, number_good_l, total_elements, n_comp, status
                        end
                        new_sc = Int(unsafe_load(start_col_u, ic1))
                        if pos_largest >= 0
                            pos_largest = pos_largest + new_sc - sc
                        end
                        sc = new_sc
                        put = sc + Int(num)
                    end
                end
                unsafe_store!(el_u, other_el, put + 1)
                unsafe_store!(idx_row_u, other_row, put + 1)
                if av > largest
                    largest = av
                    pos_largest = put
                end
                put += 1
            else
                if found_other
                    num_deleted += Int32(1)
                    w2 = s_other
                    while unsafe_load(idx_col_u, w2 + 1) != ic
                        w2 += 1
                    end
                    e_other -= 1
                    unsafe_store!(idx_col_u, unsafe_load(idx_col_u, e_other + 1), w2 + 1)
                end
            end

            new_nic = Int32(put - sc)
            unsafe_store!(nic, new_nic, ic1)
            if pos_largest >= 0
                v_l = unsafe_load(el_u, pos_largest + 1)
                r_l = unsafe_load(idx_row_u, pos_largest + 1)
                unsafe_store!(el_u, unsafe_load(el_u, sc + 1), pos_largest + 1)
                unsafe_store!(idx_row_u, unsafe_load(idx_row_u, sc + 1), pos_largest + 1)
                unsafe_store!(el_u, v_l, sc + 1)
                unsafe_store!(idx_row_u, r_l, sc + 1)
            end
            if unsafe_load(ncnt, Int(ic + nr) + 1) != Int32(-2)
                _cf_delete_link!(fc, ncnt, lcnt, ic + nr)
                _cf_add_link!(fc, ncnt, lcnt, ic + nr, new_nic)
            end
        end
        j += 1
    end

    # Get space for otherRow's row list
    nxt_or = unsafe_load(next_row, or1)
    sp = Int(unsafe_load(start_row_u, Int(nxt_or) + 1)) - e_other
    total_elements += num_added - num_deleted
    cur_len = Int32(e_other - s_other)
    number = num_added + cur_len
    if sp < Int(num_added)
        unsafe_store!(nir, cur_len, or1)
        ok, n_comp, status = cf_get_row_space!(st, other_row, number, n_comp, status)
        if !ok
            return false, length_l, number_good_l, total_elements, n_comp, status
        end
        e_other = Int(unsafe_load(start_row_u, or1)) + Int(cur_len)
    end
    unsafe_store!(nir, number, or1)
    if number != num_save
        _cf_delete_link!(fc, ncnt, lcnt, other_row)
        _cf_add_link!(fc, ncnt, lcnt, other_row, number)
    end
    j = 0
    na_i = Int(num_added)
    while j < na_i
        unsafe_store!(idx_col_u, unsafe_load(save_col, j + 1), e_other + 1)
        e_other += 1
        j += 1
    end

    _cf_delete_link!(fc, ncnt, lcnt, pivot_row)
    _cf_delete_link!(fc, ncnt, lcnt, pivot_column + nr)
    return true, length_l, number_good_l, total_elements, n_comp, status
end

# =============================================================================
# pivot<TMark> (CoinFactorization.hpp:953-1429)
# =============================================================================

function cf_pivot!(
    ::Type{TMark},
    st::CoinFactorizationFactorStatePod,
    pivot_row::Int32,
    pivot_column::Int32,
    pivot_row_pos::Int32,
    pivot_col_pos::Int32,
    work::Ptr{Float64},
    work2::Ptr{UInt32},
    increment2::Int32,
    mark_row::Ptr{TMark},
    large_integer::Int32,
    length_l::Int32,
    number_good_l::Int32,
    number_good_u::Int32,
    total_elements::Int32,
    n_comp::Int32,
    status::Int32,
)::Tuple{Bool,Int32,Int32,Int32,Int32,Int32} where {TMark<:Union{UInt16,Int32}}
    idx_col_u = st.indexColumnU
    start_col_u = st.startColumnU
    nic = st.numberInColumn
    nicp = st.numberInColumnPlus
    el_u = st.elementU
    idx_row_u = st.indexRowU
    start_row_u = st.startRowU
    nir = st.numberInRow
    el_l = st.elementL
    idx_row_l = st.indexRowL
    save_col = st.saveColumn
    next_row = st.nextRow
    last_row = st.lastRow
    next_col = st.nextColumn
    fc = st.firstCount
    ncnt = st.nextCount
    lcnt = st.lastCount
    nr = st.numberRows
    tol = st.zeroTolerance

    pr1 = Int(pivot_row) + 1
    pc1 = Int(pivot_column) + 1
    num_in_piv_row = Int(unsafe_load(nir, pr1)) - 1
    sc_piv = Int(unsafe_load(start_col_u, pc1))
    num_in_piv_col = Int(unsafe_load(nic, pc1)) - 1
    ec_piv = sc_piv + num_in_piv_col + 1
    sr_piv = Int(unsafe_load(start_row_u, pr1))
    er_piv = sr_piv + num_in_piv_row + 1

    put = 0
    pcp = Int(pivot_col_pos)
    if pcp < 0
        pcp = sr_piv
        while pcp < er_piv
            ic = unsafe_load(idx_col_u, pcp + 1)
            if ic != pivot_column
                put += 1
                unsafe_store!(save_col, ic, put)
            else
                break
            end
            pcp += 1
        end
    else
        i = sr_piv
        while i < pcp
            put += 1
            unsafe_store!(save_col, unsafe_load(idx_col_u, i + 1), put)
            i += 1
        end
    end
    pcp += 1
    while pcp < er_piv
        put += 1
        unsafe_store!(save_col, unsafe_load(idx_col_u, pcp + 1), put)
        pcp += 1
    end

    # Remove pivotRow from row linked list
    nxt = unsafe_load(next_row, pr1)
    lst = unsafe_load(last_row, pr1)
    unsafe_store!(next_row, nxt, Int(lst) + 1)
    unsafe_store!(last_row, lst, Int(nxt) + 1)
    unsafe_store!(next_row, number_good_u, pr1)
    unsafe_store!(last_row, Int32(-2), pr1)
    unsafe_store!(nir, Int32(0), pr1)

    l = Int(length_l)
    if l + num_in_piv_col > Int(st.lengthAreaL)
        return false, length_l, number_good_l, total_elements, n_comp, status
    end
    l_save = l
    start_col_l = st.startColumnL
    unsafe_store!(start_col_l, Int32(l), Int(number_good_l) + 1)
    number_good_l += Int32(1)
    unsafe_store!(start_col_l, Int32(l + num_in_piv_col), Int(number_good_l) + 1)
    length_l += Int32(num_in_piv_col)

    prp = Int(pivot_row_pos)
    if prp < 0
        prp = sc_piv
        while prp < ec_piv
            ir = unsafe_load(idx_row_u, prp + 1)
            if ir != pivot_row
                unsafe_store!(idx_row_l, ir, l + 1)
                unsafe_store!(el_l, unsafe_load(el_u, prp + 1), l + 1)
                unsafe_store!(mark_row, TMark(l - l_save), Int(ir) + 1)
                l += 1
                ir1 = Int(ir) + 1
                s = Int(unsafe_load(start_row_u, ir1))
                cur_nir = unsafe_load(nir, ir1)
                e = s + Int(cur_nir)
                w = s
                while unsafe_load(idx_col_u, w + 1) != pivot_column
                    w += 1
                end
                unsafe_store!(idx_col_u, unsafe_load(idx_col_u, e), w + 1)
                unsafe_store!(nir, cur_nir - Int32(1), ir1)
            else
                break
            end
            prp += 1
        end
    else
        i = sc_piv
        while i < prp
            ir = unsafe_load(idx_row_u, i + 1)
            unsafe_store!(mark_row, TMark(l - l_save), Int(ir) + 1)
            unsafe_store!(idx_row_l, ir, l + 1)
            unsafe_store!(el_l, unsafe_load(el_u, i + 1), l + 1)
            l += 1
            ir1 = Int(ir) + 1
            s = Int(unsafe_load(start_row_u, ir1))
            cur_nir = unsafe_load(nir, ir1)
            e = s + Int(cur_nir)
            w = s
            while unsafe_load(idx_col_u, w + 1) != pivot_column
                w += 1
            end
            unsafe_store!(idx_col_u, unsafe_load(idx_col_u, e), w + 1)
            unsafe_store!(nir, cur_nir - Int32(1), ir1)
            i += 1
        end
    end

    piv_el = unsafe_load(el_u, prp + 1)
    piv_mult = 1.0 / piv_el
    unsafe_store!(st.pivotRegion, piv_mult, Int(number_good_u) + 1)
    prp += 1
    while prp < ec_piv
        ir = unsafe_load(idx_row_u, prp + 1)
        unsafe_store!(mark_row, TMark(l - l_save), Int(ir) + 1)
        unsafe_store!(idx_row_l, ir, l + 1)
        unsafe_store!(el_l, unsafe_load(el_u, prp + 1), l + 1)
        l += 1
        ir1 = Int(ir) + 1
        s = Int(unsafe_load(start_row_u, ir1))
        cur_nir = unsafe_load(nir, ir1)
        e = s + Int(cur_nir)
        w = s
        while unsafe_load(idx_col_u, w + 1) != pivot_column
            w += 1
        end
        unsafe_store!(idx_col_u, unsafe_load(idx_col_u, e), w + 1)
        unsafe_store!(nir, cur_nir - Int32(1), ir1)
        prp += 1
    end

    unsafe_store!(mark_row, TMark(large_integer), pr1)
    unsafe_store!(nic, Int32(0), pc1)

    index_l = idx_row_l + l_save * sizeof(Int32)
    mult_l = el_l + l_save * sizeof(Float64)

    j = 0
    while j < num_in_piv_col
        unsafe_store!(mult_l, unsafe_load(mult_l, j + 1) * piv_mult, j + 1)
        j += 1
    end

    inc2 = Int(increment2)
    erase_end = inc2 * num_in_piv_row
    ie = 0
    while ie < erase_end
        unsafe_store!(work2, UInt32(0), ie + 1)
        ie += 1
    end

    added = Int32(num_in_piv_row * num_in_piv_col)
    temp2 = work2
    large_unset = large_integer + Int32(1)

    jcol = 0
    while jcol < num_in_piv_row
        ic = unsafe_load(save_col, jcol + 1)
        ic1 = Int(ic) + 1
        sc = Int(unsafe_load(start_col_u, ic1))
        ec = sc + Int(unsafe_load(nic, ic1))
        ir = unsafe_load(idx_row_u, sc + 1)
        val = unsafe_load(el_u, sc + 1)
        put = sc
        pos_largest = -1
        this_piv_val = 0.0
        mark = Int32(unsafe_load(mark_row, Int(ir) + 1))

        if mark == large_unset
            largest = abs(val)
            pos_largest = put
            put += 1
            check_largest = false
        else
            largest = 0.0
            check_largest = true
            if mark != large_integer
                unsafe_store!(work, val, Int(mark) + 1)
                word = Int(mark >> CF_SHIFT_PER_INT)
                bit = mark & CF_MASK_PER_INT
                unsafe_store!(temp2, unsafe_load(temp2, word + 1) | (UInt32(1) << bit), word + 1)
                added -= Int32(1)
            else
                this_piv_val = val
            end
        end

        i = sc + 1
        while i < ec
            ir = unsafe_load(idx_row_u, i + 1)
            val = unsafe_load(el_u, i + 1)
            mark = Int32(unsafe_load(mark_row, Int(ir) + 1))
            if mark == large_unset
                unsafe_store!(idx_row_u, ir, put + 1)
                unsafe_store!(el_u, val, put + 1)
                if check_largest
                    av = abs(val)
                    if av > largest
                        largest = av
                        pos_largest = put
                    end
                end
                put += 1
            elseif mark != large_integer
                unsafe_store!(work, val, Int(mark) + 1)
                word = Int(mark >> CF_SHIFT_PER_INT)
                bit = mark & CF_MASK_PER_INT
                unsafe_store!(temp2, unsafe_load(temp2, word + 1) | (UInt32(1) << bit), word + 1)
                added -= Int32(1)
            else
                this_piv_val = val
            end
            i += 1
        end

        # Slot in pivot
        unsafe_store!(el_u, unsafe_load(el_u, sc + 1), put + 1)
        unsafe_store!(idx_row_u, unsafe_load(idx_row_u, sc + 1), put + 1)
        if pos_largest == sc
            pos_largest = put
        end
        put += 1
        unsafe_store!(el_u, this_piv_val, sc + 1)
        unsafe_store!(idx_row_u, pivot_row, sc + 1)
        sc += 1
        unsafe_store!(nic, Int32(put - sc), ic1)
        unsafe_store!(nicp, unsafe_load(nicp, ic1) + Int32(1), ic1)
        unsafe_store!(start_col_u, Int32(sc), ic1)

        nxt_c1 = Int(unsafe_load(next_col, ic1)) + 1
        sp = Int(unsafe_load(start_col_u, nxt_c1)) - put - Int(unsafe_load(nicp, nxt_c1))
        if num_in_piv_col > sp
            ok, n_comp, status = cf_get_column_space!(st, ic, Int32(num_in_piv_col), n_comp, status)
            if !ok
                return false, length_l, number_good_l, total_elements, n_comp, status
            end
            new_sc = Int(unsafe_load(start_col_u, ic1))
            if pos_largest >= 0
                pos_largest = pos_largest + new_sc - sc
            end
            sc = new_sc
            put = sc + Int(unsafe_load(nic, ic1))
        end

        j = 0
        while j < num_in_piv_col
            # Hardware arm64 fmsub matching clang++ -O3 (fmadd with negated this_piv_val)
            v = _fmsub(this_piv_val, unsafe_load(mult_l, j + 1), unsafe_load(work, j + 1))
            av = abs(v)
            if av > tol
                unsafe_store!(work, 0.0, j + 1)
                unsafe_store!(el_u, v, put + 1)
                unsafe_store!(idx_row_u, unsafe_load(index_l, j + 1), put + 1)
                if av > largest
                    largest = av
                    pos_largest = put
                end
                put += 1
            else
                unsafe_store!(work, 0.0, j + 1)
                added -= Int32(1)
                word = j >> 5
                bit = j & 31
                wval = unsafe_load(temp2, word + 1)
                mask = UInt32(1) << bit
                if (wval & mask) != UInt32(0)
                    ir_j = unsafe_load(index_l, j + 1)
                    ir_j1 = Int(ir_j) + 1
                    s = Int(unsafe_load(start_row_u, ir_j1))
                    cur_nir = unsafe_load(nir, ir_j1)
                    e = s + Int(cur_nir)
                    w = s
                    while unsafe_load(idx_col_u, w + 1) != ic
                        w += 1
                    end
                    unsafe_store!(idx_col_u, unsafe_load(idx_col_u, e), w + 1)
                    unsafe_store!(nir, cur_nir - Int32(1), ir_j1)
                else
                    unsafe_store!(temp2, wval | mask, word + 1)
                end
            end
            j += 1
        end

        new_nic = Int32(put - sc)
        unsafe_store!(nic, new_nic, ic1)
        if pos_largest >= 0
            v_l = unsafe_load(el_u, pos_largest + 1)
            r_l = unsafe_load(idx_row_u, pos_largest + 1)
            unsafe_store!(el_u, unsafe_load(el_u, sc + 1), pos_largest + 1)
            unsafe_store!(idx_row_u, unsafe_load(idx_row_u, sc + 1), pos_largest + 1)
            unsafe_store!(el_u, v_l, sc + 1)
            unsafe_store!(idx_row_u, r_l, sc + 1)
        end
        if unsafe_load(ncnt, Int(ic + nr) + 1) != Int32(-2)
            _cf_delete_link!(fc, ncnt, lcnt, ic + nr)
            _cf_add_link!(fc, ncnt, lcnt, ic + nr, new_nic)
        end
        temp2 += inc2 * sizeof(UInt32)
        jcol += 1
    end

    # Update row lists and counts using bitmask work2
    put_base = work2
    big_loops = num_in_piv_col >> 5
    i = 0
    inc2_bytes = inc2 * sizeof(UInt32)
    unset_mark = TMark(large_unset)

    while big_loops > 0
        big_loops -= 1
        bit = 0
        while bit < 32
            put_this = put_base
            ir = unsafe_load(index_l, i + 1)
            ir1 = Int(ir) + 1
            number = Int32(0)
            jcol = 0
            while jcol < num_in_piv_row
                t = unsafe_load(put_this)
                put_this += inc2_bytes
                number += Int32(1) - Int32((t >> bit) & UInt32(1))
                jcol += 1
            end
            nxt_r = unsafe_load(next_row, ir1)
            sp = unsafe_load(start_row_u, Int(nxt_r) + 1) - unsafe_load(start_row_u, ir1)
            number += unsafe_load(nir, ir1)
            if sp < number
                ok, n_comp, status = cf_get_row_space!(st, ir, number, n_comp, status)
                if !ok
                    return false, length_l, number_good_l, total_elements, n_comp, status
                end
            end
            put_this = put_base
            nxt_r = unsafe_load(next_row, ir1)
            sr_i = unsafe_load(start_row_u, ir1)
            e = Int(sr_i + unsafe_load(nir, ir1))
            nxt_sr = Int(unsafe_load(start_row_u, Int(nxt_r) + 1))
            save_idx = unsafe_load(idx_col_u, nxt_sr + 1)
            jcol = 0
            while jcol < num_in_piv_row
                t = unsafe_load(put_this)
                put_this += inc2_bytes
                inc = Int(UInt32(1) - ((t >> bit) & UInt32(1)))
                unsafe_store!(idx_col_u, unsafe_load(save_col, jcol + 1), e + 1)
                e += inc
                jcol += 1
            end
            unsafe_store!(idx_col_u, save_idx, nxt_sr + 1)
            unsafe_store!(mark_row, unset_mark, ir1)
            new_nir = Int32(e) - sr_i
            unsafe_store!(nir, new_nir, ir1)
            _cf_delete_link!(fc, ncnt, lcnt, ir)
            _cf_add_link!(fc, ncnt, lcnt, ir, new_nir)
            i += 1
            bit += 1
        end
        put_base += sizeof(UInt32)
    end

    bit = 0
    while i < num_in_piv_col
        put_this = put_base
        ir = unsafe_load(index_l, i + 1)
        ir1 = Int(ir) + 1
        number = Int32(0)
        jcol = 0
        while jcol < num_in_piv_row
            t = unsafe_load(put_this)
            put_this += inc2_bytes
            number += Int32(1) - Int32((t >> bit) & UInt32(1))
            jcol += 1
        end
        nxt_r = unsafe_load(next_row, ir1)
        sp = unsafe_load(start_row_u, Int(nxt_r) + 1) - unsafe_load(start_row_u, ir1)
        number += unsafe_load(nir, ir1)
        if sp < number
            ok, n_comp, status = cf_get_row_space!(st, ir, number, n_comp, status)
            if !ok
                return false, length_l, number_good_l, total_elements, n_comp, status
            end
        end
        put_this = put_base
        nxt_r = unsafe_load(next_row, ir1)
        sr_i = unsafe_load(start_row_u, ir1)
        e = Int(sr_i + unsafe_load(nir, ir1))
        nxt_sr = Int(unsafe_load(start_row_u, Int(nxt_r) + 1))
        save_idx = unsafe_load(idx_col_u, nxt_sr + 1)
        jcol = 0
        while jcol < num_in_piv_row
            t = unsafe_load(put_this)
            put_this += inc2_bytes
            inc = Int(UInt32(1) - ((t >> bit) & UInt32(1)))
            unsafe_store!(idx_col_u, unsafe_load(save_col, jcol + 1), e + 1)
            e += inc
            jcol += 1
        end
        unsafe_store!(idx_col_u, save_idx, nxt_sr + 1)
        unsafe_store!(mark_row, unset_mark, ir1)
        new_nir = Int32(e) - sr_i
        unsafe_store!(nir, new_nir, ir1)
        _cf_delete_link!(fc, ncnt, lcnt, ir)
        _cf_add_link!(fc, ncnt, lcnt, ir, new_nir)
        i += 1
        bit += 1
    end

    unsafe_store!(mark_row, unset_mark, pr1)
    _cf_delete_link!(fc, ncnt, lcnt, pivot_row)
    _cf_delete_link!(fc, ncnt, lcnt, pivot_column + nr)
    total_elements += added
    return true, length_l, number_good_l, total_elements, n_comp, status
end

# =============================================================================
# factorSparseSmall / factorSparseLarge (CoinFactorization2.cpp:43-513, 1079-1519)
# =============================================================================

function cf_factor_sparse_impl!(
    ::Type{TMark},
    st_ptr::Ptr{CoinFactorizationFactorStatePod},
    mark_set::Int32,
    mark_unset::Int32,
)::Int32 where {TMark<:Union{UInt16,Int32}}
    st = unsafe_load(st_ptr)
    idx_row = st.indexRowU
    idx_col = st.indexColumnU
    el = st.elementU
    nr = Int(st.numberRows)
    nc = Int(st.numberColumns)
    nr32 = st.numberRows
    bigger_dim = st.biggerDimension
    dense_thresh = abs(st.denseThreshold)
    piv_tol = st.pivotTolerance
    num_trials = st.numberTrials
    bias_lu = st.biasLU
    time_limit = st.timeLimit

    work = st.workArea
    coin_zero_n!(work, nr)
    work2 = st.workArea2
    work_size = st.workArea2Size

    mark_row = Ptr{TMark}(st.markRow)
    unset_val = TMark(mark_unset)
    i_m = 1
    while i_m <= nr
        unsafe_store!(mark_row, unset_val, i_m)
        i_m += 1
    end

    status = Int32(0)
    nir = st.numberInRow
    nic = st.numberInColumn
    nicp = st.numberInColumnPlus
    start_col_u = st.startColumnU
    start_col_l = st.startColumnL
    start_row = st.startRowU
    piv_col = st.pivotColumn
    next_row = st.nextRow
    last_row = st.lastRow
    fc = st.firstCount
    ncnt = st.nextCount
    lcnt = st.lastCount

    num_good_u = st.numberGoodU
    num_good_l = st.numberGoodL
    length_l = st.lengthL
    total_elements = st.totalElements
    n_comp = st.numberCompressions

    # Do slacks first
    if bias_lu < Int32(3) && nc == nr
        slack_val = st.slackValue
        ipc = 0
        while ipc < nc
            ipc1 = ipc + 1
            if unsafe_load(nic, ipc1) == Int32(1)
                s = Int(unsafe_load(start_col_u, ipc1))
                v = unsafe_load(el, s + 1)
                if v == slack_val && unsafe_load(nicp, ipc1) == Int32(0)
                    ir = unsafe_load(idx_row, s + 1)
                    ir1 = Int(ir) + 1
                    cur_nir = unsafe_load(nir, ir1)
                    if cur_nir > Int32(0)
                        total_elements -= cur_nir
                        nxt = unsafe_load(next_row, ir1)
                        lst = unsafe_load(last_row, ir1)
                        unsafe_store!(next_row, nxt, Int(lst) + 1)
                        unsafe_store!(last_row, lst, Int(nxt) + 1)
                        unsafe_store!(next_row, num_good_u, ir1)
                        unsafe_store!(last_row, Int32(-2), ir1)
                        _cf_delete_link!(fc, ncnt, lcnt, ir)
                        unsafe_store!(nir, Int32(-1), ir1)
                        unsafe_store!(nic, Int32(0), ipc1)
                        num_good_l += Int32(1)
                        unsafe_store!(start_col_l, Int32(0), Int(num_good_l) + 1)
                        unsafe_store!(piv_col, Int32(ipc), Int(num_good_u) + 1)
                        num_good_u += Int32(1)
                    end
                end
            end
            ipc += 1
        end
        total_elements = cf_pre_process_core!(st, Int32(4), total_elements)
        i_m = 1
        while i_m <= nr
            unsafe_store!(mark_row, unset_val, i_m)
            i_m += 1
        end
    end

    num_slacks = num_good_u
    cf_separate_links!(fc, ncnt, lcnt, nr32, Int32(1), bias_lu > Int32(1))

    count = Int32(1)
    while count <= bigger_dim
        min_count = typemax(Int32)
        min_cost = floatmax(Float64)
        i_piv_row = Int32(-1)
        i_piv_col = Int32(-1)
        piv_row_pos = Int32(-1)
        piv_col_pos = Int32(-1)
        look = unsafe_load(fc, Int(count) + 1)
        trials = Int32(0)

        if count == Int32(1) && look >= Int32(0) && bias_lu == Int32(0)
            while look >= Int32(0)
                if look < nr32
                    look = unsafe_load(ncnt, Int(look) + 1)
                else
                    ic = look - nr32
                    s = unsafe_load(start_col_u, Int(ic) + 1)
                    ir = unsafe_load(idx_row, Int(s) + 1)
                    i_piv_row = ir
                    piv_row_pos = s
                    i_piv_col = ic
                    piv_col_pos = Int32(-1)
                    look = Int32(-1)
                    break
                end
            end
            if i_piv_row < Int32(0)
                look = unsafe_load(fc, 2)
            end
        end

        while look >= Int32(0)
            if look < nr32
                ir = look
                look = unsafe_load(ncnt, Int(look) + 1)
                rejected = false
                s = Int(unsafe_load(start_row, Int(ir) + 1))
                e = s + Int(count)
                i = s
                cm1 = count - Int32(1)
                while i < e
                    ic = unsafe_load(idx_col, i + 1)
                    ic1 = Int(ic) + 1
                    n_ic = unsafe_load(nic, ic1)
                    cost = Float64(cm1 * n_ic)
                    if cost < min_cost
                        w = Int(unsafe_load(start_col_u, ic1))
                        min_val = abs(unsafe_load(el, w + 1)) * piv_tol
                        while unsafe_load(idx_row, w + 1) != ir
                            w += 1
                        end
                        v = abs(unsafe_load(el, w + 1))
                        if v >= min_val
                            min_cost = cost
                            min_count = n_ic
                            i_piv_row = ir
                            piv_row_pos = Int32(-1)
                            i_piv_col = ic
                            piv_col_pos = Int32(i)
                            rejected = false
                            if min_count < count
                                look = Int32(-1)
                                break
                            end
                        elseif i_piv_row == Int32(-1)
                            rejected = true
                        end
                    end
                    i += 1
                end
                trials += Int32(1)
                if trials >= num_trials && i_piv_row >= Int32(0)
                    look = Int32(-1)
                    break
                end
                if rejected
                    _cf_delete_link!(fc, ncnt, lcnt, ir)
                    _cf_add_link!(fc, ncnt, lcnt, ir, bigger_dim + Int32(1))
                end
            else
                ic = look - nr32
                look = unsafe_load(ncnt, Int(look) + 1)
                ic1 = Int(ic) + 1
                s = Int(unsafe_load(start_col_u, ic1))
                e = s + Int(unsafe_load(nic, ic1))
                min_val = abs(unsafe_load(el, s + 1)) * piv_tol
                cm1 = count - Int32(1)
                i = s
                while i < e
                    v = abs(unsafe_load(el, i + 1))
                    if v >= min_val
                        ir = unsafe_load(idx_row, i + 1)
                        n_ir = unsafe_load(nir, Int(ir) + 1)
                        cost = Float64(cm1 * n_ir)
                        if cost < min_cost
                            min_cost = cost
                            min_count = n_ir
                            i_piv_row = ir
                            piv_row_pos = Int32(i)
                            i_piv_col = ic
                            piv_col_pos = Int32(-1)
                            if min_count <= count + Int32(1)
                                look = Int32(-1)
                                break
                            end
                        end
                    end
                    i += 1
                end
                trials += Int32(1)
                if trials >= num_trials && i_piv_row >= Int32(0)
                    look = Int32(-1)
                    break
                end
            end
        end

        if i_piv_row >= Int32(0)
            num_do_row = unsafe_load(nir, Int(i_piv_row) + 1) - Int32(1)
            num_do_col = unsafe_load(nic, Int(i_piv_col) + 1) - Int32(1)
            total_elements -= (num_do_row + num_do_col + Int32(1))

            if num_do_col > Int32(0)
                if num_do_row > Int32(0)
                    if num_do_col > Int32(1)
                        inc = num_do_col + Int32(5)
                        if (inc & Int32(15)) != Int32(0)
                            inc = (inc & ~Int32(15)) + Int32(16)
                        end
                        inc2 = (inc + CF_BITS_PER_INT - Int32(1)) >> CF_SHIFT_PER_INT
                        sz = inc2 * num_do_row
                        if sz > work_size
                            work_size = sz
                            if st.grow_work_area2_cb != C_NULL
                                work2 = ccall(st.grow_work_area2_cb, Ptr{UInt32}, (Ptr{Cvoid}, Cint), st.cb_ctx, Cint(work_size))
                            end
                        end
                        good_piv, length_l, num_good_l, total_elements, n_comp, status = cf_pivot!(
                            TMark, st, i_piv_row, i_piv_col, piv_row_pos, piv_col_pos,
                            work, work2, inc2, mark_row, mark_set,
                            length_l, num_good_l, num_good_u, total_elements, n_comp, status,
                        )
                        if !good_piv
                            status = Int32(-99)
                            count = bigger_dim + Int32(1)
                            break
                        end
                    else
                        good_piv, length_l, num_good_l, total_elements, n_comp, status = cf_pivot_one_other_row!(
                            st, i_piv_row, i_piv_col,
                            length_l, num_good_l, num_good_u, total_elements, n_comp, status,
                        )
                        if !good_piv
                            status = Int32(-99)
                            count = bigger_dim + Int32(1)
                            break
                        end
                    end
                else
                    good_piv, length_l, num_good_l = cf_pivot_row_singleton!(
                        st, i_piv_row, i_piv_col, length_l, num_good_l, num_good_u,
                    )
                    if !good_piv
                        status = Int32(-99)
                        count = bigger_dim + Int32(1)
                        break
                    end
                end
            else
                good_piv, num_good_l = cf_pivot_column_singleton!(
                    st, i_piv_row, i_piv_col, length_l, num_good_l, num_good_u,
                )
                if !good_piv
                    status = Int32(-99)
                    count = bigger_dim + Int32(1)
                    break
                end
            end

            unsafe_store!(piv_col, i_piv_col, Int(num_good_u) + 1)
            num_good_u += Int32(1)
            if num_good_u == nr32
                count = bigger_dim + Int32(1)
            end
            if time_limit > 0.0 && (num_good_u & Int32(63)) == Int32(0) && _cf_wallclock_time() > time_limit
                status = Int32(-100)
                count = bigger_dim + Int32(1)
                break
            end

            if dense_thresh != Int32(0)
                left_rows = nr32 - num_good_u
                full = Float64(left_rows)
                full *= full
                left_el = Float64(total_elements)
                ratio = if left_rows > Int32(2000)
                    3.5
                elseif left_rows > Int32(800)
                    2.75
                elseif left_rows > Int32(256)
                    2.0
                else
                    1.5
                end
                if ratio * left_el > full && left_rows > dense_thresh
                    if (left_rows & Int32(7)) == Int32(0)
                        if status != Int32(0)
                            break
                        end
                        chk = Int32(0)
                        ic = 0
                        while ic < nc
                            if unsafe_load(nic, ic + 1) != Int32(0)
                                chk += Int32(1)
                            end
                            ic += 1
                        end
                        if chk != left_rows && dense_thresh != Int32(0)
                            dense_thresh = Int32(0)
                        else
                            status = Int32(2)
                            break
                        end
                    end
                end
            end

            count = Int32(1)
        else
            count += Int32(1)
        end
    end

    _fs_store_u32ptr!(st_ptr, _FS_OFF_workArea2, work2)
    _fs_store_i32!(st_ptr, _FS_OFF_workArea2Size, work_size)
    _fs_store_i32!(st_ptr, _FS_OFF_numberGoodU, num_good_u)
    _fs_store_i32!(st_ptr, _FS_OFF_numberGoodL, num_good_l)
    _fs_store_i32!(st_ptr, _FS_OFF_numberSlacks, num_slacks)
    _fs_store_i32!(st_ptr, _FS_OFF_lengthL, length_l)
    _fs_store_i32!(st_ptr, _FS_OFF_totalElements, total_elements)
    _fs_store_i32!(st_ptr, _FS_OFF_numberCompressions, n_comp)
    _fs_store_i32!(st_ptr, _FS_OFF_status, status)
    return status
end

function cf_factor_sparse!(st_ptr::Ptr{CoinFactorizationFactorStatePod})::Int32
    st = unsafe_load(st_ptr)
    larger = st.numberRows < st.numberColumns ? st.numberColumns : st.numberRows
    if larger < CF_LARGELIMIT
        return cf_factor_sparse_impl!(UInt16, st_ptr, CF_SMALL_SET, CF_SMALL_UNSET)
    else
        return cf_factor_sparse_impl!(Int32, st_ptr, CF_LARGE_SET, CF_LARGE_UNSET)
    end
end

# =============================================================================
# factorDense (CoinFactorization2.cpp:517-778)
# Expects denseAreaAddress (zeroed, 256B-aligned) and densePermute allocated by caller.
# =============================================================================

function cf_factor_dense!(st_ptr::Ptr{CoinFactorizationFactorStatePod})::Int32
    st = unsafe_load(st_ptr)
    nr = Int(st.numberRows)
    nc = Int(st.numberColumns)
    num_good_u = st.numberGoodU
    num_good_l = st.numberGoodL
    num_dense = st.numberRows - num_good_u
    nd = Int(num_dense)
    full = num_dense * num_dense
    total_elements = full

    dense_area = st.denseAreaAddress
    dense_perm = st.densePermute
    next_row = st.nextRow
    last_row = st.lastRow
    nic = st.numberInColumn
    nicp = st.numberInColumnPlus
    idx_row = st.indexRowU
    el_u = st.elementU
    start_col_l = st.startColumnL
    el_l = st.elementL
    idx_row_l = st.indexRowL
    piv_col = st.pivotColumn
    piv_reg = st.pivotRegion
    start_col_u = st.startColumnU

    i = 0
    while i < nr
        if unsafe_load(last_row, i + 1) >= Int32(0)
            unsafe_store!(last_row, Int32(0), i + 1)
        end
        i += 1
    end
    which = Int32(0)
    i = 0
    while i < nr
        if unsafe_load(last_row, i + 1) == Int32(0)
            unsafe_store!(last_row, which, i + 1)
            unsafe_store!(next_row, num_good_u + which, i + 1)
            unsafe_store!(dense_perm, Int32(i), Int(which) + 1)
            which += Int32(1)
        end
        i += 1
    end

    end_l = unsafe_load(start_col_l, Int(num_good_l) + 1)
    col_ptr = dense_area
    rows_done = 0
    ic = 0
    while ic < nc
        ic1 = ic + 1
        num = Int(unsafe_load(nic, ic1))
        if num != 0
            s = Int(unsafe_load(start_col_u, ic1))
            e = s + num
            k = s
            while k < e
                ir = Int(unsafe_load(last_row, Int(unsafe_load(idx_row, k + 1)) + 1))
                unsafe_store!(col_ptr, unsafe_load(el_u, k + 1), ir + 1)
                k += 1
            end
            col_ptr += nd * sizeof(Float64)
            while unsafe_load(last_row, rows_done + 1) < Int32(0)
                rows_done += 1
            end
            unsafe_store!(next_row, num_good_u, rows_done + 1)
            rows_done += 1
            unsafe_store!(start_col_l, end_l, Int(num_good_u) + 2)
            unsafe_store!(nic, Int32(0), ic1)
            unsafe_store!(piv_col, Int32(ic), Int(num_good_u) + 1)
            unsafe_store!(piv_reg, 1.0, Int(num_good_u) + 1)
            num_good_u += Int32(1)
        end
        ic += 1
    end

    # COIN_FACTORIZATION_DENSE_CODE == 1 (LAPACK dgetrf_ via Apple Accelerate)
    if st.denseThreshold != Int32(0)
        num_good_l = st.numberRows
        if st.timeLimit > 0.0 && _cf_wallclock_time() > st.timeLimit
            _fs_store_i32!(st_ptr, _FS_OFF_numberDense, num_dense)
            _fs_store_i32!(st_ptr, _FS_OFF_totalElements, total_elements)
            _fs_store_i32!(st_ptr, _FS_OFF_numberGoodU, num_good_u)
            _fs_store_i32!(st_ptr, _FS_OFF_numberGoodL, num_good_l)
            _fs_store_i32!(st_ptr, _FS_OFF_freeDense, Int32(0))
            _fs_store_i32!(st_ptr, _FS_OFF_status, Int32(-100))
            return Int32(-100)
        end
        info = _lapack_dgetrf!(num_dense, num_dense, dense_area, num_dense, dense_perm)
        status = info != Int32(0) ? Int32(-1) : Int32(0)
        _fs_store_i32!(st_ptr, _FS_OFF_numberDense, num_dense)
        _fs_store_i32!(st_ptr, _FS_OFF_totalElements, total_elements)
        _fs_store_i32!(st_ptr, _FS_OFF_numberGoodU, num_good_u)
        _fs_store_i32!(st_ptr, _FS_OFF_numberGoodL, num_good_l)
        _fs_store_i32!(st_ptr, _FS_OFF_freeDense, Int32(0))
        _fs_store_i32!(st_ptr, _FS_OFF_status, status)
        return status
    end

    # Naive dense fallback (denseThreshold == 0, CoinFactorization2.cpp:651-777)
    num_good_u = st.numberRows - num_dense
    base = Int(num_good_u)
    num_to_do = Int(-st.denseThreshold)
    n_comp = st.numberCompressions
    status = Int32(0)
    length_l = st.lengthL
    tol = 1.0e-30
    next_col = st.nextColumn

    id = 0
    while id < num_to_do
        ic_d = unsafe_load(piv_col, base + id + 1)
        ic_d1 = Int(ic_d) + 1
        nxt_c1 = Int(unsafe_load(next_col, ic_d1)) + 1
        sp = unsafe_load(start_col_u, nxt_c1) - unsafe_load(start_col_u, ic_d1) - unsafe_load(nicp, nxt_c1)
        if Int32(id) > sp
            ok, n_comp, status = cf_get_column_space!(st, ic_d, Int32(id), n_comp, status)
            if !ok
                _fs_store_i32!(st_ptr, _FS_OFF_numberDense, num_dense)
                _fs_store_i32!(st_ptr, _FS_OFF_denseThreshold, Int32(0))
                _fs_store_i32!(st_ptr, _FS_OFF_numberCompressions, n_comp)
                _fs_store_i32!(st_ptr, _FS_OFF_freeDense, Int32(0))
                _fs_store_i32!(st_ptr, _FS_OFF_status, Int32(-99))
                return Int32(-99)
            end
        end
        unsafe_store!(nic, Int32(id), ic_d1)
        id += 1
    end

    ic_f = base + num_to_do
    while ic_f < nr
        unsafe_store!(next_row, Int32(ic_f), ic_f + 1)
        unsafe_store!(start_col_l, end_l, ic_f + 2)
        unsafe_store!(piv_reg, 1.0, ic_f + 1)
        ic_f += 1
    end

    if Float64(length_l) + Float64(full) * 0.5 > Float64(st.lengthAreaL)
        _fs_store_i32!(st_ptr, _FS_OFF_numberDense, num_dense)
        _fs_store_i32!(st_ptr, _FS_OFF_denseThreshold, Int32(0))
        _fs_store_i32!(st_ptr, _FS_OFF_numberCompressions, n_comp)
        _fs_store_i32!(st_ptr, _FS_OFF_freeDense, Int32(0))
        _fs_store_i32!(st_ptr, _FS_OFF_status, Int32(-99))
        return Int32(-99)
    end

    id = 0
    while id < num_to_do
        piv_r = -1
        col_el = dense_area + (id * nd) * sizeof(Float64)
        largest = 1.0e-12
        ir = id
        while ir < nd
            av = abs(unsafe_load(col_el, ir + 1))
            if av > largest
                largest = av
                piv_r = ir
            end
            ir += 1
        end
        if piv_r >= 0
            ic_d = unsafe_load(piv_col, base + id + 1)
            ic_d1 = Int(ic_d) + 1
            piv_el = unsafe_load(col_el, piv_r + 1)
            orig_row = unsafe_load(dense_perm, piv_r + 1)
            unsafe_store!(next_row, num_good_u, Int(orig_row) + 1)
            unsafe_store!(last_row, Int32(-2), Int(orig_row) + 1)
            unsafe_store!(dense_perm, unsafe_load(dense_perm, id + 1), piv_r + 1)
            unsafe_store!(dense_perm, orig_row, id + 1)
            sw_ptr = col_el
            jd = id
            while jd < nd
                v = unsafe_load(sw_ptr, id + 1)
                unsafe_store!(sw_ptr, unsafe_load(sw_ptr, piv_r + 1), id + 1)
                unsafe_store!(sw_ptr, v, piv_r + 1)
                sw_ptr += nd * sizeof(Float64)
                jd += 1
            end
            piv_mult = 1.0 / piv_el
            unsafe_store!(piv_reg, piv_mult, Int(num_good_u) + 1)
            l = Int(length_l)
            unsafe_store!(start_col_l, Int32(l), Int(num_good_l) + 1)
            ir = id + 1
            while ir < nd
                v = unsafe_load(col_el, ir + 1) * piv_mult
                unsafe_store!(col_el, v, ir + 1)
                if abs(v) > tol
                    unsafe_store!(idx_row_l, unsafe_load(dense_perm, ir + 1), l + 1)
                    unsafe_store!(el_l, v, l + 1)
                    l += 1
                end
                ir += 1
            end
            num_good_l += Int32(1)
            length_l = Int32(l)
            unsafe_store!(start_col_l, length_l, Int(num_good_l) + 1)
            s = Int(unsafe_load(start_col_u, ic_d1))
            orig_s = s
            ir = 0
            while ir < id
                v = unsafe_load(col_el, ir + 1)
                if abs(v) > tol
                    unsafe_store!(idx_row, unsafe_load(dense_perm, ir + 1), s + 1)
                    unsafe_store!(el_u, v, s + 1)
                    s += 1
                end
                ir += 1
            end
            unsafe_store!(nic, Int32(0), ic_d1)
            unsafe_store!(nicp, unsafe_load(nicp, ic_d1) + Int32(s - orig_s), ic_d1)
            unsafe_store!(start_col_u, Int32(s), ic_d1)

            el2 = col_el + nd * sizeof(Float64)
            jd = id + 1
            while jd < num_to_do
                v = unsafe_load(el2, id + 1)
                ir = id + 1
                while ir < nd
                    unsafe_store!(el2, _fmsub(v, unsafe_load(col_el, ir + 1), unsafe_load(el2, ir + 1)), ir + 1)
                    ir += 1
                end
                el2 += nd * sizeof(Float64)
                jd += 1
            end
            num_good_u += Int32(1)
        else
            _fs_store_i32!(st_ptr, _FS_OFF_numberDense, num_dense)
            _fs_store_i32!(st_ptr, _FS_OFF_denseThreshold, Int32(0))
            _fs_store_i32!(st_ptr, _FS_OFF_numberGoodU, num_good_u)
            _fs_store_i32!(st_ptr, _FS_OFF_numberGoodL, num_good_l)
            _fs_store_i32!(st_ptr, _FS_OFF_lengthL, length_l)
            _fs_store_i32!(st_ptr, _FS_OFF_numberCompressions, n_comp)
            _fs_store_i32!(st_ptr, _FS_OFF_freeDense, Int32(0))
            _fs_store_i32!(st_ptr, _FS_OFF_status, Int32(-1))
            return Int32(-1)
        end
        id += 1
    end

    _fs_store_i32!(st_ptr, _FS_OFF_numberDense, Int32(0))
    _fs_store_i32!(st_ptr, _FS_OFF_denseThreshold, Int32(0))
    _fs_store_i32!(st_ptr, _FS_OFF_totalElements, total_elements)
    _fs_store_i32!(st_ptr, _FS_OFF_numberGoodU, num_good_u)
    _fs_store_i32!(st_ptr, _FS_OFF_numberGoodL, num_good_l)
    _fs_store_i32!(st_ptr, _FS_OFF_lengthL, length_l)
    _fs_store_i32!(st_ptr, _FS_OFF_numberCompressions, n_comp)
    _fs_store_i32!(st_ptr, _FS_OFF_freeDense, Int32(1))
    _fs_store_i32!(st_ptr, _FS_OFF_status, status)
    return status
end

# =============================================================================
# Singular post-processing in factor() (CoinFactorization1.cpp:1102-1161)
# Note: writes final permute into st.nextRow (caller then does permute_.swap(nextRow_);
# nextRow_.conditionalDelete();).
# =============================================================================

function cf_factor_singular!(st_ptr::Ptr{CoinFactorizationFactorStatePod})::Int32
    st = unsafe_load(st_ptr)
    nr = Int(st.numberRows)
    nc = Int(st.numberColumns)
    mre = st.maximumRowsExtra
    num_good_u = Int(st.numberGoodU)
    next_row = st.nextRow
    last_row = st.lastRow
    last_col = st.lastColumn
    piv_col = st.pivotColumn
    permute_a = st.permute

    k = unsafe_load(next_row, Int(mre) + 1)
    while k != mre && k >= Int32(0)
        ir = k
        k = unsafe_load(next_row, Int(k) + 1)
        unsafe_store!(next_row, Int32(-1), Int(ir) + 1)
    end
    i = 0
    while i < nr
        igood = unsafe_load(next_row, i + 1)
        if igood >= Int32(0)
            unsafe_store!(permute_a, Int32(i), Int(igood) + 1)
        end
        i += 1
    end
    # After swap in C++, permute will point to next_row, while permute_a is old permute
    i = 0
    while i < nr
        unsafe_store!(last_row, Int32(-1), i + 1)
        i += 1
    end
    i = 0
    while i < nc
        unsafe_store!(last_col, Int32(-1), i + 1)
        i += 1
    end
    i = 0
    while i < num_good_u
        good_row = unsafe_load(permute_a, i + 1)
        good_col = unsafe_load(piv_col, i + 1)
        unsafe_store!(last_row, good_col, Int(good_row) + 1)
        unsafe_store!(last_col, good_row, Int(good_col) + 1)
        i += 1
    end
    i = 0
    while i < nr
        unsafe_store!(next_row, unsafe_load(last_row, i + 1), i + 1)
        i += 1
    end
    i = 0
    while i < nc
        unsafe_store!(piv_col, unsafe_load(last_col, i + 1), i + 1)
        i += 1
    end
    _fs_store_i32!(st_ptr, _FS_OFF_totalElements, Int32(0))
    _fs_store_i32!(st_ptr, _FS_OFF_status, Int32(-1))
    return Int32(-1)
end

# =============================================================================
# cleanup (CoinFactorization1.cpp:1705-2105)
# Caller pre-allocates permuteBack (mre+1), pivotColumnBack (mre+1),
# convertRowToColumnU (lengthAreaU + 4).
# After return, caller executes:
#   numberInColumn_.swap(numberInColumnPlus_);
#   permute_.swap(nextRow_);
#   free saveColumn_, markRow_, nextCount_, lastCount_, pivotRowL_;
#   if (!useExtraR) numberInColumnPlus_.conditionalDelete();
#   allocate startColumnR_ (and zero tail if useExtraR);
#   elementR_ = elementL_ + lengthL_; indexRowR_ = indexRowL_ + lengthL_;
#   setupPointers();
# =============================================================================

function cf_cleanup!(st_ptr::Ptr{CoinFactorizationFactorStatePod})::Nothing
    st = unsafe_load(st_ptr)
    nr = Int(st.numberRows)
    nc = Int(st.numberColumns)
    mre = st.maximumRowsExtra
    mce = st.maximumColumnsExtra
    num_slacks = Int(st.numberSlacks)

    # Compress U columns
    _, n_comp, status = cf_get_column_space!(st, Int32(0), CF_COMPRESS_SENTINEL, st.numberCompressions, st.status)

    # Local pointer swaps matching C++ numberInColumn_.swap(numberInColumnPlus_) & permute_.swap(nextRow_)
    nic = st.numberInColumnPlus
    nicp = st.numberInColumn
    permute = st.nextRow
    next_row = st.permute

    start_col_u = st.startColumnU
    last_u = Int(unsafe_load(start_col_u, Int(mce) + 1))

    # permute & permuteBack
    unsafe_store!(permute, Int32(0), nr + 1)
    permute_back = st.permuteBack
    i = 0
    while i < nr
        ir = unsafe_load(permute, i + 1)
        unsafe_store!(permute_back, Int32(i), Int(ir) + 1)
        i += 1
    end

    # Redo totalElements and shift startColumnU back to beginning of each column
    total_elements = Int32(0)
    i = 0
    while i < nc
        num = unsafe_load(nic, i + 1)
        total_elements += num
        unsafe_store!(start_col_u, unsafe_load(start_col_u, i + 1) - num, i + 1)
        i += 1
    end

    piv_col = st.pivotColumn
    piv_col_b = st.pivotColumnBack
    idx_col_u = st.indexColumnU
    idx_row_u = st.indexRowU
    el_u = st.elementU

    num_u = Int32(0)
    i = 0
    while i < nc
        ic = unsafe_load(piv_col, i + 1)
        unsafe_store!(piv_col_b, Int32(i), Int(ic) + 1)
        if ic >= Int32(0)
            if num_u != ic
                unsafe_store!(nicp, num_u, Int(ic) + 1)
            else
                unsafe_store!(nicp, Int32(-1), Int(ic) + 1)
            end
            num_u += Int32(1)
        end
        i += 1
    end

    # Cycle-permute numberInColumn & startColumnU into pivot order
    i = 0
    while i < nc
        i1 = i + 1
        num = unsafe_load(nic, i1)
        w = unsafe_load(nicp, i1)
        unsafe_store!(nicp, Int32(-1), i1)
        s = unsafe_load(start_col_u, i1)
        while w >= Int32(0)
            w1 = Int(w) + 1
            num_next = unsafe_load(nic, w1)
            w_next = unsafe_load(nicp, w1)
            s_next = unsafe_load(start_col_u, w1)
            unsafe_store!(nic, num, w1)
            unsafe_store!(nicp, Int32(-1), w1)
            unsafe_store!(start_col_u, s, w1)
            num = num_next
            w = w_next
            s = s_next
        end
        i += 1
    end

    # Sort indexRowU & elementU into pivot column order using indexColumnU
    coin_fill_n!(idx_col_u, last_u, Int32(-1))
    k = Int32(0)
    i = num_slacks
    while i < nr
        s = Int(unsafe_load(start_col_u, i + 1))
        e = s + Int(unsafe_load(nic, i + 1))
        j = s
        while j < e
            unsafe_store!(idx_col_u, k, j + 1)
            k += Int32(1)
            j += 1
        end
        i += 1
    end
    i = num_slacks
    while i < nr
        s = Int(unsafe_load(start_col_u, i + 1))
        e = s + Int(unsafe_load(nic, i + 1))
        j = s
        while j < e
            kk = unsafe_load(idx_col_u, j + 1)
            ir = unsafe_load(idx_row_u, j + 1)
            ev = unsafe_load(el_u, j + 1)
            while kk != Int32(-1)
                k1 = Int(kk) + 1
                kk_next = unsafe_load(idx_col_u, k1)
                ir_next = unsafe_load(idx_row_u, k1)
                ev_next = unsafe_load(el_u, k1)
                unsafe_store!(idx_col_u, Int32(-1), k1)
                unsafe_store!(idx_row_u, ir, k1)
                unsafe_store!(el_u, ev, k1)
                kk = kk_next
                ir = ir_next
                ev = ev_next
            end
            j += 1
        end
        i += 1
    end

    coin_zero_n!(start_col_u, num_slacks)
    k = Int32(0)
    i = num_slacks
    while i < nr
        unsafe_store!(start_col_u, k, i + 1)
        k += unsafe_load(nic, i + 1)
        i += 1
    end
    max_u = k

    total_elements += st.numberDense * st.numberDense + st.lengthL
    next_col = st.nextColumn
    last_col = st.lastColumn

    use_extra_r = Int32(0)
    if !(max_u > Int32(10) * st.numberRows || st.numberRows < Int32(200))
        use_extra_r = Int32(1)
        i = 0
        while i < nc
            unsafe_store!(last_col, Int32(i - 1), i + 1)
            unsafe_store!(next_col, Int32(i + 1), i + 1)
            unsafe_store!(nicp, Int32(0), i + 1)
            i += 1
        end
        mce1 = Int(mce) + 1
        if nc > 0
            unsafe_store!(next_col, mce, nc)
        end
        unsafe_store!(last_col, Int32(nc - 1), mce1)
        unsafe_store!(next_col, Int32(0), mce1)
        unsafe_store!(last_col, mce, 1)
    end

    num_good_u = num_u
    num_l = st.numberGoodL
    nir = st.numberInRow
    piv_reg = st.pivotRegion
    nu_i = Int(num_u)

    i = num_slacks
    while i < nu_i
        s = Int(unsafe_load(start_col_u, i + 1))
        n_c = unsafe_load(nic, i + 1)
        e = s + Int(n_c)
        total_elements += n_c
        j = s
        while j < e
            ir = unsafe_load(permute, Int(unsafe_load(idx_row_u, j + 1)) + 1)
            unsafe_store!(idx_row_u, ir, j + 1)
            ir1 = Int(ir) + 1
            unsafe_store!(nir, unsafe_load(nir, ir1) + Int32(1), ir1)
            j += 1
        end
        i += 1
    end

    # Cross-reference row copy of U
    conv = st.convertRowToColumnU
    start_row = st.startRowU
    pos = Int32(0)
    ir = 0
    while ir < nr
        unsafe_store!(start_row, pos, ir + 1)
        pos += unsafe_load(nir, ir + 1)
        ir += 1
    end
    num_in_u = pos
    coin_zero_n!(nir, nr)

    i = num_slacks
    while i < nr
        s = Int(unsafe_load(start_col_u, i + 1))
        e = s + Int(unsafe_load(nic, i + 1))
        pv = unsafe_load(piv_reg, i + 1)
        j = s
        while j < e
            ir1 = Int(unsafe_load(idx_row_u, j + 1)) + 1
            lk = unsafe_load(nir, ir1)
            unsafe_store!(nir, lk + Int32(1), ir1)
            kk = Int(unsafe_load(start_row, ir1) + lk)
            unsafe_store!(idx_col_u, Int32(i), kk + 1)
            unsafe_store!(conv, Int32(j), kk + 1)
            unsafe_store!(el_u, unsafe_load(el_u, j + 1) * pv, j + 1)
            j += 1
        end
        i += 1
    end

    last_row = st.lastRow
    j = 0
    while j < nr
        unsafe_store!(last_row, Int32(j - 1), j + 1)
        unsafe_store!(next_row, Int32(j + 1), j + 1)
        j += 1
    end
    mre1 = Int(mre) + 1
    if nr > 0
        unsafe_store!(next_row, mre, nr)
        unsafe_store!(last_row, mre, 1)
    end
    unsafe_store!(last_row, Int32(nr - 1), mre1)
    unsafe_store!(next_row, Int32(0), mre1)
    unsafe_store!(start_row, num_in_u, mre1)

    # Permute L row indices and find baseL
    first_real = st.numberRows
    start_col_l = st.startColumnL
    idx_row_l = st.indexRowL
    i = nr - 1
    while i >= 0
        s = Int(unsafe_load(start_col_l, i + 1))
        e = Int(unsafe_load(start_col_l, i + 2))
        total_elements += Int32(e - s)
        if e > s
            first_real = Int32(i)
            j = s
            while j < e
                ir = unsafe_load(permute, Int(unsafe_load(idx_row_l, j + 1)) + 1)
                unsafe_store!(idx_row_l, ir, j + 1)
                j += 1
            end
        end
        i -= 1
    end
    base_l = first_real
    num_l -= first_real
    factor_elements = total_elements

    # Setup R area sizing & areaFactor adjustment
    space = st.lengthAreaL - st.lengthL
    space_used = st.lengthL + st.lengthU
    needed = div(space_used + st.numberRows - Int32(1), st.numberRows)
    needed = needed * Int32(2) * st.maximumPivots
    if needed < Int32(2) * st.numberRows
        needed = Int32(2) * st.numberRows
    end
    if use_extra_r != Int32(0)
        space = div(space, Int32(2))
    end
    area_factor = st.areaFactor
    if space < needed
        area_factor = area_factor != 0.0 ? area_factor * 1.1 : 1.1
    end

    _fs_store_f64!(st_ptr, _FS_OFF_areaFactor, area_factor)
    _fs_store_i32!(st_ptr, _FS_OFF_numberU, num_u)
    _fs_store_i32!(st_ptr, _FS_OFF_numberGoodU, num_good_u)
    _fs_store_i32!(st_ptr, _FS_OFF_maximumU, max_u)
    _fs_store_i32!(st_ptr, _FS_OFF_numberL, num_l)
    _fs_store_i32!(st_ptr, _FS_OFF_baseL, base_l)
    _fs_store_i32!(st_ptr, _FS_OFF_numberR, Int32(0))
    _fs_store_i32!(st_ptr, _FS_OFF_lengthR, Int32(0))
    _fs_store_i32!(st_ptr, _FS_OFF_lengthAreaR, space)
    _fs_store_i32!(st_ptr, _FS_OFF_totalElements, total_elements)
    _fs_store_i32!(st_ptr, _FS_OFF_factorElements, factor_elements)
    _fs_store_i32!(st_ptr, _FS_OFF_numberCompressions, n_comp)
    _fs_store_i32!(st_ptr, _FS_OFF_status, status)
    _fs_store_i32!(st_ptr, _FS_OFF_useExtraR, use_extra_r)
    return nothing
end

# =============================================================================
# goSparse (CoinFactorization4.cpp:2816-2909)
# =============================================================================

function cf_go_sparse_thresholds!(st_ptr::Ptr{CoinFactorizationFactorStatePod})::Int32
    st = unsafe_load(st_ptr)
    nr = st.numberRows
    sp_t = st.sparseThreshold
    sp_t2 = st.sparseThreshold2
    if sp_t == Int32(0)
        if nr > Int32(300)
            if nr < Int32(10000)
                sp_t = min(div(nr, Int32(6)), Int32(500))
                sp_t2 = div(nr, Int32(2))
            else
                sp_t = Int32(500)
                sp_t2 = max(sp_t, nr >> 3)
            end
        else
            sp_t = Int32(0)
            sp_t2 = Int32(0)
        end
    else
        sp_t2 = sp_t
    end
    _fs_store_i32!(st_ptr, _FS_OFF_sparseThreshold, sp_t)
    _fs_store_i32!(st_ptr, _FS_OFF_sparseThreshold2, sp_t2)
    return sp_t
end

function cf_go_sparse_build!(st_ptr::Ptr{CoinFactorizationFactorStatePod})::Nothing
    st = unsafe_load(st_ptr)
    nr = Int(st.numberRows)
    mre = Int(st.maximumRowsExtra)

    # Zero mark bitmap at sparseArray + 3 * maximumRowsExtra
    mark_ptr = Ptr{UInt8}(st.sparseArray + 3 * mre * sizeof(Int32))
    i = 1
    while i <= mre
        unsafe_store!(mark_ptr, UInt8(0), i)
        i += 1
    end

    start_row_l = st.startRowL
    coin_zero_n!(start_row_l, nr)
    start_col_l = st.startColumnL
    el_l = st.elementL
    idx_row_l = st.indexRowL
    base_l = Int(st.baseL)
    end_l_col = base_l + Int(st.numberL)

    i = base_l
    while i < end_l_col
        s = Int(unsafe_load(start_col_l, i + 1))
        e = Int(unsafe_load(start_col_l, i + 2))
        j = s
        while j < e
            r1 = Int(unsafe_load(idx_row_l, j + 1)) + 1
            unsafe_store!(start_row_l, unsafe_load(start_row_l, r1) + Int32(1), r1)
            j += 1
        end
        i += 1
    end

    cnt = Int32(0)
    i = 0
    while i < nr
        n_r = unsafe_load(start_row_l, i + 1)
        cnt += n_r
        unsafe_store!(start_row_l, cnt, i + 1)
        i += 1
    end
    unsafe_store!(start_row_l, cnt, nr + 1)

    el_by_row_l = st.elementByRowL
    idx_col_l = st.indexColumnL
    i = end_l_col - 1
    while i >= base_l
        s = Int(unsafe_load(start_col_l, i + 1))
        e = Int(unsafe_load(start_col_l, i + 2))
        j = s
        while j < e
            r1 = Int(unsafe_load(idx_row_l, j + 1)) + 1
            pos = unsafe_load(start_row_l, r1) - Int32(1)
            unsafe_store!(start_row_l, pos, r1)
            p1 = Int(pos) + 1
            unsafe_store!(el_by_row_l, unsafe_load(el_l, j + 1), p1)
            unsafe_store!(idx_col_l, Int32(i), p1)
            j += 1
        end
        i -= 1
    end
    return nothing
end

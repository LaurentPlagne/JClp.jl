# Opaque integer-handle registry for Julia-owned objects crossing the C++ bridge.
# Strictly concrete `Vector{T}` + free-list (zero Union/Any dispatch, --trim=safe clean,
# O(1) lookup with zero heap allocation on every hot call).

const _SPARSE_LU_POOL = SparseLU[]
const _SPARSE_LU_FREE = Int32[]

function _lu_handle_create()::Int32
    if !isempty(_SPARSE_LU_FREE)
        id = pop!(_SPARSE_LU_FREE)
        @inbounds _SPARSE_LU_POOL[Int(id)] = SparseLU()
        return id
    end
    push!(_SPARSE_LU_POOL, SparseLU())
    return Int32(length(_SPARSE_LU_POOL))
end

function _lu_handle_copy(src_id::Int32)::Int32
    @inbounds src = _SPARSE_LU_POOL[Int(src_id)]
    dup = copy_lu(src)
    if !isempty(_SPARSE_LU_FREE)
        id = pop!(_SPARSE_LU_FREE)
        @inbounds _SPARSE_LU_POOL[Int(id)] = dup
        return id
    end
    push!(_SPARSE_LU_POOL, dup)
    return Int32(length(_SPARSE_LU_POOL))
end

function _lu_handle_assign!(dst_id::Int32, src_id::Int32)::Nothing
    if dst_id != src_id
        @inbounds _SPARSE_LU_POOL[Int(dst_id)] = copy_lu(_SPARSE_LU_POOL[Int(src_id)])
    end
    return nothing
end

function _lu_handle_destroy!(id::Int32)::Nothing
    if id > Int32(0) && Int(id) <= length(_SPARSE_LU_POOL)
        @inbounds _SPARSE_LU_POOL[Int(id)] = SparseLU()
        push!(_SPARSE_LU_FREE, id)
    end
    return nothing
end

@inline function _lu_get(id::Int32)::SparseLU
    @inbounds return _SPARSE_LU_POOL[Int(id)]
end

# Exact Julia mirrors of bridge/jclp_presolve_pod.h (bit-identical C ABI layout on 64-bit).

const PRESOLVE_NO_LINK   = Int32(-66666666)
const PRESOLVE_ZTOLDP    = 1.0e-12
const PRESOLVE_ZTOLDP2   = 1.0e-10
const PRESOLVE_INF_DBL   = floatmax(Float64)
const PRESOLVE_SMALL_INF = 1.0e20

const PS_STATUS_IS_FREE        = UInt8(0x00)
const PS_STATUS_BASIC          = UInt8(0x01)
const PS_STATUS_AT_UPPER_BOUND = UInt8(0x02)
const PS_STATUS_AT_LOWER_BOUND = UInt8(0x03)
const PS_STATUS_SUPER_BASIC    = UInt8(0x04)

@inline _fmadd(a::Float64, b::Float64, c::Float64)::Float64 = fma(a, b, c)

struct PresolveHLinkPod
    pre::Int32
    suc::Int32
end

struct CoinPresolvePodView
    # Column-major
    mcstrt::Ptr{Int32}
    hincol::Ptr{Int32}
    hrow::Ptr{Int32}
    colels::Ptr{Float64}
    clink::Ptr{PresolveHLinkPod}

    # Row-major
    mrstrt::Ptr{Int32}
    hinrow::Ptr{Int32}
    hcol::Ptr{Int32}
    rowels::Ptr{Float64}
    rlink::Ptr{PresolveHLinkPod}

    # Rim & solution
    clo::Ptr{Float64}
    cup::Ptr{Float64}
    rlo::Ptr{Float64}
    rup::Ptr{Float64}
    cost::Ptr{Float64}
    sol::Ptr{Float64}
    acts::Ptr{Float64}
    rowduals::Ptr{Float64}
    rcosts::Ptr{Float64}
    colstat::Ptr{UInt8}
    rowstat::Ptr{UInt8}
    integerType::Ptr{UInt8}

    # Original index maps
    originalColumn::Ptr{Int32}
    originalRow::Ptr{Int32}

    # Change & ToDo tracking
    colChanged::Ptr{UInt8}
    rowChanged::Ptr{UInt8}
    colsToDo::Ptr{Int32}
    nextColsToDo::Ptr{Int32}
    rowsToDo::Ptr{Int32}
    nextRowsToDo::Ptr{Int32}

    # Scratch work arrays
    usefulRowInt::Ptr{Int32}
    usefulColumnInt::Ptr{Int32}
    infiniteUp::Ptr{Int32}
    infiniteDown::Ptr{Int32}
    usefulRowDouble::Ptr{Float64}
    usefulColumnDouble::Ptr{Float64}
    randomNumber::Ptr{Float64}
    sumUp::Ptr{Float64}
    sumDown::Ptr{Float64}

    # 5 doubles
    dobias::Float64
    maxmin::Float64
    ztolzb::Float64
    ztoldj::Float64
    feasibilityTolerance::Float64

    # 16 int32 scalars
    nrows::Int32
    ncols::Int32
    nrows0::Int32
    ncols0::Int32
    nelems::Int32
    bulk0::Int32
    status::Int32
    presolveOptions::Int32
    anyProhibited::Int32
    anyInteger::Int32
    numberColsToDo::Int32
    numberNextColsToDo::Int32
    numberRowsToDo::Int32
    numberNextRowsToDo::Int32
    pass::Int32
    _pad::Int32
end

struct CoinPostsolvePodView
    mcstrt::Ptr{Int32}
    hincol::Ptr{Int32}
    hrow::Ptr{Int32}
    link::Ptr{Int32}
    colels::Ptr{Float64}

    clo::Ptr{Float64}
    cup::Ptr{Float64}
    rlo::Ptr{Float64}
    rup::Ptr{Float64}
    cost::Ptr{Float64}
    sol::Ptr{Float64}
    acts::Ptr{Float64}
    rowduals::Ptr{Float64}
    rcosts::Ptr{Float64}
    colstat::Ptr{UInt8}
    rowstat::Ptr{UInt8}

    maxmin::Float64
    ztolzb::Float64
    ztoldj::Float64

    nrows::Int32
    ncols::Int32
    nrows0::Int32
    ncols0::Int32
    nelems::Int32
    bulk0::Int32
    free_list::Int32
    _pad::Int32
end

# Action POD structs
struct DroppedZeroPod
    row::Int32
    col::Int32
end

struct DropEmptyColActionPod
    clo::Float64
    cup::Float64
    cost::Float64
    sol::Float64
    jcol::Int32
    _pad::Int32
end

struct DropEmptyRowActionPod
    rlo::Float64
    rup::Float64
    row::Int32
    fill_row::Int32
end

struct UselessActionPod
    rlo::Float64
    rup::Float64
    rowcols::Ptr{Int32}
    rowels::Ptr{Float64}
    row::Int32
    ninrow::Int32
end

struct RemoveFixedActionPod
    col::Int32
    start::Int32
    sol::Float64
end

struct MakeFixedActionPod
    bound::Float64
    col::Int32
    _pad::Int32
end

# B7b Action POD structs
struct TightenActionPod
    rows::Ptr{Int32}
    lbound::Ptr{Float64}
    ubound::Ptr{Float64}
    col::Int32
    nrows::Int32
    direction::Int32
    _pad::Int32
end

struct ForcingActionPod
    rowcols::Ptr{Int32}
    bounds::Ptr{Float64}
    row::Int32
    nlo::Int32
    nup::Int32
    _pad::Int32
end

struct SlackActionPod
    clo::Float64
    cup::Float64
    rlo::Float64
    rup::Float64
    coeff::Float64
    col::Int32
    row::Int32
end

struct DualActionPod
    rlo::Float64
    rup::Float64
    ndx::Int32
    _pad::Int32
end

# Compile-time byte offsets for mutable scalar write-back
const _PS_OFF_dobias             = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :dobias))
const _PS_OFF_nrows              = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :nrows))
const _PS_OFF_ncols              = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :ncols))
const _PS_OFF_nelems             = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :nelems))
const _PS_OFF_status             = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :status))
const _PS_OFF_numberColsToDo     = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :numberColsToDo))
const _PS_OFF_numberNextColsToDo = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :numberNextColsToDo))
const _PS_OFF_numberRowsToDo     = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :numberRowsToDo))
const _PS_OFF_numberNextRowsToDo = fieldoffset(CoinPresolvePodView, Base.fieldindex(CoinPresolvePodView, :numberNextRowsToDo))

const _POST_OFF_nrows     = fieldoffset(CoinPostsolvePodView, Base.fieldindex(CoinPostsolvePodView, :nrows))
const _POST_OFF_ncols     = fieldoffset(CoinPostsolvePodView, Base.fieldindex(CoinPostsolvePodView, :ncols))
const _POST_OFF_free_list = fieldoffset(CoinPostsolvePodView, Base.fieldindex(CoinPostsolvePodView, :free_list))

@inline _ps_store_i32!(p::Ptr{CoinPresolvePodView}, off::UInt, v::Int32) =
    (unsafe_store!(Ptr{Int32}(UInt(p) + off), v); nothing)
@inline _ps_store_f64!(p::Ptr{CoinPresolvePodView}, off::UInt, v::Float64) =
    (unsafe_store!(Ptr{Float64}(UInt(p) + off), v); nothing)
@inline _post_store_i32!(p::Ptr{CoinPostsolvePodView}, off::UInt, v::Int32) =
    (unsafe_store!(Ptr{Int32}(UInt(p) + off), v); nothing)

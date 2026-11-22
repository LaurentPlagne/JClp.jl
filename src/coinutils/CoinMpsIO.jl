# Faithful translation of CoinUtils/src/CoinMpsIO.{hpp,cpp} (B6):
#   - CoinMpsCardReader: osi_strtod (decimal & IEEE), cleanCard, nextBlankOr,
#     strcpyAndCompress, readToNextSection, nextField
#   - CoinMpsIO: hash, startHash, findHash, convertBoundToSense, convertSenseToBound,
#     CoinConvertDouble
# Zero-allocation on hot paths, bit-identical arm64 FMA in osi_strtod, `--trim=safe` compatible.

# --- Enums matching CoinMpsIO.hpp ---
const COIN_NO_SECTION        = Int32(0)
const COIN_NAME_SECTION      = Int32(1)
const COIN_ROW_SECTION       = Int32(2)
const COIN_COLUMN_SECTION    = Int32(3)
const COIN_RHS_SECTION       = Int32(4)
const COIN_RANGES_SECTION    = Int32(5)
const COIN_BOUNDS_SECTION    = Int32(6)
const COIN_ENDATA_SECTION    = Int32(7)
const COIN_EOF_SECTION       = Int32(8)
const COIN_QUADRATIC_SECTION = Int32(9)
const COIN_CONIC_SECTION     = Int32(10)
const COIN_QUAD_SECTION      = Int32(11)
const COIN_SOS_SECTION       = Int32(12)
const COIN_BASIS_SECTION     = Int32(13)
const COIN_INDICATOR_SECTION = Int32(14)
const COIN_UNKNOWN_SECTION   = Int32(15)

const COIN_N_ROW             = Int32(0)
const COIN_E_ROW             = Int32(1)
const COIN_L_ROW             = Int32(2)
const COIN_G_ROW             = Int32(3)
const COIN_BLANK_COLUMN      = Int32(4)
const COIN_S1_COLUMN         = Int32(5)
const COIN_S2_COLUMN         = Int32(6)
const COIN_S3_COLUMN         = Int32(7)
const COIN_INTORG            = Int32(8)
const COIN_INTEND            = Int32(9)
const COIN_SOSEND            = Int32(10)
const COIN_UNSET_BOUND       = Int32(11)
const COIN_UP_BOUND          = Int32(12)
const COIN_FX_BOUND          = Int32(13)
const COIN_LO_BOUND          = Int32(14)
const COIN_FR_BOUND          = Int32(15)
const COIN_MI_BOUND          = Int32(16)
const COIN_PL_BOUND          = Int32(17)
const COIN_BV_BOUND          = Int32(18)
const COIN_UI_BOUND          = Int32(19)
const COIN_LI_BOUND          = Int32(20)
const COIN_BOTH_BOUNDS_SET   = Int32(21)
const COIN_SC_BOUND          = Int32(22)
const COIN_S1_BOUND          = Int32(23)
const COIN_S2_BOUND          = Int32(24)
const COIN_BS_BASIS          = Int32(25)
const COIN_XL_BASIS          = Int32(26)
const COIN_XU_BASIS          = Int32(27)
const COIN_LL_BASIS          = Int32(28)
const COIN_UL_BASIS          = Int32(29)
const COIN_IF_INDICATOR      = Int32(30)
const COIN_UNKNOWN_MPS_TYPE  = Int32(31)

const MPS_STRING_VALUE = -1.234567e-101

# --- POD structs mirroring bridge/jclp_mps_pod.h ---

struct CoinHashLinkPod
    index::Int32
    next::Int32
end

struct CoinMpsCardReaderPod
    card::Ptr{UInt8}
    position::Ptr{UInt8}
    eol::Ptr{UInt8}
    rowName::Ptr{UInt8}
    columnName::Ptr{UInt8}
    valueString::Ptr{UInt8}
    gets_cb::Ptr{Cvoid}
    log_line_cb::Ptr{Cvoid}
    cb_ctx::Ptr{Cvoid}
    _pad_ptr::Ptr{Cvoid}

    value::Float64

    mpsType::Int32
    section::Int32
    cardNumber::Int32
    freeFormat::Int32
    ieeeFormat::Int32
    eightChar::Int32
    stringsAllowed::Int32
    maxCardLength::Int32
end

const _MR_OFF_position   = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :position))
const _MR_OFF_eol        = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :eol))
const _MR_OFF_value      = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :value))
const _MR_OFF_mpsType    = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :mpsType))
const _MR_OFF_section    = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :section))
const _MR_OFF_cardNumber = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :cardNumber))
const _MR_OFF_freeFormat = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :freeFormat))
const _MR_OFF_ieeeFormat = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :ieeeFormat))
const _MR_OFF_eightChar  = fieldoffset(CoinMpsCardReaderPod, Base.fieldindex(CoinMpsCardReaderPod, :eightChar))

@inline _mr_store_ptr!(p::Ptr{CoinMpsCardReaderPod}, off::UInt, v::Ptr{UInt8}) =
    (unsafe_store!(Ptr{Ptr{UInt8}}(UInt(p) + off), v); nothing)
@inline _mr_store_f64!(p::Ptr{CoinMpsCardReaderPod}, off::UInt, v::Float64) =
    (unsafe_store!(Ptr{Float64}(UInt(p) + off), v); nothing)
@inline _mr_store_i32!(p::Ptr{CoinMpsCardReaderPod}, off::UInt, v::Int32) =
    (unsafe_store!(Ptr{Int32}(UInt(p) + off), v); nothing)

# --- Constant tables (isbits tuples for zero-alloc --trim=safe indexing) ---

const _MPS_FRACTION = (
    1.0, 1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 1.0e-8,
    1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13, 1.0e-14, 1.0e-15, 1.0e-16,
    1.0e-17, 1.0e-18, 1.0e-19, 1.0e-20, 1.0e-21, 1.0e-22, 1.0e-23,
)

const _MPS_EXPONENT = (
    1.0e-9, 1.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1,
    1.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 1.0e8, 1.0e9,
)

const _MPS_MMULT = (
    Int32(262139), Int32(259459), Int32(256889), Int32(254291), Int32(251701), Int32(249133), Int32(246709), Int32(244247),
    Int32(241667), Int32(239179), Int32(236609), Int32(233983), Int32(231289), Int32(228859), Int32(226357), Int32(223829),
    Int32(221281), Int32(218849), Int32(216319), Int32(213721), Int32(211093), Int32(208673), Int32(206263), Int32(203773),
    Int32(201233), Int32(198637), Int32(196159), Int32(193603), Int32(191161), Int32(188701), Int32(186149), Int32(183761),
    Int32(181303), Int32(178873), Int32(176389), Int32(173897), Int32(171469), Int32(169049), Int32(166471), Int32(163871),
    Int32(161387), Int32(158941), Int32(156437), Int32(153949), Int32(151531), Int32(149159), Int32(146749), Int32(144299),
    Int32(141709), Int32(139369), Int32(136889), Int32(134591), Int32(132169), Int32(129641), Int32(127343), Int32(124853),
    Int32(122477), Int32(120163), Int32(117757), Int32(115361), Int32(112979), Int32(110567), Int32(108179), Int32(105727),
    Int32(103387), Int32(101021), Int32(98639),  Int32(96179),  Int32(93911),  Int32(91583),  Int32(89317),  Int32(86939),
    Int32(84521),  Int32(82183),  Int32(79939),  Int32(77587),  Int32(75307),  Int32(72959),  Int32(70793),  Int32(68447),
    Int32(66103),
)

const _MPS_START_TYPE = (
    COIN_UNKNOWN_MPS_TYPE, COIN_UNKNOWN_MPS_TYPE,
    COIN_N_ROW, COIN_BLANK_COLUMN,
    COIN_BLANK_COLUMN, COIN_BLANK_COLUMN,
    COIN_UP_BOUND, COIN_UNKNOWN_MPS_TYPE,
    COIN_UNKNOWN_MPS_TYPE,
    COIN_BLANK_COLUMN, COIN_BLANK_COLUMN, COIN_BLANK_COLUMN, COIN_S1_BOUND,
    COIN_BS_BASIS, COIN_IF_INDICATOR, COIN_UNKNOWN_MPS_TYPE,
)

const _MPS_END_TYPE = (
    COIN_UNKNOWN_MPS_TYPE, COIN_UNKNOWN_MPS_TYPE,
    COIN_BLANK_COLUMN, COIN_UNSET_BOUND,
    COIN_S1_COLUMN, COIN_S1_COLUMN,
    COIN_UNKNOWN_MPS_TYPE, COIN_UNKNOWN_MPS_TYPE,
    COIN_UNKNOWN_MPS_TYPE,
    COIN_BLANK_COLUMN, COIN_BLANK_COLUMN, COIN_BLANK_COLUMN, COIN_BS_BASIS,
    COIN_UNKNOWN_MPS_TYPE, COIN_UNKNOWN_MPS_TYPE, COIN_UNKNOWN_MPS_TYPE,
)

const _MPS_ALLOWED_LEN = (
    Int32(0), Int32(0),
    Int32(1), Int32(2),
    Int32(0), Int32(0),
    Int32(2), Int32(0),
    Int32(0), Int32(0),
    Int32(0), Int32(0),
    Int32(0), Int32(2),
    Int32(2), Int32(0),
)

# 2-byte codes packed as (UInt8(c1), UInt8(c2)) for mpsTypes[0..31]
const _MPS_TYPE_BYTES = (
    (UInt8('N'), UInt8(0)),   (UInt8('E'), UInt8(0)),   (UInt8('L'), UInt8(0)),   (UInt8('G'), UInt8(0)),
    (UInt8(' '), UInt8(' ')), (UInt8('S'), UInt8('1')), (UInt8('S'), UInt8('2')), (UInt8('S'), UInt8('3')),
    (UInt8(' '), UInt8(' ')), (UInt8(' '), UInt8(' ')), (UInt8(' '), UInt8(' ')),
    (UInt8(' '), UInt8(' ')), (UInt8('U'), UInt8('P')), (UInt8('F'), UInt8('X')), (UInt8('L'), UInt8('O')),
    (UInt8('F'), UInt8('R')), (UInt8('M'), UInt8('I')), (UInt8('P'), UInt8('L')), (UInt8('B'), UInt8('V')),
    (UInt8('U'), UInt8('I')), (UInt8('L'), UInt8('I')), (UInt8('X'), UInt8('X')), (UInt8('S'), UInt8('C')),
    (UInt8('X'), UInt8('1')), (UInt8('X'), UInt8('2')), (UInt8('B'), UInt8('S')), (UInt8('X'), UInt8('L')),
    (UInt8('X'), UInt8('U')), (UInt8('L'), UInt8('L')), (UInt8('U'), UInt8('L')), (UInt8('I'), UInt8('F')),
    (UInt8(' '), UInt8(' ')),
)

# --- Low-level C-string helpers (zero-alloc, --trim=safe) ---

@inline function _c_strlen(s::Ptr{UInt8})::Int
    return Int(ccall(:strlen, Csize_t, (Ptr{UInt8},), s))
end

@inline function _c_strcmp(a::Ptr{UInt8}, b::Ptr{UInt8})::Int32
    return Int32(ccall(:strcmp, Cint, (Ptr{UInt8}, Ptr{UInt8}), a, b))
end

@inline function _c_strcpy!(dst::Ptr{UInt8}, src::Ptr{UInt8})::Nothing
    i = 1
    while true
        b = unsafe_load(src, i)
        unsafe_store!(dst, b, i)
        b == 0x00 && break
        i += 1
    end
    return nothing
end

@inline function _c_startswith(p::Ptr{UInt8}, s::NTuple{N,UInt8})::Bool where {N}
    i = 1
    while i <= N
        unsafe_load(p, i) == s[i] || return false
        i += 1
    end
    return true
end

@inline function _c_contains(p::Ptr{UInt8}, pat::NTuple{N,UInt8})::Bool where {N}
    cur = p
    while true
        b = unsafe_load(cur)
        b == 0x00 && return false
        if b == pat[1]
            ok = true
            j = 2
            while j <= N
                if unsafe_load(cur, j) != pat[j]
                    ok = false
                    break
                end
                j += 1
            end
            ok && return true
        end
        cur += 1
    end
end

@inline function _match_section(card::Ptr{UInt8})::Int32
    # Checks section[COIN_ROW_SECTION (2) .. COIN_INDICATOR_SECTION (14)]
    # "ROW", "COLUMN", "RHS", "RANGES", "BOUNDS", "ENDATA", " ",
    # "QSECTION", "CSECTION", "QUADOBJ", "SOS", "BASIS", "INDICATORS"
    b0 = unsafe_load(card, 1)
    if b0 == UInt8('R')
        _c_startswith(card, (UInt8('R'), UInt8('O'), UInt8('W'))) && return COIN_ROW_SECTION
        _c_startswith(card, (UInt8('R'), UInt8('H'), UInt8('S'))) && return COIN_RHS_SECTION
        _c_startswith(card, (UInt8('R'), UInt8('A'), UInt8('N'), UInt8('G'), UInt8('E'), UInt8('S'))) && return COIN_RANGES_SECTION
    elseif b0 == UInt8('C')
        _c_startswith(card, (UInt8('C'), UInt8('O'), UInt8('L'), UInt8('U'), UInt8('M'), UInt8('N'))) && return COIN_COLUMN_SECTION
        _c_startswith(card, (UInt8('C'), UInt8('S'), UInt8('E'), UInt8('C'), UInt8('T'), UInt8('I'), UInt8('O'), UInt8('N'))) && return COIN_CONIC_SECTION
    elseif b0 == UInt8('B')
        _c_startswith(card, (UInt8('B'), UInt8('O'), UInt8('U'), UInt8('N'), UInt8('D'), UInt8('S'))) && return COIN_BOUNDS_SECTION
        _c_startswith(card, (UInt8('B'), UInt8('A'), UInt8('S'), UInt8('I'), UInt8('S'))) && return COIN_BASIS_SECTION
    elseif b0 == UInt8('E')
        _c_startswith(card, (UInt8('E'), UInt8('N'), UInt8('D'), UInt8('A'), UInt8('T'), UInt8('A'))) && return COIN_ENDATA_SECTION
    elseif b0 == UInt8(' ')
        return COIN_EOF_SECTION
    elseif b0 == UInt8('Q')
        _c_startswith(card, (UInt8('Q'), UInt8('S'), UInt8('E'), UInt8('C'), UInt8('T'), UInt8('I'), UInt8('O'), UInt8('N'))) && return COIN_QUADRATIC_SECTION
        _c_startswith(card, (UInt8('Q'), UInt8('U'), UInt8('A'), UInt8('D'), UInt8('O'), UInt8('B'), UInt8('J'))) && return COIN_QUAD_SECTION
    elseif b0 == UInt8('S')
        _c_startswith(card, (UInt8('S'), UInt8('O'), UInt8('S'))) && return COIN_SOS_SECTION
    elseif b0 == UInt8('I')
        _c_startswith(card, (UInt8('I'), UInt8('N'), UInt8('D'), UInt8('I'), UInt8('C'), UInt8('A'), UInt8('T'), UInt8('O'), UInt8('R'), UInt8('S'))) && return COIN_INDICATOR_SECTION
    end
    return COIN_UNKNOWN_SECTION
end

# =============================================================================
# 1. CoinMpsCardReader::osi_strtod (CoinMpsIO.cpp:38-226)
# =============================================================================

function mps_osi_strtod_string(
    ptr::Ptr{UInt8},
    out_ptr::Ptr{Ptr{UInt8}},
    strings_allowed::Bool,
    value_string::Ptr{UInt8},
)::Float64
    save = ptr
    val = -1.0e100
    if !strings_allowed
        unsafe_store!(out_ptr, save)
    else
        while true
            b = unsafe_load(ptr)
            (b == UInt8(' ') || b == UInt8('\t')) || break
            ptr += 1
        end
        if unsafe_load(ptr) == UInt8('=')
            _c_strcpy!(value_string, ptr)
            val = MPS_STRING_VALUE
            unsafe_store!(out_ptr, ptr + _c_strlen(ptr))
        else
            unsafe_store!(out_ptr, save)
        end
    end
    return val
end

@inline function _decode_ieee_6bit(b::UInt8)::Int32
    if b >= UInt8('0') && b <= UInt8('9')
        return Int32(b - UInt8('0'))
    elseif b >= UInt8('a') && b <= UInt8('z')
        return Int32(b - UInt8('a')) + Int32(10)
    elseif b >= UInt8('A') && b <= UInt8('Z')
        return Int32(b - UInt8('A')) + Int32(36)
    elseif b == UInt8('*') || b == UInt8('+')
        return Int32(b - UInt8('*')) + Int32(62)
    else
        return Int32(-1)
    end
end

function mps_osi_strtod(
    ptr::Ptr{UInt8},
    out_ptr::Ptr{Ptr{UInt8}},
    fmt_type::Int32,
    strings_allowed::Bool,
    value_string::Ptr{UInt8},
)::Float64
    val = 0.0
    save = ptr
    while true
        b = unsafe_load(ptr)
        (b == UInt8(' ') || b == UInt8('\t')) || break
        ptr += 1
    end

    if fmt_type == Int32(0)
        sign1 = 1.0
        b = unsafe_load(ptr)
        if b == UInt8('-')
            sign1 = -1.0
            ptr += 1
        elseif b == UInt8('+')
            ptr += 1
        end
        while true
            b = unsafe_load(ptr)
            (b == UInt8(' ') || b == UInt8('\t')) || break
            ptr += 1
        end
        this_char = 0x00
        while val < 1.0e30
            this_char = unsafe_load(ptr)
            ptr += 1
            if this_char >= UInt8('0') && this_char <= UInt8('9')
                # Match clang++ -O3 arm64: fmadd(val, 10.0, Float64(this_char)) + (-48.0)
                val = fma(val, 10.0, Float64(this_char)) - 48.0
            else
                break
            end
        end
        if val < 1.0e30
            if this_char == UInt8('.')
                val2 = 0.0
                nfrac = 0
                while nfrac < 24
                    this_char = unsafe_load(ptr)
                    ptr += 1
                    if this_char >= UInt8('0') && this_char <= UInt8('9')
                        val2 = fma(val2, 10.0, Float64(this_char)) - 48.0
                        nfrac += 1
                    else
                        break
                    end
                end
                if nfrac < 24
                    # Match clang++ -O3 arm64: fmadd(val2, fraction[nfrac], val)
                    val = fma(val2, @inbounds(_MPS_FRACTION[nfrac + 1]), val)
                else
                    this_char = UInt8('x')
                end
            end
            if this_char == UInt8('e') || this_char == UInt8('E')
                sign2 = Int32(1)
                eb = unsafe_load(ptr)
                if eb == UInt8('-')
                    sign2 = Int32(-1)
                    ptr += 1
                elseif eb == UInt8('+')
                    ptr += 1
                end
                val3 = Int32(0)
                while val3 < Int32(1000)
                    this_char = unsafe_load(ptr)
                    ptr += 1
                    if this_char >= UInt8('0') && this_char <= UInt8('9')
                        val3 = val3 * Int32(10) + Int32(this_char - UInt8('0'))
                    else
                        break
                    end
                end
                if val3 < Int32(300)
                    val3 *= sign2
                    if abs(val3) < Int32(10)
                        val *= @inbounds(_MPS_EXPONENT[Int(val3) + 10])
                    else
                        val *= ccall(:pow, Float64, (Float64, Float64), 10.0, Float64(val3))
                    end
                elseif sign2 < Int32(0)
                    val = 0.0
                else
                    val = floatmax(Float64)
                end
            end
            if this_char == 0x00 || this_char == UInt8('\t') || this_char == UInt8(' ')
                unsafe_store!(out_ptr, ptr)
            else
                val = mps_osi_strtod_string(save, out_ptr, strings_allowed, value_string)
                sign1 = 1.0
            end
        else
            val = mps_osi_strtod_string(save, out_ptr, strings_allowed, value_string)
            sign1 = 1.0
        end
        val *= sign1
    else
        # IEEE base-64 decoding: 4 x 16-bit chunks from 12 chars
        unsafe_store!(out_ptr, ptr + 12)
        u64 = UInt64(0)
        if fmt_type == Int32(1)
            # INTEL (little-endian): i = 3 down to 0 -> chunk i at bit shift 16*i
            i = 3
            while i >= 0
                v16 = UInt16(0)
                j = 2
                while j >= 0
                    v16 = v16 << 6
                    d = _decode_ieee_6bit(unsafe_load(ptr, j + 1))
                    if d >= Int32(0)
                        v16 |= UInt16(d)
                    else
                        unsafe_store!(out_ptr, save)
                    end
                    j -= 1
                end
                ptr += 3
                u64 |= UInt64(v16) << (16 * i)
                i -= 1
            end
        else
            i = 0
            while i < 4
                v16 = UInt16(0)
                j = 2
                while j >= 0
                    v16 = v16 << 6
                    d = _decode_ieee_6bit(unsafe_load(ptr, j + 1))
                    if d >= Int32(0)
                        v16 |= UInt16(d)
                    else
                        unsafe_store!(out_ptr, save)
                    end
                    j -= 1
                end
                ptr += 3
                u64 |= UInt64(v16) << (16 * i)
                i += 1
            end
        end
        val = reinterpret(Float64, u64)
    end
    return val
end

# =============================================================================
# 2. CoinMpsCardReader: nextBlankOr, strcpyAndCompress, cleanCard,
#    readToNextSection, nextField (CoinMpsIO.cpp:275-932)
# =============================================================================

function mps_next_blank_or(image::Ptr{UInt8})::Ptr{UInt8}
    save_image = image
    while true
        b = unsafe_load(image)
        if b == UInt8(' ') || b == UInt8('\t')
            break
        end
        if b == 0x00
            return Ptr{UInt8}(C_NULL)
        end
        image += 1
    end
    if (UInt(image) - UInt(save_image)) == UInt(1)
        sb = unsafe_load(save_image)
        if sb == UInt8('+') || sb == UInt8('-')
            while true
                b = unsafe_load(image)
                (b == UInt8(' ') || b == UInt8('\t')) || break
                image += 1
            end
            image = mps_next_blank_or(image)
        end
    end
    return image
end

function mps_strcpy_and_compress!(to::Ptr{UInt8}, from::Ptr{UInt8})::Nothing
    n = _c_strlen(from)
    nto = 0
    i = 1
    while i <= n
        b = unsafe_load(from, i)
        if b != UInt8(' ') && b != UInt8('\t')
            nto += 1
            unsafe_store!(to, b, nto)
        end
        i += 1
    end
    if nto == 0
        nto = 1
        unsafe_store!(to, UInt8(' '), 1)
    end
    unsafe_store!(to, 0x00, nto + 1)
    return nothing
end

function mps_clean_card!(r_ptr::Ptr{CoinMpsCardReaderPod})::Int32
    r = unsafe_load(r_ptr)
    card = r.card
    got = ccall(r.gets_cb, Ptr{UInt8}, (Ptr{Cvoid}, Ptr{UInt8}, Cint), r.cb_ctx, card, Cint(r.maxCardLength))
    if got == C_NULL
        return Int32(1)
    end
    _mr_store_i32!(r_ptr, _MR_OFF_cardNumber, r.cardNumber + Int32(1))
    last_nb = card - 1
    img = card
    tabs = false
    while true
        b = unsafe_load(img)
        b == 0x00 && break
        if b != UInt8('\t') && b < UInt8(' ')
            break
        elseif b != UInt8('\t') && b != UInt8(' ')
            last_nb = img
        elseif b == UInt8('\t')
            tabs = true
        end
        img += 1
    end
    unsafe_store!(last_nb + 1, 0x00)
    if tabs && r.section == COIN_BOUNDS_SECTION && r.freeFormat == Int32(0) && r.eightChar != Int32(0)
        len = Int(UInt(last_nb + 1) - UInt(card))
        i = 0
        while i < len
            unsafe_store!(card, unsafe_load(card, i + 1), i + 83)
            i += 1
        end
        stops = (1, 4, 14, 24, 1000)
        put = 0
        tab = 1
        i = 0
        while i < len
            look = unsafe_load(card, i + 83)
            if look != UInt8('\t')
                unsafe_store!(card, look, put + 1)
                put += 1
            else
                while tab <= 5
                    st_pos = @inbounds stops[tab]
                    if put < st_pos
                        while put < st_pos
                            unsafe_store!(card, UInt8(' '), put + 1)
                            put += 1
                        end
                        break
                    end
                    tab += 1
                end
            end
            i += 1
        end
        unsafe_store!(card, 0x00, put + 1)
    end
    return Int32(0)
end

@inline function _mps_log_line!(r::CoinMpsCardReaderPod, card_num::Int32)::Nothing
    if r.log_line_cb != C_NULL
        ccall(r.log_line_cb, Cvoid, (Ptr{Cvoid}, Cint, Ptr{UInt8}), r.cb_ctx, Cint(card_num), r.card)
    end
    return nothing
end

function mps_read_to_next_section!(r_ptr::Ptr{CoinMpsCardReaderPod})::Int32
    while true
        if mps_clean_card!(r_ptr) != Int32(0)
            _mr_store_i32!(r_ptr, _MR_OFF_section, COIN_EOF_SECTION)
            return COIN_EOF_SECTION
        end
        r = unsafe_load(r_ptr)
        card = r.card
        is_hdr = _c_startswith(card, (UInt8('N'), UInt8('A'), UInt8('M'), UInt8('E'))) ||
                 _c_startswith(card, (UInt8('T'), UInt8('I'), UInt8('M'), UInt8('E'))) ||
                 _c_startswith(card, (UInt8('B'), UInt8('A'), UInt8('S'), UInt8('I'), UInt8('S'))) ||
                 _c_startswith(card, (UInt8('S'), UInt8('T'), UInt8('O'), UInt8('C'), UInt8('H')))
        if is_hdr
            _mr_store_i32!(r_ptr, _MR_OFF_section, COIN_NAME_SECTION)
            next = card + 5
            eol = card + _c_strlen(card)
            _mr_store_ptr!(r_ptr, _MR_OFF_position, eol)
            _mr_store_ptr!(r_ptr, _MR_OFF_eol, eol)
            _mps_log_line!(r, r.cardNumber)
            while UInt(next) < UInt(eol)
                b = unsafe_load(next)
                (b == UInt8(' ') || b == UInt8('\t')) || break
                next += 1
            end
            if UInt(next) < UInt(eol)
                nb = mps_next_blank_or(next)
                if nb != C_NULL
                    save = unsafe_load(nb)
                    unsafe_store!(nb, 0x00)
                    _c_strcpy!(r.columnName, next)
                    unsafe_store!(nb, save)
                    if _c_contains(nb, (UInt8('F'), UInt8('R'), UInt8('E'), UInt8('E'), UInt8('I'), UInt8('E'), UInt8('E'), UInt8('E')))
                        _mr_store_i32!(r_ptr, _MR_OFF_freeFormat, Int32(1))
                        _mr_store_i32!(r_ptr, _MR_OFF_ieeeFormat, Int32(1))
                    elseif _c_contains(nb, (UInt8('F'), UInt8('R'), UInt8('E'), UInt8('E')))
                        _mr_store_i32!(r_ptr, _MR_OFF_freeFormat, Int32(1))
                    elseif _c_contains(nb, (UInt8('V'), UInt8('A'), UInt8('L'), UInt8('U'), UInt8('E'), UInt8('S')))
                        _mr_store_i32!(r_ptr, _MR_OFF_freeFormat, Int32(1))
                    elseif _c_contains(nb, (UInt8('I'), UInt8('E'), UInt8('E'), UInt8('E')))
                        _mr_store_i32!(r_ptr, _MR_OFF_ieeeFormat, Int32(1))
                    end
                else
                    _c_strcpy!(r.columnName, next)
                end
            else
                # "no_name"
                col = r.columnName
                unsafe_store!(col, UInt8('n'), 1); unsafe_store!(col, UInt8('o'), 2)
                unsafe_store!(col, UInt8('_'), 3); unsafe_store!(col, UInt8('n'), 4)
                unsafe_store!(col, UInt8('a'), 5); unsafe_store!(col, UInt8('m'), 6)
                unsafe_store!(col, UInt8('e'), 7); unsafe_store!(col, 0x00, 8)
            end
            return COIN_NAME_SECTION
        else
            b0 = unsafe_load(card)
            if b0 != UInt8('*') && b0 != UInt8('#') && b0 != 0x00
                _mps_log_line!(r, r.cardNumber)
                sec = _match_section(card)
                _mr_store_ptr!(r_ptr, _MR_OFF_position, card)
                _mr_store_ptr!(r_ptr, _MR_OFF_eol, card)
                _mr_store_i32!(r_ptr, _MR_OFF_section, sec)
                return sec
            end
        end
    end
end

@inline function _is_eight_spaces(p::Ptr{UInt8})::Bool
    i = 1
    while i <= 8
        unsafe_load(p, i) == UInt8(' ') || return false
        i += 1
    end
    return true
end

@inline function _write_eight_spaces!(dst::Ptr{UInt8})::Nothing
    i = 1
    while i <= 8
        unsafe_store!(dst, UInt8(' '), i)
        i += 1
    end
    unsafe_store!(dst, 0x00, 9)
    return nothing
end

function mps_next_field!(r_ptr::Ptr{CoinMpsCardReaderPod})::Int32
    r = unsafe_load(r_ptr)
    card = r.card
    pos = r.position
    eol = r.eol
    row_name = r.rowName
    col_name = r.columnName
    val_str = r.valueString
    mps_type = COIN_BLANK_COLUMN
    section = r.section
    free_fmt = r.freeFormat != Int32(0)
    ieee_fmt = r.ieeeFormat
    eight_char = r.eightChar != Int32(0)
    str_allowed = r.stringsAllowed != Int32(0)
    val = r.value

    next = pos
    while next != eol
        b = unsafe_load(next)
        (b == UInt8(' ') || b == UInt8('\t')) || break
        next += 1
    end
    got_card = (next != eol)

    after_ref = Ref{Ptr{UInt8}}(C_NULL)

    while !got_card
        # Store mutable flags before cleanCard reads section/freeFormat/eightChar
        _mr_store_i32!(r_ptr, _MR_OFF_section, section)
        _mr_store_i32!(r_ptr, _MR_OFF_freeFormat, free_fmt ? Int32(1) : Int32(0))
        _mr_store_i32!(r_ptr, _MR_OFF_eightChar, eight_char ? Int32(1) : Int32(0))
        if mps_clean_card!(r_ptr) != Int32(0)
            _mr_store_i32!(r_ptr, _MR_OFF_mpsType, mps_type)
            return COIN_EOF_SECTION
        end
        b0 = unsafe_load(card)
        if b0 == UInt8(' ') || b0 == 0x00
            pos = card
            eol = card + _c_strlen(card)
            next = card
            while next != eol
                b = unsafe_load(next)
                (b == UInt8(' ') || b == UInt8('\t')) || break
                next += 1
            end
            if next != eol
                nb = mps_next_blank_or(next)
                nchar = nb != C_NULL ? Int32(Int(UInt(nb) - UInt(next))) : Int32(-1)
                mps_type = COIN_BLANK_COLUMN
                if (section != COIN_RHS_SECTION && section != COIN_RANGES_SECTION) ||
                   free_fmt || !_is_eight_spaces(card + 4)
                    if section == COIN_COLUMN_SECTION &&
                       !_c_contains(next, (UInt8('\''), UInt8('M'), UInt8('A'), UInt8('R'), UInt8('K'), UInt8('E'), UInt8('R'), UInt8('\'')))
                        nchar = Int32(-1)
                    end
                    if section == COIN_SOS_SECTION
                        if _c_startswith(card, (UInt8(' '), UInt8('S'), UInt8('1')))
                            mps_type = COIN_S1_BOUND
                            break
                        elseif _c_startswith(card, (UInt8(' '), UInt8('S'), UInt8('2')))
                            mps_type = COIN_S2_BOUND
                            break
                        end
                    end
                    sec1 = Int(section) + 1
                    if nchar == @inbounds(_MPS_ALLOWED_LEN[sec1])
                        st_i = @inbounds _MPS_START_TYPE[sec1]
                        en_i = @inbounds _MPS_END_TYPE[sec1]
                        c0 = unsafe_load(next, 1)
                        c1 = nchar == Int32(2) ? unsafe_load(next, 2) : 0x00
                        ti = st_i
                        while ti < en_i
                            tb = @inbounds _MPS_TYPE_BYTES[Int(ti) + 1]
                            if tb[1] == c0 && (nchar == Int32(1) || tb[2] == c1)
                                mps_type = ti
                                break
                            end
                            ti += Int32(1)
                        end
                        if mps_type != COIN_BLANK_COLUMN
                            next = nb
                            while next != eol
                                b = unsafe_load(next)
                                (b == UInt8(' ') || b == UInt8('\t')) || break
                                next += 1
                            end
                            if next == eol
                                pos = eol
                                mps_type = COIN_UNKNOWN_MPS_TYPE
                            else
                                nb = mps_next_blank_or(next)
                            end
                        elseif section == COIN_BOUNDS_SECTION
                            if unsafe_load(card, 1) == UInt8(' ') && unsafe_load(card, 4) == UInt8(' ') &&
                               (unsafe_load(card, 2) != UInt8(' ') || unsafe_load(card, 3) != UInt8(' '))
                                mps_type = COIN_S3_COLUMN
                                next = nb
                                while next != eol
                                    b = unsafe_load(next)
                                    (b == UInt8(' ') || b == UInt8('\t')) || break
                                    next += 1
                                end
                                if next == eol
                                    pos = eol
                                    mps_type = COIN_UNKNOWN_MPS_TYPE
                                else
                                    nb = mps_next_blank_or(next)
                                end
                            end
                        end
                    end

                    if mps_type != COIN_UNKNOWN_MPS_TYPE
                        if section != COIN_BOUNDS_SECTION || free_fmt || !_is_eight_spaces(card + 4)
                            save = UInt8('?')
                            if !free_fmt && eight_char && next == card + 4
                                if Int(UInt(eol) - UInt(next)) >= 8
                                    b8 = unsafe_load(next, 9)
                                    if b8 != UInt8(' ') && b8 != 0x00
                                        eight_char = false
                                    else
                                        nb = next + 8
                                    end
                                    if nb != C_NULL
                                        save = unsafe_load(nb)
                                        unsafe_store!(nb, 0x00)
                                    end
                                else
                                    nb = Ptr{UInt8}(C_NULL)
                                end
                            else
                                if nb != C_NULL
                                    save = unsafe_load(nb)
                                    unsafe_store!(nb, 0x00)
                                end
                            end
                            mps_strcpy_and_compress!(col_name, next)
                            if next != card + 4
                                free_fmt = true
                            end
                            if nb != C_NULL
                                unsafe_store!(nb, save)
                                next = nb
                            else
                                next = eol
                            end
                        else
                            _write_eight_spaces!(col_name)
                        end

                        while next != eol
                            b = unsafe_load(next)
                            (b == UInt8(' ') || b == UInt8('\t')) || break
                            next += 1
                        end
                        if next == eol
                            pos = eol
                            val = -1.0e100
                            if section != COIN_ROW_SECTION && section != COIN_CONIC_SECTION
                                mps_type = COIN_UNKNOWN_MPS_TYPE
                            else
                                _mr_store_ptr!(r_ptr, _MR_OFF_position, pos)
                                _mr_store_ptr!(r_ptr, _MR_OFF_eol, eol)
                                _mr_store_f64!(r_ptr, _MR_OFF_value, val)
                                _mr_store_i32!(r_ptr, _MR_OFF_mpsType, mps_type)
                                _mr_store_i32!(r_ptr, _MR_OFF_freeFormat, free_fmt ? Int32(1) : Int32(0))
                                _mr_store_i32!(r_ptr, _MR_OFF_eightChar, eight_char ? Int32(1) : Int32(0))
                                return section
                            end
                        else
                            nb = mps_next_blank_or(next)
                        end

                        if section != COIN_ROW_SECTION
                            save = UInt8('?')
                            if !free_fmt && eight_char && next == card + 14
                                if Int(UInt(eol) - UInt(next)) >= 8
                                    b8 = unsafe_load(next, 9)
                                    if b8 != UInt8(' ') && b8 != 0x00
                                        eight_char = false
                                    else
                                        nb = next + 8
                                    end
                                    if nb != C_NULL
                                        save = unsafe_load(nb)
                                        unsafe_store!(nb, 0x00)
                                    end
                                else
                                    nb = Ptr{UInt8}(C_NULL)
                                end
                            else
                                free_fmt = true
                                if nb != C_NULL
                                    save = unsafe_load(nb)
                                    unsafe_store!(nb, 0x00)
                                end
                            end
                            mps_strcpy_and_compress!(row_name, next)
                            if nb != C_NULL
                                unsafe_store!(nb, save)
                                next = nb
                            else
                                next = eol
                            end
                            while next != eol
                                b = unsafe_load(next)
                                (b == UInt8(' ') || b == UInt8('\t')) || break
                                next += 1
                            end
                            if section == COIN_COLUMN_SECTION &&
                               _c_startswith(row_name, (UInt8('\''), UInt8('M'), UInt8('A'), UInt8('R'), UInt8('K'), UInt8('E'), UInt8('R'), UInt8('\''))) &&
                               next != eol
                                if _c_startswith(next, (UInt8('\''), UInt8('I'), UInt8('N'), UInt8('T'), UInt8('O'), UInt8('R'), UInt8('G'), UInt8('\'')))
                                    mps_type = COIN_INTORG
                                elseif _c_startswith(next, (UInt8('\''), UInt8('I'), UInt8('N'), UInt8('T'), UInt8('E'), UInt8('N'), UInt8('D'), UInt8('\'')))
                                    mps_type = COIN_INTEND
                                elseif _c_startswith(next, (UInt8('\''), UInt8('S'), UInt8('O'), UInt8('S'), UInt8('O'), UInt8('R'), UInt8('G'), UInt8('\'')))
                                    if mps_type == COIN_BLANK_COLUMN
                                        mps_type = COIN_S1_COLUMN
                                    end
                                elseif _c_startswith(next, (UInt8('\''), UInt8('S'), UInt8('O'), UInt8('S'), UInt8('E'), UInt8('N'), UInt8('D'), UInt8('\'')))
                                    mps_type = COIN_SOSEND
                                else
                                    mps_type = COIN_UNKNOWN_MPS_TYPE
                                end
                                pos = eol
                                _mr_store_ptr!(r_ptr, _MR_OFF_position, pos)
                                _mr_store_ptr!(r_ptr, _MR_OFF_eol, eol)
                                _mr_store_f64!(r_ptr, _MR_OFF_value, val)
                                _mr_store_i32!(r_ptr, _MR_OFF_mpsType, mps_type)
                                _mr_store_i32!(r_ptr, _MR_OFF_freeFormat, free_fmt ? Int32(1) : Int32(0))
                                _mr_store_i32!(r_ptr, _MR_OFF_eightChar, eight_char ? Int32(1) : Int32(0))
                                return section
                            end
                            if next == eol
                                pos = eol
                                if section != COIN_BOUNDS_SECTION
                                    if section != COIN_BASIS_SECTION
                                        mps_type = COIN_UNKNOWN_MPS_TYPE
                                    end
                                    val = -1.0e100
                                else
                                    val = 0.0
                                end
                            else
                                nb = mps_next_blank_or(next)
                                if nb != C_NULL
                                    save = unsafe_load(nb)
                                    unsafe_store!(nb, 0x00)
                                end
                                GC.@preserve after_ref begin
                                    val = mps_osi_strtod(next, Base.unsafe_convert(Ptr{Ptr{UInt8}}, after_ref), ieee_fmt, str_allowed, val_str)
                                end
                                if UInt(after_ref[]) > UInt(next)
                                    if nb != C_NULL
                                        unsafe_store!(nb, save)
                                        pos = nb
                                    else
                                        pos = eol
                                    end
                                else
                                    pos = eol
                                    mps_type = COIN_UNKNOWN_MPS_TYPE
                                    val = -1.0e100
                                end
                            end
                        end
                    end
                else
                    # Blank name in RHS or RANGES
                    _write_eight_spaces!(col_name)
                    save = UInt8('?')
                    if !free_fmt && eight_char && next == card + 14
                        if Int(UInt(eol) - UInt(next)) >= 8
                            b8 = unsafe_load(next, 9)
                            if b8 != UInt8(' ') && b8 != 0x00
                                eight_char = false
                            else
                                nb = next + 8
                            end
                            if nb != C_NULL
                                save = unsafe_load(nb)
                                unsafe_store!(nb, 0x00)
                            end
                        else
                            nb = Ptr{UInt8}(C_NULL)
                        end
                    else
                        if nb != C_NULL
                            save = unsafe_load(nb)
                            unsafe_store!(nb, 0x00)
                        end
                    end
                    mps_strcpy_and_compress!(row_name, next)
                    if nb != C_NULL
                        unsafe_store!(nb, save)
                        next = nb
                    else
                        next = eol
                    end
                    while next != eol
                        b = unsafe_load(next)
                        (b == UInt8(' ') || b == UInt8('\t')) || break
                        next += 1
                    end
                    if next == eol
                        pos = eol
                        val = -1.0e100
                        mps_type = COIN_UNKNOWN_MPS_TYPE
                    else
                        nb = mps_next_blank_or(next)
                        if nb != C_NULL
                            save = unsafe_load(nb)
                            unsafe_store!(nb, 0x00)
                        end
                        GC.@preserve after_ref begin
                            val = mps_osi_strtod(next, Base.unsafe_convert(Ptr{Ptr{UInt8}}, after_ref), ieee_fmt, str_allowed, val_str)
                        end
                        if UInt(after_ref[]) > UInt(next)
                            if nb != C_NULL
                                unsafe_store!(nb, save)
                                pos = nb
                            else
                                pos = eol
                            end
                        else
                            pos = eol
                            mps_type = COIN_UNKNOWN_MPS_TYPE
                            val = -1.0e100
                        end
                    end
                end
            else
                continue
            end
            _mr_store_ptr!(r_ptr, _MR_OFF_position, pos)
            _mr_store_ptr!(r_ptr, _MR_OFF_eol, eol)
            _mr_store_f64!(r_ptr, _MR_OFF_value, val)
            _mr_store_i32!(r_ptr, _MR_OFF_mpsType, mps_type)
            _mr_store_i32!(r_ptr, _MR_OFF_freeFormat, free_fmt ? Int32(1) : Int32(0))
            _mr_store_i32!(r_ptr, _MR_OFF_eightChar, eight_char ? Int32(1) : Int32(0))
            return section
        elseif b0 != UInt8('*')
            r_cur = unsafe_load(r_ptr)
            _mps_log_line!(r_cur, r_cur.cardNumber)
            section = _match_section(card)
            pos = card
            eol = card
            _mr_store_ptr!(r_ptr, _MR_OFF_position, pos)
            _mr_store_ptr!(r_ptr, _MR_OFF_eol, eol)
            _mr_store_i32!(r_ptr, _MR_OFF_mpsType, mps_type)
            _mr_store_i32!(r_ptr, _MR_OFF_section, section)
            return section
        end
    end

    # Second field on the same card (or SOS break)
    save = UInt8('?')
    nb = mps_next_blank_or(next)
    if !free_fmt && eight_char && next == card + 39
        if Int(UInt(eol) - UInt(next)) >= 8
            b8 = unsafe_load(next, 9)
            if b8 != UInt8(' ') && b8 != 0x00
                eight_char = false
            else
                nb = next + 8
            end
            if nb != C_NULL
                save = unsafe_load(nb)
                unsafe_store!(nb, 0x00)
            end
        else
            nb = Ptr{UInt8}(C_NULL)
        end
    else
        free_fmt = true
        if nb != C_NULL
            save = unsafe_load(nb)
            unsafe_store!(nb, 0x00)
        end
    end
    mps_strcpy_and_compress!(row_name, next)
    if nb != C_NULL
        unsafe_store!(nb, save)
        next = nb
    else
        next = eol
    end
    while next != eol
        b = unsafe_load(next)
        (b == UInt8(' ') || b == UInt8('\t')) || break
        next += 1
    end
    if next == eol && section != COIN_SOS_SECTION
        pos = eol
        mps_type = COIN_UNKNOWN_MPS_TYPE
    else
        nb = mps_next_blank_or(next)
    end
    if nb != C_NULL
        save = unsafe_load(nb)
        unsafe_store!(nb, 0x00)
    end
    GC.@preserve after_ref begin
        val = mps_osi_strtod(next, Base.unsafe_convert(Ptr{Ptr{UInt8}}, after_ref), ieee_fmt, str_allowed, val_str)
    end
    if UInt(after_ref[]) > UInt(next)
        if nb != C_NULL
            unsafe_store!(nb, save)
            pos = nb
        else
            pos = eol
        end
    else
        pos = eol
        if mps_type != COIN_S1_BOUND && mps_type != COIN_S2_BOUND
            mps_type = COIN_UNKNOWN_MPS_TYPE
        end
        val = -1.0e100
    end

    _mr_store_ptr!(r_ptr, _MR_OFF_position, pos)
    _mr_store_ptr!(r_ptr, _MR_OFF_eol, eol)
    _mr_store_f64!(r_ptr, _MR_OFF_value, val)
    _mr_store_i32!(r_ptr, _MR_OFF_mpsType, mps_type)
    _mr_store_i32!(r_ptr, _MR_OFF_freeFormat, free_fmt ? Int32(1) : Int32(0))
    _mr_store_i32!(r_ptr, _MR_OFF_eightChar, eight_char ? Int32(1) : Int32(0))
    return section
end

# =============================================================================
# 3. CoinMpsIO Hash Table (CoinMpsIO.cpp:1197-1367)
# =============================================================================

@inline function mps_hash(name::Ptr{UInt8}, maxsiz::Int32, len::Int32)::Int32
    n = Int32(0)
    j = 0
    l = Int(len)
    sname = Ptr{Int8}(name)
    while j < l
        iname = Int32(unsafe_load(sname, j + 1))
        m = @inbounds _MPS_MMULT[rem(j, 81) + 1]
        n += m * iname
        j += 1
    end
    return rem(abs(n), maxsiz)
end

function mps_start_hash!(
    names::Ptr{Ptr{UInt8}},
    number::Int32,
    hash_tbl::Ptr{CoinHashLinkPod},
)::Nothing
    maxhash = Int32(4) * number
    mh = Int(maxhash)
    empty_link = CoinHashLinkPod(Int32(-1), Int32(-1))
    i = 1
    while i <= mh
        unsafe_store!(hash_tbl, empty_link, i)
        i += 1
    end

    num = Int(number)
    i = 0
    while i < num
        this_name = unsafe_load(names, i + 1)
        len = Int32(_c_strlen(this_name))
        ipos = Int(mps_hash(this_name, maxhash, len))
        lnk = unsafe_load(hash_tbl, ipos + 1)
        if lnk.index == Int32(-1)
            unsafe_store!(hash_tbl, CoinHashLinkPod(Int32(i), lnk.next), ipos + 1)
        end
        i += 1
    end

    iput = Int32(-1)
    i = 0
    while i < num
        this_name = unsafe_load(names, i + 1)
        len = Int32(_c_strlen(this_name))
        ipos = mps_hash(this_name, maxhash, len)
        while true
            lnk = unsafe_load(hash_tbl, Int(ipos) + 1)
            j1 = lnk.index
            if j1 == Int32(i)
                break
            else
                this_name2 = unsafe_load(names, Int(j1) + 1)
                if _c_strcmp(this_name, this_name2) == Int32(0)
                    break
                else
                    k = lnk.next
                    if k == Int32(-1)
                        while true
                            iput += Int32(1)
                            if iput > number
                                break
                            end
                            if unsafe_load(hash_tbl, Int(iput) + 1).index == Int32(-1)
                                break
                            end
                        end
                        unsafe_store!(hash_tbl, CoinHashLinkPod(j1, iput), Int(ipos) + 1)
                        lnk_put = unsafe_load(hash_tbl, Int(iput) + 1)
                        unsafe_store!(hash_tbl, CoinHashLinkPod(Int32(i), lnk_put.next), Int(iput) + 1)
                        break
                    else
                        ipos = k
                    end
                end
            end
        end
        i += 1
    end
    return nothing
end

function mps_find_hash(
    name::Ptr{UInt8},
    names::Ptr{Ptr{UInt8}},
    number::Int32,
    hash_tbl::Ptr{CoinHashLinkPod},
)::Int32
    maxhash = Int32(4) * number
    if maxhash == Int32(0)
        return Int32(-1)
    end
    len = Int32(_c_strlen(name))
    ipos = mps_hash(name, maxhash, len)
    while true
        lnk = unsafe_load(hash_tbl, Int(ipos) + 1)
        j1 = lnk.index
        if j1 >= Int32(0)
            this_name2 = unsafe_load(names, Int(j1) + 1)
            if _c_strcmp(name, this_name2) != Int32(0)
                k = lnk.next
                if k != Int32(-1)
                    ipos = k
                else
                    return Int32(-1)
                end
            else
                return j1
            end
        else
            return Int32(-1)
        end
    end
end

# =============================================================================
# 4. Bound/Sense Conversions & CoinConvertDouble (CoinMpsIO.cpp:3926-4096, 4946-5004)
# =============================================================================

@inline function mps_convert_bound_to_sense(
    lower::Float64,
    upper::Float64,
    infinity::Float64,
)::Tuple{UInt8,Float64,Float64}
    range_v = 0.0
    if lower > -infinity
        if upper < infinity
            right_v = upper
            if upper == lower
                return UInt8('E'), right_v, range_v
            else
                return UInt8('R'), right_v, upper - lower
            end
        else
            return UInt8('G'), lower, range_v
        end
    else
        if upper < infinity
            return UInt8('L'), upper, range_v
        else
            return UInt8('N'), 0.0, range_v
        end
    end
end

@inline function mps_convert_sense_to_bound(
    sense::UInt8,
    right::Float64,
    range_v::Float64,
    infinity::Float64,
)::Tuple{Float64,Float64}
    if sense == UInt8('E')
        return right, right
    elseif sense == UInt8('L')
        return -infinity, right
    elseif sense == UInt8('G')
        return right, infinity
    elseif sense == UInt8('R')
        return right - range_v, right
    elseif sense == UInt8('N')
        return -infinity, infinity
    else
        return -infinity, infinity
    end
end

@inline function _encode_ieee_6bit(part::UInt16)::UInt8
    if part < UInt16(10)
        return UInt8(part) + UInt8('0')
    elseif part < UInt16(36)
        return UInt8(part - UInt16(10)) + UInt8('a')
    elseif part < UInt16(62)
        return UInt8(part - UInt16(36)) + UInt8('A')
    else
        return UInt8(part - UInt16(62)) + UInt8('*')
    end
end

@inline function _snprintf_fixed12!(out::Ptr{UInt8}, dec::Int32, value::Float64)::Nothing
    if dec <= Int32(0)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.0f", value)
    elseif dec == Int32(1)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.1f", value)
    elseif dec == Int32(2)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.2f", value)
    elseif dec == Int32(3)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.3f", value)
    elseif dec == Int32(4)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.4f", value)
    elseif dec == Int32(5)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.5f", value)
    elseif dec == Int32(6)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.6f", value)
    elseif dec == Int32(7)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.7f", value)
    elseif dec == Int32(8)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.8f", value)
    elseif dec == Int32(9)
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.9f", value)
    else
        ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...), out, Csize_t(24), "%12.10f", value)
    end
    return nothing
end

function coin_convert_double!(
    section::Int32,
    format_type::Int32,
    value::Float64,
    out::Ptr{UInt8},
)::Nothing
    if format_type == Int32(0)
        strip_zeros = true
        if abs(value) < 1.0e40
            if value >= 0.0
                p10 = value == 0.0 ? typemin(Int32) : Int32(unsafe_trunc(Int32, log10(value)))
                if p10 < Int32(9) && p10 > Int32(-4)
                    dec = min(Int32(10), Int32(10) - p10)
                    _snprintf_fixed12!(out, dec, value)
                else
                    ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...),
                          out, Csize_t(24), "%13.7g", value)
                    strip_zeros = false
                end
            else
                p10 = Int32(unsafe_trunc(Int32, log10(-value))) + Int32(1)
                if p10 < Int32(8) && p10 > Int32(-3)
                    dec = min(Int32(9), Int32(9) - p10)
                    _snprintf_fixed12!(out, dec, value)
                else
                    ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...),
                          out, Csize_t(24), "%13.6g", value)
                    strip_zeros = false
                end
            end
            if strip_zeros
                j = 11
                while j >= 0
                    if unsafe_load(out, j + 1) == UInt8('0')
                        unsafe_store!(out, UInt8(' '), j + 1)
                    else
                        break
                    end
                    j -= 1
                end
            else
                e_pos = -1
                k = 0
                while k < 24
                    b = unsafe_load(out, k + 1)
                    b == 0x00 && break
                    if b == UInt8('e')
                        e_pos = k
                        break
                    end
                    k += 1
                end
                if e_pos < 0
                    b12 = unsafe_load(out, 13)
                    if b12 != UInt8(' ') && b12 != 0x00
                        j = 0
                        while j < 12
                            unsafe_store!(out, unsafe_load(out, j + 2), j + 1)
                            j += 1
                        end
                    end
                    unsafe_store!(out, 0x00, 13)
                else
                    j = e_pos + 1
                    put = j + 1
                    j = put
                    while j < 14
                        if unsafe_load(out, j + 1) != UInt8('0')
                            break
                        end
                        j += 1
                    end
                    if j == put
                        if unsafe_load(out, 1) == UInt8(' ')
                            j = 1
                            put = 0
                        else
                            put -= 3
                            j -= 2
                        end
                    end
                    while j < 14
                        unsafe_store!(out, unsafe_load(out, j + 1), put + 1)
                        put += 1
                        j += 1
                    end
                end
            end
            if abs(value) < 1.0e-20
                unsafe_store!(out, UInt8('0'), 1)
                unsafe_store!(out, UInt8('.'), 2)
                unsafe_store!(out, UInt8('0'), 3)
                unsafe_store!(out, 0x00, 4)
            end
        else
            if section == Int32(2)
                unsafe_store!(out, 0x00, 1)
            else
                ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...),
                      out, Csize_t(24), "%12.6g", value)
            end
        end
        i = 0
        while i < 12
            unsafe_load(out, i + 1) == 0x00 && break
            i += 1
        end
        while i < 12
            unsafe_store!(out, UInt8(' '), i + 1)
            i += 1
        end
        unsafe_store!(out, 0x00, 13)
    elseif format_type == Int32(1)
        if abs(value) < 1.0e40
            i = 1
            while i <= 24
                unsafe_store!(out, UInt8(' '), i)
                i += 1
            end
            ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...),
                  out, Csize_t(24), "%.16g", value)
            ii = 0
            j = 0
            while j < 23
                b = unsafe_load(out, j + 1)
                if b != UInt8(' ')
                    unsafe_store!(out, b, ii + 1)
                    ii += 1
                end
                j += 1
            end
            unsafe_store!(out, 0x00, ii + 1)
        else
            if section == Int32(2)
                unsafe_store!(out, 0x00, 1)
            else
                ccall(:snprintf, Cint, (Ptr{UInt8}, Csize_t, Cstring, Cdouble...),
                      out, Csize_t(24), "%12.6g", value)
            end
        end
    else
        # IEEE 6-bit encoding
        u64 = reinterpret(UInt64, value)
        unsafe_store!(out, 0x00, 13)
        pos = 1
        if format_type == Int32(2)
            i = 3
            while i >= 0
                v16 = UInt16((u64 >> (16 * i)) & UInt64(0xffff))
                j = 0
                while j < 3
                    unsafe_store!(out, _encode_ieee_6bit(v16 & UInt16(63)), pos)
                    v16 >>= 6
                    pos += 1
                    j += 1
                end
                i -= 1
            end
        else
            i = 0
            while i < 4
                v16 = UInt16((u64 >> (16 * i)) & UInt64(0xffff))
                j = 0
                while j < 3
                    unsafe_store!(out, _encode_ieee_6bit(v16 & UInt16(63)), pos)
                    v16 >>= 6
                    pos += 1
                    j += 1
                end
                i += 1
            end
        end
    end
    return nothing
end

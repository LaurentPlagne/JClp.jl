# Exact Julia mirror of bridge/jclp_cf_pod.h (bit-identical C ABI layout on 64-bit).

struct CoinIndexedVecPod
    elements::Ptr{Float64}
    indices::Ptr{Int32}
    n_elements::Int32
    capacity::Int32
    packed_mode::Int32
    _pad::Int32
end

struct CoinFactorizationPodView
    # 8-byte pointers
    elementU::Ptr{Float64}
    indexRowU::Ptr{Int32}
    startColumnU::Ptr{Int32}
    numberInColumn::Ptr{Int32}
    numberInColumnPlus::Ptr{Int32}
    startRowU::Ptr{Int32}
    numberInRow::Ptr{Int32}
    indexColumnU::Ptr{Int32}
    convertRowToColumnU::Ptr{Int32}

    elementL::Ptr{Float64}
    indexRowL::Ptr{Int32}
    startColumnL::Ptr{Int32}
    elementByRowL::Ptr{Float64}
    indexColumnL::Ptr{Int32}
    startRowL::Ptr{Int32}

    elementR::Ptr{Float64}
    indexRowR::Ptr{Int32}
    startColumnR::Ptr{Int32}

    pivotRegion::Ptr{Float64}
    pivotColumn::Ptr{Int32}
    pivotColumnBack::Ptr{Int32}
    permute::Ptr{Int32}
    permuteBack::Ptr{Int32}
    nextColumn::Ptr{Int32}
    lastColumn::Ptr{Int32}

    sparseArray::Ptr{Int32}
    denseAreaAddress::Ptr{Float64}
    densePermute::Ptr{Int32}

    p_numberBtranCounts::Ptr{Int32}
    p_btranCountInput::Ptr{Float64}
    p_btranCountAfterU::Ptr{Float64}
    p_btranCountAfterR::Ptr{Float64}
    p_btranCountAfterL::Ptr{Float64}

    p_numberFtranCounts::Ptr{Int32}
    p_ftranCountInput::Ptr{Float64}
    p_ftranCountAfterL::Ptr{Float64}
    p_ftranCountAfterR::Ptr{Float64}
    p_ftranCountAfterU::Ptr{Float64}

    p_lengthR::Ptr{Int32}
    p_numberR::Ptr{Int32}

    # 8-byte doubles
    zeroTolerance::Float64
    slackValue::Float64
    btranAverageAfterU::Float64
    btranAverageAfterR::Float64
    btranAverageAfterL::Float64
    ftranAverageAfterL::Float64
    ftranAverageAfterR::Float64
    ftranAverageAfterU::Float64

    # 4-byte int32 scalars (22 fields -> 88 bytes, multiple of 8)
    numberRows::Int32
    numberRowsExtra::Int32
    maximumRowsExtra::Int32
    numberColumnsExtra::Int32
    maximumColumnsExtra::Int32
    numberGoodU::Int32
    numberU::Int32
    numberL::Int32
    baseL::Int32
    numberR::Int32
    lengthR::Int32
    lengthAreaR::Int32
    lengthAreaU::Int32
    numberDense::Int32
    numberSlacks::Int32
    numberPivots::Int32
    maximumPivots::Int32
    sparseThreshold::Int32
    sparseThreshold2::Int32
    doForrestTomlin::Int32
    collectStatistics::Int32
    _pad::Int32
end

mutable struct CoinFactorizationReplaceState
    nextRow::Ptr{Int32}
    lastRow::Ptr{Int32}
    relaxCheck::Float64
    numberRowsExtra::Int32
    numberColumnsExtra::Int32
    numberU::Int32
    numberGoodU::Int32
    numberPivots::Int32
    numberR::Int32
    lengthR::Int32
    lengthU::Int32
    maximumU::Int32
    totalElements::Int32
    factorElements::Int32
    lengthAreaL::Int32
    numberCompressions::Int32
    status::Int32
    dropNumberInColumnPlus::Int32
    _pad::Int32
end

const CF_BITS_PER_CHECK = Int32(8)
const CF_CHECK_SHIFT = Int32(3)

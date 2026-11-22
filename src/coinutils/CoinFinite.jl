# Faithful translation of CoinUtils/src/CoinFinite.{hpp,cpp}

@inline function coin_finite(val::Float64)::Bool
    return isfinite(val)
end

@inline function coin_isnan(val::Float64)::Bool
    return isnan(val)
end

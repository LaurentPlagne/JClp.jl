using JClp
using Test

@testset "JClp.jl" begin
    include("test_b0_bridge.jl")
    include("test_b1_coin_basics.jl")
    include("test_b2_indexed_vector.jl")
    include("test_b3_packed_matrix.jl")
    include("test_b4_dense_factorization.jl")
    include("test_b5a_btran.jl")
    include("test_b5a_ftran.jl")
    include("test_b5b_replace.jl")
end

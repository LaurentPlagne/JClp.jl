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
    include("test_b5c_factor.jl")
    include("test_b6_mps_io.jl")
    include("test_b7a_presolve_basic.jl")
    include("test_b7b_presolve_medium.jl")
    include("test_b7c1_presolve_adv1.jl")
    include("test_b7c2_presolve_adv2.jl")
end

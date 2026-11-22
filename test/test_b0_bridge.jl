@testset "B0 Bridge Smoke Kernels" begin
    @test JClp.jclp_bridge_version() == Cint(1)
    @test JClp.jclp_smoke_add(1.5, 2.25) === 3.75

    x = Float64[1.0, -2.0, 3.5, 4.0]
    y = Float64[2.0, 3.0, -4.0, 0.5]
    res = GC.@preserve x y JClp.jclp_smoke_dot(Cint(length(x)), pointer(x), pointer(y))
    @test res === (1.0 * 2.0 + (-2.0) * 3.0 + 3.5 * (-4.0) + 4.0 * 0.5)
    @test GC.@preserve x y JClp.jclp_smoke_dot(Cint(0), pointer(x), pointer(y)) === 0.0

    allocs = GC.@preserve x y @allocated JClp.jclp_smoke_dot(Cint(length(x)), pointer(x), pointer(y))
    @test allocs == 0
end

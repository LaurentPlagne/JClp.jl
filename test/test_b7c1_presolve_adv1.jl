using Test
using JClp

function _test_alloc_dbl(n::Int32)::Ptr{Float64}
    return Ptr{Float64}(Libc.malloc(max(Int(n), 1) * sizeof(Float64)))
end

@testset "B7c1: CoinPresolveDoubleton + Tripleton + ImpliedFree" begin
    @testset "ABI layout" begin
        @test sizeof(JClp.DoubletonActionPod) == 88
        @test sizeof(JClp.TripletonActionPod) == 120
        @test sizeof(JClp.ImpliedFreeActionPod) == 64
    end

    @testset "Doubleton presolve + postsolve" begin
        # Row 0 (doubleton): x0 + 2*x1 == 4.0 (coeffy=2 > coeffx=1 => eliminate x1, keep x0)
        # Row 1:             x0 + x1 <= 10.0  -> after subst x1 = 2 - 0.5*x0: 0.5*x0 <= 8.0
        # Bounds: x0 in [0, 10], x1 in [0, 3] => 0 <= (4 - x0)/2 <= 3 => -2 <= x0 <= 4 => x0 in [0, 4]
        # Costs: c0 = 1.0, c1 = 2.0 => c0' = 1 + 2*(-1)/2 = 0.0, dobias += 2*4/2 = 4.0
        INF = JClp.PRESOLVE_INF_DBL
        mcstrt = Int32[0, 2, 10]
        hincol = Int32[2, 2]
        hrow   = zeros(Int32, 16); hrow[1:4] .= Int32[0, 1, 0, 1]
        colels = zeros(Float64, 16); colels[1:4] .= Float64[1.0, 1.0, 2.0, 1.0]
        mrstrt = Int32[0, 2, 10]
        hinrow = Int32[2, 2]
        hcol   = zeros(Int32, 16); hcol[1:4] .= Int32[0, 1, 0, 1]
        rowels = zeros(Float64, 16); rowels[1:4] .= Float64[1.0, 2.0, 1.0, 1.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)

        clo = Float64[0.0, 0.0]
        cup = Float64[10.0, 3.0]
        rlo = Float64[4.0, -INF]
        rup = Float64[4.0, 10.0]
        cost = Float64[1.0, 2.0]
        integerType = UInt8[0, 0]
        colChanged = zeros(UInt8, 2)
        rowChanged = zeros(UInt8, 2)
        rowsToDo = Int32[0, 1]
        nextColsToDo = zeros(Int32, 2)
        nextRowsToDo = zeros(Int32, 2)
        usefulColInt = zeros(Int32, 6)
        actions = Vector{JClp.DoubletonActionPod}(undef, 2)
        out_na = Ref(Int32(0))
        out_nz = Ref(Int32(0))
        out_nf = Ref(Int32(0))
        out_nm = Ref(Int32(0))
        out_ic = Ref(Int32(-1))
        out_lo = Ref(0.0)
        out_up = Ref(0.0)
        alloc_cb = @cfunction(_test_alloc_dbl, Ptr{Float64}, (Int32,))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup cost integerType colChanged rowChanged rowsToDo nextColsToDo nextRowsToDo usefulColInt actions out_na out_nz out_nf out_nm out_ic out_lo out_up begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(2))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, pointer(integerType),
                C_NULL, C_NULL,
                pointer(colChanged), pointer(rowChanged),
                C_NULL, pointer(nextColsToDo), pointer(rowsToDo), pointer(nextRowsToDo),
                C_NULL, pointer(usefulColInt), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(16),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(2), Int32(0), Int32(0), Int32(3),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_doubleton_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    pointer(actions), alloc_cb,
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_nz),
                    Base.unsafe_convert(Ptr{Int32}, out_nf),
                    Base.unsafe_convert(Ptr{Int32}, out_nm),
                    Base.unsafe_convert(Ptr{Int32}, out_ic),
                    Base.unsafe_convert(Ptr{Float64}, out_lo),
                    Base.unsafe_convert(Ptr{Float64}, out_up),
                )
                @test out_na[] == Int32(1)
                @test out_nm[] == Int32(0)
                @test out_ic[] == Int32(-1)
                @test clo[1] == 0.0 && cup[1] == 4.0
                @test cost[1] ≈ 0.0
                @test ps_ref[].dobias ≈ 4.0
                @test rup[2] ≈ 8.0
                @test hincol[2] == Int32(0) && hinrow[1] == Int32(0)
            end

            # Postsolve: x0 = 2.0 (basic), acts[1] = 0.5*2 = 1.0, rowduals = [0, 0]
            # Expected: x1 = (4 - 2)/2 = 1.0, rup[1] restored to 10.0, acts[1] = 3.0, rowduals[0] = 2/2 = 1.0, rcosts[0] = 1 - 1*1 = 0
            p_mcstrt = Int32[0, JClp.PRESOLVE_NO_LINK]
            p_hincol = Int32[1, 0]
            p_hrow   = zeros(Int32, 16); p_hrow[1] = Int32(1)
            p_colels = zeros(Float64, 16); p_colels[1] = 0.5
            p_link   = Int32[JClp.PRESOLVE_NO_LINK, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, JClp.PRESOLVE_NO_LINK]
            p_sol    = Float64[2.0, 0.0]
            p_acts   = Float64[0.0, 1.0]
            p_duals  = Float64[0.0, 0.0]
            p_rcosts = Float64[0.0, 0.0]
            p_cstat  = UInt8[JClp.PS_STATUS_BASIC, JClp.PS_STATUS_AT_LOWER_BOUND]
            p_rstat  = UInt8[JClp.PS_STATUS_AT_LOWER_BOUND, JClp.PS_STATUS_BASIC]
            index1   = zeros(Int32, 2)
            element1 = zeros(Float64, 2)
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(p_mcstrt), pointer(p_hincol), pointer(p_hrow), pointer(p_link), pointer(p_colels),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                pointer(p_sol), pointer(p_acts), pointer(p_duals), pointer(p_rcosts),
                pointer(p_cstat), pointer(p_rstat),
                1.0, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(16), Int32(1), Int32(0),
            ))
            GC.@preserve p_mcstrt p_hincol p_hrow p_colels p_link p_sol p_acts p_duals p_rcosts p_cstat p_rstat index1 element1 post_ref begin
                JClp.ps_doubleton_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[], pointer(index1), pointer(element1),
                )
                @test p_sol[2] ≈ 1.0
                @test rup[2] ≈ 10.0
                @test p_acts[1] ≈ 4.0 && p_acts[2] ≈ 3.0
                @test p_duals[1] ≈ 1.0
                @test p_rcosts[1] ≈ 0.0 && p_rcosts[2] ≈ 0.0
                @test p_cstat[2] == JClp.PS_STATUS_BASIC
                @test p_hincol[1] == Int32(2) && p_hincol[2] == Int32(2)
            end
            for k in 1:Int(out_na[])
                Libc.free(actions[k].colel)
            end
        end
    end

    @testset "Tripleton presolve + postsolve" begin
        # Row 0 (tripleton): x0 - y + x2 == 0 (cx=1, cz=1, y=x0+x2)
        # Row 1: x0 + y + x2 <= 10 -> after subst: 2*x0 + 2*x2 <= 10
        # Bounds: x0 in [0, 3], x2 in [0, 4], y in [0, 10] (implied y in [0, 7] subset [0, 10]!)
        # Costs: c0=1, cy=2, c2=3 -> c0'=3, c2'=5
        INF = JClp.PRESOLVE_INF_DBL
        mcstrt = Int32[0, 2, 4, 10]
        hincol = Int32[2, 2, 2]
        hrow   = zeros(Int32, 24); hrow[1:6] .= Int32[0, 1, 0, 1, 0, 1]
        colels = zeros(Float64, 24); colels[1:6] .= Float64[1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
        mrstrt = Int32[0, 3, 12]
        hinrow = Int32[3, 3]
        hcol   = zeros(Int32, 24); hcol[1:6] .= Int32[0, 1, 2, 0, 1, 2]
        rowels = zeros(Float64, 24); rowels[1:6] .= Float64[1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 4)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)

        clo = Float64[0.0, 0.0, 0.0]
        cup = Float64[3.0, 10.0, 4.0]
        rlo = Float64[0.0, -INF]
        rup = Float64[0.0, 10.0]
        cost = Float64[1.0, 2.0, 3.0]
        integerType = zeros(UInt8, 3)
        colChanged = zeros(UInt8, 3)
        rowChanged = zeros(UInt8, 2)
        rowsToDo = Int32[0, 1]
        nextColsToDo = zeros(Int32, 3)
        nextRowsToDo = zeros(Int32, 2)
        usefulColInt = zeros(Int32, 9)
        actions = Vector{JClp.TripletonActionPod}(undef, 2)
        out_na = Ref(Int32(0))
        out_nz = Ref(Int32(0))
        out_nm = Ref(Int32(0))
        alloc_cb = @cfunction(_test_alloc_dbl, Ptr{Float64}, (Int32,))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup cost integerType colChanged rowChanged rowsToDo nextColsToDo nextRowsToDo usefulColInt actions out_na out_nz out_nm begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(3))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, pointer(integerType),
                C_NULL, C_NULL,
                pointer(colChanged), pointer(rowChanged),
                C_NULL, pointer(nextColsToDo), pointer(rowsToDo), pointer(nextRowsToDo),
                C_NULL, pointer(usefulColInt), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(3), Int32(2), Int32(3), Int32(6), Int32(24),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(2), Int32(0), Int32(0), Int32(3),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_tripleton_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    pointer(actions), alloc_cb,
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_nz),
                    Base.unsafe_convert(Ptr{Int32}, out_nm),
                )
                @test out_na[] == Int32(1)
                @test out_nm[] == Int32(0)
                @test cost[1] ≈ 3.0 && cost[3] ≈ 5.0
                @test hincol[2] == Int32(0) && hinrow[1] == Int32(0)
            end

            # Postsolve
            p_mcstrt = Int32[0, JClp.PRESOLVE_NO_LINK, 1]
            p_hincol = Int32[1, 0, 1]
            p_hrow   = zeros(Int32, 24); p_hrow[1:2] .= Int32[1, 1]
            p_colels = zeros(Float64, 24); p_colels[1:2] .= Float64[2.0, 2.0]
            p_link   = Int32[i < 24 ? Int32(i) : JClp.PRESOLVE_NO_LINK for i in 1:24]
            p_link[1] = JClp.PRESOLVE_NO_LINK; p_link[2] = JClp.PRESOLVE_NO_LINK
            p_sol    = Float64[1.0, 0.0, 2.0]
            p_acts   = Float64[0.0, 6.0]
            p_duals  = Float64[0.0, 0.0]
            p_rcosts = Float64[3.0, 0.0, 5.0]
            p_cstat  = UInt8[JClp.PS_STATUS_BASIC, JClp.PS_STATUS_AT_LOWER_BOUND, JClp.PS_STATUS_BASIC]
            p_rstat  = UInt8[JClp.PS_STATUS_AT_LOWER_BOUND, JClp.PS_STATUS_BASIC]
            index1 = zeros(Int32, 2); element1 = zeros(Float64, 2)
            index2 = zeros(Int32, 2); element2 = zeros(Float64, 2)
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(p_mcstrt), pointer(p_hincol), pointer(p_hrow), pointer(p_link), pointer(p_colels),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                pointer(p_sol), pointer(p_acts), pointer(p_duals), pointer(p_rcosts),
                pointer(p_cstat), pointer(p_rstat),
                1.0, 1e-7, 1e-7,
                Int32(2), Int32(3), Int32(2), Int32(3), Int32(6), Int32(24), Int32(2), Int32(0),
            ))
            GC.@preserve p_mcstrt p_hincol p_hrow p_colels p_link p_sol p_acts p_duals p_rcosts p_cstat p_rstat index1 element1 index2 element2 post_ref begin
                JClp.ps_tripleton_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[], pointer(index1), pointer(element1),
                    pointer(index2), pointer(element2),
                )
                @test p_sol[2] ≈ 3.0
                @test cost[1] ≈ 1.0 && cost[2] ≈ 2.0 && cost[3] ≈ 3.0
                @test p_duals[1] ≈ -2.0 # djy / coeffy = 2.0 / (-1.0)
                @test p_rcosts[2] ≈ 0.0
                @test p_cstat[2] == JClp.PS_STATUS_BASIC
                @test p_hincol == Int32[2, 2, 2]
            end
            for k in 1:Int(out_na[])
                Libc.free(actions[k].colel)
            end
        end
    end

    @testset "ImpliedFree presolve + postsolve + check_row" begin
        # Row 0: x0 + x1 == 5.0 with x0 in [0, 2], x1 in [0, 10] (x1 is natural singleton, implied in [3, 5] subset [0, 10]!)
        # Costs: c0 = 1.0, c1 = 2.0 => c0' = 1 - 2*1/1 = -1.0, dobias += 2*5/1 = 10.0
        mcstrt = Int32[0, 1, 2]
        hincol = Int32[1, 1]
        hrow   = Int32[0, 0, 0, 0]
        colels = Float64[1.0, 1.0, 0.0, 0.0]
        mrstrt = Int32[0, 2]
        hinrow = Int32[2]
        hcol   = Int32[0, 1, 0, 0]
        rowels = Float64[1.0, 1.0, 0.0, 0.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 2)

        clo = Float64[0.0, 0.0]
        cup = Float64[2.0, 10.0]
        rlo = Float64[5.0]
        rup = Float64[5.0]
        cost = Float64[1.0, 2.0]
        integerType = zeros(UInt8, 2)
        colChanged = zeros(UInt8, 2)
        rowChanged = zeros(UInt8, 1)
        colsToDo = Int32[0, 1]
        nextColsToDo = zeros(Int32, 2)
        usefulRowInt = zeros(Int32, 3)
        usefulColInt = zeros(Int32, 6)
        infDown = zeros(Int32, 1)
        infUp   = zeros(Int32, 1)
        maxDown = zeros(Float64, 1)
        maxUp   = zeros(Float64, 1)
        actions = Vector{JClp.ImpliedFreeActionPod}(undef, 2)
        fill_level = Ref(Int32(2))
        out_na = Ref(Int32(0))
        out_un = Ref(Int32(0))
        out_nf = Ref(Int32(0))
        out_ir = Ref(Int32(-1))
        alloc_cb = @cfunction(_test_alloc_dbl, Ptr{Float64}, (Int32,))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup cost integerType colChanged rowChanged colsToDo nextColsToDo usefulRowInt usefulColInt infDown infUp maxDown maxUp actions fill_level out_na out_un out_nf out_ir begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(2))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(1))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, pointer(integerType),
                C_NULL, C_NULL,
                pointer(colChanged), pointer(rowChanged),
                pointer(colsToDo), pointer(nextColsToDo), C_NULL, C_NULL,
                pointer(usefulRowInt), pointer(usefulColInt), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(1), Int32(2), Int32(1), Int32(2), Int32(2), Int32(4),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(2), Int32(0), Int32(0), Int32(0), Int32(0), Int32(3),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_implied_free_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    Base.unsafe_convert(Ptr{Int32}, fill_level),
                    pointer(actions), pointer(infDown), pointer(infUp),
                    pointer(maxDown), pointer(maxUp), alloc_cb,
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_un),
                    Base.unsafe_convert(Ptr{Int32}, out_nf),
                    Base.unsafe_convert(Ptr{Int32}, out_ir),
                )
                @test out_ir[] == Int32(-1)
                @test out_na[] == Int32(1)
                @test out_un[] == Int32(0)
                @test cost[1] ≈ -1.0 && cost[2] == 0.0
                @test ps_ref[].dobias ≈ 10.0
                @test hinrow[1] == Int32(0) && hincol[2] == Int32(0)
            end

            # Postsolve: x0 = 2.0 (at UB) -> x1 = 3.0 (basic), row 0 dual = 2.0, AT_LOWER_BOUND
            p_mcstrt = Int32[JClp.PRESOLVE_NO_LINK, JClp.PRESOLVE_NO_LINK]
            p_hincol = Int32[0, 0]
            p_hrow   = zeros(Int32, 8)
            p_colels = zeros(Float64, 8)
            p_link   = Int32[1, 2, 3, 4, 5, 6, 7, JClp.PRESOLVE_NO_LINK]
            p_sol    = Float64[2.0, 0.0]
            p_acts   = Float64[0.0]
            p_duals  = Float64[0.0]
            p_rcosts = Float64[-1.0, 0.0]
            p_cstat  = UInt8[JClp.PS_STATUS_AT_UPPER_BOUND, JClp.PS_STATUS_AT_LOWER_BOUND]
            p_rstat  = UInt8[JClp.PS_STATUS_BASIC]
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(p_mcstrt), pointer(p_hincol), pointer(p_hrow), pointer(p_link), pointer(p_colels),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                pointer(p_sol), pointer(p_acts), pointer(p_duals), pointer(p_rcosts),
                pointer(p_cstat), pointer(p_rstat),
                1.0, 1e-7, 1e-7,
                Int32(1), Int32(2), Int32(1), Int32(2), Int32(2), Int32(8), Int32(0), Int32(0),
            ))
            GC.@preserve p_mcstrt p_hincol p_hrow p_colels p_link p_sol p_acts p_duals p_rcosts p_cstat p_rstat post_ref begin
                JClp.ps_implied_free_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[],
                )
                @test cost[1] ≈ 1.0 && cost[2] ≈ 2.0
                @test p_sol[2] ≈ 3.0
                @test p_acts[1] ≈ 5.0
                @test p_duals[1] ≈ 2.0
                @test p_cstat[2] == JClp.PS_STATUS_BASIC
                @test p_rstat[1] == JClp.PS_STATUS_AT_LOWER_BOUND
                @test p_rcosts[2] == 0.0
                @test p_hincol == Int32[1, 1]
            end

            for k in 1:Int(out_na[])
                Libc.free(actions[k].rowels)
                if actions[k].costs != C_NULL
                    Libc.free(actions[k].costs)
                end
            end
        end
    end
end

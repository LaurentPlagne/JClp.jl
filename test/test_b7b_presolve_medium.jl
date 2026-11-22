using Test
using JClp

function _test_tighten_alloc(slot::Ptr{JClp.TightenActionPod}, n::Int32)::Cvoid
    r = Ptr{Int32}(Libc.malloc(Int(n) * sizeof(Int32)))
    l = Ptr{Float64}(Libc.malloc(Int(n) * sizeof(Float64)))
    u = Ptr{Float64}(Libc.malloc(Int(n) * sizeof(Float64)))
    unsafe_store!(slot, JClp.TightenActionPod(r, l, u, Int32(0), Int32(0), Int32(0), Int32(0)))
    return nothing
end

function _test_forcing_alloc(slot::Ptr{JClp.ForcingActionPod}, n::Int32)::Cvoid
    rc = Ptr{Int32}(Libc.malloc(Int(n) * sizeof(Int32)))
    bn = Ptr{Float64}(Libc.malloc(Int(n) * sizeof(Float64)))
    unsafe_store!(slot, JClp.ForcingActionPod(rc, bn, Int32(0), Int32(0), Int32(0), Int32(0)))
    return nothing
end

@testset "B7b: CoinPresolveTighten + Forcing + Singleton + Dual" begin
    @testset "ABI layout" begin
        @test sizeof(JClp.TightenActionPod) == 40
        @test sizeof(JClp.ForcingActionPod) == 32
        @test sizeof(JClp.SlackActionPod) == 48
        @test sizeof(JClp.DualActionPod) == 24
    end

    @testset "Tighten presolve + postsolve" begin
        # 2 rows, 3 cols:
        # Row 0: x0 + x1 >= 2.0 (rlo=2.0, rup=INF)
        # Row 1: -x2 <= -1.0    (rlo=-INF, rup=-1.0) -> coeff=-1, rub<1e28, rlb<-1e28 => jflag = 1 (up towards feas)
        # Col 0: cost=0, clo=0, cup=5 -> iflag=1 (up), cup<1e10 -> fixup
        # Col 1: cost=0, clo=0, cup=INF -> iflag=1 (up), unbounded -> free tighten action (row 0 becomes useless)
        # Col 2: cost=0, clo=-3, cup=INF -> in row 1, coeff=-1, rlb=-INF => jflag=1 (up), unbounded -> free tighten action
        INF = JClp.PRESOLVE_INF_DBL
        mcstrt = Int32[0, 1, 2, 3]
        hincol = Int32[1, 1, 1]
        hrow   = Int32[0, 0, 1, 0, 0, 0]
        colels = Float64[1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        mrstrt = Int32[0, 2, 3]
        hinrow = Int32[2, 1]
        hcol   = Int32[0, 1, 2, 0, 0, 0]
        rowels = Float64[1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 4)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)

        clo = Float64[0.0, 0.0, -3.0]
        cup = Float64[5.0, INF, INF]
        rlo = Float64[2.0, -INF]
        rup = Float64[INF, -1.0]
        cost = Float64[0.0, 0.0, 0.0]
        integerType = UInt8[0, 0, 0]
        colChanged = zeros(UInt8, 3)
        rowChanged = zeros(UInt8, 2)
        colsToDo = Int32[0, 1, 2]
        nextColsToDo = zeros(Int32, 3)
        rowsToDo = Int32[0, 1]
        nextRowsToDo = zeros(Int32, 2)
        usefulRowInt = zeros(Int32, 6)
        usefulColInt = zeros(Int32, 9)

        actions = Vector{JClp.TightenActionPod}(undef, 3)
        # Pool buffers owned by test for alloc_cb
        pool_rows = zeros(Int32, 16)
        pool_lb = zeros(Float64, 16)
        pool_ub = zeros(Float64, 16)
        pool_off = Ref(0)

        out_na = Ref(Int32(0))
        out_nu = Ref(Int32(0))
        out_fu = Ref(Int32(0))
        out_fd = Ref(Int32(0))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup cost integerType colChanged rowChanged colsToDo nextColsToDo rowsToDo nextRowsToDo usefulRowInt usefulColInt actions pool_rows pool_lb pool_ub pool_off out_na out_nu out_fu out_fd begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(3))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))

            alloc_cb = @cfunction(_test_tighten_alloc, Cvoid, (Ptr{JClp.TightenActionPod}, Int32))

            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, pointer(integerType),
                C_NULL, C_NULL,
                pointer(colChanged), pointer(rowChanged),
                pointer(colsToDo), pointer(nextColsToDo), pointer(rowsToDo), pointer(nextRowsToDo),
                pointer(usefulRowInt), pointer(usefulColInt), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(3), Int32(2), Int32(3), Int32(3), Int32(6),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(3), Int32(0), Int32(2), Int32(0), Int32(0), Int32(0),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_tighten_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    pointer(actions), alloc_cb, C_NULL, C_NULL,
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_nu),
                    Base.unsafe_convert(Ptr{Int32}, out_fu),
                    Base.unsafe_convert(Ptr{Int32}, out_fd),
                )
            end
            @test out_fu[] == Int32(1)
            @test usefulColInt[1] == Int32(0)
            @test out_fd[] == Int32(3)
            @test out_na[] == Int32(2)
            @test out_nu[] == Int32(2)
            @test rlo[1] == -INF && rup[1] == INF
            @test rlo[2] == -INF && rup[2] == INF

            # Postsolve: x1=0, x2=0, acts=[0.0, 0.0], colstat nonbasic, rowstat basic
            link = Int32[JClp.PRESOLVE_NO_LINK, JClp.PRESOLVE_NO_LINK, JClp.PRESOLVE_NO_LINK, 4, 5, JClp.PRESOLVE_NO_LINK]
            sol = Float64[0.0, 0.0, 0.0]
            acts = Float64[0.0, 0.0]
            colstat = UInt8[JClp.PS_STATUS_AT_LOWER_BOUND, JClp.PS_STATUS_AT_LOWER_BOUND, JClp.PS_STATUS_AT_LOWER_BOUND]
            rowstat = UInt8[JClp.PS_STATUS_BASIC, JClp.PS_STATUS_BASIC]
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(link), pointer(colels),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                pointer(sol), pointer(acts), C_NULL, C_NULL, pointer(colstat), pointer(rowstat),
                1.0, 1e-7, 1e-7,
                Int32(2), Int32(3), Int32(2), Int32(3), Int32(3), Int32(6), Int32(3), Int32(0),
            ))
            GC.@preserve link sol acts colstat rowstat post_ref begin
                JClp.ps_tighten_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[],
                )
                @test rlo[1] == 2.0 && rup[1] == INF
                @test rlo[2] == -INF && rup[2] == -1.0
                @test sol[2] == 2.0
                @test sol[3] == 1.0
                @test acts[1] == 2.0
                @test acts[2] == -1.0
                @test colstat[2] == JClp.PS_STATUS_BASIC
                @test rowstat[1] == JClp.PS_STATUS_AT_LOWER_BOUND
                @test colstat[3] == JClp.PS_STATUS_BASIC
                @test rowstat[2] == JClp.PS_STATUS_AT_UPPER_BOUND

                # Zero-alloc check on postsolve
                allocs = @allocated JClp.ps_tighten_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[],
                )
                @test allocs == 0
            end

            for k in 1:Int(out_na[])
                Libc.free(actions[k].rows)
                Libc.free(actions[k].lbound)
                Libc.free(actions[k].ubound)
            end
        end
    end

    @testset "Forcing presolve + postsolve + canBeFeasible" begin
        # Row 0: x0 + 2*x1 >= 5.0 with x0 in [0,1], x1 in [0,2] => maxup = 1 + 4 = 5 == rlo => forcing (both at UB!)
        # Row 1: x0 + x1 <= 10.0 => maxup = 3 <= 10 => useless!
        INF = JClp.PRESOLVE_INF_DBL
        mcstrt = Int32[0, 2, 4]
        hincol = Int32[2, 2]
        hrow   = Int32[0, 1, 0, 1, 0, 0]
        colels = Float64[1.0, 1.0, 2.0, 1.0, 0.0, 0.0]
        mrstrt = Int32[0, 2, 4]
        hinrow = Int32[2, 2]
        hcol   = Int32[0, 1, 0, 1, 0, 0]
        rowels = Float64[1.0, 2.0, 1.0, 1.0, 0.0, 0.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)

        clo = Float64[0.0, 0.0]
        cup = Float64[1.0, 2.0]
        rlo = Float64[5.0, -INF]
        rup = Float64[INF, 10.0]
        sol = Float64[0.5, 1.0]
        colChanged = zeros(UInt8, 2)
        rowChanged = zeros(UInt8, 2)
        rowsToDo = Int32[0, 1]
        nextColsToDo = zeros(Int32, 2)
        actions = Vector{JClp.ForcingActionPod}(undef, 2)
        fixed_cols = zeros(Int32, 2)
        useless_rows = zeros(Int32, 2)

        out_na = Ref(Int32(0))
        out_nu = Ref(Int32(0))
        out_nf = Ref(Int32(0))
        out_ir = Ref(Int32(-1))

        alloc_cb = @cfunction(_test_forcing_alloc, Cvoid, (Ptr{JClp.ForcingActionPod}, Int32))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup sol colChanged rowChanged rowsToDo nextColsToDo actions fixed_cols useless_rows out_na out_nu out_nf out_ir begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(2))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), C_NULL,
                pointer(sol), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL,
                pointer(colChanged), pointer(rowChanged),
                C_NULL, pointer(nextColsToDo), pointer(rowsToDo), C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(6),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(2), Int32(0), Int32(0), Int32(0),
            ))
            GC.@preserve ps_ref begin
                pp = Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref)
                @test JClp.ps_can_be_feasible(pp, Int32(0)) == Int32(1)
                JClp.ps_forcing_presolve!(
                    pp, pointer(actions), pointer(fixed_cols), pointer(useless_rows), alloc_cb,
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_nu),
                    Base.unsafe_convert(Ptr{Int32}, out_nf),
                    Base.unsafe_convert(Ptr{Int32}, out_ir),
                )
            end
            @test out_ir[] == Int32(-1)
            @test out_na[] == Int32(1)
            @test out_nu[] == Int32(1)
            @test useless_rows[1] == Int32(1)
            @test out_nf[] == Int32(2)
            @test clo == [1.0, 2.0] && cup == [1.0, 2.0]
            @test sol == [1.0, 2.0]

            # Postsolve with positive rcost on x1 (violates NBUB -> dual adjustment needed!)
            link = Int32[1, JClp.PRESOLVE_NO_LINK, 3, JClp.PRESOLVE_NO_LINK, 5, JClp.PRESOLVE_NO_LINK]
            rcosts = Float64[-1.0, 4.0]
            acts = Float64[5.0, 3.0]
            rowduals = Float64[0.0, 0.0]
            colstat = UInt8[JClp.PS_STATUS_AT_UPPER_BOUND, JClp.PS_STATUS_AT_UPPER_BOUND]
            rowstat = UInt8[JClp.PS_STATUS_BASIC, JClp.PS_STATUS_BASIC]
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(link), pointer(colels),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), C_NULL,
                pointer(sol), pointer(acts), pointer(rowduals), pointer(rcosts),
                pointer(colstat), pointer(rowstat),
                1.0, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(6), Int32(4), Int32(0),
            ))
            GC.@preserve link rcosts acts rowduals colstat rowstat post_ref begin
                JClp.ps_forcing_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[],
                )
                @test clo == [0.0, 0.0] && cup == [1.0, 2.0]
                @test rowduals[1] ≈ 2.0 # cbar1 / a01 = 4.0 / 2.0
                @test rcosts[2] ≈ 0.0
                @test rcosts[1] ≈ -3.0 # -1.0 - 2.0*1.0
                @test colstat[2] == JClp.PS_STATUS_BASIC
                @test rowstat[1] == JClp.PS_STATUS_AT_LOWER_BOUND

                allocs = @allocated JClp.ps_forcing_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[],
                )
                @test allocs == 0
            end

            for k in 1:Int(out_na[])
                Libc.free(actions[k].rowcols)
                Libc.free(actions[k].bounds)
            end
        end
    end

    @testset "Slack doubleton & slack singleton presolve + postsolve" begin
        INF = JClp.PRESOLVE_INF_DBL
        # Slack doubleton: Row 0 is singleton: -2*x0 <= -4  (rlo=-INF, rup=-4) => x0 >= 2
        mcstrt = Int32[0, 1]
        hincol = Int32[1]
        hrow   = Int32[0, 0, 0, 0]
        colels = Float64[-2.0, 0.0, 0.0, 0.0]
        mrstrt = Int32[0, 1]
        hinrow = Int32[1]
        hcol   = Int32[0, 0, 0, 0]
        rowels = Float64[-2.0, 0.0, 0.0, 0.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 2)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 2)
        clo = Float64[0.0]
        cup = Float64[10.0]
        rlo = Float64[-INF]
        rup = Float64[-4.0]
        integerType = UInt8[0]
        rowsToDo = Int32[0]
        nextColsToDo = zeros(Int32, 1)
        colChanged = zeros(UInt8, 1)
        usefulColInt = zeros(Int32, 3)
        actions = Vector{JClp.SlackActionPod}(undef, 2)
        out_na = Ref(Int32(0))
        out_nf = Ref(Int32(0))
        out_ic = Ref(Int32(-1))
        out_lo = Ref(0.0)
        out_up = Ref(0.0)

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup integerType rowsToDo nextColsToDo colChanged usefulColInt actions out_na out_nf out_ic out_lo out_up begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(1))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(1))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, pointer(integerType),
                C_NULL, C_NULL,
                pointer(colChanged), C_NULL, C_NULL, pointer(nextColsToDo), pointer(rowsToDo), C_NULL,
                C_NULL, pointer(usefulColInt), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(1), Int32(1), Int32(1), Int32(1), Int32(1), Int32(4),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(1), Int32(0), Int32(0), Int32(0),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_slack_doubleton_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    pointer(actions),
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_nf),
                    Base.unsafe_convert(Ptr{Int32}, out_ic),
                    Base.unsafe_convert(Ptr{Float64}, out_lo),
                    Base.unsafe_convert(Ptr{Float64}, out_up),
                )
            end
            @test out_ic[] == Int32(-1)
            @test out_na[] == Int32(1)
            @test clo[1] == 2.0 && cup[1] == 10.0
            @test hinrow[1] == Int32(0) && hincol[1] == Int32(0)

            # Postsolve: sol[0] = 2.0 (at tightened LB, not orig LB 0.0), rcost[0] = 3.0 -> must become basic, row dual = 3/(-2) = -1.5
            link = Int32[1, 2, 3, JClp.PRESOLVE_NO_LINK]
            mcstrt[1] = JClp.PRESOLVE_NO_LINK
            sol = Float64[2.0]
            acts = Float64[0.0]
            rcosts = Float64[3.0]
            rowduals = Float64[0.0]
            colstat = UInt8[JClp.PS_STATUS_AT_LOWER_BOUND]
            rowstat = UInt8[JClp.PS_STATUS_BASIC]
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(link), pointer(colels),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), C_NULL,
                pointer(sol), pointer(acts), pointer(rowduals), pointer(rcosts),
                pointer(colstat), pointer(rowstat),
                1.0, 1e-7, 1e-7,
                Int32(1), Int32(1), Int32(1), Int32(1), Int32(1), Int32(4), Int32(0), Int32(0),
            ))
            GC.@preserve link sol acts rcosts rowduals colstat rowstat post_ref begin
                JClp.ps_slack_doubleton_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[],
                )
                @test clo[1] == 0.0 && cup[1] == 10.0
                @test rlo[1] == -INF && rup[1] == -4.0
                @test acts[1] == -4.0
                @test rowduals[1] ≈ -1.5
                @test rcosts[1] == 0.0
                @test colstat[1] == JClp.PS_STATUS_BASIC
                @test rowstat[1] == JClp.PS_STATUS_AT_UPPER_BOUND
                @test hincol[1] == Int32(1)
            end
        end

        # Slack singleton: Row 0: x0 + 2*x1 == 6, x1 is costless singleton in [0, 2]
        # Eliminating x1 gives 2 <= x0 <= 6
        mcstrt2 = Int32[0, 1, 2]
        hincol2 = Int32[1, 1]
        hrow2   = Int32[0, 0, 0, 0]
        colels2 = Float64[1.0, 2.0, 0.0, 0.0]
        mrstrt2 = Int32[0, 2]
        hinrow2 = Int32[2]
        hcol2   = Int32[0, 1, 0, 0]
        rowels2 = Float64[1.0, 2.0, 0.0, 0.0]
        clink2 = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlink2 = Vector{JClp.PresolveHLinkPod}(undef, 2)
        clo2 = Float64[0.0, 0.0]
        cup2 = Float64[10.0, 2.0]
        rlo2 = Float64[6.0]
        rup2 = Float64[6.0]
        cost2 = Float64[1.0, 0.0]
        sol2 = Float64[4.0, 1.0]
        acts2 = Float64[6.0]
        colstat2 = UInt8[JClp.PS_STATUS_BASIC, JClp.PS_STATUS_BASIC]
        rowstat2 = UInt8[JClp.PS_STATUS_AT_LOWER_BOUND]
        colChanged2 = zeros(UInt8, 2)
        rowChanged2 = zeros(UInt8, 1)
        colsToDo2 = Int32[0, 1]
        nextRowsToDo2 = zeros(Int32, 1)
        fixed_cols2 = zeros(Int32, 2)
        actions2 = Vector{JClp.SlackActionPod}(undef, 2)
        act_ptr_ref = Ref(pointer(actions2))
        out_na2 = Ref(Int32(0))
        out_nf2 = Ref(Int32(0))

        GC.@preserve mcstrt2 hincol2 hrow2 colels2 mrstrt2 hinrow2 hcol2 rowels2 clink2 rlink2 clo2 cup2 rlo2 rup2 cost2 sol2 acts2 colstat2 rowstat2 colChanged2 rowChanged2 colsToDo2 nextRowsToDo2 fixed_cols2 actions2 act_ptr_ref out_na2 out_nf2 begin
            JClp.ps_make_memlists!(pointer(hincol2), pointer(clink2), Int32(2))
            JClp.ps_make_memlists!(pointer(hinrow2), pointer(rlink2), Int32(1))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt2), pointer(hincol2), pointer(hrow2), pointer(colels2), pointer(clink2),
                pointer(mrstrt2), pointer(hinrow2), pointer(hcol2), pointer(rowels2), pointer(rlink2),
                pointer(clo2), pointer(cup2), pointer(rlo2), pointer(rup2), pointer(cost2),
                pointer(sol2), pointer(acts2), C_NULL, C_NULL, pointer(colstat2), pointer(rowstat2), C_NULL,
                C_NULL, C_NULL,
                pointer(colChanged2), pointer(rowChanged2),
                pointer(colsToDo2), C_NULL, C_NULL, pointer(nextRowsToDo2),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(1), Int32(2), Int32(1), Int32(2), Int32(2), Int32(4),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(2), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_slack_singleton_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    Ptr{Float64}(C_NULL), Base.unsafe_convert(Ptr{Ptr{JClp.SlackActionPod}}, act_ptr_ref),
                    Int32(2), pointer(fixed_cols2), C_NULL,
                    Base.unsafe_convert(Ptr{Int32}, out_na2),
                    Base.unsafe_convert(Ptr{Int32}, out_nf2),
                )
            end
            @test out_na2[] == Int32(1)
            @test out_nf2[] == Int32(1)
            @test rlo2[1] == 2.0 && rup2[1] == 6.0
            @test sol2[2] == 0.0 && acts2[1] == 4.0

            # Postsolve
            link2 = Int32[JClp.PRESOLVE_NO_LINK, 2, 3, JClp.PRESOLVE_NO_LINK]
            mcstrt2[2] = JClp.PRESOLVE_NO_LINK
            rcosts2 = Float64[0.0, 0.0]
            rowduals2 = Float64[0.0]
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(mcstrt2), pointer(hincol2), pointer(hrow2), pointer(link2), pointer(colels2),
                pointer(clo2), pointer(cup2), pointer(rlo2), pointer(rup2), pointer(cost2),
                pointer(sol2), pointer(acts2), pointer(rowduals2), pointer(rcosts2),
                pointer(colstat2), pointer(rowstat2),
                1.0, 1e-7, 1e-7,
                Int32(1), Int32(2), Int32(1), Int32(2), Int32(2), Int32(4), Int32(1), Int32(0),
            ))
            GC.@preserve link2 rcosts2 rowduals2 post_ref begin
                JClp.ps_slack_singleton_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions2), out_na2[],
                )
                @test rlo2[1] == 6.0 && rup2[1] == 6.0
                @test clo2[2] == 0.0 && cup2[2] == 2.0
                @test sol2[2] ≈ 1.0
                @test acts2[1] ≈ 6.0
                @test colstat2[2] == JClp.PS_STATUS_BASIC
                @test hincol2[2] == Int32(1)
            end
        end
    end

    @testset "Dual presolve (phase1 + phase2 + apply) + postsolve" begin
        # Min 2*x0 + 5*x1 s.t. x0 + x1 >= 3, x0 >= 0, x1 in [0, 10]
        # x0 is a non-singleton? Make 2 rows so x0 is singleton in row 0?
        # Even simpler:
        # Row 0: x0 + x1 >= 3 (rlo=3, rup=INF => ymin[0]=0, ymax[0]=INF)
        # Row 1: x0 <= 10 (rlo=-INF, rup=10 => ymin[1]=-INF, ymax[1]=0)
        # Wait: let's test:
        # Row 0: x0 + x1 >= 3 (ymin=0, ymax=INF)
        # Col 2: singleton in Row 0 with coeff 1.0, cost 1.0, clo=0, cup=INF => ymax[0] <= 1.0
        # Col 1: coeff 1.0 in Row 0, cost 5.0, clo=0, cup=10 => cbarmin[1] = 5 - 1*1 = 4 > 0 => fixdown col 1!
        # Also ymin[0] = 0 -> if we make Col 3 singleton with coeff -1, cost -2, clo=0, cup=INF => ymin[0] >= 2.0 > 0 => Row 0 forced to equality!
        INF = JClp.PRESOLVE_INF_DBL
        mcstrt = Int32[0, 1, 2, 3, 4]
        hincol = Int32[1, 1, 1, 1]
        hrow   = Int32[0, 0, 0, 0]
        colels = Float64[1.0, 1.0, 1.0, -1.0]
        mrstrt = Int32[0, 4]
        hinrow = Int32[4]
        hcol   = Int32[0, 1, 2, 3]
        rowels = Float64[1.0, 1.0, 1.0, -1.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 5)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 2)

        clo = Float64[0.0, 0.0, 0.0, 0.0]
        cup = Float64[10.0, 10.0, INF, INF]
        rlo = Float64[3.0]
        rup = Float64[INF]
        cost = Float64[5.0, 5.0, 3.0, -2.0] # col 2 gives ymax[0] <= 3; col 3 gives ymin[0] >= 2 => cbarmin[0]=5-3=2>0, cbarmin[1]=2>0
        integerType = zeros(UInt8, 4)
        colChanged = zeros(UInt8, 4)
        rowChanged = zeros(UInt8, 1)
        colsToDo = Int32[0, 1, 2, 3]
        nextRowsToDo = zeros(Int32, 1)
        usefulRowInt = zeros(Int32, 3)
        usefulColInt = zeros(Int32, 12)
        usefulRowDouble = zeros(Float64, 2)
        cbarmin = zeros(Float64, 4)
        cbarmax = zeros(Float64, 4)
        out_fu = Ref(Int32(0))
        out_fd = Ref(Int32(0))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup cost integerType colChanged rowChanged colsToDo nextRowsToDo usefulRowInt usefulColInt usefulRowDouble cbarmin cbarmax out_fu out_fd begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(4))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(1))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, pointer(integerType),
                C_NULL, C_NULL,
                pointer(colChanged), pointer(rowChanged),
                pointer(colsToDo), C_NULL, C_NULL, pointer(nextRowsToDo),
                pointer(usefulRowInt), pointer(usefulColInt), C_NULL, C_NULL,
                pointer(usefulRowDouble), C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(1), Int32(4), Int32(1), Int32(4), Int32(4), Int32(4),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(4), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
            ))
            GC.@preserve ps_ref begin
                pp = Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref)
                JClp.ps_dual_presolve_phase1!(
                    pp, pointer(cbarmin), pointer(cbarmax), C_NULL, C_NULL,
                    Base.unsafe_convert(Ptr{Int32}, out_fu),
                    Base.unsafe_convert(Ptr{Int32}, out_fd),
                )
                @test out_fu[] == Int32(0)
                @test out_fd[] == Int32(2) # cols 0 and 1 fixed down
                @test usefulRowDouble[1] == 2.0 # ymin[0]
                @test usefulRowDouble[2] == 1.0 # ymax[0] tightened by PRESOLVE_TIGHTEN_DUALS

                # Now test Phase 2 + apply + postsolve on a <= row forced to equality (canFix = 1):
                # Row 0: x0 + x1 + x2 - x3 <= 3 (rlo = -INF, rup = 3), ymax[0] = -1.0 < 0
                rlo[1] = -INF
                rup[1] = 3.0
                usefulRowDouble[1] = -INF
                usefulRowDouble[2] = -1.0
                cost[3] = -1.0 # singleton col 2 (coeff=1, cost=-1) -> sets canFix[0]=1
                cost[4] = 1.0  # singleton col 3 (coeff=-1, cost=1) -> sets canFix[0]=1
                meq = JClp.ps_dual_presolve_phase2!(pp)
                @test meq == Int32(1)

                recs = Vector{JClp.DualActionPod}(undef, 1)
                GC.@preserve recs begin
                    JClp.ps_dual_presolve_apply!(pp, meq, pointer(recs))
                    @test rlo[1] == 3.0 && rup[1] == 3.0
                    @test recs[1].rlo == -INF && recs[1].rup == 3.0 && recs[1].ndx == Int32(0)

                    # Postsolve: row 0 was AT_LOWER_BOUND as equality, must flip to AT_UPPER_BOUND when rlo -> -INF
                    rowstat = UInt8[JClp.PS_STATUS_AT_LOWER_BOUND]
                    post_ref = Ref(JClp.CoinPostsolvePodView(
                        C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                        C_NULL, C_NULL, pointer(rlo), pointer(rup), C_NULL,
                        C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, pointer(rowstat),
                        1.0, 1e-7, 1e-7,
                        Int32(1), Int32(4), Int32(1), Int32(4), Int32(4), Int32(4), Int32(-1), Int32(0),
                    ))
                    GC.@preserve rowstat post_ref begin
                        JClp.ps_dual_postsolve!(
                            Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                            pointer(recs), Int32(1),
                        )
                        @test rlo[1] == -INF && rup[1] == 3.0
                        @test rowstat[1] == JClp.PS_STATUS_AT_UPPER_BOUND

                        allocs = @allocated JClp.ps_dual_postsolve!(
                            Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                            pointer(recs), Int32(1),
                        )
                        @test allocs == 0
                    end
                end
            end
        end
    end
end

using Test
using JClp

function _test_subst_alloc(slot::Ptr{JClp.SubstActionPod}, nc::Int32, ntot::Int32, ncost::Int32)::Cvoid
    rlos = Ptr{Float64}(Libc.malloc(Int(nc) * sizeof(Float64)))
    rups = Ptr{Float64}(Libc.malloc(Int(nc) * sizeof(Float64)))
    cxs  = Ptr{Float64}(Libc.malloc(Int(nc) * sizeof(Float64)))
    rows = Ptr{Int32}(Libc.malloc(Int(nc) * sizeof(Int32)))
    nxs  = Ptr{Int32}(Libc.malloc(Int(nc) * sizeof(Int32)))
    rcs  = Ptr{Int32}(Libc.malloc(Int(ntot) * sizeof(Int32)))
    rels = Ptr{Float64}(Libc.malloc(Int(ntot) * sizeof(Float64)))
    csx  = ncost > Int32(0) ? Ptr{Float64}(Libc.malloc(Int(ncost) * sizeof(Float64))) : Ptr{Float64}(C_NULL)
    unsafe_store!(slot, JClp.SubstActionPod(rlos, rups, cxs, rows, nxs, rcs, rels, csx, Int32(0), Int32(0), Int32(0), Int32(0)))
    return nothing
end

function _test_gub_alloc(slot::Ptr{JClp.GubrowActionPod}, ndrop::Int32, ninrow::Int32)::Cvoid
    dr = Ptr{Int32}(Libc.malloc((Int(ndrop) + 1) * sizeof(Int32)))
    re = Ptr{Float64}(Libc.malloc((Int(ndrop) + 1) * sizeof(Float64)))
    ix = Ptr{Int32}(Libc.malloc(Int(ninrow) * sizeof(Int32)))
    unsafe_store!(slot, JClp.GubrowActionPod(0.0, dr, re, ix, Int32(0), Int32(0)))
    return nothing
end

function _test_dupcol_alloc_dbl(n::Int32)::Ptr{Float64}
    return Ptr{Float64}(Libc.malloc(max(Int(n), 1) * sizeof(Float64)))
end

@testset "B7c2: CoinPresolveSubst + Dupcol + Duprow + Gubrow + Twoxtwo" begin
    @testset "ABI layout" begin
        @test sizeof(JClp.SubstActionPod) == 80
        @test sizeof(JClp.DupcolActionPod) == 56
        @test sizeof(JClp.GubrowActionPod) == 40
        @test sizeof(JClp.TwoxtwoActionPod) == 64
    end

    @testset "coin_init_random_vec deterministic & zero-alloc" begin
        w = zeros(Float64, 8)
        GC.@preserve w begin
            JClp.ps_coin_init_random_vec!(pointer(w), Int32(8))
            @test all(x -> 0.0 < x < 1.0, w)
            # First step: mod(12345678 * 16807, 2147483647) / 2147483647
            expected1 = mod(12345678 * 16807, 2147483647) / 2147483647.0
            @test w[1] ≈ expected1
            allocs = @allocated JClp.ps_coin_init_random_vec!(pointer(w), Int32(8))
            @test allocs == 0
        end
    end

    @testset "Subst presolve + postsolve" begin
        # Row 0 (tgtrow): x0 + x1 + x2 == 6.0  (tgtcol = x2, coeff = 1.0)
        # Row 1:          2*x0 + x2 <= 10.0    -> after subst x2 = 6 - x0 - x1: x0 - x1 <= 4.0
        # Costs: c0 = 1.0, c1 = 2.0, c2 = 3.0  -> c0' = -2, c1' = -1, c2' = 0, dobias += 3*6 = 18
        INF = JClp.PRESOLVE_INF_DBL
        mcstrt = Int32[0, 2, 3, 5, 12]
        hincol = Int32[2, 1, 2, 0]
        hrow   = zeros(Int32, 24); hrow[1:5] .= Int32[0, 1, 0, 0, 1]
        colels = zeros(Float64, 24); colels[1:5] .= Float64[1.0, 2.0, 1.0, 1.0, 1.0]
        mrstrt = Int32[0, 3, 12]
        hinrow = Int32[3, 2]
        hcol   = zeros(Int32, 24); hcol[1:5] .= Int32[0, 1, 2, 0, 2]
        rowels = zeros(Float64, 24); rowels[1:5] .= Float64[1.0, 1.0, 1.0, 2.0, 1.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 5)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)

        rlo = Float64[6.0, -INF]
        rup = Float64[6.0, 10.0]
        cost = Float64[1.0, 2.0, 3.0, 0.0]
        colChanged = zeros(UInt8, 4)
        rowChanged = zeros(UInt8, 2)
        nextColsToDo = zeros(Int32, 4)
        nextRowsToDo = zeros(Int32, 2)
        usefulRowInt = zeros(Int32, 6)
        implied_free = Int32[0]
        whichFree = Int32[2]
        actions = Vector{JClp.SubstActionPod}(undef, 4)
        zerocols = zeros(Int32, 10)
        x_to_y = zeros(Int32, 10)
        out_na = Ref(Int32(0))
        out_nz = Ref(Int32(0))
        out_nm = Ref(Int32(0))
        alloc_cb = @cfunction(_test_subst_alloc, Cvoid, (Ptr{JClp.SubstActionPod}, Int32, Int32, Int32))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink rlo rup cost colChanged rowChanged nextColsToDo nextRowsToDo usefulRowInt implied_free whichFree actions zerocols x_to_y out_na out_nz out_nm begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(4))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                C_NULL, C_NULL, pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL,
                pointer(colChanged), pointer(rowChanged),
                C_NULL, pointer(nextColsToDo), C_NULL, pointer(nextRowsToDo),
                pointer(usefulRowInt), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(4), Int32(2), Int32(4), Int32(5), Int32(24),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(3),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_subst_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    pointer(implied_free), pointer(whichFree), Int32(1), Int32(3),
                    pointer(actions), pointer(zerocols), pointer(x_to_y), alloc_cb,
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_nz),
                    Base.unsafe_convert(Ptr{Int32}, out_nm),
                )
                @test out_na[] == Int32(1)
                @test out_nm[] == Int32(0)
                @test rup[2] ≈ 4.0
                @test cost[1] ≈ -2.0 && cost[2] ≈ -1.0 && cost[3] == 0.0
                @test ps_ref[].dobias ≈ 18.0
                @test hinrow[1] == Int32(0) && hincol[3] == Int32(0)
            end

            # Postsolve: x0 = 2.0, x1 = 1.0 => x2 = 6 - 2 - 1 = 3.0; row1 act = 2*2 + 3 = 7.0
            p_mcstrt = Int32[0, 1, JClp.PRESOLVE_NO_LINK]
            p_hincol = Int32[1, 1, 0]
            p_hrow   = zeros(Int32, 24); p_hrow[1:2] .= Int32[1, 1]
            p_colels = zeros(Float64, 24); p_colels[1:2] .= Float64[1.0, -1.0]
            p_link   = Int32[i < 24 ? Int32(i) : JClp.PRESOLVE_NO_LINK for i in 1:24]
            p_link[1] = JClp.PRESOLVE_NO_LINK; p_link[2] = JClp.PRESOLVE_NO_LINK
            p_sol    = Float64[2.0, 1.0, 0.0]
            p_acts   = Float64[0.0, 1.0]
            p_duals  = Float64[0.0, 0.0]
            p_rcosts = Float64[-2.0, -1.0, 0.0]
            p_cstat  = UInt8[JClp.PS_STATUS_AT_UPPER_BOUND, JClp.PS_STATUS_AT_LOWER_BOUND, JClp.PS_STATUS_AT_LOWER_BOUND]
            p_rstat  = UInt8[JClp.PS_STATUS_AT_LOWER_BOUND, JClp.PS_STATUS_BASIC]
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(p_mcstrt), pointer(p_hincol), pointer(p_hrow), pointer(p_link), pointer(p_colels),
                C_NULL, C_NULL, pointer(rlo), pointer(rup), pointer(cost),
                pointer(p_sol), pointer(p_acts), pointer(p_duals), pointer(p_rcosts),
                pointer(p_cstat), pointer(p_rstat),
                1.0, 1e-7, 1e-7,
                Int32(2), Int32(3), Int32(2), Int32(3), Int32(5), Int32(24), Int32(2), Int32(0),
            ))
            GC.@preserve p_mcstrt p_hincol p_hrow p_colels p_link p_sol p_acts p_duals p_rcosts p_cstat p_rstat post_ref begin
                JClp.ps_subst_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[],
                )
                @test p_sol[3] ≈ 3.0
                @test p_acts[1] ≈ 6.0 && p_acts[2] ≈ 7.0
                @test rup[2] ≈ 10.0
                @test cost[1:3] == [1.0, 2.0, 3.0]
                @test p_duals[1] ≈ 3.0
                @test p_cstat[3] == JClp.PS_STATUS_BASIC
                @test p_rstat[1] == JClp.PS_STATUS_AT_LOWER_BOUND
                @test p_hincol == Int32[2, 1, 2]
            end

            for k in 1:Int(out_na[])
                a = actions[k]
                Libc.free(a.rlos); Libc.free(a.rups); Libc.free(a.coeffxs)
                Libc.free(a.rows); Libc.free(a.ninrowxs); Libc.free(a.rowcolsxs)
                Libc.free(a.rowelsxs)
                if a.costsx != C_NULL; Libc.free(a.costsx); end
            end
        end
    end

    @testset "Dupcol presolve + postsolve" begin
        # Col 0 and Col 1 are identical: [2.0; 1.0], equal cost 3.0, bounds [0, 4] and [0, 5]
        # Should combine into one column with bounds [0, 9]
        INF = JClp.PRESOLVE_INF_DBL
        mcstrt = Int32[0, 2, 4]
        hincol = Int32[2, 2]
        hrow   = Int32[0, 1, 0, 1]
        colels = Float64[2.0, 1.0, 2.0, 1.0]
        mrstrt = Int32[0, 2, 4]
        hinrow = Int32[2, 2]
        hcol   = Int32[0, 1, 0, 1]
        rowels = Float64[2.0, 2.0, 1.0, 1.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        clo = Float64[0.0, 0.0]
        cup = Float64[4.0, 5.0]
        rlo = Float64[-INF, -INF]
        rup = Float64[10.0, 10.0]
        cost = Float64[3.0, 3.0]
        usefulColInt = zeros(Int32, 6)
        usefulColDbl = zeros(Float64, 4)
        usefulRowDbl = zeros(Float64, 4)
        rowmul = zeros(Float64, 2)
        actions = Vector{JClp.DupcolActionPod}(undef, 2)
        fdown = zeros(Int32, 2)
        fup = zeros(Int32, 2)
        out_na = Ref(Int32(0)); out_fd = Ref(Int32(0)); out_fu = Ref(Int32(0))
        alloc_cb = @cfunction(_test_dupcol_alloc_dbl, Ptr{Float64}, (Int32,))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup cost usefulColInt usefulColDbl usefulRowDbl rowmul actions fdown fup out_na out_fd out_fu begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(2))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, pointer(usefulColInt), C_NULL, C_NULL,
                pointer(usefulRowDbl), pointer(usefulColDbl), C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(4),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(3),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_dupcol_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    pointer(rowmul), Ptr{UInt8}(C_NULL), pointer(actions), pointer(fdown), pointer(fup), alloc_cb,
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_fd),
                    Base.unsafe_convert(Ptr{Int32}, out_fu),
                )
                @test out_na[] == Int32(1)
                @test out_fd[] == Int32(0) && out_fu[] == Int32(0)
                kept = Int(actions[1].ilast) + 1
                elim = Int(actions[1].ithis) + 1
                @test cup[kept] == 9.0 && clo[kept] == 0.0
                @test hincol[elim] == Int32(0)
            end

            # Postsolve: kept sol = 6.0 -> splits into elim=0 (or 4) and kept=6 (or 2) within bounds!
            p_mcstrt = Int32[JClp.PRESOLVE_NO_LINK, JClp.PRESOLVE_NO_LINK]
            p_hincol = Int32[0, 0]
            kept = Int(actions[1].ilast) + 1
            elim = Int(actions[1].ithis) + 1
            p_mcstrt[kept] = Int32(0)
            p_hincol[kept] = Int32(2)
            p_hrow   = zeros(Int32, 8); p_hrow[1:2] .= Int32[0, 1]
            p_colels = zeros(Float64, 8); p_colels[1:2] .= Float64[2.0, 1.0]
            p_link   = Int32[1, JClp.PRESOLVE_NO_LINK, 3, 4, 5, 6, 7, JClp.PRESOLVE_NO_LINK]
            p_sol    = Float64[0.0, 0.0]; p_sol[kept] = 6.0
            p_rcosts = Float64[0.0, 0.0]
            p_cstat  = UInt8[JClp.PS_STATUS_AT_LOWER_BOUND, JClp.PS_STATUS_AT_LOWER_BOUND]
            p_cstat[kept] = JClp.PS_STATUS_BASIC
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(p_mcstrt), pointer(p_hincol), pointer(p_hrow), pointer(p_link), pointer(p_colels),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                pointer(p_sol), C_NULL, C_NULL, pointer(p_rcosts), pointer(p_cstat), C_NULL,
                1.0, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(8), Int32(2), Int32(0),
            ))
            GC.@preserve p_mcstrt p_hincol p_hrow p_colels p_link p_sol p_rcosts p_cstat post_ref begin
                JClp.ps_dupcol_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(actions), out_na[],
                )
                @test p_sol[1] + p_sol[2] ≈ 6.0
                @test clo[1] <= p_sol[1] <= cup[1]
                @test clo[2] <= p_sol[2] <= cup[2]
                @test p_hincol == Int32[2, 2]
            end
            for k in 1:Int(out_na[])
                Libc.free(actions[k].colels)
            end
        end
    end

    @testset "Dupcol nonzero lower bounds (USE_LBS=1, nesm regression)" begin
        INF = JClp.PRESOLVE_INF_DBL
        mcstrt = Int32[0, 2, 4]
        hincol = Int32[2, 2]
        hrow   = Int32[0, 1, 0, 1]
        colels = Float64[1.0, -1.0, 1.0, -1.0]
        mrstrt = Int32[0, 2, 4]
        hinrow = Int32[2, 2]
        hcol   = Int32[0, 1, 0, 1]
        rowels = Float64[1.0, 1.0, -1.0, -1.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        clo = Float64[0.7, 0.0]
        cup = Float64[12.7, 21.5]
        rlo = Float64[-INF, -INF]
        rup = Float64[100.0, 100.0]
        cost = Float64[0.0, 0.0]
        usefulColInt = zeros(Int32, 6)
        usefulColDbl = zeros(Float64, 4)
        usefulRowDbl = zeros(Float64, 4)
        rowmul = zeros(Float64, 2)
        actions = Vector{JClp.DupcolActionPod}(undef, 2)
        fdown = zeros(Int32, 2)
        fup = zeros(Int32, 2)
        out_na = Ref(Int32(0)); out_fd = Ref(Int32(0)); out_fu = Ref(Int32(0))
        alloc_cb = @cfunction(_test_dupcol_alloc_dbl, Ptr{Float64}, (Int32,))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup cost usefulColInt usefulColDbl usefulRowDbl rowmul actions fdown fup out_na out_fd out_fu begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(2))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, pointer(usefulColInt), C_NULL, C_NULL,
                pointer(usefulRowDbl), pointer(usefulColDbl), C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(4),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(3),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_dupcol_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    pointer(rowmul), Ptr{UInt8}(C_NULL), pointer(actions), pointer(fdown), pointer(fup), alloc_cb,
                    Base.unsafe_convert(Ptr{Int32}, out_na),
                    Base.unsafe_convert(Ptr{Int32}, out_fd),
                    Base.unsafe_convert(Ptr{Int32}, out_fu),
                )
                @test out_na[] == Int32(1)
                kept = Int(actions[1].ilast) + 1
                @test clo[kept] ≈ 0.7 && cup[kept] ≈ 34.2
            end
            for k in 1:Int(out_na[])
                Libc.free(actions[k].colels)
            end
        end
    end

    @testset "Duprow & Gubrow presolve + postsolve" begin
        INF = JClp.PRESOLVE_INF_DBL
        # Duprow: Row 0: x0 + 2*x1 <= 10; Row 1: x0 + 2*x1 <= 5 -> Row 0 redundant
        mrstrt = Int32[0, 2, 4]
        hinrow = Int32[2, 2]
        hcol   = Int32[0, 1, 0, 1]
        rowels = Float64[1.0, 2.0, 1.0, 2.0]
        rlo = Float64[-INF, -INF]
        rup = Float64[10.0, 5.0]
        sort_arr = zeros(Int32, 2)
        workrow = zeros(Float64, 3)
        workcol = zeros(Float64, 3)
        out_nu = Ref(Int32(0)); out_ir = Ref(Int32(-1))
        GC.@preserve mrstrt hinrow hcol rowels rlo rup sort_arr workrow workcol out_nu out_ir begin
            ps_ref = Ref(JClp.CoinPresolvePodView(
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), C_NULL,
                C_NULL, C_NULL, pointer(rlo), pointer(rup), C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(4),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(3),
            ))
            GC.@preserve ps_ref begin
                JClp.ps_duprow_presolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                    pointer(sort_arr), pointer(workrow), pointer(workcol),
                    Base.unsafe_convert(Ptr{Int32}, out_nu),
                    Base.unsafe_convert(Ptr{Int32}, out_ir),
                )
                @test out_ir[] == Int32(-1)
                @test out_nu[] == Int32(1)
                @test sort_arr[1] == Int32(0)
            end
        end

        # Gubrow: Row 0: x0 + x1 == 1.0; Row 1: 3*x0 + 3*x1 + x2 <= 5.0 => drop x0,x1 from Row 1, rhs becomes 2.0
        mcstrt = Int32[0, 2, 4, 5]
        hincol = Int32[2, 2, 1]
        hrow   = Int32[0, 1, 0, 1, 1]
        colels = Float64[1.0, 3.0, 1.0, 3.0, 1.0]
        mrstrt2 = Int32[0, 2, 5]
        hinrow2 = Int32[2, 3]
        hcol2   = Int32[0, 1, 0, 1, 2]
        rowels2 = Float64[1.0, 1.0, 3.0, 3.0, 1.0]
        clink2 = Vector{JClp.PresolveHLinkPod}(undef, 4)
        rlink2 = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlo2 = Float64[1.0, -INF]
        rup2 = Float64[1.0, 5.0]
        usefulRowInt2 = zeros(Int32, 6)
        usefulRowDbl2 = zeros(Float64, 4)
        usefulColInt2 = zeros(Int32, 6)
        gub_acts = Vector{JClp.GubrowActionPod}(undef, 2)
        gub_cb = @cfunction(_test_gub_alloc, Cvoid, (Ptr{JClp.GubrowActionPod}, Int32, Int32))

        GC.@preserve mcstrt hincol hrow colels mrstrt2 hinrow2 hcol2 rowels2 clink2 rlink2 rlo2 rup2 usefulRowInt2 usefulRowDbl2 usefulColInt2 gub_acts begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink2), Int32(3))
            JClp.ps_make_memlists!(pointer(hinrow2), pointer(rlink2), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink2),
                pointer(mrstrt2), pointer(hinrow2), pointer(hcol2), pointer(rowels2), pointer(rlink2),
                C_NULL, C_NULL, pointer(rlo2), pointer(rup2), C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                pointer(usefulRowInt2), pointer(usefulColInt2), C_NULL, C_NULL,
                pointer(usefulRowDbl2), C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(3), Int32(2), Int32(3), Int32(5), Int32(5),
                Int32(0), Int32(0), Int32(0), Int32(0),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(3),
            ))
            na = GC.@preserve ps_ref JClp.ps_gubrow_presolve!(
                Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref),
                pointer(gub_acts), gub_cb,
            )
            @test na == Int32(1)
            @test hinrow2[2] == Int32(1)
            @test rup2[2] ≈ 2.0

            # Postsolve
            p_mcstrt = Int32[0, 1, 2]
            p_hincol = Int32[1, 1, 1]
            p_hrow   = zeros(Int32, 8); p_hrow[1:3] .= Int32[0, 0, 1]
            p_colels = zeros(Float64, 8); p_colels[1:3] .= Float64[1.0, 1.0, 1.0]
            p_link   = Int32[JClp.PRESOLVE_NO_LINK, JClp.PRESOLVE_NO_LINK, JClp.PRESOLVE_NO_LINK, 4, 5, 6, 7, JClp.PRESOLVE_NO_LINK]
            p_acts   = Float64[1.0, 2.0]
            p_duals  = Float64[1.0, -2.0]
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(p_mcstrt), pointer(p_hincol), pointer(p_hrow), pointer(p_link), pointer(p_colels),
                C_NULL, C_NULL, pointer(rlo2), pointer(rup2), C_NULL,
                C_NULL, pointer(p_acts), pointer(p_duals), C_NULL, C_NULL, C_NULL,
                1.0, 1e-7, 1e-7,
                Int32(2), Int32(3), Int32(2), Int32(3), Int32(5), Int32(8), Int32(3), Int32(0),
            ))
            GC.@preserve p_mcstrt p_hincol p_hrow p_colels p_link p_acts p_duals post_ref begin
                JClp.ps_gubrow_postsolve!(
                    Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref),
                    pointer(gub_acts), na,
                )
                @test rup2[2] ≈ 5.0
                @test p_acts[2] ≈ 5.0
                @test p_duals[1] ≈ 7.0 # 1.0 - (3.0 * (-2.0)) / 1.0
                @test p_hincol == Int32[2, 2, 1]
            end
            for k in 1:Int(na)
                Libc.free(gub_acts[k].deletedRow)
                Libc.free(gub_acts[k].rowels)
                Libc.free(gub_acts[k].indices)
            end
        end
    end
end

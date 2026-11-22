using Test
using JClp

@testset "B7a: CoinPresolve Core Helpers + Zeros + Empty + Isolated + Useless + Fixed" begin
    @testset "ABI layout" begin
        @test sizeof(JClp.PresolveHLinkPod) == 8
        @test sizeof(JClp.CoinPresolvePodView) == 416
        @test sizeof(JClp.CoinPostsolvePodView) == 200
        @test sizeof(JClp.DroppedZeroPod) == 8
        @test sizeof(JClp.DropEmptyColActionPod) == 40
        @test sizeof(JClp.DropEmptyRowActionPod) == 24
        @test sizeof(JClp.UselessActionPod) == 40
        @test sizeof(JClp.RemoveFixedActionPod) == 16
        @test sizeof(JClp.MakeFixedActionPod) == 16
        @test JClp.jclp_ps_pod_sizeof() == Cint(416)
        @test JClp.jclp_post_pod_sizeof() == Cint(200)
    end

    @testset "ps_make_memlists! & ps_expand_major!" begin
        # 3 columns: lengths [2, 0, 2], starts [0, 2, 2, 10] (bulkCap = 10)
        starts = Int32[0, 2, 2, 10]
        lens = Int32[2, 0, 2]
        els = zeros(Float64, 10); els[1:4] .= [1.0, 2.0, 3.0, 4.0]
        rows = zeros(Int32, 10); rows[1:4] .= Int32[0, 1, 0, 2]
        lnks = Vector{JClp.PresolveHLinkPod}(undef, 4)
        GC.@preserve starts lens els rows lnks begin
            JClp.ps_make_memlists!(pointer(lens), pointer(lnks), Int32(3))
            @test lnks[1] == JClp.PresolveHLinkPod(JClp.PRESOLVE_NO_LINK, Int32(2))
            @test lnks[2] == JClp.PresolveHLinkPod(JClp.PRESOLVE_NO_LINK, JClp.PRESOLVE_NO_LINK)
            @test lnks[3] == JClp.PresolveHLinkPod(Int32(0), Int32(3))
            @test lnks[4] == JClp.PresolveHLinkPod(Int32(2), JClp.PRESOLVE_NO_LINK)

            # Expand col 0 (must move to end after col 2 at index 4)
            fail = JClp.ps_expand_major!(pointer(starts), pointer(els), pointer(rows), pointer(lens), pointer(lnks), Int32(3), Int32(0))
            @test !fail
            @test starts[1] == Int32(4)
            @test els[5:6] == [1.0, 2.0]
            @test rows[5:6] == Int32[0, 1]
            @test lnks[4].pre == Int32(0)

            # Expand col 0 again -> in-place (4+2+1 < 10)
            fail2 = JClp.ps_expand_major!(pointer(starts), pointer(els), pointer(rows), pointer(lens), pointer(lnks), Int32(3), Int32(0))
            @test !fail2
            @test starts[1] == Int32(4)

            # Zero-alloc check
            allocs = @allocated JClp.ps_expand_major!(pointer(starts), pointer(els), pointer(rows), pointer(lens), pointer(lnks), Int32(3), Int32(0))
            @test allocs == 0
        end
    end

    @testset "ps_delete_from_major2! (threaded postsolve matrix)" begin
        # Col 0 threaded: 0 -> 1 -> 2 -> NO_LINK, rows [10, 20, 30]
        mcstrt = Int32[0]
        hincol = Int32[3]
        hrow = Int32[10, 20, 30, 0]
        link = Int32[1, 2, JClp.PRESOLVE_NO_LINK, JClp.PRESOLVE_NO_LINK]
        fl = Ref(Int32(3))
        GC.@preserve mcstrt hincol hrow link fl begin
            flp = Base.unsafe_convert(Ptr{Int32}, fl)
            # Delete middle entry (row 20 at slot 1)
            JClp.ps_delete_from_major2!(Int32(0), Int32(20), pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(link), flp)
            @test hincol[1] == Int32(2)
            @test fl[] == Int32(1)
            @test link[1] == Int32(2)
            # Delete head entry (row 10 at slot 0)
            JClp.ps_delete_from_major2!(Int32(0), Int32(10), pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(link), flp)
            @test hincol[1] == Int32(1)
            @test mcstrt[1] == Int32(2)
            @test fl[] == Int32(0)
        end
    end

    @testset "Zeros presolve + postsolve" begin
        # 2x2 matrix with 1 explicit zero at (row=1, col=0)
        mcstrt = Int32[0, 2, 3]
        hincol = Int32[2, 1]
        hrow   = Int32[0, 1, 1, 0, 0, 0]
        colels = Float64[2.0, 1e-15, 3.0, 0.0, 0.0, 0.0]
        mrstrt = Int32[0, 1, 3]
        hinrow = Int32[1, 2]
        hcol   = Int32[0, 0, 1, 0, 0, 0]
        rowels = Float64[2.0, 1e-15, 3.0, 0.0, 0.0, 0.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        checkcols = Int32[1, 0, 0] # unsorted with duplicate
        ncheck = Ref(Int32(3))
        actions = Vector{JClp.DroppedZeroPod}(undef, 4)

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink checkcols ncheck actions begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(2))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(3), Int32(6),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
            ))
            GC.@preserve ps_ref begin
                pp = Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref)
                nz = JClp.ps_zeros_count!(pp, pointer(checkcols), Base.unsafe_convert(Ptr{Int32}, ncheck))
                @test nz == Int32(1)
                @test ncheck[] == Int32(1)
                @test checkcols[1] == Int32(0)
                na = JClp.ps_zeros_drop!(pp, pointer(checkcols), ncheck[], pointer(actions))
                @test na == Int32(1)
                @test actions[1] == JClp.DroppedZeroPod(Int32(1), Int32(0))
                @test hincol[1] == Int32(1) && hinrow[2] == Int32(1)
            end

            # Postsolve
            post_link = Int32[JClp.PRESOLVE_NO_LINK, 3, JClp.PRESOLVE_NO_LINK, 4, 5, JClp.PRESOLVE_NO_LINK]
            post_ref = Ref(JClp.CoinPostsolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(post_link), pointer(colels),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                1.0, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(3), Int32(6), Int32(1), Int32(0),
            ))
            GC.@preserve post_link post_ref begin
                JClp.ps_zeros_postsolve!(Base.unsafe_convert(Ptr{JClp.CoinPostsolvePodView}, post_ref), pointer(actions), Int32(1))
                @test hincol[1] == Int32(2)
                @test post_ref[].free_list == Int32(3)
                @test colels[2] == 0.0 && hrow[2] == Int32(1)
            end
        end
    end

    @testset "RemoveFixed & TransferCosts & TestRedundant" begin
        # 2x2: row 0: 2*x0 + x1 <= 10; row 1: x0 + x1 <= 100 (redundant when x0=2, x1 in [0, 4])
        # x0 fixed at 2.0, x1 in [0, 4], cost = [3.0, 5.0]
        mcstrt = Int32[0, 2, 4]
        hincol = Int32[2, 2]
        hrow   = Int32[0, 1, 0, 1, 0, 0]
        colels = Float64[2.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        mrstrt = Int32[0, 2, 4]
        hinrow = Int32[2, 2]
        hcol   = Int32[0, 1, 0, 1, 0, 0]
        rowels = Float64[2.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        clink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        rlink = Vector{JClp.PresolveHLinkPod}(undef, 3)
        clo = Float64[2.0, 0.0]; cup = Float64[2.0, 4.0]
        rlo = Float64[-JClp.PRESOLVE_INF_DBL, -JClp.PRESOLVE_INF_DBL]
        rup = Float64[10.0, 100.0]
        cost = Float64[3.0, 5.0]
        int_type = UInt8[0, 0]
        col_ch = UInt8[0, 0]; row_ch = UInt8[0, 0]
        cols_todo = Int32[0, 0]; next_cols = Int32[0, 0]
        rows_todo = Int32[0, 0]; next_rows = Int32[0, 0]
        useful_ri = zeros(Int32, 6); useful_ci = zeros(Int32, 4)

        col_lo = zeros(Float64, 2); col_hi = zeros(Float64, 2)
        out_nu = Ref(Int32(0)); out_nf = Ref(Int32(0)); out_ir = Ref(Int32(-1))

        GC.@preserve mcstrt hincol hrow colels mrstrt hinrow hcol rowels clink rlink clo cup rlo rup cost int_type col_ch row_ch cols_todo next_cols rows_todo next_rows useful_ri useful_ci col_lo col_hi out_nu out_nf out_ir begin
            JClp.ps_make_memlists!(pointer(hincol), pointer(clink), Int32(2))
            JClp.ps_make_memlists!(pointer(hinrow), pointer(rlink), Int32(2))
            ps_ref = Ref(JClp.CoinPresolvePodView(
                pointer(mcstrt), pointer(hincol), pointer(hrow), pointer(colels), pointer(clink),
                pointer(mrstrt), pointer(hinrow), pointer(hcol), pointer(rowels), pointer(rlink),
                pointer(clo), pointer(cup), pointer(rlo), pointer(rup), pointer(cost),
                C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, pointer(int_type),
                C_NULL, C_NULL,
                pointer(col_ch), pointer(row_ch), pointer(cols_todo), pointer(next_cols), pointer(rows_todo), pointer(next_rows),
                pointer(useful_ri), pointer(useful_ci), C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL, C_NULL,
                0.0, 1.0, 1e-7, 1e-7, 1e-7,
                Int32(2), Int32(2), Int32(2), Int32(2), Int32(4), Int32(6),
                Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0),
            ))
            GC.@preserve ps_ref begin
                pp = Base.unsafe_convert(Ptr{JClp.CoinPresolvePodView}, ps_ref)
                # testRedundant should identify both rows as redundant (row 0 max = 2*2+4=8 <= 10, row 1 max = 2+4=6 <= 100)
                ninf = JClp.ps_test_redundant_core!(
                    pp, pointer(col_lo), pointer(col_hi),
                    Base.unsafe_convert(Ptr{Int32}, out_nu),
                    Base.unsafe_convert(Ptr{Int32}, out_nf),
                    Base.unsafe_convert(Ptr{Int32}, out_ir),
                )
                @test ninf == Int32(0)
                @test out_nu[] == Int32(2)

                # Test remove_fixed_presolve on col 0
                fcols = Int32[0]
                rf_acts = Vector{JClp.RemoveFixedActionPod}(undef, 2)
                rf_els = zeros(Float64, 2); rf_rows = zeros(Int32, 2)
                rstrt = zeros(Int32, 3); col_buf = zeros(Int32, 2); mark = zeros(UInt8, 2)
                GC.@preserve fcols rf_acts rf_els rf_rows rstrt col_buf mark begin
                    asize = JClp.ps_remove_fixed_presolve!(
                        pp, pointer(fcols), Int32(1),
                        pointer(rf_acts), pointer(rf_els), pointer(rf_rows),
                        pointer(rstrt), pointer(col_buf), pointer(mark),
                    )
                    @test asize == Int32(2)
                    @test hincol[1] == Int32(0)
                    @test hinrow[1] == Int32(1) && hinrow[2] == Int32(1)
                    @test rup[1] == 6.0 && rup[2] == 98.0
                end
            end
        end
    end
end

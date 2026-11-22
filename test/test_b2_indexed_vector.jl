@testset "B2 CoinIndexedVector" begin
    @testset "Construction, reserve, insert, clear (sparse/odd/even/dense/packed)" begin
        v = JClp.CoinIndexedVector(20)
        @test JClp.capacity(v) == Int32(20)
        @test JClp.get_num_elements(v) == Int32(0)
        @test JClp.check_clear(v)

        # Sparse odd count (3 * 3 < 20)
        JClp.quick_insert!(v, Int32(2), 1.5)
        JClp.quick_insert!(v, Int32(7), -3.25)
        JClp.quick_insert!(v, Int32(19), 4.0)
        @test JClp.get_num_elements(v) == Int32(3)
        @test v.elements[3] == 1.5 && v.elements[8] == -3.25 && v.elements[20] == 4.0
        JClp.clear!(v)
        @test JClp.check_clear(v)

        # Sparse even count (3 * 4 < 20)
        for idx in Int32[0, 5, 10, 15]
            JClp.quick_insert!(v, idx, Float64(idx + 1))
        end
        JClp.clear!(v)
        @test JClp.check_clear(v)

        # Dense clear path (3 * 8 >= 20)
        for idx in Int32(0):Int32(7)
            JClp.quick_insert!(v, idx, Float64(idx + 1))
        end
        JClp.clear!(v)
        @test JClp.check_clear(v)

        # Auto-reserve on insert!
        JClp.insert!(v, Int32(35), 9.5)
        @test JClp.capacity(v) == Int32(36)
        @test v.elements[36] == 9.5
        JClp.clear!(v)
        @test JClp.check_clear(v)
    end

    @testset "add / quick_add tiny cancellation semantics" begin
        v = JClp.CoinIndexedVector(10)
        # Below TINY (1e-50) on empty slot -> ignored
        JClp.add!(v, Int32(3), 1e-60)
        @test JClp.get_num_elements(v) == Int32(0)
        @test v.elements[4] == 0.0

        # Normal add + exact cancellation -> becomes REALLY_TINY (1e-100), stays in indices
        JClp.add!(v, Int32(3), 2.5)
        @test JClp.get_num_elements(v) == Int32(1)
        @test v.elements[4] === 2.5
        JClp.add!(v, Int32(3), -2.5)
        @test JClp.get_num_elements(v) == Int32(1)
        @test v.elements[4] === JClp.COIN_INDEXED_REALLY_TINY_ELEMENT

        # zero_entry!
        JClp.add!(v, Int32(5), 7.0)
        JClp.zero_entry!(v, Int32(5))
        @test v.elements[6] === JClp.COIN_INDEXED_REALLY_TINY_ELEMENT

        # clean! removes tiny entries and zeros dense slots
        n_left = JClp.clean!(v, 1e-12)
        @test n_left == Int32(0)
        @test JClp.check_clear(v)
    end

    @testset "scan, clean_and_pack, scan_and_pack, expand, create_packed/unpacked" begin
        v = JClp.CoinIndexedVector(12)
        v.elements[2] = 1.0       # idx 1
        v.elements[5] = 1e-14     # idx 4 (below tol)
        v.elements[8] = -2.5      # idx 7
        v.elements[12] = 3.0      # idx 11

        # scan with tolerance
        n = JClp.scan!(v, 1e-10)
        @test n == Int32(3)
        @test v.indices[1:3] == Int32[1, 7, 11]
        @test v.elements[5] == 0.0 # cleaned below tol

        # clean_and_pack!
        n2 = JClp.clean_and_pack!(v, 1.5)
        @test n2 == Int32(2)
        @test JClp.packed_mode(v)
        @test v.indices[1:2] == Int32[7, 11]
        @test v.elements[1:2] == Float64[-2.5, 3.0]
        @test v.elements[8] == 0.0 && v.elements[12] == 0.0

        # expand! back to unpacked
        JClp.expand!(v)
        @test !JClp.packed_mode(v)
        @test v.elements[8] == -2.5 && v.elements[12] == 3.0
        @test v.elements[1] == 0.0 && v.elements[2] == 0.0

        # scan_and_pack! with tolerance
        v.elements[4] = 1e-15
        n3 = JClp.scan_and_pack!(v, 1e-10)
        @test n3 == Int32(2)
        @test JClp.packed_mode(v)
        @test v.indices[1:2] == Int32[7, 11]
        @test v.elements[1:2] == Float64[-2.5, 3.0]
        JClp.clear!(v)
        @test JClp.check_clear(v)

        # create_packed! + sort_packed! + copy_indexed!
        JClp.create_packed!(v, Int32(3), Int32[9, 2, 5], Float64[90.0, 20.0, 50.0])
        JClp.sort_packed!(v)
        @test v.indices[1:3] == Int32[2, 5, 9]
        @test v.elements[1:3] == Float64[20.0, 50.0, 90.0]

        v2 = JClp.CoinIndexedVector(12)
        JClp.copy_indexed!(v2, v, -2.0)
        @test JClp.packed_mode(v2)
        @test JClp.get_num_elements(v2) == Int32(3)
        @test v2.indices[1:3] == Int32[2, 5, 9]
        @test v2.elements[1:3] == Float64[-40.0, -100.0, -180.0]

        # copy_indexed! with tiny multiplier clamps to REALLY_TINY
        JClp.copy_indexed!(v2, v, 0.0)
        @test all(i -> v2.elements[i] === JClp.COIN_INDEXED_REALLY_TINY_ELEMENT, 1:3)
    end

    @testset "Zero-allocation checks on hot C ABI kernels" begin
        function _bench_civ_alloc(el::Vector{Float64}, idx::Vector{Int32})
            a1 = GC.@preserve el idx @allocated JClp.jclp_civ_clear(pointer(el), pointer(idx), Cint(3), Cint(length(el)), Cint(0))
            el[2] = 1.0; el[5] = -2.0; el[9] = 1e-15
            a2 = GC.@preserve el idx @allocated JClp.jclp_civ_scan_tol(pointer(el), pointer(idx), Cint(0), Cint(length(el)), Cint(0), Cint(length(el)), 1e-10)
            a3 = GC.@preserve el idx @allocated JClp.jclp_civ_clean(pointer(el), pointer(idx), Cint(2), 1e-10)
            a4 = GC.@preserve el idx @allocated JClp.jclp_civ_clean_and_pack(pointer(el), pointer(idx), Cint(2), 1e-10)
            return a1 + a2 + a3 + a4
        end
        el = zeros(Float64, 32)
        idx = Int32[1, 4, 8]
        resize!(idx, 32)
        _bench_civ_alloc(el, idx) # warmup
        @test _bench_civ_alloc(el, idx) == 0
    end
end

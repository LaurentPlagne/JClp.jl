@testset "B1 CoinUtils Basics (Finite, Random, Helpers, Sort)" begin
    @testset "CoinFinite / CoinIsnan" begin
        for x in (0.0, -0.0, 1.0, -1e300, 1e-300, floatmax(Float64), -floatmax(Float64))
            @test JClp.coin_finite(x) === true
            @test JClp.coin_isnan(x) === false
            @test JClp.jclp_coin_finite(x) === Cint(1)
            @test JClp.jclp_coin_isnan(x) === Cint(0)
        end
        for x in (Inf, -Inf)
            @test JClp.coin_finite(x) === false
            @test JClp.coin_isnan(x) === false
            @test JClp.jclp_coin_finite(x) === Cint(0)
            @test JClp.jclp_coin_isnan(x) === Cint(0)
        end
        @test JClp.coin_finite(NaN) === false
        @test JClp.coin_isnan(NaN) === true
        @test JClp.jclp_coin_finite(NaN) === Cint(0)
        @test JClp.jclp_coin_isnan(NaN) === Cint(1)
    end

    @testset "CoinThreadRandom (32-bit LCG bit-exact)" begin
        rng = JClp.CoinThreadRandom()
        @test JClp.get_seed(rng) === UInt32(12345678)

        # Exact C++ COIN_OWN_RANDOM_32 sequence verification
        s = UInt32(12345678)
        for _ in 1:100
            s = UInt32(1664525) * s + UInt32(1013904223)
            expected = Float64(s) / 4294967296.0
            got = JClp.random_double!(rng)
            @test got === expected
            @test JClp.get_seed(rng) === s
            @test 0.0 <= got < 1.0
        end

        # Raw-pointer C ABI matches struct version identically & 0 allocs
        seed_box = Ref{UInt32}(UInt32(987654321))
        rng2 = JClp.CoinThreadRandom(UInt32(987654321))
        for _ in 1:50
            v1 = JClp.random_double!(rng2)
            v2 = GC.@preserve seed_box JClp.jclp_random_double(Base.unsafe_convert(Ptr{UInt32}, seed_box))
            @test v1 === v2
            @test JClp.get_seed(rng2) === seed_box[]
        end

        # randomize! with n=0 (uses seed & 255) and explicit n
        JClp.set_seed!(rng2, 42)
        seed_box[] = UInt32(42)
        JClp.randomize!(rng2, Int32(0))
        GC.@preserve seed_box JClp.jclp_randomize(Base.unsafe_convert(Ptr{UInt32}, seed_box), Cint(0))
        @test JClp.get_seed(rng2) === seed_box[]

        allocs = GC.@preserve seed_box @allocated JClp.jclp_random_double(Base.unsafe_convert(Ptr{UInt32}, seed_box))
        @test allocs == 0
    end

    @testset "CoinHelperFunctions" begin
        @test JClp.coin_max(3, 7) == 7
        @test JClp.coin_min(3.5, -1.2) == -1.2

        a = Float64[1.0, 2.0, 3.0, 4.0, 5.0]
        b = zeros(Float64, 5)
        GC.@preserve a b JClp.coin_memcpy_n!(pointer(a), 5, pointer(b))
        @test b == a

        # Overlapping memmove via coin_copy_n!
        GC.@preserve b JClp.coin_copy_n!(pointer(b), 3, pointer(b) + sizeof(Float64))
        @test b == Float64[1.0, 1.0, 2.0, 3.0, 5.0]

        JClp.coin_zero_n!(b, 3)
        @test b == Float64[0.0, 0.0, 0.0, 3.0, 5.0]

        JClp.coin_fill_n!(b, 4, -7.25)
        @test b == Float64[-7.25, -7.25, -7.25, -7.25, 5.0]

        idx = Vector{Int32}(undef, 6)
        JClp.coin_iota_n!(idx, 6, Int32(10))
        @test idx == Int32[10, 11, 12, 13, 14, 15]
        @test JClp.coin_is_sorted(idx) === true
        idx[4] = Int32(5)
        @test JClp.coin_is_sorted(idx) === false
        @test JClp.coin_is_sorted(Int32[]) === true
        @test JClp.coin_is_sorted(Int32[42]) === true
    end

    @testset "CoinSort_2 & CoinSort_3 (correctness, edge cases, 0 allocs)" begin
        # Edge sizes: 0, 1, 2, 3, 10, 11, 12, 100, 1500
        for n in (0, 1, 2, 3, 5, 10, 11, 12, 31, 100, 1500)
            # 1) Already sorted
            k = Int32[i for i in 1:n]
            v = Float64[Float64(i) * 1.5 for i in 1:n]
            JClp.coin_sort_2!(k, v)
            @test JClp.coin_is_sorted(k)
            @test all(i -> v[i] === Float64(k[i]) * 1.5, 1:n)

            # 2) Reverse sorted
            k = Int32[n - i + 1 for i in 1:n]
            v = Float64[Float64(k[i]) * -2.0 for i in 1:n]
            JClp.coin_sort_2!(k, v)
            @test JClp.coin_is_sorted(k)
            @test all(i -> v[i] === Float64(k[i]) * -2.0, 1:n)

            # 3) All equal keys (adversarial for naive quicksort!)
            k = fill(Int32(7), n)
            v = Float64[Float64(i) for i in 1:n]
            JClp.coin_sort_2!(k, v)
            @test all(==(Int32(7)), k)
            @test sort(v) == Float64[Float64(i) for i in 1:n]

            # 4) Pseudo-random with duplicates (deterministic LCG)
            rng = JClp.CoinThreadRandom(UInt32(1000 + n))
            kf = Float64[floor(JClp.random_double!(rng) * max(1, n ÷ 3)) for _ in 1:n]
            vi = Int32[i for i in 1:n]
            pairs_before = sort(collect(zip(kf, vi)); by = first)
            kf_std = copy(kf)
            vi_std = copy(vi)
            JClp.coin_sort_2!(kf, vi)
            @test JClp.coin_is_sorted(kf)
            @test sort(collect(zip(kf, vi))) == sort(pairs_before)
            JClp.coin_sort_2_std!(kf_std, vi_std)
            @test JClp.coin_is_sorted(kf_std)
            @test sort(collect(zip(kf_std, vi_std))) == sort(pairs_before)

            # 5) Descending sort
            kf_d = copy(kf)
            vi_d = copy(vi)
            reverse!(kf_d)
            reverse!(vi_d)
            JClp.coin_sort_2_desc!(kf_d, vi_d)
            @test all(i -> kf_d[i] >= kf_d[i + 1], 1:(n - 1))
            @test sort(collect(zip(kf_d, vi_d))) == sort(collect(zip(kf, vi)))

            # 6) CoinSort_3
            k3 = Int32[(i * 37) % max(1, n) for i in 1:n]
            v3 = Int32[i for i in 1:n]
            w3 = Float64[Float64(k3[i]) + 0.001 * i for i in 1:n]
            triples_before = sort(collect(zip(k3, v3, w3)))
            JClp.coin_sort_3!(k3, v3, w3)
            @test JClp.coin_is_sorted(k3)
            @test sort(collect(zip(k3, v3, w3))) == triples_before
        end

        # Zero-allocation check on C ABI sort entrypoints (inside typed local function)
        function _bench_sort_alloc(k::Vector{Int32}, v::Vector{Float64})
            n = Cint(length(k))
            return GC.@preserve k v @allocated JClp.jclp_coin_sort_2_i32_f64(pointer(k), pointer(v), n)
        end
        k = Int32[5, 2, 9, 1, 7, 3, 8, 4, 6, 0, 12, 11, 10, 15, 14, 13]
        v = Float64[Float64(x) for x in k]
        _bench_sort_alloc(k, v) # warmup
        k .= Int32[5, 2, 9, 1, 7, 3, 8, 4, 6, 0, 12, 11, 10, 15, 14, 13]
        @test _bench_sort_alloc(k, v) == 0
        @test JClp.coin_is_sorted(k)
    end
end

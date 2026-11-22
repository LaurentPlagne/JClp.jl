using Test
using JClp

function _cstr(s::String)
    v = Vector{UInt8}(undef, ncodeunits(s) + 1)
    copyto!(v, 1, codeunits(s), 1, ncodeunits(s))
    v[end] = 0x00
    return v
end

function _read_cstr(v::Vector{UInt8})
    k = findfirst(==(0x00), v)
    return String(v[1:(k === nothing ? length(v) : k - 1)])
end

@testset "B6: CoinMpsIO" begin
    @testset "ABI layout" begin
        @test sizeof(JClp.CoinHashLinkPod) == 8
        @test sizeof(JClp.CoinMpsCardReaderPod) == 120
        @test JClp.jclp_mps_reader_pod_sizeof() == Cint(120)
    end

    @testset "osi_strtod decimal, exponent, strings, IEEE round-trip" begin
        val_str = zeros(UInt8, 160)
        out_ref = Ref{Ptr{UInt8}}(C_NULL)

        for (s, expected) in [
            ("0", 0.0),
            ("  -42.5  ", -42.5),
            (" + 123.456 ", 123.456),
            ("1.25e3", 1250.0),
            ("3.5E-4", 3.5e-4),
            ("-9.87654321e+20", -9.87654321e20),
            ("1e-350", 0.0),
            ("1e350", floatmax(Float64)),
            (".125", 0.125),
            ("-0.00000123456789", -0.00000123456789),
        ]
            buf = _cstr(s)
            GC.@preserve buf val_str out_ref begin
                p = pointer(buf)
                v = JClp.mps_osi_strtod(p, Base.unsafe_convert(Ptr{Ptr{UInt8}}, out_ref), Int32(0), false, pointer(val_str))
                @test UInt(out_ref[]) > UInt(p)
                @test isapprox(v, expected; rtol=1e-14, atol=0.0) || v == expected
            end
        end

        # Invalid string without stringsAllowed
        bad = _cstr("abc")
        GC.@preserve bad val_str out_ref begin
            p = pointer(bad)
            v = JClp.mps_osi_strtod(p, Base.unsafe_convert(Ptr{Ptr{UInt8}}, out_ref), Int32(0), false, pointer(val_str))
            @test out_ref[] == p
            @test v == -1.0e100
        end

        # String expression with stringsAllowed = true
        expr = _cstr("  =2*x+1")
        GC.@preserve expr val_str out_ref begin
            p = pointer(expr)
            v = JClp.mps_osi_strtod(p, Base.unsafe_convert(Ptr{Ptr{UInt8}}, out_ref), Int32(0), true, pointer(val_str))
            @test UInt(out_ref[]) > UInt(p)
            @test v == JClp.MPS_STRING_VALUE
            @test _read_cstr(val_str) == "=2*x+1"
        end

        # IEEE round-trip: CoinConvertDouble (formatType=2/3) <-> osi_strtod (fmt_type=1/2)
        out24 = zeros(UInt8, 24)
        for x in [0.0, -0.0, 1.0, -3.141592653589793, 1.2345678901234567e-123, -9.87654321e+200]
            for (enc_fmt, dec_fmt) in [(Int32(2), Int32(1)), (Int32(3), Int32(2))]
                GC.@preserve out24 val_str out_ref begin
                    JClp.coin_convert_double!(Int32(0), enc_fmt, x, pointer(out24))
                    y = JClp.mps_osi_strtod(pointer(out24), Base.unsafe_convert(Ptr{Ptr{UInt8}}, out_ref), dec_fmt, false, pointer(val_str))
                    @test reinterpret(UInt64, y) == reinterpret(UInt64, x)
                end
            end
        end
    end

    @testset "CoinConvertDouble format 0 & 1" begin
        out24 = zeros(UInt8, 24)
        GC.@preserve out24 begin
            JClp.coin_convert_double!(Int32(0), Int32(0), 12.5, pointer(out24))
            s0 = _read_cstr(out24)
            @test length(s0) == 12
            @test parse(Float64, strip(s0)) == 12.5

            JClp.coin_convert_double!(Int32(0), Int32(0), 1e-25, pointer(out24))
            @test strip(_read_cstr(out24)) == "0.0"

            JClp.coin_convert_double!(Int32(0), Int32(1), -3.141592653589793, pointer(out24))
            s1 = _read_cstr(out24)
            @test !occursin(' ', s1)
            @test isapprox(parse(Float64, s1), -3.141592653589793; rtol=1e-15)

            JClp.coin_convert_double!(Int32(2), Int32(0), 1e50, pointer(out24))
            @test strip(_read_cstr(out24)) == ""
        end
    end

    @testset "nextBlankOr & strcpyAndCompress" begin
        s = _cstr("-  12.5 rest")
        GC.@preserve s begin
            nb = JClp.mps_next_blank_or(pointer(s))
            @test nb != C_NULL
            @test Int(UInt(nb) - UInt(pointer(s))) == 7
        end
        dst = zeros(UInt8, 32)
        src1 = _cstr(" A B\tC ")
        src2 = _cstr("   \t  ")
        GC.@preserve dst src1 src2 begin
            JClp.mps_strcpy_and_compress!(pointer(dst), pointer(src1))
            @test _read_cstr(dst) == "ABC"
            JClp.mps_strcpy_and_compress!(pointer(dst), pointer(src2))
            @test _read_cstr(dst) == " "
        end
    end

    @testset "Hash table (startHash / findHash)" begin
        n = 500
        name_bufs = [_cstr("ROW_$(i)_NAME") for i in 1:n]
        name_ptrs = [pointer(b) for b in name_bufs]
        hash_tbl = Vector{JClp.CoinHashLinkPod}(undef, 4 * n)
        miss = _cstr("DOES_NOT_EXIST")
        GC.@preserve name_bufs name_ptrs hash_tbl miss begin
            JClp.mps_start_hash!(pointer(name_ptrs), Int32(n), pointer(hash_tbl))
            for i in 1:n
                idx = JClp.mps_find_hash(name_ptrs[i], pointer(name_ptrs), Int32(n), pointer(hash_tbl))
                @test idx == Int32(i - 1)
            end
            @test JClp.mps_find_hash(pointer(miss), pointer(name_ptrs), Int32(n), pointer(hash_tbl)) == Int32(-1)

            # Zero allocation check on findHash
            q = name_ptrs[250]
            allocs = @allocated JClp.mps_find_hash(q, pointer(name_ptrs), Int32(n), pointer(hash_tbl))
            @test allocs == 0
        end
    end

    @testset "convertBoundToSense & convertSenseToBound" begin
        inf = 1.0e30
        for (lo, hi, exp_s, exp_rhs, exp_rng) in [
            (5.0, 5.0, UInt8('E'), 5.0, 0.0),
            (-inf, 7.0, UInt8('L'), 7.0, 0.0),
            (3.0, inf, UInt8('G'), 3.0, 0.0),
            (2.0, 9.0, UInt8('R'), 9.0, 7.0),
            (-inf, inf, UInt8('N'), 0.0, 0.0),
        ]
            s, r, rng = JClp.mps_convert_bound_to_sense(lo, hi, inf)
            @test s == exp_s && r == exp_rhs && rng == exp_rng
            lo2, hi2 = JClp.mps_convert_sense_to_bound(s, r, rng, inf)
            @test lo2 == lo && hi2 == hi
        end
    end

    @testset "CardReader tokenization on afiro.mps + markers + bounds" begin
        mps_text = """
        * Comment line
        NAME          TESTPROB
        ROWS
         N  COST
         E  R1
         L  R2
        COLUMNS
            X1        COST                 1.5   R1                   2.0
            M1        'MARKER'                 'INTORG'
            X2        R1                  -3.0   R2                   4.25
            M2        'MARKER'                 'INTEND'
        RHS
            RHS1      R1                  10.0   R2                  20.0
        BOUNDS
         UP BND1      X1                   5.0
         FX BND1      X2                   2.0
        ENDATA
        """
        lines = split(mps_text, '\n')
        line_idx = Ref(1)
        gets_fn = @cfunction((ctx, buf, max_len) -> begin
            st = unsafe_pointer_to_objref(ctx)::Base.RefValue{Tuple{Vector{SubString{String}},Base.RefValue{Int}}}
            ls, idx = st[]
            if idx[] > length(ls)
                return Ptr{UInt8}(C_NULL)
            end
            ln = ls[idx[]]
            idx[] += 1
            n = min(ncodeunits(ln), Int(max_len) - 2)
            for k in 1:n
                unsafe_store!(buf, codeunit(ln, k), k)
            end
            unsafe_store!(buf, UInt8('\n'), n + 1)
            unsafe_store!(buf, 0x00, n + 2)
            return buf
        end, Ptr{UInt8}, (Ptr{Cvoid}, Ptr{UInt8}, Cint))

        ctx_obj = Ref((lines, line_idx))
        card = zeros(UInt8, 880)
        row_name = zeros(UInt8, 160)
        col_name = zeros(UInt8, 160)
        val_str = zeros(UInt8, 160)

        r_ref = Ref(JClp.CoinMpsCardReaderPod(
            pointer(card), pointer(card), pointer(card),
            pointer(row_name), pointer(col_name), pointer(val_str),
            gets_fn, C_NULL, pointer_from_objref(ctx_obj), C_NULL,
            0.0, JClp.COIN_UNKNOWN_MPS_TYPE, JClp.COIN_EOF_SECTION,
            Int32(0), Int32(0), Int32(0), Int32(1), Int32(0), Int32(880),
        ))

        GC.@preserve ctx_obj card row_name col_name val_str r_ref begin
            rp = Base.unsafe_convert(Ptr{JClp.CoinMpsCardReaderPod}, r_ref)
            @test JClp.mps_read_to_next_section!(rp) == JClp.COIN_NAME_SECTION
            @test _read_cstr(col_name) == "TESTPROB"

            # ROWS header + 3 rows
            @test JClp.mps_next_field!(rp) == JClp.COIN_ROW_SECTION
            rows_seen = Tuple{Int32,String}[]
            while JClp.mps_next_field!(rp) == JClp.COIN_ROW_SECTION
                push!(rows_seen, (r_ref[].mpsType, _read_cstr(col_name)))
            end
            @test rows_seen == [
                (JClp.COIN_N_ROW, "COST"),
                (JClp.COIN_E_ROW, "R1"),
                (JClp.COIN_L_ROW, "R2"),
            ]
            @test r_ref[].section == JClp.COIN_COLUMN_SECTION

            # COLUMNS entries + markers
            cols_seen = Tuple{Int32,String,String,Float64}[]
            while JClp.mps_next_field!(rp) == JClp.COIN_COLUMN_SECTION
                push!(cols_seen, (r_ref[].mpsType, _read_cstr(col_name), _read_cstr(row_name), r_ref[].value))
            end
            @test length(cols_seen) == 6
            @test cols_seen[1] == (JClp.COIN_BLANK_COLUMN, "X1", "COST", 1.5)
            @test cols_seen[2] == (JClp.COIN_BLANK_COLUMN, "X1", "R1", 2.0)
            @test cols_seen[3][1] == JClp.COIN_INTORG
            @test cols_seen[4] == (JClp.COIN_BLANK_COLUMN, "X2", "R1", -3.0)
            @test cols_seen[5] == (JClp.COIN_BLANK_COLUMN, "X2", "R2", 4.25)
            @test cols_seen[6][1] == JClp.COIN_INTEND
            @test r_ref[].section == JClp.COIN_RHS_SECTION

            # RHS
            rhs_seen = Tuple{String,Float64}[]
            while JClp.mps_next_field!(rp) == JClp.COIN_RHS_SECTION
                push!(rhs_seen, (_read_cstr(row_name), r_ref[].value))
            end
            @test rhs_seen == [("R1", 10.0), ("R2", 20.0)]
            @test r_ref[].section == JClp.COIN_BOUNDS_SECTION

            # BOUNDS
            bnds_seen = Tuple{Int32,String,Float64}[]
            while JClp.mps_next_field!(rp) == JClp.COIN_BOUNDS_SECTION
                push!(bnds_seen, (r_ref[].mpsType, _read_cstr(row_name), r_ref[].value))
            end
            @test bnds_seen == [
                (JClp.COIN_UP_BOUND, "X1", 5.0),
                (JClp.COIN_FX_BOUND, "X2", 2.0),
            ]
            @test r_ref[].section == JClp.COIN_ENDATA_SECTION
        end
    end
end

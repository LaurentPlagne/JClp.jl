# Pure 1-based Julia MPS reader (fixed and free format).
# Zero Ptr, zero unsafe_load/unsafe_store!, zero PodView.
# Supports NAME, OBJSENSE, ROWS, COLUMNS, RHS, RANGES, BOUNDS, ENDATA
# matching Clp/CoinMpsIO LP-relaxation conventions across Sample & Netlib suites.

const _MPS_INF = floatmax(Float64)

@inline function _mps_parse_float(s::AbstractString)::Float64
    v = tryparse(Float64, s)
    if v !== nothing
        return v
    end
    # Handle Fortran 'D'/'d' exponents if present
    if occursin('D', s) || occursin('d', s)
        s2 = replace(s, 'D' => 'e', 'd' => 'e')
        v2 = tryparse(Float64, s2)
        return v2 === nothing ? 0.0 : v2
    end
    return 0.0
end

function _mps_tokenize!(tokens::Vector{SubString{String}}, line::AbstractString)::Vector{SubString{String}}
    # Split on ASCII whitespace; strip inline comments starting with '$'
    empty!(tokens)
    s = line isa String ? line : String(line)
    n = ncodeunits(s)
    i = 1
    @inbounds while i <= n
        while i <= n && (codeunit(s, i) == 0x20 || codeunit(s, i) == 0x09 || codeunit(s, i) == 0x0d || codeunit(s, i) == 0x0a)
            i += 1
        end
        i > n && break
        if codeunit(s, i) == UInt8('$')
            break
        end
        j = i
        while j <= n && !(codeunit(s, j) == 0x20 || codeunit(s, j) == 0x09 || codeunit(s, j) == 0x0d || codeunit(s, j) == 0x0a)
            j += 1
        end
        push!(tokens, SubString(s, i, j - 1))
        i = j
    end
    return tokens
end

@inline function _mps_fixed_field(line::String, a::Int, b::Int)::SubString{String}
    n = ncodeunits(line)
    a > n && return SubString(line, 1, 0)
    return strip(SubString(line, a, min(b, n)))
end

@inline function _mps_norm_btype(t::SubString{String})::Symbol
    if ncodeunits(t) == 2
        b1 = codeunit(t, 1)
        b2 = codeunit(t, 2)
        (0x61 <= b1 <= 0x7a) && (b1 -= 0x20)
        (0x61 <= b2 <= 0x7a) && (b2 -= 0x20)
        if b1 == UInt8('F') && b2 == UInt8('R')
            return :FR
        elseif b1 == UInt8('M') && b2 == UInt8('I')
            return :MI
        elseif b1 == UInt8('P') && b2 == UInt8('L')
            return :PL
        elseif b1 == UInt8('B') && b2 == UInt8('V')
            return :BV
        elseif b1 == UInt8('L') && b2 == UInt8('O')
            return :LO
        elseif b1 == UInt8('L') && b2 == UInt8('I')
            return :LI
        elseif b1 == UInt8('U') && b2 == UInt8('P')
            return :UP
        elseif b1 == UInt8('U') && b2 == UInt8('I')
            return :UI
        elseif b1 == UInt8('F') && b2 == UInt8('X')
            return :FX
        end
    end
    return :OTHER
end

@inline function _mps_finalize_rows!(
    rows_finalized::Bool,
    row_types::Vector{UInt8},
    rhs_vals::Vector{Float64},
    range_vals::Vector{Float64},
    has_range::Vector{Bool},
)::Bool
    if !rows_finalized
        nr = length(row_types)
        resize!(rhs_vals, nr); fill!(rhs_vals, 0.0)
        resize!(range_vals, nr); fill!(range_vals, 0.0)
        resize!(has_range, nr); fill!(has_range, false)
    end
    return true
end

@inline function _mps_finalize_cols!(
    cols_finalized::Bool,
    obj_vec::Vector{Float64},
    col_is_int::Vector{Bool},
    col_lo::Vector{Float64},
    col_hi::Vector{Float64},
    col_has_lo::Vector{Bool},
    col_has_hi::Vector{Bool},
)::Bool
    if !cols_finalized
        nc = length(obj_vec)
        resize!(col_lo, nc); fill!(col_lo, 0.0)
        resize!(col_hi, nc)
        @inbounds for j in 1:nc
            col_hi[j] = col_is_int[j] ? 1.0 : _MPS_INF
        end
        resize!(col_has_lo, nc); fill!(col_has_lo, false)
        resize!(col_has_hi, nc); fill!(col_has_hi, false)
    end
    return true
end

function _clp_read_mps_io(io::IO)
    prob_name = ""
    obj_sense = 1.0  # 1.0 = minimize, -1.0 = maximize
    obj_offset = 0.0

    row_map = Dict{String,Int}()
    row_types = UInt8[]
    obj_row_name = ""

    col_map = Dict{String,Int}()
    trip_col = Int32[]
    trip_row = Int32[]
    trip_val = Float64[]
    col_counts = Int32[]
    obj_vec = Float64[]
    col_is_int = Bool[]
    in_intorg = false

    rhs_name = ""
    rhs_vals = Float64[]
    range_name = ""
    range_vals = Float64[]
    has_range = Bool[]
    bound_name = ""
    col_lo = Float64[]
    col_hi = Float64[]
    col_has_lo = Bool[]
    col_has_hi = Bool[]

    section = :NONE
    rows_finalized = false
    cols_finalized = false
    toks = SubString{String}[]

    for raw_line in eachline(io)
        isempty(raw_line) && continue
        c1 = codeunit(raw_line, 1)
        if c1 == UInt8('*')
            continue
        end
        # Section header starts in column 1 (non-whitespace)
        if c1 != 0x20 && c1 != 0x09
            _mps_tokenize!(toks, raw_line)
            isempty(toks) && continue
            hdr = uppercase(String(toks[1]))
            if hdr == "NAME"
                prob_name = length(toks) >= 2 ? String(toks[2]) : ""
                section = :NAME
            elseif hdr == "OBJSENSE"
                section = :OBJSENSE
            elseif hdr == "OBJNAME"
                section = :OBJNAME
            elseif hdr == "ROWS"
                section = :ROWS
            elseif hdr == "COLUMNS"
                rows_finalized = _mps_finalize_rows!(rows_finalized, row_types, rhs_vals, range_vals, has_range)
                section = :COLUMNS
            elseif hdr == "RHS"
                rows_finalized = _mps_finalize_rows!(rows_finalized, row_types, rhs_vals, range_vals, has_range)
                cols_finalized = _mps_finalize_cols!(cols_finalized, obj_vec, col_is_int, col_lo, col_hi, col_has_lo, col_has_hi)
                section = :RHS
            elseif hdr == "RANGES"
                rows_finalized = _mps_finalize_rows!(rows_finalized, row_types, rhs_vals, range_vals, has_range)
                cols_finalized = _mps_finalize_cols!(cols_finalized, obj_vec, col_is_int, col_lo, col_hi, col_has_lo, col_has_hi)
                section = :RANGES
            elseif hdr == "BOUNDS"
                rows_finalized = _mps_finalize_rows!(rows_finalized, row_types, rhs_vals, range_vals, has_range)
                cols_finalized = _mps_finalize_cols!(cols_finalized, obj_vec, col_is_int, col_lo, col_hi, col_has_lo, col_has_hi)
                section = :BOUNDS
            elseif hdr == "ENDATA"
                break
            else
                section = :OTHER
            end
            continue
        end

        _mps_tokenize!(toks, raw_line)
        isempty(toks) && continue

        if section === :OBJSENSE
            s_u = uppercase(String(toks[1]))
            if startswith(s_u, "MAX")
                obj_sense = -1.0
            elseif startswith(s_u, "MIN")
                obj_sense = 1.0
            end
        elseif section === :OBJNAME
            obj_row_name = String(toks[1])
        elseif section === :ROWS
            length(toks) < 2 && continue
            rt = uppercase(toks[1][1])
            rname_sub = length(toks) == 2 ? toks[2] : _mps_fixed_field(raw_line, 5, 12)
            isempty(rname_sub) && (rname_sub = toks[2])
            rname = String(rname_sub)
            if rt == 'N'
                if isempty(obj_row_name)
                    obj_row_name = rname
                end
                # Store objective row as 0 (and other N rows as -1 to ignore)
                row_map[rname] = (rname == obj_row_name) ? 0 : -1
            else
                push!(row_types, UInt8(rt))
                row_map[rname] = length(row_types)
            end
        elseif section === :COLUMNS
            # Track integer markers: 'MARKER' 'INTORG'/'INTEND'
            if length(toks) >= 3 && toks[2] == "'MARKER'"
                if toks[3] == "'INTORG'"
                    in_intorg = true
                elseif toks[3] == "'INTEND'"
                    in_intorg = false
                end
                continue
            end
            length(toks) < 3 && continue
            # Detect fixed-format embedded spaces in column/row names
            use_fixed = !haskey(row_map, toks[2]) && ncodeunits(raw_line) >= 25
            if use_fixed
                cname_sub = _mps_fixed_field(raw_line, 5, 12)
                isempty(cname_sub) && continue
                cidx = get(col_map, cname_sub, 0)
                if cidx == 0
                    cidx = length(obj_vec) + 1
                    col_map[String(cname_sub)] = cidx
                    push!(col_counts, Int32(0))
                    push!(obj_vec, 0.0)
                    push!(col_is_int, in_intorg)
                end
                for (fa, fb, va, vb) in ((15, 22, 25, 36), (40, 47, 50, 61))
                    rn = _mps_fixed_field(raw_line, fa, fb)
                    vs = _mps_fixed_field(raw_line, va, vb)
                    if !isempty(rn) && !isempty(vs)
                        ridx = get(row_map, rn, -1)
                        val = _mps_parse_float(vs)
                        if ridx == 0
                            obj_vec[cidx] += val
                        elseif ridx > 0 && val != 0.0
                            push!(trip_col, Int32(cidx))
                            push!(trip_row, Int32(ridx))
                            push!(trip_val, val)
                            col_counts[cidx] += Int32(1)
                        end
                    end
                end
            else
                cname_sub = toks[1]
                cidx = get(col_map, cname_sub, 0)
                if cidx == 0
                    cidx = length(obj_vec) + 1
                    col_map[String(cname_sub)] = cidx
                    push!(col_counts, Int32(0))
                    push!(obj_vec, 0.0)
                    push!(col_is_int, in_intorg)
                end
                k = 2
                ntoks = length(toks)
                while k + 1 <= ntoks
                    val = _mps_parse_float(toks[k + 1])
                    ridx = get(row_map, toks[k], -1)
                    if ridx == 0
                        obj_vec[cidx] += val
                    elseif ridx > 0 && val != 0.0
                        push!(trip_col, Int32(cidx))
                        push!(trip_row, Int32(ridx))
                        push!(trip_val, val)
                        col_counts[cidx] += Int32(1)
                    end
                    k += 2
                end
            end
        elseif section === :RHS
            # Handle fixed-format embedded spaces in row names if free-format fails
            use_fixed_rhs = false
            if iseven(length(toks)) && haskey(row_map, toks[1])
                rset = SubString(raw_line, 1, 0)
                start_k = 1
            elseif length(toks) >= 2 && haskey(row_map, toks[2])
                rset = toks[1]
                start_k = 2
            elseif ncodeunits(raw_line) >= 25
                use_fixed_rhs = true
                rset = _mps_fixed_field(raw_line, 5, 12)
                start_k = 1
            else
                rset = toks[1]
                start_k = 2
            end
            if isempty(rhs_name) && !isempty(rset)
                rhs_name = String(rset)
            end
            if !isempty(rhs_name) && rset != rhs_name && !isempty(rset)
                continue
            end
            if use_fixed_rhs
                for (fa, fb, va, vb) in ((15, 22, 25, 36), (40, 47, 50, 61))
                    rn = _mps_fixed_field(raw_line, fa, fb)
                    vs = _mps_fixed_field(raw_line, va, vb)
                    if !isempty(rn) && !isempty(vs)
                        ridx = get(row_map, rn, -1)
                        val = _mps_parse_float(vs)
                        if ridx == 0
                            obj_offset = val
                        elseif ridx > 0
                            rhs_vals[ridx] = val
                        end
                    end
                end
            else
                k = start_k
                ntoks = length(toks)
                while k + 1 <= ntoks
                    val = _mps_parse_float(toks[k + 1])
                    ridx = get(row_map, toks[k], -1)
                    if ridx == 0
                        obj_offset = val
                    elseif ridx > 0
                        rhs_vals[ridx] = val
                    end
                    k += 2
                end
            end
        elseif section === :RANGES
            use_fixed_rng = false
            if iseven(length(toks)) && haskey(row_map, toks[1])
                rset = SubString(raw_line, 1, 0)
                start_k = 1
            elseif length(toks) >= 2 && haskey(row_map, toks[2])
                rset = toks[1]
                start_k = 2
            elseif ncodeunits(raw_line) >= 25
                use_fixed_rng = true
                rset = _mps_fixed_field(raw_line, 5, 12)
                start_k = 1
            else
                rset = toks[1]
                start_k = 2
            end
            if isempty(range_name) && !isempty(rset)
                range_name = String(rset)
            end
            if !isempty(range_name) && rset != range_name && !isempty(rset)
                continue
            end
            if use_fixed_rng
                for (fa, fb, va, vb) in ((15, 22, 25, 36), (40, 47, 50, 61))
                    rn = _mps_fixed_field(raw_line, fa, fb)
                    vs = _mps_fixed_field(raw_line, va, vb)
                    if !isempty(rn) && !isempty(vs)
                        ridx = get(row_map, rn, -1)
                        if ridx > 0
                            range_vals[ridx] = _mps_parse_float(vs)
                            has_range[ridx] = true
                        end
                    end
                end
            else
                k = start_k
                ntoks = length(toks)
                while k + 1 <= ntoks
                    val = _mps_parse_float(toks[k + 1])
                    ridx = get(row_map, toks[k], -1)
                    if ridx > 0
                        range_vals[ridx] = val
                        has_range[ridx] = true
                    end
                    k += 2
                end
            end
        elseif section === :BOUNDS
            length(toks) < 2 && continue
            btype = _mps_norm_btype(toks[1])
            # Bounds line: btype [bname] cname [val]
            # For FR, MI, PL, BV: 2 tokens (blank bname) or 3 tokens (named bname)
            # For LO, UP, FX, LI, UI: 3 tokens (blank bname) or 4 tokens (named bname)
            bset = SubString(raw_line, 1, 0)
            cname_sub = SubString(raw_line, 1, 0)
            bval = 0.0
            if btype === :FR || btype === :MI || btype === :PL || btype === :BV
                if length(toks) == 2
                    cname_sub = toks[2]
                else
                    bset = toks[2]
                    cname_sub = toks[3]
                end
            else
                if length(toks) == 3
                    cname_sub = toks[2]
                    bval = _mps_parse_float(toks[3])
                elseif length(toks) >= 4
                    bset = toks[2]
                    cname_sub = toks[3]
                    bval = _mps_parse_float(toks[4])
                else
                    continue
                end
            end
            cidx = get(col_map, cname_sub, 0)
            if cidx == 0 && ncodeunits(raw_line) >= 15
                # Fixed-format fallback for column names with embedded spaces
                bset_f = _mps_fixed_field(raw_line, 5, 12)
                cname_f = _mps_fixed_field(raw_line, 15, 22)
                cidx = get(col_map, cname_f, 0)
                if cidx > 0
                    bset = bset_f
                    v_f = _mps_fixed_field(raw_line, 25, 36)
                    if !isempty(v_f)
                        bval = _mps_parse_float(v_f)
                    end
                end
            end
            if isempty(bound_name) && !isempty(bset)
                bound_name = String(bset)
            end
            if !isempty(bound_name) && bset != bound_name && !isempty(bset)
                continue
            end
            cidx == 0 && continue
            if btype === :LO || btype === :LI
                col_lo[cidx] = bval
                col_has_lo[cidx] = true
                if col_hi[cidx] < bval && !col_has_hi[cidx]
                    col_hi[cidx] = _MPS_INF
                end
            elseif btype === :UP || btype === :UI
                col_hi[cidx] = bval
                col_has_hi[cidx] = true
                if !col_has_lo[cidx] && bval < 0.0
                    col_lo[cidx] = -_MPS_INF
                end
            elseif btype === :FX
                col_lo[cidx] = bval
                col_hi[cidx] = bval
                col_has_lo[cidx] = true
                col_has_hi[cidx] = true
            elseif btype === :FR
                col_lo[cidx] = -_MPS_INF
                col_hi[cidx] = _MPS_INF
                col_has_lo[cidx] = true
                col_has_hi[cidx] = true
            elseif btype === :MI
                col_lo[cidx] = -_MPS_INF
                col_has_lo[cidx] = true
                if !col_has_hi[cidx]
                    col_hi[cidx] = 0.0
                end
            elseif btype === :PL
                col_hi[cidx] = _MPS_INF
                col_has_hi[cidx] = true
            elseif btype === :BV
                col_lo[cidx] = 0.0
                col_hi[cidx] = 1.0
                col_has_lo[cidx] = true
                col_has_hi[cidx] = true
            end
        end
    end

    rows_finalized = _mps_finalize_rows!(rows_finalized, row_types, rhs_vals, range_vals, has_range)
    cols_finalized = _mps_finalize_cols!(cols_finalized, obj_vec, col_is_int, col_lo, col_hi, col_has_lo, col_has_hi)

    nrows = length(row_types)
    ncols = length(obj_vec)

    # Convert row_types + rhs_vals + range_vals to row_lower / row_upper
    row_lower = Vector{Float64}(undef, nrows)
    row_upper = Vector{Float64}(undef, nrows)
    @inbounds for i in 1:nrows
        rt = Char(row_types[i])
        rhs = rhs_vals[i]
        if rt == 'E'
            if has_range[i]
                r = range_vals[i]
                if r >= 0.0
                    row_lower[i] = rhs
                    row_upper[i] = rhs + r
                else
                    row_lower[i] = rhs + r
                    row_upper[i] = rhs
                end
            else
                row_lower[i] = rhs
                row_upper[i] = rhs
            end
        elseif rt == 'L'
            row_upper[i] = rhs
            row_lower[i] = has_range[i] ? (rhs - abs(range_vals[i])) : -_MPS_INF
        elseif rt == 'G'
            row_lower[i] = rhs
            row_upper[i] = has_range[i] ? (rhs + abs(range_vals[i])) : _MPS_INF
        else
            row_lower[i] = -_MPS_INF
            row_upper[i] = _MPS_INF
        end
    end

    # Assemble 1-based CSC arrays via linear counting sort (merging duplicate row indices in a column)
    col_starts = Vector{Int32}(undef, ncols + 1)
    pos_i32 = Int32(1)
    @inbounds for j in 1:ncols
        col_starts[j] = pos_i32
        pos_i32 += col_counts[j]
        col_counts[j] = col_starts[j]
    end
    col_starts[ncols + 1] = pos_i32

    nnz_total = length(trip_col)
    col_indices = Vector{Int32}(undef, nnz_total)
    col_elements = Vector{Float64}(undef, nnz_total)
    @inbounds for t in 1:nnz_total
        c = Int(trip_col[t])
        p = Int(col_counts[c])
        col_counts[c] = Int32(p + 1)
        col_indices[p] = trip_row[t]
        col_elements[p] = trip_val[t]
    end

    out_pos = 1
    row_seen = zeros(Int32, nrows)
    next_raw_start = Int(col_starts[1])
    @inbounds for j in 1:ncols
        raw_start = next_raw_start
        next_raw_start = Int(col_starts[j + 1])
        raw_end = next_raw_start - 1
        col_out_start = out_pos
        col_starts[j] = Int32(col_out_start)
        for k in raw_start:raw_end
            ir = Int(col_indices[k])
            v = col_elements[k]
            prev = Int(row_seen[ir])
            if prev >= col_out_start
                col_elements[prev] += v
            else
                row_seen[ir] = Int32(out_pos)
                col_indices[out_pos] = Int32(ir)
                col_elements[out_pos] = v
                out_pos += 1
            end
        end
        # Compact any exact zeros produced by merging duplicates
        write_pos = col_out_start
        for p in col_out_start:(out_pos - 1)
            row_seen[Int(col_indices[p])] = Int32(0)
            if col_elements[p] != 0.0
                if write_pos != p
                    col_indices[write_pos] = col_indices[p]
                    col_elements[write_pos] = col_elements[p]
                end
                write_pos += 1
            end
        end
        out_pos = write_pos
    end
    col_starts[ncols + 1] = Int32(out_pos)
    resize!(col_indices, out_pos - 1)
    resize!(col_elements, out_pos - 1)

    return (
        name = prob_name,
        nrows = nrows,
        ncols = ncols,
        col_starts = col_starts,
        col_indices = col_indices,
        col_elements = col_elements,
        col_lower = col_lo,
        col_upper = col_hi,
        objective = obj_vec,
        row_lower = row_lower,
        row_upper = row_upper,
        obj_offset = obj_offset,
        obj_sense = obj_sense,
    )
end

"""
    clp_read_mps(path::AbstractString) -> NamedTuple

Read an LP in MPS format (fixed or free) into 1-based CSC representation:
`(name, nrows, ncols, col_starts, col_indices, col_elements,
  col_lower, col_upper, objective, row_lower, row_upper,
  obj_offset, obj_sense)`.
"""
function clp_read_mps(path::AbstractString)
    if endswith(lowercase(path), ".gz")
        return open(`gzip -dc $path`, "r") do io
            _clp_read_mps_io(io)
        end
    else
        return open(path, "r") do io
            _clp_read_mps_io(io)
        end
    end
end

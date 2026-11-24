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
    s2 = replace(s, 'D' => 'e', 'd' => 'e')
    v2 = tryparse(Float64, s2)
    return v2 === nothing ? 0.0 : v2
end

function _mps_tokenize(line::AbstractString)::Vector{SubString{String}}
    # Split on ASCII whitespace; strip inline comments starting with '$'
    s = String(line)
    tokens = SubString{String}[]
    n = ncodeunits(s)
    i = 1
    while i <= n
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

"""
    clp_read_mps(path::AbstractString) -> NamedTuple

Read an LP in MPS format (fixed or free) into 1-based CSC representation:
`(name, nrows, ncols, col_starts, col_indices, col_elements,
  col_lower, col_upper, objective, row_lower, row_upper,
  obj_offset, obj_sense)`.
"""
function clp_read_mps(path::AbstractString)
    prob_name = ""
    obj_sense = 1.0  # 1.0 = minimize, -1.0 = maximize
    obj_offset = 0.0

    row_map = Dict{String,Int}()
    row_types = UInt8[]
    obj_row_name = ""

    col_map = Dict{String,Int}()
    col_rows = Vector{Int32}[]
    col_vals = Vector{Float64}[]
    obj_vec = Float64[]

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

    function _finalize_rows!()
        if !rows_finalized
            nr = length(row_types)
            resize!(rhs_vals, nr); fill!(rhs_vals, 0.0)
            resize!(range_vals, nr); fill!(range_vals, 0.0)
            resize!(has_range, nr); fill!(has_range, false)
            rows_finalized = true
        end
    end

    function _finalize_cols!()
        if !cols_finalized
            nc = length(obj_vec)
            resize!(col_lo, nc); fill!(col_lo, 0.0)
            resize!(col_hi, nc); fill!(col_hi, _MPS_INF)
            resize!(col_has_lo, nc); fill!(col_has_lo, false)
            resize!(col_has_hi, nc); fill!(col_has_hi, false)
            cols_finalized = true
        end
    end

    open(path, "r") do io
        for raw_line in eachline(io)
            isempty(raw_line) && continue
            c1 = codeunit(raw_line, 1)
            if c1 == UInt8('*')
                continue
            end
            # Section header starts in column 1 (non-whitespace)
            if c1 != 0x20 && c1 != 0x09
                toks = _mps_tokenize(raw_line)
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
                    _finalize_rows!()
                    section = :COLUMNS
                elseif hdr == "RHS"
                    _finalize_rows!()
                    _finalize_cols!()
                    section = :RHS
                elseif hdr == "RANGES"
                    _finalize_rows!()
                    _finalize_cols!()
                    section = :RANGES
                elseif hdr == "BOUNDS"
                    _finalize_rows!()
                    _finalize_cols!()
                    section = :BOUNDS
                elseif hdr == "ENDATA"
                    break
                else
                    section = :OTHER
                end
                continue
            end

            toks = _mps_tokenize(raw_line)
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
                rt = uppercase(String(toks[1]))[1]
                rname = String(toks[2])
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
                # Skip integer markers: 'MARKER' 'INTORG'/'INTEND'
                if length(toks) >= 2 && toks[2] == "'MARKER'"
                    continue
                end
                length(toks) < 3 && continue
                cname = String(toks[1])
                cidx = get(col_map, cname, 0)
                if cidx == 0
                    cidx = length(obj_vec) + 1
                    col_map[cname] = cidx
                    push!(col_rows, Int32[])
                    push!(col_vals, Float64[])
                    push!(obj_vec, 0.0)
                end
                k = 2
                while k + 1 <= length(toks)
                    rname = String(toks[k])
                    val = _mps_parse_float(toks[k + 1])
                    ridx = get(row_map, rname, -1)
                    if ridx == 0
                        obj_vec[cidx] += val
                    elseif ridx > 0 && val != 0.0
                        push!(col_rows[cidx], Int32(ridx))
                        push!(col_vals[cidx], val)
                    end
                    k += 2
                end
            elseif section === :RHS
                # Handle both named RHS (odd/even tokens) and blank RHS name
                # In standard MPS, RHS has: [rhs_name] row1 val1 [row2 val2]
                # If toks[1] is a known row name and length(toks) is even, rhs_name is blank.
                if iseven(length(toks)) && haskey(row_map, String(toks[1]))
                    rset = ""
                    start_k = 1
                else
                    rset = String(toks[1])
                    start_k = 2
                end
                if isempty(rhs_name) && !isempty(rset)
                    rhs_name = rset
                end
                if !isempty(rhs_name) && rset != rhs_name && !isempty(rset)
                    continue
                end
                k = start_k
                while k + 1 <= length(toks)
                    rname = String(toks[k])
                    val = _mps_parse_float(toks[k + 1])
                    ridx = get(row_map, rname, -1)
                    if ridx == 0
                        obj_offset = -val
                    elseif ridx > 0
                        rhs_vals[ridx] = val
                    end
                    k += 2
                end
            elseif section === :RANGES
                if iseven(length(toks)) && haskey(row_map, String(toks[1]))
                    rset = ""
                    start_k = 1
                else
                    rset = String(toks[1])
                    start_k = 2
                end
                if isempty(range_name) && !isempty(rset)
                    range_name = rset
                end
                if !isempty(range_name) && rset != range_name && !isempty(rset)
                    continue
                end
                k = start_k
                while k + 1 <= length(toks)
                    rname = String(toks[k])
                    val = _mps_parse_float(toks[k + 1])
                    ridx = get(row_map, rname, -1)
                    if ridx > 0
                        range_vals[ridx] = val
                        has_range[ridx] = true
                    end
                    k += 2
                end
            elseif section === :BOUNDS
                length(toks) < 2 && continue
                btype = uppercase(String(toks[1]))
                # Bounds line: btype [bname] cname [val]
                # For FR, MI, PL, BV: 2 tokens (blank bname) or 3 tokens (named bname)
                # For LO, UP, FX, LI, UI: 3 tokens (blank bname) or 4 tokens (named bname)
                bset = ""
                cname = ""
                bval = 0.0
                if btype == "FR" || btype == "MI" || btype == "PL" || btype == "BV"
                    if length(toks) == 2
                        bset = ""
                        cname = String(toks[2])
                    else
                        bset = String(toks[2])
                        cname = String(toks[3])
                    end
                else
                    if length(toks) == 3
                        bset = ""
                        cname = String(toks[2])
                        bval = _mps_parse_float(toks[3])
                    elseif length(toks) >= 4
                        bset = String(toks[2])
                        cname = String(toks[3])
                        bval = _mps_parse_float(toks[4])
                    else
                        continue
                    end
                end
                if isempty(bound_name) && !isempty(bset)
                    bound_name = bset
                end
                if !isempty(bound_name) && bset != bound_name && !isempty(bset)
                    continue
                end
                cidx = get(col_map, cname, 0)
                cidx == 0 && continue
                if btype == "LO" || btype == "LI"
                    col_lo[cidx] = bval
                    col_has_lo[cidx] = true
                    if col_hi[cidx] < bval && !col_has_hi[cidx]
                        col_hi[cidx] = _MPS_INF
                    end
                elseif btype == "UP" || btype == "UI"
                    col_hi[cidx] = bval
                    col_has_hi[cidx] = true
                    if !col_has_lo[cidx] && bval < 0.0
                        col_lo[cidx] = -_MPS_INF
                    end
                elseif btype == "FX"
                    col_lo[cidx] = bval
                    col_hi[cidx] = bval
                    col_has_lo[cidx] = true
                    col_has_hi[cidx] = true
                elseif btype == "FR"
                    col_lo[cidx] = -_MPS_INF
                    col_hi[cidx] = _MPS_INF
                    col_has_lo[cidx] = true
                    col_has_hi[cidx] = true
                elseif btype == "MI"
                    col_lo[cidx] = -_MPS_INF
                    col_has_lo[cidx] = true
                    if !col_has_hi[cidx]
                        col_hi[cidx] = 0.0
                    end
                elseif btype == "PL"
                    col_hi[cidx] = _MPS_INF
                    col_has_hi[cidx] = true
                elseif btype == "BV"
                    col_lo[cidx] = 0.0
                    col_hi[cidx] = 1.0
                    col_has_lo[cidx] = true
                    col_has_hi[cidx] = true
                end
            end
        end
    end

    _finalize_rows!()
    _finalize_cols!()

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

    # Assemble 1-based CSC arrays (merging any duplicate row indices in a column)
    col_starts = Vector{Int32}(undef, ncols + 1)
    nnz_total = 0
    for j in 1:ncols
        nnz_total += length(col_rows[j])
    end
    col_indices = Vector{Int32}(undef, nnz_total)
    col_elements = Vector{Float64}(undef, nnz_total)
    pos = 1
    row_seen = zeros(Int32, nrows)
    @inbounds for j in 1:ncols
        col_starts[j] = Int32(pos)
        rj = col_rows[j]
        vj = col_vals[j]
        col_start = pos
        for k in 1:length(rj)
            ir = Int(rj[k])
            prev = Int(row_seen[ir])
            if prev >= col_start
                col_elements[prev] += vj[k]
            else
                row_seen[ir] = Int32(pos)
                col_indices[pos] = Int32(ir)
                col_elements[pos] = vj[k]
                pos += 1
            end
        end
        # Compact any exact zeros produced by merging duplicates
        write_pos = col_start
        for p in col_start:(pos - 1)
            row_seen[Int(col_indices[p])] = Int32(0)
            if col_elements[p] != 0.0
                if write_pos != p
                    col_indices[write_pos] = col_indices[p]
                    col_elements[write_pos] = col_elements[p]
                end
                write_pos += 1
            end
        end
        pos = write_pos
    end
    col_starts[ncols + 1] = Int32(pos)
    resize!(col_indices, pos - 1)
    resize!(col_elements, pos - 1)

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

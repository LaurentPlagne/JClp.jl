using Test
using LinearAlgebra
using JClp

@testset "Pure Julia SparseLU (1-based, zero-alloc)" begin
    function build_csc(A::Matrix{Float64})
        m, n = size(A)
        starts = Vector{Int32}(undef, n)
        lens = Vector{Int32}(undef, n)
        rows = Int32[]
        vals = Float64[]
        rc = zeros(Int32, m)
        for j in 1:n
            starts[j] = Int32(length(vals) + 1)
            cnt = 0
            for i in 1:m
                v = A[i, j]
                if v != 0.0
                    push!(rows, Int32(i))
                    push!(vals, v)
                    rc[i] += Int32(1)
                    cnt += 1
                end
            end
            lens[j] = Int32(cnt)
        end
        return starts, lens, rows, vals, rc
    end

    function set_dense_rhs!(v::JClp.IndexedVector, b::Vector{Float64}; packed::Bool=false)
        JClp.clear!(v)
        v.packed = packed
        nnz = 0
        for i in eachindex(b)
            if b[i] != 0.0
                nnz += 1
                v.indices[nnz] = Int32(i)
                v.values[packed ? nnz : i] = b[i]
            end
        end
        v.nnz = nnz
        return v
    end

    function to_dense(v::JClp.IndexedVector, n::Int)
        out = zeros(Float64, n)
        for k in 1:v.nnz
            i = Int(v.indices[k])
            out[i] = v.packed ? v.values[k] : v.values[i]
        end
        return out
    end

    @testset "Small sparse + FTRAN/BTRAN + Forrest-Tomlin updates" begin
        B = Float64[
            -1.0  0.0  2.0  0.0  1.5;
             0.0 -1.0  1.0  3.0  0.0;
             0.0  0.0  4.0 -1.0  2.0;
             0.0  0.0  0.5  2.5 -1.0;
             0.0  0.0  0.0  1.0  3.5
        ]
        n = size(B, 1)
        starts, lens, rows, vals, rc = build_csc(B)
        lu = JClp.SparseLU()
        JClp.allocate_areas!(lu, n, n, 200, 200)
        JClp.load_csc!(lu, n, n, length(vals), starts, lens, rows, vals, rc)
        JClp.preprocess!(lu, 2)
        st = JClp.factor!(lu)
        @test st == 0
        # Record basis column mapping: pivot_var[j] = original column index at position j
        pivot_var = Vector{Int}(undef, n)
        for c in 1:n
            j = Int(lu.permute_back[Int(lu.pivot_col_back[c])])
            pivot_var[j] = c
        end
        B_ordered = B[:, pivot_var]
        JClp.finalize_basis_permutation!(lu)

        work = JClp.IndexedVector(n + 50)
        rhs = JClp.IndexedVector(n + 50)

        # FTRAN unpacked & packed
        a = Float64[1.0, -2.0, 3.0, 0.5, -1.5]
        for packed in (false, true)
            set_dense_rhs!(rhs, a; packed=packed)
            JClp.ftran!(lu, work, rhs)
            x = to_dense(rhs, n)
            @test norm(B_ordered * x - a) < 1e-12
            @test work.nnz == 0
        end

        # BTRAN unpacked & packed
        c = Float64[0.0, 1.0, -0.5, 2.0, 0.0]
        for packed in (false, true)
            set_dense_rhs!(rhs, c; packed=packed)
            JClp.btran!(lu, work, rhs)
            y = to_dense(rhs, n)
            @test norm(B_ordered' * y - c) < 1e-12
            @test work.nnz == 0
        end

        # 5 consecutive Forrest-Tomlin updates
        B_cur = copy(B_ordered)
        for step in 1:5
            new_col = Float64[0.5 * step, -1.0, 1.5, 0.25 * step, 2.0 - 0.1 * step]
            set_dense_rhs!(rhs, new_col; packed=true)
            res_ft = JClp.ftran_ft!(lu, work, rhs)
            @test res_ft > 0
            alpha = to_dense(rhs, n)
            @test norm(B_cur * alpha - new_col) < 1e-11
            piv_row = argmax(abs.(alpha))
            ret = JClp.replace_column!(lu, work, piv_row, alpha[piv_row]; check_before=true)
            @test ret == 0 || ret == 1
            B_cur[:, piv_row] = new_col

            set_dense_rhs!(rhs, a; packed=false)
            JClp.ftran!(lu, work, rhs)
            @test norm(B_cur * to_dense(rhs, n) - a) < 1e-10

            set_dense_rhs!(rhs, c; packed=true)
            JClp.btran!(lu, work, rhs)
            @test norm(B_cur' * to_dense(rhs, n) - c) < 1e-10
        end
    end

    @testset "Hypersparse (n=360) + go_sparse! + zero-allocation hot paths" begin
        n = 360
        # Tridiagonal + sparse off-diagonals + slacks -> exercises hypersparse DFS, bitmask, row-R
        B = zeros(Float64, n, n)
        for i in 1:80
            B[i, i] = -1.0
        end
        for i in 81:n
            B[i, i] = 4.0 + 0.01 * (i % 17)
            if i > 81
                B[i - 1, i] = -0.5
            end
            if i < n
                B[i + 1, i] = 0.3
            end
            if i % 15 == 0
                B[1 + (i % 70), i] = 1.2
            end
        end
        starts, lens, rows, vals, rc = build_csc(B)
        lu = JClp.SparseLU()
        JClp.allocate_areas!(lu, n, n, 40000, 40000)
        JClp.load_csc!(lu, n, n, length(vals), starts, lens, rows, vals, rc)
        JClp.preprocess!(lu, 3)
        @test JClp.factor!(lu) == 0

        pivot_var = Vector{Int}(undef, n)
        for c in 1:n
            j = Int(lu.permute_back[Int(lu.pivot_col_back[c])])
            pivot_var[j] = c
        end
        B_ord = B[:, pivot_var]
        JClp.finalize_basis_permutation!(lu)
        JClp.go_sparse!(lu)
        @test lu.has_sparse
        @test lu.has_row_l
        @test lu.has_row_r

        # Check weights
        w = zeros(Int32, n)
        JClp.get_weights!(lu, w)
        @test all(w .>= Int32(1))

        work = JClp.IndexedVector(n + 250)
        rhs = JClp.IndexedVector(n + 250)
        e_vec = zeros(Float64, n)
        e_vec[120] = 1.0
        e_vec[250] = -2.0

        # Warmup
        set_dense_rhs!(rhs, e_vec; packed=true)
        JClp.btran!(lu, work, rhs)
        @test norm(B_ord' * to_dense(rhs, n) - e_vec) < 1e-11

        set_dense_rhs!(rhs, e_vec; packed=true)
        JClp.ftran!(lu, work, rhs)
        @test norm(B_ord * to_dense(rhs, n) - e_vec) < 1e-11

        # Verify ZERO allocations on hot BTRAN, FTRAN, FTRAN_FT, REPLACE
        set_dense_rhs!(rhs, e_vec; packed=true)
        alloc_btran = @allocated JClp.btran!(lu, work, rhs)
        @test alloc_btran == 0

        set_dense_rhs!(rhs, e_vec; packed=true)
        alloc_ftran = @allocated JClp.ftran!(lu, work, rhs)
        @test alloc_ftran == 0

        new_col = zeros(Float64, n)
        new_col[100] = 2.0
        new_col[101] = -0.5
        new_col[200] = 1.5
        set_dense_rhs!(rhs, new_col; packed=true)
        # Warmup FT + replace once
        JClp.ftran_ft!(lu, work, rhs)
        alpha = to_dense(rhs, n)
        piv_r = argmax(abs.(alpha))
        @test JClp.replace_column!(lu, work, piv_r, alpha[piv_r]) == 0
        B_ord[:, piv_r] = new_col

        new_col2 = zeros(Float64, n)
        new_col2[150] = 3.0
        new_col2[151] = 0.7
        set_dense_rhs!(rhs, new_col2; packed=true)
        alloc_ft = @allocated JClp.ftran_ft!(lu, work, rhs)
        @test alloc_ft == 0
        alpha2 = to_dense(rhs, n)
        piv_r2 = argmax(abs.(alpha2))
        alloc_rep = @allocated JClp.replace_column!(lu, work, piv_r2, alpha2[piv_r2])
        @test alloc_rep == 0
        B_ord[:, piv_r2] = new_col2

        # Post-update accuracy check
        set_dense_rhs!(rhs, e_vec; packed=false)
        JClp.ftran!(lu, work, rhs)
        @test norm(B_ord * to_dense(rhs, n) - e_vec) < 1e-10
        set_dense_rhs!(rhs, e_vec; packed=true)
        JClp.btran!(lu, work, rhs)
        @test norm(B_ord' * to_dense(rhs, n) - e_vec) < 1e-10
    end

    @testset "Dense LAPACK trailing block & Singular matrix detection" begin
        # Dense 24x24 (multiple of 8, > dense_threshold=8)
        n = 24
        B = [i == j ? 10.0 : 0.1 * ((i * 7 + j * 13) % 11 - 5) for i in 1:n, j in 1:n]
        starts, lens, rows, vals, rc = build_csc(B)
        lu = JClp.SparseLU()
        lu.dense_threshold = 8
        JClp.allocate_areas!(lu, n, n, 4000, 4000)
        JClp.load_csc!(lu, n, n, length(vals), starts, lens, rows, vals, rc)
        JClp.preprocess!(lu, 2)
        @test JClp.factor!(lu) == 0
        @test lu.ndense > 0

        pivot_var = Vector{Int}(undef, n)
        for c in 1:n
            j = Int(lu.permute_back[Int(lu.pivot_col_back[c])])
            pivot_var[j] = c
        end
        B_ord = B[:, pivot_var]
        JClp.finalize_basis_permutation!(lu)

        work = JClp.IndexedVector(n + 50)
        rhs = JClp.IndexedVector(n + 50)
        a = Float64.(1:n)
        set_dense_rhs!(rhs, a; packed=false)
        JClp.ftran!(lu, work, rhs)
        @test norm(B_ord * to_dense(rhs, n) - a) < 1e-10
        set_dense_rhs!(rhs, a; packed=true)
        JClp.btran!(lu, work, rhs)
        @test norm(B_ord' * to_dense(rhs, n) - a) < 1e-10

        # Singular 3x3
        S = Float64[1.0 2.0 3.0; 2.0 4.0 6.0; 0.0 1.0 1.0]
        starts, lens, rows, vals, rc = build_csc(S)
        lu_s = JClp.SparseLU()
        JClp.allocate_areas!(lu_s, 3, 3, 100, 100)
        JClp.load_csc!(lu_s, 3, 3, length(vals), starts, lens, rows, vals, rc)
        JClp.preprocess!(lu_s, 2)
        @test JClp.factor!(lu_s) == -1
    end
end

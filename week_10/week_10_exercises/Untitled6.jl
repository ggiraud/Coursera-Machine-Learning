# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Julia 1.1.0
#     language: julia
#     name: julia-1.1
# ---

using Distributed, OnlineStats, LibPQ, DataStreams, DataFrames, SharedArrays, Suppressor, Distributions, SparseArrays, IJulia, Plots, Logging, HDF5

# ### Config Logger

logger = ConsoleLogger(stdout, Logging.Debug)
global_logger(logger)

# ### Create workers

addprocs(4)
@everywhere using LinearAlgebra, LibPQ, DataFrames, SparseArrays, SharedArrays, OnlineStats, DataStreams, TimerOutputs

# ### Cost function

# +
@inline function predict(i::Int64, u::Int64, μ::Float64, P::SharedMatrix{Float64,}, Q::SharedMatrix{Float64})::Float64
    @fastmath @inbounds @views μ + P[i,2] + Q[u,2] + dot(P[i, 3:end],Q[u, 3:end])
end

function db_get_rows(conn::LibPQ.Connection, table::Symbol, limit::Int64, offset::Int64)::LibPQ.Result
    LibPQ.execute(conn, """
        SELECT
            "movieId",
            "userId",
            "rating"
        FROM $table
        LIMIT $limit
        OFFSET $offset;
    """);
end

function cost(conn::LibPQ.Connection, table::Symbol,
        item_map::SparseVector{Int64,Int64}, user_map::SparseVector{Int64,Int64},
        P::SharedMatrix{Float64,}, Q::SharedMatrix{Float64},
        μ::Float64)::Float64

    limit::Int64 = 10000
    offset::Int64 = 0
    total::Float64 = 0.0
    nrows::Int64 = 0

    while true
        res = db_get_rows(conn, table, limit, offset)
        nrows = num_rows(res)
        total += mapreduce((row)->begin
                @fastmath @inbounds abs2(predict(item_map[row.movieId], user_map[row.userId], μ, P, Q) - row.rating)
            end, +, Data.rows(res))
        
        if nrows < limit
            break
        end

        offset += limit
    end

    total
end
# -

# ### Lstq function

# +
@everywhere function db_select_rows(conn::LibPQ.Connection, table::Symbol,
        src_column::Symbol, dst_column::Symbol,
        bounds::UnitRange{Int64},
        limit::Int64, offset::Int64)::LibPQ.Result
    LibPQ.execute(conn, """
        SELECT
            "$src_column",
            "$dst_column",
            "rating"
        FROM $table
        WHERE "$dst_column" BETWEEN $(minimum(bounds)) AND $(maximum(bounds))
        ORDER BY ("$dst_column", "$src_column")
        LIMIT $limit
        OFFSET $offset;
    """);
end

@everywhere function lstq(table::Symbol, bounds::UnitRange{Int64},
        src_column::Symbol, dst_column::Symbol,
        src_map::SparseVector{Int64,Int64}, dst_map::SparseVector{Int64,Int64},
        S::SharedMatrix{Float64,}, D::SharedMatrix{Float64},
        μ::Float64, reg::Float64)::Nothing

    global conn
    
    limit::Int64 = 10000
    offset::Int64 = 0
    current_id::Int64 = minimum(bounds)
    src_index::Int64 = 0
    m::Int64, n::Int64 = size(S)
    X = zeros(n-1, n-1)
    y = zeros(n-1)
    Z = zeros(n-1, n-1)
    eye::Matrix{Float64} = Diagonal(ones(n-1))
    nrows::Int64 = 0
    
    
    while true
        res = db_select_rows(conn, table, src_column, dst_column, bounds, limit, offset)
        nrows = num_rows(res)
        
        @fastmath @inbounds for row = Data.rows(res)
            if row[dst_column] != current_id
                BLAS.axpy!(reg,eye,X) # X += reg*I
                LAPACK.sysv!('U', X, y) # y = X \ y
                @views D[dst_map[current_id], 2:end] .= y
                current_id = row[dst_column]
                BLAS.blascopy!(length(X),Z,1,X,1) # fill!(X, zero(Float64))
                BLAS.blascopy!(length(y),Z[:,1],1,y,1) # fill!(y, zero(Float64))
            end
            src_index = src_map[row[src_column]]
            a = S[src_index, [1; 3:end]]
            b = row[:rating] .- μ .- S[src_index, 2]

            BLAS.syr!('U', 1.0, a, X) # X += a * a'
            BLAS.axpy!(b,a,y) # y += a * b
        end
        
        @fastmath @inbounds if nrows < limit
            BLAS.axpy!(reg,eye,X)
            LAPACK.sysv!('U', X, y)
            @views D[dst_map[current_id], 2:end] .= y
            break
        end
        
        offset += limit
    end
end
# -

# ### ALS function

# +
@everywhere function db_set_work_memory(conn::LibPQ.Connection)::Nothing
    LibPQ.execute(conn, """ 
        SET work_mem TO '1 GB';
    """);
    nothing
end

function db_mean(conn::LibPQ.Connection, table::Symbol, column::Symbol)::Float64
    collect(skipmissing(fetch!(NamedTuple, LibPQ.execute(conn, """
        SELECT
            AVG("$column") as mean
        FROM $table;
    """))[:mean]))[1];
end

function db_unique_ids(conn::LibPQ.Connection, table::Symbol, column::Symbol)::Vector{Int64}
    collect(skipmissing(fetch!(NamedTuple, LibPQ.execute(conn, """
        SELECT DISTINCT
            "$column"
        FROM $table
        ORDER BY "$column";
    """))[column]));
end

function db_split_ids(conn::LibPQ.Connection, table::Symbol, column::Symbol, chunks::Int)::Vector{UnitRange{Int64}}
    collect(UnitRange{Int64}(row.min, row.max) for row in skipmissing(Data.rows(LibPQ.execute(conn, """
        WITH a AS (
            SELECT DISTINCT
                "$column",
                trunc(
                    cume_dist() OVER (
                        ORDER BY "$column"
                    ) * (\$1 -  0.00001)
                )::int4 + 1 AS chunk
            FROM $table
            ORDER BY "$column"
        )

        SELECT
            min("$column") as min,
            max("$column") as max--,
            --array_agg("$column")::int8[] AS ids,
            --chunk 
        FROM a
        GROUP BY chunk
        ORDER BY chunk;
    """, [chunks]))));
end

function fit_factors(conn::LibPQ.Connection, table::Symbol, nepochs::Int64, μ::Float64, reg::Float64,
    item_chunks::Vector{UnitRange{Int64}}, user_chunks::Vector{UnitRange{Int64}},
    item_column::Symbol, user_column::Symbol,
    item_map::SparseVector{Int64,Int64}, user_map::SparseVector{Int64,Int64},
    P::SharedMatrix{Float64}, Q::SharedMatrix{Float64},
    cb::Union{Nothing, Function}=nothing,)::Nothing

    for epoch::Int64 in 1:nepochs
        @sync for (i,w) in enumerate(workers())
            @async remotecall_wait(lstq, w, [table, user_chunks[i], item_column, user_column, item_map, user_map, P, Q, μ, reg]...)
        end

        @sync for (i,w) in enumerate(workers())
            @async remotecall_wait(lstq, w, [table, item_chunks[i], user_column, item_column, user_map, item_map, Q, P, μ, reg]...)
        end

        if cb !== nothing
            cb(epoch, cost(conn, table, item_map, user_map, P, Q, μ))
        end
    end
    nothing
end

function als(dbstr::String, table::Symbol, k::Int64=10;
        nepochs::Int64=10,
        reg::Float64=0.0,
        cb::Union{Nothing, Function}=nothing)
    
    type_map::Dict{Symbol, Type} = Dict(:int4=>Int64, :float4=>Float64)
    
    conn = LibPQ.Connection(dbstr; type_map=type_map)
    db_set_work_memory(conn)
    
    μ::Float64 = db_mean(conn, table, :rating)
    user_ids::Vector{Int64} = db_unique_ids(conn, table, :userId)
    item_ids::Vector{Int64} = db_unique_ids(conn, table, :movieId)
    user_chunks::Vector{UnitRange{Int64}} = db_split_ids(conn, table, :userId, nworkers())
    item_chunks::Vector{UnitRange{Int64}} = db_split_ids(conn, table, :movieId, nworkers())

    m::Int64, n::Int64 = length(item_ids), length(user_ids)
    
    item_map::SparseVector{Int64,Int64} = sparsevec(item_ids, 1:m);
    user_map::SparseVector{Int64,Int64} = sparsevec(user_ids, 1:n);
    
    temp_dir_name = tempdir()
    P_path = joinpath(temp_dir_name, "P")
    @info "Created temp file $P_path"
    Q_path = joinpath(temp_dir_name, "Q")
    @info "Created temp file $Q_path"
    
    P::SharedMatrix{Float64} = SharedMatrix{Float64}(P_path, (m,k+2), mode="w+");
    Q::SharedMatrix{Float64} = SharedMatrix{Float64}(Q_path, (n,k+2), mode="w+");
    
    P[:,:] .= [ones(m) zeros(m) rand(Normal(0.0, 1.0), m, k)];
    Q[:,:] .= [ones(n) zeros(n) rand(Normal(0.0, 1.0), n, k)];
    
    @everywhere workers() begin
        conn = LibPQ.Connection($dbstr; type_map=$type_map)
        db_set_work_memory(conn)
    end
    
    fit_factors(conn, table, nepochs, μ, reg, item_chunks, user_chunks, :movieId, :userId, item_map, user_map, P, Q, cb)
        
    @everywhere workers() close(conn)
    close(conn)
    
    # overwrite existing contents ("cw" otherwise)
    h5open("factors.h5", "w") do fid
        write(fid, "P", P)
        write(fid, "Q", Q)
    end
    @info "Serialized P and Q factors to factors.h5"
    
    rm(P_path)
    @info "Removed temp file $P_path"
    rm(Q_path)
    @info "Removed temp file $Q_path"
    nothing
end
# -

# ### Compute Latent Factors

# +
costs = []
    
@time als("host=localhost dbname=postgres", :ml_small, 100;
    nepochs=10,
    reg=0.0001,
    cb=(epoch, cost)->begin
        #IJulia.clear_output(true)
        @info "epoch: $(epoch), cost: $(cost)"
        push!(costs, cost)
        end)

plot(costs)

# +
costs = []
    
@time als("host=localhost dbname=postgres", :ml_small, 100;
    nepochs=10,
    reg=0.0001)
# -

# ### Destroy workers

@suppress_err rmprocs(workers())

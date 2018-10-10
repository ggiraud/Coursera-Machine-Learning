using Distributed, Profile

addprocs(4)

using CSV, OnlineStats, Random, Distributions
@everywhere using LinearAlgebra, SparseArrays, SharedArrays

filepath = "/Users/guillaume/Downloads/ml-latest-small/ratings.csv"
f = CSV.File(filepath, use_mmap=true)

mutable struct Rating
    itemId::Int64
    userId::Int64
    value::Float64
end

mutable struct Item
    weights::Vector{Float64}
    bias::Float64
end

mutable struct User
    weights::Vector{Float64}
    bias::Float64
end

@everywhere function lsq(F::DenseArray{Float64,2}, rows::Vector{Int64}, ratings::Vector{Float64}, μ::Float64, reg::Float64)::Vector{Float64}
    @views A::Matrix{Float64} = [ones(length(rows)) F[rows, 2:end]]
    @views b::Vector{Float64} = ratings .- μ .- F[rows, 1]
    Symmetric(A'A + reg*I) \ (A'b)
end

@inline function predict(i::Int64, u::Int64, μ::Float64, P::DenseArray{Float64,2}, Q::DenseArray{Float64,2})::Float64
    @views μ + P[i,1] + Q[u,1] + dot(P[i,2:end],Q[u,2:end])
end

function cost(R::AbstractArray{Float64,2}, P::DenseArray{Float64,2}, Q::DenseArray{Float64,2}, μ::Float64=mean(nonzeros(R)))::Float64
    map(zip(findnz(R)...)) do (i,u,r)
        abs2(predict(i,u,μ,P,Q) - r)
    end |> sum
end

function als_processes(ratings, k::Int64=10;
        nepochs::Int64=10,
        reg::Float64=0.0,
        cb::Union{Nothing, Function}=nothing)
    
    o = Mean()
    R::SparseMatrixCSC{Float64,Int64} = let items::Vector{Int64} = Int64[],
                                            users::Vector{Int64} = Int64[],
                                            values::Vector{Float64} = Float64[]
        for r::Rating in ratings
            push!(items, r.itemId)
            push!(users, r.userId)
            push!(values, r.value)
            fit!(o, r.value)
        end
        sparse(items, users, values)
    end
    μ::Float64 = value(o)
    
    P::SharedMatrix{Float64} = SharedMatrix{Float64}([zeros(R.m) rand(Normal(0.0, 1e-4), R.m, k)])
    Q::SharedMatrix{Float64} = SharedMatrix{Float64}([zeros(R.n) rand(Normal(0.0, 1e-4), R.n, k)])
    
    rated_items::Vector{Int64} = unique(sort(findnz(R)[1]))
    rating_users::Vector{Int64} = unique(sort(findnz(R)[2]))
    
    
    for epoch::Int64 in 1:nepochs
        @sync @distributed for u::Int64 in rating_users
            items_rated_by_user::Vector{Int64}, ratings_given_by_user::Vector{Float64} = findnz(R[:,u])
            q::Vector{Float64} = lsq(P, items_rated_by_user, ratings_given_by_user, μ, reg)
            @views Q[u,:] .= q[:]
        end
        
        @sync @distributed for i::Int64 in rated_items
            users_who_rated_item::Vector{Int64}, ratings_given_to_item::Vector{Float64} = findnz(R[i,:])
            p::Vector{Float64} = lsq(Q, users_who_rated_item, ratings_given_to_item, μ, reg)
            @views P[i,:] .= p[:]
        end

        if cb !== nothing
            cb(epoch, cost(R,P,Q,μ))
        end
    end
    
    return P, Q
end

costs = []
ratings = (Rating(r.movieId, r.userId, r.rating) for r in f)

als_processes(ratings, 100;
    nepochs=10,
    reg=0.001)
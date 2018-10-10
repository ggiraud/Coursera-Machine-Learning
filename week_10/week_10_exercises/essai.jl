using Distributed, ClusterManagers
using CSV, LinearAlgebra, OnlineStats, Random, Distributions
using SparseArrays, SharedArrays, DistributedArrays, HDF5

f = CSV.File("ratings.csv", use_mmap=true);


##########
# CONFIG #
##########

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

function create_h5_file(filename)
	fid = h5open(filename, "cw")
	close(fid)
end

function initialize_h5_dataset(filename::String, dataset::String,
			       dims::Tuple{Int,Int})
	m, n = dims
	fid = h5open(filename, "r+")
	d = d_create(fid, dataset, datatype(Float64), dataspace(m, n)) 
	for i in 1:m
		d[i,:] = rand(Normal(0.0, 1e-4), n)
	end
	close(fid)
end

function open_h5_dataset(fun::Function, filename::String, dataset::String)
	fid = h5open(filename, "r+")
	d = fid["$(dataset)"]
	if ismmappable(d)
		d = readmmap(d)
		fun(d)
	end
	close(fid)
end

function alsbiased2(ratings, k::Int64=10;
		    nepochs::Int64=10,
		    reg::Float64=0.0,
		    cb::Union{Nothing, Function}=nothing)

	R::SparseMatrixCSC{Float64,Int64}, μ::Float64 = let 
		items::Vector{Int64} = Int64[]
		users::Vector{Int64} = Int64[]
		values::Vector{Float64} = Float64[]

		o = Mean()
		for r in ratings
			push!(items, r.itemId)
			push!(users, r.userId)
			push!(values, r.value)
			fit!(o, r.value)
		end

		sparse(items, users, values), value(o)
	end

	P::Matrix{Float64} = open("/tmp/P.bin", "w+") do f
		write(f, rand(Normal(0.0, 1e-4), R.m, k))
		Mmap.mmap(f, Matrix{Float64}, (R.m, k))
	end

	Q::Matrix{Float64} = open("/tmp/Q.bin", "w+") do f
		write(f, rand(Normal(0.0, 1e-4), R.n, k))
		Mmap.mmap(f, Matrix{Float64}, (R.n, k))
	end

	items_bias::SharedVector{Float64} = SharedVector{Float64}(zeros(Float64, R.m))
	users_bias::SharedVector{Float64} = SharedVector{Float64}(zeros(Float64, R.n))

	rated_items::Vector{Int64} = unique(sort(findnz(R)[1]))
	rating_users::Vector{Int64} = unique(sort(findnz(R)[2]))

	for epoch::Int64 in 1:nepochs
		cost::Float64=0.0

		@sync @distributed for u in rating_users
			items_rated_by_user = R[:,u].nzind
			ratings_given_by_user = R[:,u].nzval
			A = [ones(length(items_rated_by_user)) P[items_rated_by_user,:]]
			b = ratings_given_by_user .- μ .- items_bias[items_rated_by_user]
			x = Symmetric(A'A + reg*I) \ (A'b)
			users_bias[u] = x[1]
			Q[u,:] .= x[2:end]
		end

		@sync @distributed for i in rated_items
			users_who_rated_item = R[i,:].nzind
			ratings_given_to_item = R[i,:].nzval
			A = [ones(length(users_who_rated_item)) Q[users_who_rated_item,:]]
			b = ratings_given_to_item .- μ .- users_bias[users_who_rated_item]
			x = Symmetric(A'A + reg*I) \ (A'b)
			items_bias[i] = x[1]
			P[i,:] .= x[2:end]
		end

		i, u, v = findnz(R)
		for n in 1:nnz(R)
			cost += abs2(dot(P[i[n],:], Q[u[n],:]) + μ + items_bias[i[n]] + users_bias[u[n]] - v[n])
		end

		if cb !== nothing
			cb(epoch, cost)
		end
	end

	return P, Q
end



###########
# COMPUTE #
###########

addprocs(4)

println(workers())

@everywhere using SparseArrays, SharedArrays, LinearAlgebra

costs = []
ratings = (Rating(r.movieId, r.userId, r.rating) for r in f);

@time P, Q = alsbiased2(ratings, 100;
			nepochs=10,
			reg=0.001,
			cb=(epoch, cost)->begin
				println("epoch: $(epoch), cost: $(cost)")
				push!(costs, cost)
			end)

rmprocs(workers());


module RecommenderSystems

using Distributions: Normal, mean, MvNormal
using LinearAlgebra: dot
using Random: shuffle, GLOBAL_RNG
using SharedArrays: SharedArray
using Distributed#: @distributed, RemoteChannel
import Base.show

export Rating, MatrixFactorization, Item, User, SVDModel, Solver, SGD, ALS, itembiases, userbiases, fit!, predict, score, learningcurves
export SplitMethod, TrainTestSplit, LeaveOneOut, splitcv


"""
    Rating

User's rating of an item.
"""
mutable struct Rating
    item::Int64
    user::Int64
    value::Float64
end

function Base.show(io::IO, rating::Rating)
    fields = fieldnames(Rating)
    kwstr = join(["$(f)=$(getfield(rating, f))" for f in fields], ", ")
    print(io, "Rating($(kwstr))")
end


"""
    Item

An Item's weights and bias.
"""
mutable struct Item
    weights::Vector{Float64}
    bias::Float64
end


"""
    User

A User's weights and bias.
"""
mutable struct User
    weights::Vector{Float64}
    bias::Float64
end


"""
    Solver

Abstract supertype for minimizing algorithms.
"""
abstract type Solver end


"""
    SGD <: Solver

Stochastic Gradient Descent algorithm.
"""
mutable struct SGD <: Solver
    nepochs::Int64
    lr::Float64
    reg::Float64
end

"""
    SGD(; <keyword arguments>)

# Arguments
- `nepochs:::Int64=20`: number of epochs processed during stochastic gradient descent optimization.
- `lr::Float64=0.005`: learning rate used during gradient descent optimization.
- `reg::Float64=0.02`: regularization parameter used during gradient descent optimization.
"""
function SGD(;
        nepochs::Int64=10,
        lr::Float64=0.001,
        reg::Float64=0.0)
    SGD(nepochs, lr, reg)
end


"""
    ALS <: Solver

Alternating Least Squares algorithm.
"""
mutable struct ALS <: Solver
    nepochs::Int64
    reg::Float64
end

"""
    ALS(; <keyword arguments>)

# Arguments
- `nepochs:::Int64=20`: number of epochs processed during alternating least squares optimization.
- `reg::Float64=0.02`: regularization parameter used during alternating least squares optimization.
"""
function ALS(;
    nepochs::Int64=10,
    reg::Float64=0.0)
    ALS(nepochs, reg)
end

"""
    MatrixFactorization

Abstract supertype for matrix factorization based recommender systems models.
"""
abstract type MatrixFactorization end


"""
    SVDModel <: MatrixFactorization

SVD based recommender systems model.
"""
mutable struct SVDModel <: MatrixFactorization
    # Factorization
    items::Dict{Int64, Item}
    users::Dict{Int64, User}
    k::Int64
    
    # Solver
    solver::Solver
    
    # Initialization
    mean::Float64
    std::Float64
    
    # Baseline
    bias::Float64

end


"""
    SVDModel(; <keyword arguments>)

# Arguments
- `k::Int64=10`: the number of factors.
- `mean::Float64=0.0`: mean of the normal distribution used to initialize the factorization matrices.
- `std::Float64=1e-4`: standard deviation of the normal distribution used to initialize the factorization matrices.
- `nepochs:::Int64=20`: number of epochs processed during stochastic gradient descent optimization.
- `lr::Float64=0.005`: learning rate used during gradient descent optimization.
- `reg::Float64=0.02`: regularization parameter used during gradient descent optimization.
"""
function SVDModel(;
        k::Int64=10,
        solver::Solver=SGD(),
        mean::Float64=0.0,
        std::Float64=1e-4)
    
    items = Dict{Int64, Item}()
    users = Dict{Int64, User}()
    bias = 0.0
    
    SVDModel(items,users,k,solver,mean,std,bias)
end

function Base.show(io::IO, model::SVDModel)
    fields = [:k, :mean, :std]
    kwstr = join(["$(f)=$(getfield(model, f))" for f in fields], ", ")
    print(io, "SVD($(kwstr),...)")
end

function User(model::SVDModel)
    User(rand(MvNormal(model.k, model.std)), 0.0)
    #User(rand(Normal(model.mean, model.std), model.k), 0.0)
end

function Item(model::SVDModel)
    Item(rand(MvNormal(model.k, model.std)), 0.0)
    #Item(rand(Normal(model.mean, model.std), model.k), 0.0)
end

function userbiases(model::SVDModel)
    (get(model.users,i,User(model)).bias for i in 1:maximum(keys(model.users)))
end

function itembiases(model::SVDModel)
    (get(model.items,i,Item(model)).bias for i in 1:maximum(keys(model.items)))
end


function fit!(model::SVDModel, ratings::Vector{Rating}; cb::Union{Function, Nothing}=nothing)
    fit!(model, ratings, model.solver; cb=cb)
end

"""
    fit!(model::SVDModel, ratings::Vector{Ratings}; cb::Union{Function, Nothing}=nothing)

Fit the model by optimizing a regularized SSE(Summed Squared Error) through stochastic gradient descent.

If passed as the keyword argument `cb`, a callback function will be called at the end of each epoch
with arguments `nepoch::Int64` and `cost::Float64`.
"""
function fit!(model::SVDModel, ratings::Vector{Rating}, solver::SGD; cb::Union{Function, Nothing}=nothing)
    model.bias = mean(r.value for r in ratings)
    for r in ratings
        if !(r.item in keys(model.items))
            model.items[r.item] = Item(model)
        end
        if !(r.user in keys(model.users))
            model.users[r.user] = User(model)
        end
    end
    
    for epoch in 1:solver.nepochs
        currentcost = 0
        for r in shuffle(ratings)
            item = model.items[r.item]
            user = model.users[r.user]

            e = model.bias + item.bias + user.bias + dot(item.weights, user.weights) - r.value

            currentcost += abs2(e)
            
            item.weights .-= solver.lr .* 2 .* (e .* user.weights .+ solver.reg .* item.weights)
            user.weights .-= solver.lr .* 2 .* (e .* item.weights .+ solver.reg .* user.weights)
            item.bias -= solver.lr * 2 * (e + solver.reg * item.bias)
            user.bias -= solver.lr * 2 * (e + solver.reg * user.bias)
        end
        
        if cb !== nothing
            cb(epoch, currentcost)
        end
    end
end


"""
    predict(model::SVDModel, item::Int64, user::Int64)

Predict the rating of an item by a user.
"""
function predict(model::SVDModel, item::Int64, user::Int64)
    item = get(model.items, item, Item(zeros(Float64, model.k), 0.0))
    user = get(model.users, user, User(zeros(Float64, model.k), 0.0))

    model.bias + item.bias + user.bias + dot(item.weights, user.weights)
end


"""
    score(model::SVDModel, ratings::Vector{Rating})

Compute the score of the fitted model using SSE(Summed Squared Error).
"""
function score(model::SVDModel, ratings::Vector{Rating})
    sum(abs2, (predict(model, r.item, r.user) - r.value) for r in ratings)
end


"""
    learningcurves(model::SVDModel, train::Vector{Rating}, test::Vector{Rating}, step::Int64=1; cb::Union{Function, Nothing}=nothing)

Successively compute the train and test scores necessary to plot the learning curves of the model.
"""
function learningcurves(model::SVDModel, train::Vector{Rating}, test::Vector{Rating}, step::Int64=1; cb::Union{Function, Nothing}=nothing)
    sizes = collect(1:step:length(train))
    trainscores = SharedArray{Float64}(length(sizes))
    testscores = SharedArray{Float64}(length(sizes))
    
    done = RemoteChannel(() -> Channel{Bool}(32))

    @distributed for (i,s) in collect(enumerate(sizes))
        m = deepcopy(model)
        fit!(m, train[1:s])
        trainscores[i] = score(m, train[1:s])
        testscores[i] = score(m, test)
        put!(done, true)
    end
    
    for i in 1:length(sizes)
        take!(done)
        if cb !== nothing
            cb(i, length(sizes))
        end
    end
    
    return (sizes, trainscores, testscores)
end

# BEGIN: ModelSelection module
"""
    SplitMethod

Abstract supertype for dataset splitting methods.
"""
abstract type SplitMethod end


"""
    TrainTestSplit <: SplitMethod

Split a vector of ratings into a training and a test set, according to a percentage ratio.
"""
mutable struct TrainTestSplit <: SplitMethod
    ratio::Float64
    shuffle::Bool
    rng
end

"""
    TrainTestSplit(ratio::Float64; shuffle::Bool=false, rng=GLOBAL_RNG)

Split a vector of ratings into a training and a test set, according to a percentage ratio.

# Arguments
- `ratio::Float64`: the percentage of ratings to put in training set.
- `shuffle::Bool=false`: whether to randomly shuffle the ratings before splitting or not.
- `rng=GLOBAL_RNG`: random number generator to shuffle with.
"""
function TrainTestSplit(ratio::Float64; shuffle::Bool=false, rng=GLOBAL_RNG)
    TrainTestSplit(ratio, shuffle, rng)
end

"""
    split(method::TrainTestSplit, ratings::Vector{Rating})::NTuple{2, Vector{Rating}}

Split a vector of ratings into a training and a test set, according to a percentage ratio.
"""
function splitcv(method::TrainTestSplit, ratings::Vector{Rating})::NTuple{2, Vector{Rating}}
    if method.shuffle
        ratings = shuffle(method.rng, ratings)
    end
    idx = round(Int, length(ratings) * method.ratio)
    return (ratings[1:idx], ratings[idx+1:end])
end

"""
    LeaveOneOut <: SplitMethod

Split a vector of ratings into a training and a test set, each user has exactly one rating in the test set.
"""
mutable struct LeaveOneOut <: SplitMethod
    minratings::Int64
    shuffle::Bool
    rng
end

"""
    LeaveOneOut(;minratings::Int64=0, shuffle::Bool=false, rng=GLOBAL_RNG)

Split a vector of ratings into a training and a test set, each user has exactly one rating in the test set.

minratings sets the minimum number of ratings for each user in the trainset, others will be discarded.

# Arguments
- `rng=GLOBAL_RNG`: random number generator to shuffle with.
"""
function LeaveOneOut(;minratings::Int64=0, shuffle::Bool=false, rng=GLOBAL_RNG)
    LeaveOneOut(minratings, shuffle, rng)
end

"""
    splitcv(method::LeaveOneOut, ratings::Vector{Rating})::NTuple{2, Vector{Rating}}

Split a vector of ratings into a training and a test set, each user has exactly one rating in the test set.
"""
function splitcv(method::LeaveOneOut, ratings::Vector{Rating})::NTuple{2, Vector{Rating}}
    if method.shuffle
        ratings = shuffle(method.rng, ratings)
    end
    trainset = Rating[]
    testset = Rating[]
    users = Dict{Int64, NamedTuple{(:count, :done), Tuple{Int64, Bool}}}()
    
    for r in ratings
        user = get!(users, r.user, (count=0, done=false))
        users[r.user] = (count=user.count+1, done=user.done)
    end
    
    for r in ratings
        user = users[r.user]
        if user.count >= method.minratings
            if !user.done
                push!(testset, r)
                users[r.user] = (count=user.count, done=true)
            else
                push!(trainset, r)
            end
        end
    end
            
    return (trainset, testset)
end
# END: ModelSelection module

end
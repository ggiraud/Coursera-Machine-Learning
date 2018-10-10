"""
    MatrixFactorization

Abstract supertype for matrix factorization based recommender systems models.
"""
abstract type MatrixFactorization end


"""
    SVDModel{N,S<:Solver} <: MatrixFactorization

SVD based recommender systems model.
"""
mutable struct SVDModel{N,S <: Solver} <: MatrixFactorization
    # Factorization
    items::Union{SparseVector{Item{N},Int64}, Nothing}
    users::Union{SparseVector{User{N},Int64}, Nothing}
    
    # Solver
    solver::S
    
    # Initialization
    mean::Float64
    std::Float64
    
    # Baseline
    bias::Float64
end


"""
    SVDModel{N,S}(; <keyword arguments>) where {N,S<:Solver}

# Arguments
- `solver::S`: optimization algorithm.
- `mean::Float64=0.0`: mean of the normal distribution used to initialize the factorization matrices.
- `std::Float64=1e-4`: standard deviation of the normal distribution used to initialize the factorization matrices.
"""
function SVDModel(k::Int64, solver::S;
        mean::Float64=zero(Float64),
        std::Float64=1e-4) where {S <: Solver}
    
    items = nothing
    users = nothing
    bias = zero(Float64)
    
    SVDModel{k,S}(items,users,solver,mean,std,bias)
end

function Base.show(io::IO, model::SVDModel{N,S}) where {N,S <: Solver}
    fields = [:mean, :std]
    kwstr = join(["$(f)=$(getfield(model, f))" for f in fields], ", ")
    print(io, "SVD{$N,$S}($(kwstr),...)")
end

function nitems(model::SVDModel)
    model.items === nothing ? 0 : length(model.items)
end

function nusers(model::SVDModel)
    model.users === nothing ? 0 : length(model.users)
end

function userbiases(model::SVDModel)
    (u.bias for u in model.users)
end

function itembiases(model::SVDModel)
    (i.bias for i in model.items)
end


"""
    fit!(model::SVDModel{N,SGD}, ratings::Vector{Rating}; cb::Union{Function, Nothing}=nothing) where {N}

Fit the model by optimizing a regularized SSE(Summed Squared Error) through stochastic gradient descent.

If passed as the keyword argument `cb`, a callback function will be called at the end of each epoch
with arguments `nepoch::Int64` and `cost::Float64`.
"""
function fit!(model::SVDModel{N,SGD}, ratings::Vector{Rating}; cb::Union{Function, Nothing}=nothing) where {N}
    solver::SGD = model.solver
    model.bias = mean(r.value for r in ratings)
    
    # initialize model's items and users sparse vectors from ratings
    items, users = Dict{Int64,Item{N}}(), Dict{Int64,User{N}}()
    foreach(ratings) do r
        items[r.item] = Item{N}(model.std)
        users[r.user] = User{N}(model.std)
    end
    model.items, model.users = sparsevec(items), sparsevec(users)
    
    for epoch in 1:solver.nepochs
        currentcost::Float64 = zero(Float64)
        for i in randperm(length(ratings))
            r = ratings[i]
            item::Item{N} = model.items[r.item]
            user::User{N} = model.users[r.user]

            e::Float64 = model.bias + item.bias + user.bias + dot(item.weights, user.weights) - r.value

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
    fit!!(model::SVDModel{N,ALS}, ratings::Vector{Rating}; cb::Union{Function, Nothing}=nothing) where {N}

Fit the model using Alternating Least Square method.

If passed as the keyword argument `cb`, a callback function will be called at the end of each epoch
with arguments `nepoch::Int64` and `cost::Float64`.
"""
function fit!(model::SVDModel{N,ALS}, ratings::Vector{Rating}; cb::Union{Function, Nothing}=nothing) where {N}
    solver::ALS = model.solver
    
    # build sparse matrix from ratings
    R::SparseMatrixCSC{Float64, Int64} = dropzeros(
        sparse(
            [r.item for r in ratings],
            [r.user for r in ratings],
            [r.value for r in ratings]
        ))
    ni, nu = R.m, R.n
    
    # initialize model's items and users sparse vectors from ratings
    items, users = Dict{Int64,Item{N}}(), Dict{Int64,User{N}}()
    foreach(ratings) do r
        items[r.item] = Item{N}(model.std)
        users[r.user] = User{N}(model.std)
    end
    model.items, model.users = sparsevec(items), sparsevec(users)
    
    # create baseline
    model.bias = mean(nonzeros(R))
    bi = SharedArray{Float64}(ni)
    bu = SharedArray{Float64}(nu)
    
    # initialize factorization matrices
    P = SharedArray{Float64}(ni, N)
    P .= rand(Normal(model.mean, model.std), ni, N)
    Q = SharedArray{Float64}(nu, N)
    Q .= rand(Normal(model.mean, model.std), nu, N)
    
    function updatefactors()
        for i in model.items.nzind
            model.items[i] = Item{N}(P[i,:], bi[i])
        end
        for u in model.users.nzind
            model.users[u]  = User{N}(Q[u,:], bu[u])
        end
    end
    
    R_mu = R .- model.bias
    rtol = sqrt(eps(real(float(one(Float64)))))
    
    for epoch in 1:solver.nepochs
        P_biased = [ones(ni) P]
        R_mu_bi = R_mu .- bi
        @sync @distributed for u in 1:nu
            rated_items_indices = R[:, u].nzind
            P_biased_truncated = P_biased[rated_items_indices, :]
            A = P_biased_truncated' * P_biased_truncated + solver.reg * I
            b = P_biased_truncated' * R_mu_bi[rated_items_indices, u]
            #x = pinv(A, rtol) * b
            x = A \ b
            bu[u], Q[u,:] = x[1], x[2:end]
        end
        
        Q_biased = [ones(nu) Q]
        R_mu_bu = R_mu .- bu'
        @sync @distributed for i in 1:ni
            rating_users_indices = R[i, :].nzind
            Q_biased_truncated = Q_biased[rating_users_indices, :]
            A = Q_biased_truncated' * Q_biased_truncated + solver.reg * I
            b = Q_biased_truncated' * R_mu_bu[i, rating_users_indices]
            #x = pinv(A, rtol) * b
            x = A \ b
            bi[i], P[i,:] = x[1], x[2:end]
        end
        
        if cb !== nothing
            updatefactors()
            currentcost = score(model, ratings)
            cb(epoch, currentcost)
        end
    end
    
    updatefactors()
end


"""
    predict(model::SVDModel, item::Int64, user::Int64)

Predict the rating of an item by a user.
"""
function predict(model::SVDModel{N,S}, item::Int64, user::Int64) where {N,S}
    item = item > length(model.items) ? zero(Item{N}) : model.items[item]
    user = user > length(model.users) ? zero(User{N}) : model.users[user]
    model.bias + item.bias + user.bias + dot(item.weights, user.weights)
end


"""
    score(model::SVDModel, ratings::Vector{Rating})

Compute the score of the fitted model using SSE(Summed Squared Error).
"""
function score(model::SVDModel, ratings::Vector{Rating})
    sum(abs2, (predict(model, r.item, r.user) - r.value) for r in ratings)
end
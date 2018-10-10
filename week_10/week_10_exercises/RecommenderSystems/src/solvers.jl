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
- `reg::Float64=0.0`: regularization parameter used during gradient descent optimization.
"""
function SGD(;
        nepochs::Int64=10,
        lr::Float64=0.001,
        reg::Float64=zero(Float64))
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
- `reg::Float64=0.0`: regularization parameter used during alternating least squares optimization.
"""
function ALS(;
    nepochs::Int64=10,
    reg::Float64=0.001)
    ALS(nepochs, reg)
end
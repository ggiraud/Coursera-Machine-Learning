"""
    User

A User's weights and bias.
"""
mutable struct User{N}
    weights::Vector{Float64}
    bias::Float64
end

"""
    User{N}(std::Float64=1.0) where {N}

Create a new User.

`weights` field is a `Vector` initialized with a random multivariate normal distribution of `N` features and `std` standard deviation.
`bias` field is initialized to `zero(Float64)`.
"""
function User{N}(std::Float64=1.0) where {N}
    User{N}(rand(MvNormal(N, std)), zero(Float64))
end

"""
    zero(t::Type{User{N}}) where {N}

Create a new User with `weight`and `bias` sets to zero.

`weights` field is a `Vector` of `N` elements initialized to `zero(Float64`.
`bias` field is initialized to `zero(Float64`.
"""
function zero(t::Type{User{N}}) where {N}
    User{N}(zeros(Float64, N), zero(Float64))
end
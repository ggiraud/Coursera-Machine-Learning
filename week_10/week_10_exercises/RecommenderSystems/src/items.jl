"""
    Item

An Item's weights and bias.
"""
mutable struct Item{N}
    weights::Vector{Float64}
    bias::Float64
end

"""
    Item{N}(std::Float64=1.0) where {N}

Create a new Item.

`weights` field is a `Vector` initialized with a random multivariate normal distribution of `N` features and `std` standard deviation.
`bias` field is initialized to `zero(Float64)`.
"""
function Item{N}(std::Float64=1.0) where {N}
    Item{N}(rand(MvNormal(N, std)), zero(Float64))
end

"""
    zero(t::Type{Item{N}}) where {N}

Create a new Item with `weight`and `bias` sets to zero.

`weights` field is a `Vector` of `N` elements initialized to `zero(Float64`.
`bias` field is initialized to `zero(Float64`.
"""
function zero(t::Type{Item{N}}) where {N}
    Item{N}(zeros(Float64, N), zero(Float64))
end

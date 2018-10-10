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
    Ratings

Source of ratings.
"""
mutable struct Ratings{T}
    src::T
end

function Ratings{Vector{Rating}}(vec::Vector{Rating})
    Ratings{Vector{Rating}}(vec)
end

function Base.iterate(ratings::Ratings{Vector{Rating}}, state=1)
     return iterate(ratings.src, state)
end


function Ratings{NextTable}(table::NextTable)
	Ratings{NextTable}(table)
end

function Base.iterate(ratings::Ratings{NextTable}, state=1)
    return iterate(rows(ratings.src), state)
end


#=
function Base.iterate(ratings::Ratings{IOStream}, state=ratings.src)
    s::IOStream = ratings.src
    
    if eof(s)
        return nothing
    end
    Vector{Rating}}(ratings::Vector{Rating})
    Ratings{Vector{Rating}}(ratings)
end

function Base.iterate(ratings::Ratings{Vector{Rating}}, state=1)
     return iterate(ratings.src, state)
end

    return (readline(s), ratings.src)
end

function Base.IteratorSize(t::Type{Ratings{IOStream}})
    return Base.SizeUnknown
end
=#
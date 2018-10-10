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
"""
    learningcurves(model::SVDModel, train::Vector{Rating}, test::Vector{Rating}, step::Int64=1; cb::Union{Function, Nothing}=nothing)

Successively compute the train and test scores necessary to plot the learning curves of the model.
"""
function learningcurves(model::SVDModel{N,S}, train::Vector{Rating}, test::Vector{Rating}, step::Int64=1; cb::Union{Function, Nothing}=nothing) where {N,S <: Solver}
    sizes = collect(1:step:length(train))
    trainscores = SharedArray{Float64}(length(sizes))
    testscores = SharedArray{Float64}(length(sizes))
    
    done = RemoteChannel(() -> Channel{Bool}(32))

    @distributed for (i,s) in collect(enumerate(sizes))
        m::SVDModel{N,S} = deepcopy(model)
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
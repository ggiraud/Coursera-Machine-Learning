module RecommenderSystems

using JuliaDB
import Base.iterate, Base.IteratorSize, Base.SizeUnknown
export Rating
# refs:
include("./ratings.jl")

import Base.show
export Solver, SGD, ALS
# refs:
include("./solvers.jl")

using Random
using Random: GLOBAL_RNG
export SplitMethod, TrainTestSplit, LeaveOneOut, splitcv
# refs: .Rating
include("./splits.jl")

using Distributions
import Base.zero
export Item
# refs:
include("./items.jl")

using Distributions
import Base.zero
export User
# refs:
include("./users.jl")

using LinearAlgebra
using Distributions
using Random
using SparseArrays
using SharedArrays
using Distributed
export MatrixFactorization, SVDModel, itembiases, userbiases, fit!, predict, score
# refs: .Rating, .Solver, .SGD, .User, .Item
include("./mf.jl")

using SharedArrays: SharedArray
using Distributed: @distributed, RemoteChannel
export learningcurves
# refs: .SVDModel, .Rating
include("./model_selection.jl")

end

__precompile__()

module EmpiricalRiskMinimization

using Compat

# Losses
export SquaredLoss, AbsoluteLoss, HuberLoss
include("losses.jl")

# Regularizers
export L1Reg
include("regularizers.jl")

export minimize
include("optimizer.jl")

end # module

__precompile__()

module EmpiricalRiskMinimization

using Compat

# Losses
export SquaredLoss, AbsoluteLoss, HuberLoss, HingeLoss, LogisticLoss
include("losses.jl")

# Regularizers
export L1Reg, L2Reg, L1L2Reg
include("regularizers.jl")

# GD function
export minimize
include("optimizer.jl")

end # module

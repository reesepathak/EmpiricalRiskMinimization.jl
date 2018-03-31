__precompile__()

module EmpiricalRiskMinimization

using Compat

# Losses
export SquaredLoss, AbsoluteLoss, HuberLoss, HingeLoss, LogisticLoss, FrobeniusLoss
include("losses.jl")

# Regularizers
export L1Reg, L2Reg, L1L2Reg
include("regularizers.jl")

# GD function
export minimize
include("optimizer.jl")

# Models
export Model, fit!, status, final_risk, weight
include("model.jl")

# Utility functions
export sigm
include("util.jl")

end # module

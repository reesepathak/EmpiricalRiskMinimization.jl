__precompile__()

module EmpiricalRiskMinimization

using Compat

# Losses
export SquaredLoss, AbsoluteLoss, HuberLoss, HingeLoss, LogisticLoss, FrobeniusLoss
include("losses.jl")

# Regularizers
export L1Reg, L2Reg, L1L2Reg, NoReg, PosReg
include("regularizers.jl")

# GD function
export minimize, minimize_unsupervised
include("optimizer.jl")

# Models
export Model, fit!, fit_unsupervised!, status, final_risk, parameters
include("model.jl")

# Utility functions
export sigm
include("util.jl")

end # module

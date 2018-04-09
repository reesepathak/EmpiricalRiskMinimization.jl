__precompile__()

module EmpiricalRiskMinimization

using Compat

# Losses
export SquareLoss, AbsoluteLoss, HuberLoss, HingeLoss, LogisticLoss, FrobeniusLoss
include("losses.jl")

# Regularizers
export L1Reg, L2Reg, L1L2Reg, NoReg, PosReg
include("regularizers.jl")

# Embeddings
export Standardize, IdentityEmbed, PolyEmbed, AppendOneEmbed
include("embeddings.jl")

# GD function
export minimize, minimize_unsupervised
include("optimizer.jl")

# Models
export Model, Mldata, FoldedData, train, trainpath, status
export predicttest, predicttrain, predictu, setfeatures
export thetaopt, trainloss, testloss, lambda, lambdaopt
export Ytrain, Ytest, Xtrain, Xtest, thetamatrix
include("model.jl")

# Utility functions
export sigm
include("util.jl")

end # module

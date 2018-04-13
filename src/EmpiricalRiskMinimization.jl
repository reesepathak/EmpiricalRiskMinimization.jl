__precompile__()

module EmpiricalRiskMinimization

using Compat

# Losses
export loss
export SquareLoss, AbsoluteLoss, HuberLoss, HingeLoss, LogisticLoss, FrobeniusLoss
include("losses.jl")

# Regularizers
export L1Reg, L2Reg, L1L2Reg, NoReg, PosReg
include("regularizers.jl")

# Embeddings
export Standardize, IdentityEmbed, PolyEmbed, AppendOneEmbed, PiecewiseEmbed, embed
include("embeddings.jl")

# GD function
export minimize, minimize_unsupervised
include("optimizer.jl")

# Models
export Model, Mldata, FoldedData
export train, trainpath, trainfolds, status
export setfeatures
export predict
export predict_y_from_test, predict_y_from_train, predict_v_from_test, predict_v_from_train
export predict_y_from_u, predict_y_from_u, predict_v_from_u, predict_v_from_u
export thetaopt, trainloss, testloss, lambda, lambdaopt
export lambdapath, testlosspath, trainlosspath, thetapath
export Ytrain, Ytest, Xtrain, Xtest
include("model.jl")

# Utility functions
export sigm, matrix
include("util.jl")

end # module

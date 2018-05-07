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

include("source.jl")

# GD function
export minimize
include("optimizer.jl")

# Models
export Model, Mldata, FoldedData
export train, trainpath, trainfolds, status
export setfeatures
export Ytrain, Ytest, Xtrain, Xtest
include("model.jl")

# prediction
export predict
export predict_y_from_test, predict_y_from_train, predict_v_from_test, predict_v_from_train
export predict_y_from_u, predict_y_from_u, predict_v_from_u, predict_v_from_u
include("prediction.jl")

# validation
export thetaopt, trainloss, testloss, lambda, lambdaopt
export lambdapath, testlosspath, trainlosspath, thetapath
include("validation.jl")

# Utility functions
export sigm, matrix
include("util.jl")

end # module

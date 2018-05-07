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

# Internal imports (private)
include("internal/source.jl") 
include("internal/results_and_data.jl") # Results, Data
include("internal/optimizer.jl") # Gradient descent, QR, CVX solvers

# Models
export Model, Mldata, FoldedData
export train, trainpath, trainfolds, status
export setfeatures
export Ytrain, Ytest, Xtrain, Xtest
include("model.jl")

# Validation
export thetaopt, trainloss, testloss, lambda, lambdaopt
export lambdapath, testlosspath, trainlosspath, thetapath
include("validation.jl")

# Prediction
export predict
export predict_y_from_test, predict_y_from_train, predict_v_from_test, predict_v_from_train
export predict_y_from_u, predict_y_from_u, predict_v_from_u, predict_v_from_u
include("prediction.jl")

# Convenience functions
export sigm, matrix
include("convenience.jl")

end # module

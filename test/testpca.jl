include("../src/EmpiricalRiskMinimization.jl")

using EmpiricalRiskMinimization

X = randn(10, 5)
k = 3

EmpiricalRiskMinimization.minimize_unsupervised(FrobeniusLoss(), X, k)
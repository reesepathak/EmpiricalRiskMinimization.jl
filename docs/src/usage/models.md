# [Models and training](@id usage_models)

The basic primitive of EmpiricalRiskMinimization.jl is the model type.

## Creating a Model

There are many ways to instantiate a model. You use the function `Model(...)` do so.
It has the following definition

```julia
function Model(U, V; loss=SquareLoss(), reg=L2Reg(),
               Unames = nothing, Vnames = nothing,
               embedall = false, verbose=false, kwargs...)
```

The first two arguments specify the data: inputs `U` and targets `V`.
There are many keyword arguments:
- `loss` specifies the [loss](@ref usage_losses) (also called risk) function that will be used to train your model.
- `reg` specifies the [regularizer](@ref usage_regularizers) on the model parameters.
- `Unames` and `Vnames` specify names for the columns of `U` and `V`, respectively. This mainly applies to
datasets that are not entirely numerical.
- `embedall` determines whether all of the data in `U` and `V` will be used to train. If it is set to `true`, then
`U` and `V` are standardized and a constant feature is added to `U`, giving `X` and `Y`.
- `verbose` allows users to see more of the progress that occurs during model usage. It will automatically call
`status()` at the end of model actions (see below). 

### Default parameters
Suppose you create a model on `U` and `v` using the following line of code.
```julia
M = Model(U, V)
```
This specifies a squared loss, and l2 regularization. Additionally, it will create `X` by standardizing `U` and
adding a constant feature, and it will create `Y` by standardizing `V`. Addditionally, it will put regularization on
all the weights except for the first weight, corresponding to the constant feature. The simplest way to describe this model
is a regularized least squares regression model with regularization on the non-constant parameters.

### Specifying different models

## Training

## Status


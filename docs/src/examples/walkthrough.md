# Walkthrough

This page gives you a very basic example of how to use EmpiricalRiskMinimization.jl,
with links to other documentation pages to learn more about advanced functionality available within the package.

Suppose we want to solve a regularized least square linear regression problem.
Let's first generate some data.

```@setup LS-example
using EmpiricalRiskMinimization
srand(123)
```

```@example LS-example
n = 2000; k = 30;
d = k + 1;
U = randn(n, k); theta = randn(d);
v = [ones(n) U] * theta + 0.5 * randn(n);
```

So, we've generated 2000 random raw data points, occuping the
rows of `U`. These data points have 30 features. Additionally,
we generated targets `v`, so that `v[i]` is the label associated
with example `U[i, :]`.

Formulating and solving (regularized) least square linear regression
with ERM.jl is simple. The first step is to instantiate the model
```@example LS-example
M = Model(U, v, embedall=true);
```
The option `embedall=true` takes `U` and compiles
our true training data `X`, by appending the constant feature
to the rows of `U`. Additionally, it standardizes our data for us.
There are many more features available for [training, embedding,
and modelling](@ref usage_models). Of course, to specify a different
model, users must specify different [losses](@ref usage_losses) and
[regularizers](@ref usage_regularizers). 

Training the model and getting the output is two lines of code.
```@example LS-example
train(M)
status(M)
```

This training summary is useful, and is the most basic
[validation](@ref usage_validation) tool that ERM provides;
cross-validation and repeated out-of-sample validation are also available.

To assess the accuracy of the model on the train and test sets, we
can compute the (average) train and test losses.
```@example LS-example
println("Training error = $(trainloss(M))")
println("Testing error = $(testloss(M))")
```

Finally, suppose we actually want to retrieve our predictions
on the test data.
```@example LS-example
v_test_pred = predict_v_from_test(M);
```
There are more [prediction functions](@ref usage_prediction) available. These
allow you to provide alternative model parameters, unembed predictions,
and test on various other datasets. 



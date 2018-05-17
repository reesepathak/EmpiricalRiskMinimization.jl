# [Prediction](@id usage_prediction)

After training and validating a model, we often want to use models on unseen data.
In this page, we describe the prediction functions available in `ERM`.

## Using a trained model on new data
The most basic way to do prediction is to use `predict`.
```julia
predict(M, x)
````
The code above takes as input a model `M` and data point `x`, from which a prediction is
formed according to the model's parameters. You can also input a different choice of `theta`,
as in `predict(M, x, theta)`. By default, it is set to the optimal `theta` stored in `M`.

Of course, in many settings you may want to predict on a certain set of data.

For example the following two lines of code will allow you to compute predictions on the training
set. 
```julia
predict_y_from_train(M)
predict_v_from_train(M)
```
These functions allow you to compute embedded and unembedded predictions (corresponding to `y` and `v`, respectively) on the
train set of `M`.

```julia
predict_y_from_test(M)
predict_v_from_test(M)
```
These functions allow you to compute embedded and unembedded predictions (corresponding to `y` and `v`, respectively) on the
test set of `M`.

Additionally, if you would rather provide a single raw input `u`, we provide all the prediction functions
you could ever want. 
```julia
predict_y_from_u(M)
predict_v_from_u(M)
```

## Recovering losses

You can compute the train and test losses using `trainloss(M)` and `testloss(M)`, respectively. 

You often also want to compute a confusion matrix when solving classification problems.
```julia
confusion_train(M)
confusion_test(M)
```
These two functions compute confusion matrices on the train and test sets (respectively) for `M`.





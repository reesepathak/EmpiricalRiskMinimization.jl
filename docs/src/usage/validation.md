# [Validation and out-of-sample testing](@id usage_validation)

## Recovering losses and optimal parameters

To view the train and test loss on a particular model `M`, you simply
invoke `trainloss` and `testloss` on the model.

To get the optimal `theta` and `lambda` recovered from training, you
use `thetaopt(M)` and `lambdaopt(M)`. 

To recover the list of `lambda` values used for a regularization path, you
simply use `lambdapath(M)`. Similarly, the optimal thetas are found via
`thetapath(M)`, and you can find the corresponding training and test losses
through `trainloss(M)` and `testloss(M)`, respectively. 

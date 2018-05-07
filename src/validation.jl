##############################################################################
# Validation
##############################################################################

testloss(R::PointResults) = R.testloss
trainloss(R::PointResults) = R.trainloss
thetaopt(R::PointResults) = R.theta
lambda(R::PointResults) = R.lambda

testloss(R::FoldResults,i) = testloss(R.results[i])
trainloss(R::FoldResults,i) = trainloss(R.results[i])
thetaopt(R::FoldResults,i) = thetaopt(R.results[i])
lambda(R::FoldResults,i) = lambda(R.results[i])


lambdapath(R::RegPathResults) = [r.lambda for r in R.results]
testlosspath(R::RegPathResults) = [r.testloss for r in R.results]
trainlosspath(R::RegPathResults) = [r.trainloss for r in R.results]

testloss(R::RegPathResults) = testloss(R.results[R.imin])
trainloss(R::RegPathResults) = trainloss(R.results[R.imin])
thetaopt(R::RegPathResults) = thetaopt(R.results[R.imin])
lambdaopt(R::RegPathResults) = lambda(R.results[R.imin])

function thetapath(R::RegPathResults)
    r = length(R.results)
    d = length(R.results[1].theta)
    T = zeros(d,r)
    for i=1:r
        T[:,i] = R.results[i].theta
    end
    return T'
end

testloss(M::Model,i) = testloss(M.D.results,i)
trainloss(M::Model,i) = trainloss(M.D.results,i)
thetaopt(M::Model,i) = thetaopt(M.D.results,i)
lambda(M::Model,i) = lambda(M.D.results,i)

"`testloss(M)` returns the average testing loss"
testloss(M::Model) = testloss(M.D.results)
"`trainloss(M)` returns the average training loss"
trainloss(M::Model) = trainloss(M.D.results)
"`thetaopt(M)` returns the optimal `theta` chosen during training"
thetaopt(M::Model) = thetaopt(M.D.results)
"`lambda(M)` returns the `lambda` used during training. If multiple `lambda` values were specified,
returns the optimal one"
lambda(M::Model) = lambda(M.D.results)

"`lambdaopt(M)` returns the optimal lambda computed during training of M"
lambdaopt(M::Model) = lambdaopt(M.D.results)

"`lambdapath(M)` returns the vector of lambdas used"
lambdapath(M::Model) = lambdapath(M.D.results)
"`testlosspath(M)` returns the vector of test losses at various lambdas"
testlosspath(M::Model) = testlosspath(M.D.results)
"`trainlosspath(M)` returns the vector of train losses at various lambdas"
trainlosspath(M::Model) = trainlosspath(M.D.results)
"`thetapath(M)` returns the vector of optimal thetas at various lambdas"
thetapath(M::Model) = thetapath(M.D.results)
##############################################################################

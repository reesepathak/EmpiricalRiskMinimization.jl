
##############################################################################
# Results

abstract type ErmResults end


mutable struct RegPathResults<:ErmResults
    theta      # list of thetas 
    lambda     # list of numbers
    trainloss  # list of numbers
    testloss   # list of numbers
    imin       # index of lambda that minimizes test loss
end

mutable struct PointResults<:ErmResults
    theta      # optimal theta
    lambda     # lambda used
    trainloss  # number
    testloss   # number
end

mutable struct FoldResults<:ErmResults
    theta      # list of thetas, one per fold
    lambda     # lambda used
    trainloss  # list of trainlosses, one per fold
    testloss   # list of testlosses, one per fold
end

mutable struct NoResults<:ErmResults end


##############################################################################
# Data

abstract type ErmData end


mutable struct FoldedData<:ErmData
    X
    Y
    nfolds
    foldrows
    results::ErmResults
end

# Note both X and Y are 2 dimensional Arrays
mutable struct SplitData<:ErmData 
    Xtrain
    Ytrain
    Xtest
    Ytest
    trainfrac
    results::ErmResults
end

mutable struct NoData<:ErmData end

##############################################################################
# Model

mutable struct Model
    D::ErmData
    loss::Loss
    regularizer::Regularizer
    solver::Solver
    X
    Y
    Xembed
    Yembed
    hasconstfeature::Bool        # true if the first feature is 1
    featurelist                  # a list of the columns of X that are of interest
    regweights
    istrained::Bool
end

##############################################################################



function embed(U, V, Xembed, Yembed)
    if Xembed == false
        Xembed = [AppendOneEmbed(), Standardize()]
    end
    if Yembed == false
        Yembed = [Standardize()]
    end
    hasconstfeature = false
    if isa(Xembed[1], AppendOneEmbed)
        hasconstfeature = true
    end
    
    Y = embed(Yembed, matrix(V))
    X = embed(Xembed, matrix(U))
    return X, Y, Xembed, Yembed, hasconstfeature
end

function getfoldrows(n, nfolds)
    # split into groups
    groups = [convert(Int64, round(x*n/nfolds)) for x in 1:nfolds]
    unshift!(groups,0)
    # i'th fold has indices groups[i]+1:groups[i+1]
    p = randperm(n)
    foldrows=Any[]
    for i=1:nfolds
        push!(foldrows, sort(p[groups[i]+1:groups[i+1]]))
    end
    return foldrows
end

function FoldedData(U, V, nfolds; Xembed = false, Yembed = false)
    Xall, Yall, Xembed, Yembed, hasconstfeature = embed(U, V, Xembed, Yembed)
    n = size(Xall,1)
    X = Any[]
    Y = Any[]
    for i=1:nfolds
        push!(X, Xall[foldrows[i], :])
        push!(Y, Yall[foldrows[i], :])
    end
    d = FoldedData(X, Y, nfolds, nothing, hasconstfeature)
end
    
function splitrows(n, trainfrac; splitmethod=0)
    if splitmethod == 0
        ntrain = convert(Int64, round(trainfrac*n))
        p = randperm(n)
        trainrows = sort(p[1:ntrain])
        testrows = sort(p[ntrain+1:n])
    else
        # pick by Bernoulli
        testrows = Int64[]
        trainrows = Int64[]
        for i=1:n
            if rand()>trainfrac
                push!(testrows,i)
            else
                push!(trainrows,i)
            end
        end
    end
    return trainrows, testrows
end



##############################################################################



##############################################################################
# predict



predict(M::Model, X::Array{Float64,2}, theta) = X*theta
predicttest(M::Model, theta) = predict(M, Xtest(M), theta)
predicttrain(M::Model, theta) = predict(M, Xtrain(M), theta)
predicttest(M::Model) = predict(M, Xtest(M), thetaopt(M.D.results))
predicttrain(M::Model) = predict(M, Xtrain(M), thetaopt(M.D.results))
predictx(M::Model, x) = predict(M, x, thetaopt(M.D.results))
predictu(M::Model, u::Array{Float64,2}) =  predictx(M::Model, embed(M.Xembed, u))



##############################################################################
# fit

function setsolver(M::Model)
    if isa(M.solver, DefaultSolver)
        M.solver = getsolver(M.loss, M.regularizer)
    end
end

function trainx(M::Model,  lambda::Number)
    theta = solve(M.solver, M.loss, M.regularizer, M.regweights,
                  Xtrain(M), Ytrain(M), lambda)
    trainloss = loss(M.loss, predicttrain(M, theta), Ytrain(M.D))
    testloss = loss(M.loss,  predicttest(M, theta), Ytest(M.D))
    return theta, trainloss, testloss
end

function trainpoint(M::Model, lambda::Number; quiet=false)
    M.lambda = lambda
    M.theta, M.trainloss, M.testloss = trainx(M,  lambda)
    M.istrained = true
    M.hasregpath = false
    if !quiet
        @printf("lambda = %f, training loss = %.4f\n", lambda, M.trainloss)
        @printf("lambda = %f, test loss = %.4f\n", lambda, M.testloss)
    end
end

function trainfolds(M::Model, lambda::Number; quiet=false)
    setsolver(M)
    M.lambda = lambda
    M.theta = Array{Any}(M.D.nfolds)
    M.trainloss = zeros(M.D.nfolds)
    M.testloss = zeros(M.D.nfolds)
    for i=1:M.D.nfolds
        theta = solve(M.solver, M.loss, M.regularizer, M.regweights,
                      Xtrain(M.D), train(M.D), lambda)
        M.trainloss[i] = loss(M.loss, predicttrain(M, theta), Ytrain(M.D))
        M.testloss[i] = loss(M.loss,  predicttest(M, theta), Ytest(M.D))
    end

end


function train(M::Model, lambda::Number; quiet=false)
    if isa(M.D, FoldedData)
        trainfolds(M)
    else
        trainpoint(M, lambda; quiet=quiet)
    end
end

train(M::Model; kwargs...) = train(M, 1e-10; kwargs...)    

function train(M::Model, lambda::Array; quiet=false, kwargs...)
    setsolver(M)
    nl = length(lambda)
    M.lambda = lambda
    M.theta = Array{Any}(nl)
    M.trainloss = zeros(nl)
    M.testloss = zeros(nl)
    for i=1:nl
        M.theta[i], M.trainloss[i], M.testloss[i] = trainx(M, lambda[i])
    end
    M.imin =  findmin(M.testloss)[2]
    M.lambdamin = M.lambda[M.imin]
    M.istrained = true
    M.hasregpath = true
    if !quiet
        @printf("optimal lambda = %.3f\n", M.lambdamin)
        @printf("optimal test loss = %.3f\n", M.testloss[M.imin])
    end
end

##############################################################################


function Model(U, V, loss, reg; Xembed = false, Yembed = false)
    X, Y, Xembed, Yembed, hasconstfeature = embed(U, V, Xembed, Yembed)
    regweights = ones(size(X,2))
    return  Model(NoData(), loss, reg, DefaultSolver(), X, Y, Xembed, Yembed,
                  hasconstfeature, nothing, regweights, false)
end

function SplitData(X, Y, trainfrac)
    trainrows, testrows = splitrows(size(X,1), trainfrac)
    Xtrain = X[trainrows,:]
    Ytrain = Y[trainrows,:]
    Xtest = X[testrows,:]
    Ytest = Y[testrows,:]
    return SplitData(Xtrain, Ytrain, Xtest, Ytest, trainfrac, NoResults())
end

function FoldedData(X, Y, nfolds)
    foldrows = getfoldrows(size(X,1), nfolds)
    return FoldedData(X, Y, nfolds, foldrows, NoResults())
end



function splittraintest(M::Model, trainfrac)
    if isa(M.D, NoData) || M.D.trainfrac != trainfrac
        M.D = SplitData(M.X, M.Y, trainfrac)
    end
end

function splitfolds(M::Model, nfolds)
    if isa(M.D, NoData) || M.D.nfolds != nfolds
        M.D = FoldedData(M.X, M.Y, nfolds)
    end
end

function trainfolds(M::Model; nfolds=5)
    splitfolds(M, nfolds)
    trainfoldsx(M, nfolds)
end

function trainpath(M::Model; trainfrac=0.8, lambda=logspace(-5,5,100))
    splittraintest(M, trainfrac)
    trainpathx(M, lambda)
end


function train(M::Model; trainfrac=0.8, lambda=1e-10)
    splittraintest(M, trainfrac)
    setsolver(M)
    theta, trainloss, testloss = trainx(M, lambda)
    M.D.results  = PointResults(theta, lambda, trainloss, testloss)
    M.istrained = true
end



function setfeaturesx(M::Model, f)
    if f == "all"
        M.featurelist = nothing
    else
        M.featurelist = f
    end
end
    
function setfeatures(M::Model, f)
    setfeaturesx(M, f)
    M.regweights = ones(length(f))
    if M.hasconstfeature
        M.regweights[1] = 0
    end
end

Ytest(M::Model) = Ytest(M.D)
Ytrain(M::Model) = Ytrain(M.D)

function Xtest(M::Model)
    if M.featurelist == nothing
        return Xtest(M.D)
    end
    return Xtest(M.D)[:, M.featurelist]
end

function Xtrain(M::Model)
    if M.featurelist == nothing
        return Xtrain(M.D)
    end
    return Xtrain(M.D)[:, M.featurelist]
end



function Xtrain(D::SplitData)
    return D.Xtrain
end

function Xtest(D::SplitData)
    return D.Xtest
end


Ytrain(D::SplitData) = D.Ytrain
Ytest(D::SplitData) = D.Ytest

testloss(R::PointResults) = R.testloss
trainloss(R::PointResults) = R.trainloss
thetaopt(R::PointResults) = R.theta
lambdaopt(R::PointResults) = R.lambda


##############################################################################




"""
Prints and returns the status of the model.
"""
function status(M::Model)
    println("training samples: ", length(Ytrain(M)))
    println("test samples: ", length(Ytest(M)))
    if M.istrained
        if M.hasregpath
            println("optimal lambda: ",  M.lambdamin)
            println("optimal test loss: ", M.testloss[M.imin])
        else
            println("training  loss: ", trainloss(M))
            println("test loss: ", testloss(M))
        end
    end
    return 
end



"""
Returns the final test loss of the model.
"""
function testloss(M::Model)
    return testloss(M.D.results)
end

"""
Returns the final training loss of the model.
"""
function trainloss(M::Model)
    return trainloss(M.D.results)
end

"""
Returns model parameters 
"""
function thetaopt(M::Model)
    return thetaopt(M.D.results)
end


function thetamatrix(M::Model)
    r = length(M.lambda)
    d = length(M.theta[1])
    T = zeros(d,r)
    for i=1:r
        T[:,i] = M.theta[i]
    end
    return T'
end

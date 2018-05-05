
##############################################################################
# Results

abstract type ErmResults end


mutable struct RegPathResults<:ErmResults
    results    # list of PointResults
    imin       # index of lambda that minimizes test loss
end

mutable struct PointResults<:ErmResults
    theta      # optimal theta
    lambda     # lambda used
    trainloss  # number
    testloss   # number
end

mutable struct FoldResults<:ErmResults
    results    # list of results, one per fold
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
    nonfoldrows
    results::ErmResults
end

# Note both X and Y are 2 dimensional Arrays
mutable struct SplitData<:ErmData 
    Xtrain
    Ytrain
    Xtest
    Ytest
    trainrows
    testrows
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
    S::DataSource
    X
    Y
    featurelist                  # a list of the columns of X that are of interest
    regweights
    istrained::Bool
    verbose::Bool
    xydataisinvalid::Bool
    disinvalid::Bool
end

##############################################################################



function getfoldrows(n, nfolds)
    # split into groups
    groups = [convert(Int64, round(x*n/nfolds)) for x in 1:nfolds]
    unshift!(groups,0)
    # i'th fold has indices groups[i]+1:groups[i+1]
    p = randperm(n)
    foldrows = Any[]
    nonfoldrows = Any[]
    for i=1:nfolds
        push!(foldrows, sort(p[groups[i]+1:groups[i+1]]))
        push!(nonfoldrows, sort([ p[1:groups[i]] ; p[groups[i+1]+1:end]]))
    end
    return foldrows, nonfoldrows
end

function splitrows(n, trainfrac::Array{Int64,1})
    trainrows = trainfrac
    allrows = Set(1:n)
    trainset = Set(copy(trainrows))
    testrows = sort(collect(setdiff(allrows, trainset)))
    return trainrows, testrows
end

function splitrows(n, trainfrac::Number; splitmethod=0)
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




# if x happens to be a scalar, do we want this to work?
# predict(M::Model, x::Number,            theta=thetaopt(M))   = [x*theta]

# "predict" could be called "predict_y_from_x"
# for each record, x,y,u,v are always vectors
predict(M::Model, x::Array{Float64, 1}, theta=thetaopt(M))   = [dot(x, theta)]

# following could be defined using rowwise, but efficiency might dictate otherwise
predict(M::Model, X::Array{Float64, 2}, theta=thetaopt(M))   = X*theta

predict_y_from_test(M::Model,                   theta=thetaopt(M))   = predict(M, Xtest(M), theta)
predict_y_from_train(M::Model,                  theta=thetaopt(M))   = predict(M, Xtrain(M), theta)
predict_v_from_test(M::Model,                   theta=thetaopt(M))   = unembedY(M.S, predict(M, Xtest(M), theta))
predict_v_from_train(M::Model,                  theta=thetaopt(M))   = unembedY(M.S, predict(M, Xtrain(M), theta))

predict_y_from_u(M::Model, u::Array{T,1}, theta=thetaopt(M)) where {T<:Any} =  predict(M::Model, embedU(M.S, u), theta)
predict_y_from_u(M::Model, U::Array{T,2}, theta=thetaopt(M)) where {T<:Any} =  rowwise(u -> predict_y_from_u(M, u, theta), U)

predict_v_from_u(M::Model, u::Array{T,1}, theta=thetaopt(M)) where {T<:Any} =  unembedY(M.S, predict_y_from_u(M, u, theta))
predict_v_from_u(M::Model, U::Array{T,2}, theta=thetaopt(M)) where {T<:Any} =  rowwise(u -> predict_v_from_u(M, u, theta), U)






##############################################################################

function setdata(M::Model)
    M.X, M.Y  = getXY(M.S)
    M.xydataisinvalid = false
    setregweights(M)
end

function setregweights(M::Model)
    if M.xydataisinvalid
        return
    end
    X = selectfeatures(M, M.X)
    nf = size(X,2)
    M.regweights = ones(nf)
    for i=1:nf
        if var(X[:,i]) == 0
            M.regweights[i] = 0
        end
    end
end


function defaultembedding(M::Model; stand=true)
    addfeatureV(M, 1, stand=stand)
    addfeatureU(M, etype="one")
    d = size(getU(M),2) 
    for i=1:d
        addfeatureU(M, i, stand=stand)
    end
end

function Model(S::DataSource, loss, reg, verbose)
    M =  Model(NoData(), loss, reg, DefaultSolver(), S,
               nothing, nothing, #X,Y
               "all", # featurelist
               nothing, #regweights
               false, #istrained
               verbose, #verbose
               true, # xydataisinvalid
               true  # disinvalid 
               )
    setdata(M)
    return M
end

##############################################################################

function Model(U, V; loss=SquareLoss(), reg=L2Reg(),
               Unames = nothing, Vnames = nothing,
               embedall = false, verbose=false, kwargs...)
    S = makeFrameSource(U, V, Unames, Vnames)
    M = Model(S, loss, reg, verbose)
    if embedall
        if M.verbose
            println("Model: applying default embedding")
        end
        defaultembedding(M; kwargs...)
    end
    return M
end




##############################################################################


function SplitData(X, Y, trainfrac)
    trainrows, testrows = splitrows(size(X,1), trainfrac)
    Xtrain = X[trainrows,:]
    Ytrain = Y[trainrows,:]
    Xtest = X[testrows,:]
    Ytest = Y[testrows,:]
    return SplitData(Xtrain, Ytrain, Xtest, Ytest, trainrows, testrows, trainfrac, NoResults())
end




function FoldedData(X, Y, nfolds)
    foldrows, nonfoldrows = getfoldrows(size(X,1), nfolds)
    return FoldedData(X, Y, nfolds, foldrows, nonfoldrows, NoResults())
end

##############################################################################


function splittraintestx(M, trainfrac)
    setdata(M)
    if M.verbose
        println("Model: splitting data")
    end
    M.D = SplitData(M.X, M.Y, usetrainfrac(M, trainfrac))
    M.disinvalid = false
end

function usetrainfrac(M::Model, trainfrac)
    if trainfrac==nothing
        if isa(M.D, SplitData)
            return M.D.trainfrac
        end
        return 0.8
    end
    return trainfrac
end
        
function splittraintest(M::Model; trainfrac=nothing, resplit=false, force=false)
    if resplit || force
        M.disinvalid = true
    end
    if !M.disinvalid && isa(M.D, SplitData)
        if trainfrac != nothing && trainfrac != M.D.trainfrac
            M.disinvalid = true
        end
    end
    if M.disinvalid ||  !isa(M.D, SplitData)
        splittraintestx(M, trainfrac)
    end
end

function splitfolds(M::Model, nfolds; resplit=false, force=false)
    if resplit || force 
        M.disinvalid = true
    end
    if !M.disinvalid && isa(M.D, FoldedData)
        if M.D.nfolds != nfolds
            M.disinvalid = true
        end
    end

    if M.disinvalid || !isa(M.D, FoldedData)
        setdata(M)
        M.D = FoldedData(M.X, M.Y, nfolds)
        M.disinvalid = false
    end
end




##############################################################################
# fit

function trainx(M::Model, lambda, Xtrain, Xtest, Ytrain, Ytest; theta_guess = nothing)
    assignsolver(M)
    if M.verbose
        println("Model: calling solver: ", M.solver)
        for i=1:length(M.regweights)
            if M.regweights[i] == 0
                println("Model: Not regularizing constant feature X[:,$(i)]")
            end
        end
    end
    theta = solve(M.solver, M.loss, M.regularizer, M.regweights, Xtrain, Ytrain, lambda;
                  theta_guess = theta_guess)
    trainloss = loss(M.loss, predict(M, Xtrain, theta), Ytrain)
    testloss = loss(M.loss,  predict(M, Xtest,  theta), Ytest)
    return PointResults(theta, lambda, trainloss, testloss)
end


function trainfoldsx(M::Model, lambda, nfolds)
    results = Array{PointResults}(nfolds)
    for i=1:nfolds
        results[i] =  trainx(M, lambda, Xtrain(M,i), Xtest(M,i), Ytrain(M,i), Ytest(M,i))
    end
    return FoldResults(results)
end


function trainpathx(M::Model, lambda::Array; quiet=true, kwargs...)
    m = length(lambda)
    results = Array{PointResults}(m)
    for i=1:m
        if i>1
            tg = results[i-1].theta
        else
            tg = nothing
        end
        results[i] = trainx(M, lambda[i], Xtrain(M), Xtest(M), Ytrain(M), Ytest(M);
                            theta_guess = tg)
    end
    imin =  findmin([x.testloss for x in results])[2]
    return RegPathResults(results, imin)
end


function trainfolds(M::Model; lambda=1e-10, nfolds=5,
                    resplit=false, features=nothing, kwargs...)
    splitfolds(M, nfolds, resplit)
    if features != nothing
        setfeatures(M, features)
    end
    M.D.results = trainfoldsx(M, lambda, nfolds)
    M.istrained = true
    if M.verbose
        status(M)
    end
end


function trainpath(M::Model; lambda=logspace(-5,5,100), trainfrac=0.8,
                   resplit=false, features=nothing, kwargs...)
    splittraintest(M; trainfrac=trainfrac, resplit=resplit)
    if features != nothing
        setfeatures(M, features)
    end
    M.D.results = trainpathx(M, lambda)
    M.istrained = true
    if M.verbose
        status(M)
    end
end


function train(M::Model; lambda=1e-10, trainfrac=nothing,
               resplit=false, features=nothing, kwargs...)
    splittraintest(M; trainfrac=trainfrac, resplit=resplit)
    if features != nothing
        setfeatures(M, features)
    end
    M.D.results = trainx(M, lambda, Xtrain(M), Xtest(M), Ytrain(M), Ytest(M))
    M.istrained = true
    if M.verbose
        status(M)
    end
end

##############################################################################

function assignsolver(M::Model, force=false)
    if force || isa(M.solver, DefaultSolver) 
        M.solver = getsolver(M.loss, M.regularizer)
    end
end


function setfeatures(M::Model, f)
    M.featurelist = f
    setregweights(M)
end

function setloss(M::Model, l)
    M.loss = l
    assignsolver(M, true)
end
function setreg(M::Model, r)
    M.regularizer = r
    assignsolver(M, true)
end
function setsolver(M::Model, s)
    if s == "default"
        assignsolver(M, true)
        return
    end
    M.solver = s
end
    


##############################################################################
# querying
function selectfeatures(M::Model, X)
    if M.featurelist == "all" 
        return X
    end
    if size(X,2) ==0
        return X
    end
    return X[:,M.featurelist]
end

getU(M::Model) = getU(M.S)
getV(M::Model) = getV(M.S)
getXY(M::Model) = getXY(M.S)

#getd(M::Model) = getd(M.D)
#getd(D::SplitData) = size(D.Xtrain,2)
#getd(D::FoldedData) = size(D.X,2)
    


Xtest(M::Model) = selectfeatures(M, Xtest(M.D))
Xtrain(M::Model) = selectfeatures(M, Xtrain(M.D))
Xtrain(D::SplitData) = D.Xtrain
Xtest(D::SplitData) = D.Xtest
Utrain(M::Model) = getU(M.S)[M.D.trainrows,:]
Utest(M::Model)  = getU(M.S)[M.D.testrows,:]
Vtrain(M::Model) = getV(M.S)[M.D.trainrows,:]
Vtest(M::Model)  = getV(M.S)[M.D.testrows,:]


Xtest(M::Model, fold) = selectfeatures(M, Xtest(M.D, fold))
Xtrain(M::Model, fold) = selectfeatures(M, Xtrain(M.D, fold))
Xtrain(D::FoldedData, fold) = D.X[D.nonfoldrows[fold],:]
Ytrain(D::FoldedData, fold) = D.Y[D.nonfoldrows[fold],:]
Xtest(D::FoldedData, fold) = D.X[D.foldrows[fold],:]
Ytest(D::FoldedData, fold) = D.Y[D.foldrows[fold],:]
Ytest(M::Model, fold) = Ytest(M.D, fold)
Ytrain(M::Model, fold) = Ytrain(M.D, fold)

Ytest(M::Model) = Ytest(M.D)
Ytrain(M::Model) = Ytrain(M.D)
Ytrain(D::SplitData) = D.Ytrain
Ytest(D::SplitData) = D.Ytest

import Base.split
split(M::Model; kwargs...) = splittraintest(M; force=true, kwargs...)

##############################################################################
# querying results

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

testloss(M::Model) = testloss(M.D.results)
trainloss(M::Model) = trainloss(M.D.results)
thetaopt(M::Model) = thetaopt(M.D.results)
lambda(M::Model) = lambda(M.D.results)

lambdaopt(M::Model) = lambdaopt(M.D.results)

lambdapath(M::Model) = lambdapath(M.D.results)
testlosspath(M::Model) = testlosspath(M.D.results)
trainlosspath(M::Model) = trainlosspath(M.D.results)
thetapath(M::Model) = thetapath(M.D.results)

##############################################################################

function addfeatureU(M::Model; rebuild=true, kwargs...)
    M.xydataisinvalid = true
    M.disinvalid = true
    setfeatures(M, "all")
    addfeatureU(M.S, nothing; kwargs...)
    if rebuild
        setdata(M)
    end
end

function addfeatureV(M::Model; rebuild=true, kwargs...)
    M.xydataisinvalid = true
    M.disinvalid = true
    setfeatures(M, "all")
    addfeatureV(M.S, nothing; kwargs...)
    if rebuild
        setdata(M)
    end
end

function addfeatureU(M::Model, col; rebuild=true, kwargs...)
    M.xydataisinvalid = true
    M.disinvalid = true
    setfeatures(M, "all")
    addfeatureU(M.S, col; kwargs...)
    if rebuild
        setdata(M)
    end
end

function addfeatureV(M::Model, col; rebuild=true, kwargs...)
    M.xydataisinvalid = true
    M.disinvalid = true
    setfeatures(M, "all")
    addfeatureV(M.S, col; kwargs...)
    if rebuild
        setdata(M)
    end
end



##############################################################################

function status(io::IO, R::RegPathResults)
    println(io, "----------------------------------------")
    println(io, "Optimal results along regpath")
    println(io, "  optimal lambda: ",  lambdaopt(R))
    println(io, "  optimal test loss: ", testloss(R))
end

function status(io::IO, R::PointResults)
    println(io, "----------------------------------------")
    println(io, "Results for single train/test")
    println(io, "  training  loss: ", trainloss(R))
    println(io, "  test loss: ", testloss(R))
end

"""
Prints and returns the status of the model.
"""
function status(io::IO, M::Model)
    status(io, M.D.results)
    println(io, "  training samples: ", length(Ytrain(M)))
    println(io, "  test samples: ", length(Ytest(M)))
    println(io, "  columns in X: ", size(M.X,2))
    println(io, "----------------------------------------")
end

status(M::Model; kwargs...)  = status(STDOUT, M; kwargs...)

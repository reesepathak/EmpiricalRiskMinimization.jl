

##############################################################################
# Data

# Note both X and Y are 2 dimensional Arrays

mutable struct Mldata
    Xtrain
    Ytrain
    Xtest
    Ytest
    Xembed
    Yembed
    # feature list is a list of the columns of X that are of interest
    featurelist
    # true if the first feature is 1
    hasconstfeature::Bool
end

function Xtrain(D::Mldata)
    if D.featurelist == nothing
        return D.Xtrain
    end
    return D.Xtrain[:, D.featurelist]
end

function Xtest(D::Mldata)
    if D.featurelist == nothing
        return D.Xtest
    end
    return D.Xtest[:, D.featurelist]
end


Ytrain(D::Mldata) = D.Ytrain
Ytest(D::Mldata) = D.Ytest



function setfeatures(D::Mldata, f)
    if f == "all"
        D.featurelist = nothing
    else
        D.featurelist = f
    end
end
    


# apply embeddings
function Mldata(U, V; Xembed = false, Yembed = false, trainfrac=0.5, splitmethod=0)
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
    Xtrain, Ytrain, Xtest, Ytest = splittestandtrain(X, Y;
                                                     trainfrac=trainfrac,
                                                     splitmethod=splitmethod)        
    d = Mldata(Xtrain, Ytrain, Xtest, Ytest, Xembed, Yembed, nothing, hasconstfeature)
    return d
end

# split data
function splittestandtrain(X, Y; trainfrac=0.5, splitmethod=0)
    # split into test and train
    if splitmethod == 0
        n = size(X,1)
        ntrain = convert(Int64, round(trainfrac*n))
        p = randperm(n)
        trainrows = p[1:ntrain]
        testrows = p[ntrain+1:n]
    else
        # pick by Bernoulli
        testrows = Int64[]
        trainrows = Int64[]
        for i=1:size(X,1)
            if rand()>trainfrac
                push!(testrows,i)
            else
                push!(trainrows,i)
            end
        end
    end
    
    Xtrain = X[trainrows,:]
    Ytrain = Y[trainrows,:]
    Xtest = X[testrows,:]
    Ytest = Y[testrows,:]
    return Xtrain, Ytrain, Xtest, Ytest
end


##############################################################################

mutable struct Model
    D::Mldata
    loss::Loss
    regularizer::Regularizer
    solver::Solver
    theta
    lambda
    trainloss
    testloss
    lambdamin
    imin
    regweights   # e.g. regularization = sum_i regweights(i) \abs(\theta_i)
                 # so we can set regweights(1) = 0 when x_1 = 1
    istrained::Bool
    hasregpath::Bool
    

    # todo: each solve on the regularization path should have its own status
    status::String


end

setfeatures(M::Model, f) = setfeatures(M.D, f)
Ytest(M::Model) = Ytest(M.D)
Ytrain(M::Model) = Ytrain(M.D)
Xtest(M::Model) = Xtest(M.D)
Xtrain(M::Model) = Xtrain(M.D)

"""
Alternative constructor for the Model class, simply requiring 
specification of data, loss and regularizer.
"""
function Model(D, loss, reg)
    # dimension of theta
    n = size(Xtrain(D),2)
    regweights = ones(n)
    if D.hasconstfeature
        regweights[1] = 0
    end
    return  Model(D, loss, reg, DefaultSolver(),
                  0,0,0,0,0,0,
                  regweights,
                  false, false, "initial status")
end

##############################################################################
# predict


function gettheta(M::Model)
    if !M.istrained
        error("Attempt to retrieve theta from untrained ERM")
    end
    if M.hasregpath
        return M.theta[M.imin]
    end
    return M.theta
end

function getlambda(M::Model)
    if !M.istrained
        error("Attempt to retrieve lambda from untrained ERM")
    end
    if M.hasregpath
        return M.lambda[M.imin]
    end
    return M.lambda
end


predict(M::Model, X::Array{Float64,2}, theta) = X*theta

predicttest(M::Model, theta) = predict(M, Xtest(M.D), theta)
predicttrain(M::Model, theta) = predict(M, Xtrain(M.D), theta)

predicttest(M::Model) = predict(M, Xtest(M.D), gettheta(M))
predicttrain(M::Model) = predict(M, Xtrain(M.D), gettheta(M))

predictx(M::Model, x) = predict(M, x, gettheta(M))

function predictu(M::Model, u::Array{Float64,2})
    x = embed(M.D.Xembed, u)
    predictx(M::Model, x)
end


##############################################################################
# fit


function trainx(M::Model,  lambda::Number)
    theta = solve(M.solver, M.loss, M.regularizer, M.regweights,
                  Xtrain(M.D), M.D.Ytrain, lambda)
    trainloss = loss(M.loss, predicttrain(M, theta), M.D.Ytrain)
    testloss = loss(M.loss,  predicttest(M, theta), M.D.Ytest)
    return theta, trainloss, testloss
end

function train(M::Model, lambda::Number; quiet=false)
    if isa(M.solver, DefaultSolver)
        M.solver = getsolver(M.loss, M.regularizer)
    end
    M.lambda = lambda
    M.theta, M.trainloss, M.testloss = trainx(M,  lambda)
    M.istrained = true
    M.hasregpath = false
    if !quiet
        @printf("lambda = %f, training loss = %.4f\n", lambda, M.trainloss)
        @printf("lambda = %f, test loss = %.4f\n", lambda, M.testloss)
    end
end

train(M::Model; kwargs...) = train(M, 1e-10; kwargs...)    

function train(M::Model, lambda::Array; quiet=false, kwargs...)
    if isa(M.solver, DefaultSolver)
        M.solver = getsolver(M.loss, M.regularizer)
    end
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




"""
Prints and returns the status of the model.
"""
function status(M::Model)
    #println("The model status is: $(M.status).")
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
    
    return M.status
end



"""
Returns the final test loss of the model.
"""
function testloss(M::Model)
    return M.testloss
end

"""
Returns the final training loss of the model.
"""
function trainloss(M::Model)
    return M.trainloss
end

"""
Returns model parameters 
"""
function thetaopt(M::Model)
    return M.theta
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

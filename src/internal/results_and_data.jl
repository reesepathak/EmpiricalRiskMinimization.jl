##############################################################################
# Results

abstract type Results end


mutable struct RegPathResults<:Results
    results    # list of PointResults
    imin       # index of lambda that minimizes test loss
end

mutable struct PointResults<:Results
    theta      # optimal theta
    lambda     # lambda used
    trainloss  # number
    testloss   # number
end

mutable struct FoldResults<:Results
    results    # list of results, one per fold
end

mutable struct NoResults<:Results end

##############################################################################
# Data

abstract type Data end


mutable struct FoldedData<:Data
    X
    Y
    nfolds
    foldrows
    nonfoldrows
    results::Results
end

# Note both X and Y are 2 dimensional Arrays
mutable struct SplitData<:Data 
    Xtrain
    Ytrain
    Xtest
    Ytest
    trainrows
    testrows
    trainfrac
    results::Results
end

mutable struct NoData<:Data end

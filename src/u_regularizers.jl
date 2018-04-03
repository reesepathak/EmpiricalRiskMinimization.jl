
abstract type RegularizerUnsupervised <: Regularizer end

##############################################################################
# todo: replace with function of xhat,x

#########################################
# Trivial Matrix Regularizer
#########################################
struct NoReg <: RegularizerUnsupervised
end

eval(R::NoReg, X) = 0
prox(R::NoReg, X) = X


#########################################
# Nonnegative Matrix Regularizer
#########################################
struct PosReg <: RegularizerUnsupervised
end

eval(R::PosReg, X) = any(X .< 0) ? Inf : 0

function prox(R::PosReg, X)
    Q = copy(X)
    Q[Q.<0] = 0
    return Q
end


abstract type LossUnsupervised <: Loss end


##############################################################################
# todo: loss should be a function of x and xhat vectors

#########################################
# Frobenius norm loss
#########################################

struct FrobeniusLoss <: LossUnsupervised
end

function eval(L::FrobeniusLoss, C, X, Y)
    return vecnorm(X*Y' - C)^2
end

# TODO: Find some way of more nicely enumerating these things
function deriv(L::FrobeniusLoss, C, X, Y, which="X")
    if which=="X"
        return 2*(X*Y' - C)*Y
    end
    if which == "Y"
        return 2*(Y*X' - C')*X
    end
    throw("which is defined as $(which), not one of X or Y")
end

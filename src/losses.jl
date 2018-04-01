abstract type Loss end
abstract type LossUnsupervised <: Loss end
abstract type LossNonDiff <: Loss end

#########################################
# Square Loss
#########################################
struct SquaredLoss <: Loss
    weight::Float64
end

SquaredLoss() = SquaredLoss(1.0)

function eval(L::SquaredLoss, X, y, theta)
    return L.weight*norm(X*theta - y)^2
end

function deriv(L::SquaredLoss, X, y, theta)
    return L.weight*(2*X'*(X*theta - y))
end

#########################################
# L1 Loss
#########################################
struct AbsoluteLoss <: LossNonDiff
    weight::Float64
end

AbsoluteLoss() = AbsoluteLoss(1.0)

function eval(L::AbsoluteLoss, X, y, theta)
    return L.weight*norm(X*theta - y, 1)
end

function deriv(L::AbsoluteLoss, X, y, theta)
    return L.weight*X'*sign.(X*theta - y)
end

margin(X, y, theta) = X*theta .* y

#########################################
# Hinge Loss
#########################################
struct HingeLoss <: LossNonDiff
    weight::Float64
end

HingeLoss() = HingeLoss(1.0)

eval(L::HingeLoss, X, y, theta) = L.weight*sum(max.(1 - margin(X, y, theta), 0))

function deriv(L::HingeLoss, X, y, theta)
    u = y.*(1.0*(margin(X, y, theta) .<= 1.0))
    return -L.weight*X'*u
end

#########################################
# Logistic Loss
#########################################
struct LogisticLoss <: Loss
    weight::Float64
end

LogisticLoss() = LogisticLoss(1.0)

eval(L::LogisticLoss, X, y, theta) = L.weight*sum(log.(1 + exp.(-margin(X, y, theta))))

function deriv(L::LogisticLoss, X, y, theta)
    return -L.weight*X'*(y./(1 + exp.(margin(X, y, theta))))
end


#########################################
# Huber Loss
#########################################
struct HuberLoss <: Loss
    delta::Float64
    weight::Float64
end

HuberLoss() = HuberLoss(1.0, 1.0)
HuberLoss(t) = HuberLoss(t, 1.0)

function eval(L::HuberLoss, X, y, theta)
    u = X*theta - y
    abs_error = abs.(u)
    quadratic = min.(abs_error, L.delta)
    linear = abs_error - quadratic
    losses = 0.5 * quadratic .^ 2 + L.delta*linear
    return L.weight*sum(losses)
end

function deriv(L::HuberLoss, X, y, theta)
    u = X*theta - y
    ind_sq = abs.(u) .<= L.delta
    ind_abs = abs.(u) .> L.delta
    X_sq = ind_sq .* X
    X_abs = ind_abs .* X
    sq_deriv = X_sq'*X_sq*theta - X_sq'*y
    abs_deriv = L.delta*sum(sign.(u) .* X_abs, 1)'
    return L.weight*(sq_deriv + abs_deriv)
end


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

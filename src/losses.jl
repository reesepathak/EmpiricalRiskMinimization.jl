
abstract type Loss end
abstract type LossNonDiff <: Loss end

#########################################
# Square Loss
#########################################
struct SquareLoss <: Loss end

# assume  y and yhat are m-dimensional vectors (even if m=1)
function loss(L::SquareLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return dot(yhat - y, yhat-y)
end

# derivative wrt yhat
function derivloss(L::SquareLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return 2*(yhat - y)
end

#########################################
# L1 Loss
#########################################
struct AbsoluteLoss <: LossNonDiff end

function loss(L::AbsoluteLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return norm(yhat - y, 1)
end

function derivloss(L::AbsoluteLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return sign.(yhat - y)
end


#########################################
# Hinge Loss
#########################################
struct HingeLoss <: LossNonDiff end

loss(L::HingeLoss, yhat::Array{Float64,1}, y::Array{Float64,1}) = sum(max.(1 - yhat.*y, 0))

function derivloss(L::HingeLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return -y.*(1.0*(yhat.*y .<= 1.0))
end

#########################################
# Logistic Loss
#########################################
struct LogisticLoss <: Loss end

loss(L::LogisticLoss, yhat::Array{Float64,1}, y::Array{Float64,1}) = sum(log.(1 + exp.(-yhat.*y)))

function derivloss(L::LogisticLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return -(y./(1 + exp.(yhat.*y)))
end

#########################################
# Huber Loss
#########################################
struct HuberLoss <: Loss
    delta::Float64
end

HuberLoss() = HuberLoss(1.0)

function loss(L::HuberLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    u = yhat - y
    abs_error = abs.(u)
    quadratic = min.(abs_error, L.delta)
    linear = abs_error - quadratic
    losses = 0.5 * quadratic .^ 2 + L.delta*linear
    return L.weight*sum(losses)
end

function derivloss(L::HuberLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    u = yhat - y
    ind_sq = abs.(u) .<= L.delta
    ind_abs = abs.(u) .> L.delta
    X_sq = ind_sq 
    X_abs = ind_abs 
    sq_deriv = X_sq'*X_sq - X_sq'*y
    abs_deriv = L.delta*sum(sign.(u) .* X_abs, 1)'
    return sq_deriv + abs_deriv
end

##############################################################################
# average loss

function loss(L::Loss, Yhat::Array{Float64,2}, Y::Array{Float64,2})
    l = 0
    n = length(Yhat)
    for i=1:n
        l += loss(L, Yhat[i,:], Y[i,:])
    end
    return l/n
end

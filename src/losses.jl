
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

# gradient wrt yhat
# returns the transpose of the derivative
function derivloss(L::SquareLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return 2*(yhat - y)
end

function cvxloss(L::SquareLoss,  yhat, y)
    return sumsquares(yhat - y)
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

function cvxloss(L::AbsoluteLoss,  yhat, y)
    return abs(yhat - y)
end


#########################################
# Tilted Loss
#########################################
struct TiltedLoss <: LossNonDiff
    tau
end

# only defined for scalars
function loss(L::TiltedLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    if length(y)>1
        error("tilted loss only applies to scalars")
    end
    e = yhat[1] - y[1]
    return 0.5*abs(e) + (L.tau - 0.5)*e
end

function derivloss(L::TiltedLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    e = yhat[1] - y[1]
    if e>0
        return L.tau
    end
    return L.tau - 1
end

function cvxloss(L::TiltedLoss, yhat, y)
    e = yhat - y
    return 0.5*abs(e) + (L.tau - 0.5)*e
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
    alpha::Float64
end

HuberLoss() = HuberLoss(1.0)

function loss(L::HuberLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    if length(yhat)>1
        error("Huber loss applies only to scalars")
    end
    u = yhat[1]-y[1]
    if abs(u) < L.alpha
        return u*u
    end
    return L.alpha*(2*abs(u)-L.alpha)
end


function derivloss(L::HuberLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    u = yhat[1] - y[1]
    if abs(u) < L.alpha
        return 2*u
    end
    return 2*L.alpha*sign(u)
end

function cvxloss(L::HuberLoss,  yhat, y)
    return huber(yhat - y, L.alpha)
end


##############################################################################
# Nonconvex Huber

struct NonconvexHuberLoss <: Loss
    alpha::Float64
end

NonconvexHuberLoss() = NonconvexHuberLoss(1.0)

function loss(L::NonconvexHuberLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    if length(yhat)>1
        error("Huber loss applies only to scalars")
    end
    u = yhat[1]-y[1]
    if abs(u) < L.alpha
        return u*u
    end
    y = L.alpha*L.alpha*(1-2*log(L.alpha) + 2*log(abs(u)) )
    return y
end


##############################################################################
#  Deadzone loss
struct DeadzoneLoss <: Loss
    alpha::Float64
end

DeadzoneLoss() = DeadzoneLoss(1.0)

function loss(L::DeadzoneLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    if length(yhat)>1
        error("Deadzone loss applies only to scalars")
    end
    u = yhat[1]-y[1]
    return max(abs(u)-L.alpha, 0)
end

function cvxloss(L::DeadzoneLoss,  yhat, y)
    e = yhat - y
    return max(abs(e)- L.alpha, 0)
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

############################################################################## 

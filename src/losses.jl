abstract type Loss end
abstract type LossNonDiff <: Loss end 
abstract type LossDiff <: Loss end

#########################################
# Square Loss
#########################################
"`SquareLoss()` constructs the l2/squared loss. Use with `Model()`."
struct SquareLoss <: LossDiff end

# assume  y and yhat are m-dimensional vectors (even if m=1)
function loss(L::SquareLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return dot(yhat - y, yhat - y)
end

# gradient wrt yhat
# returns the transpose of the derivative
function derivloss(L::SquareLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return 2*(yhat - y)
end

function cvxloss(L::SquareLoss,  yhat, y)
    return sumsquares(yhat - y)
end

##############################################################################
# multiclass support functions

function margin(pi, pj, y)
    return dot(pj-pi,y) + 0.5*(dot(pi,pi) - dot(pj,pj))
end

# gradient of margin wrt y
function dmargin(pi, pj, y)
    return pj-pi
end

function findclosest(reps, y)
    dsqmin = Inf
    imin = 0
    for i=1:length(reps)
        di = dot(y-reps[i], y-reps[i])
        if di < dsqmin
            imin = i
            dsqmin = di
        end
    end
    return imin
end

##############################################################################
# multiclass hinge loss


struct MultiHingeLoss <: LossNonDiff
    reps  # list of representative points in R^m
end


function loss(L::MultiHingeLoss, yhat::Array{T,1}, y::Array{Float64,1}) where {T<:Any}
    # psij = y
    K = length(L.reps)
    j = findclosest(L.reps, y)
    yrep = L.reps[j]
    s = zeros(K)
    for i=1:K
        if i != j
            s[i] = max(1-margin(L.reps[i], yrep, yhat), 0)
        else
            s[i] = -Inf
        end
    end
    return maximum(s)
end

function cvxloss(L::MultiHingeLoss, yhat, y)
    K = length(L.reps)
    j = findclosest(L.reps, y)
    yrep = L.reps[j]
    s = 0
    for i=1:K
        if i != j
            m = margin(L.reps[i], yrep, yhat)
            s = max(1-m, s)
        end
    end
    return s
end


########################################
# multiclass logistic loss


struct MultiLogisticLoss <: LossDiff
    reps  # list of representative points in R^m
end


function loss(L::MultiLogisticLoss, yhat::Array{T,1}, y::Array{Float64,1}) where {T<:Any}
    # psij = y
    K = length(L.reps)
    j = findclosest(L.reps, y)
    yrep = L.reps[j]
    s = 0.0
    for i=1:K
        s = s + exp(1-margin(L.reps[i], yrep, yhat))
    end
    return log(s)
end


function derivloss(L::MultiLogisticLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    K = length(L.reps)
    j = findclosest(L.reps, y)
    yrep = L.reps[j]
    s = 0.0
    se = 0.0
    for i=1:K
        e = exp(1-margin(L.reps[i], yrep, yhat))
        se += e
        s  += e * dmargin(L.reps[i], yrep, yhat)
    end
    return -s/se
end

#########################################
# L1 Loss
#########################################
"`AbsoluteLoss()` constructs the l1/absolute loss. Use with `Model()`."
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
# Huber Loss
#########################################
"""`HuberLoss()` constructs Huber loss. 
Use with `Model()`. 
Can also invoke as `HuberLoss(alpha)`, which allows specification 
of the tradeoff parameter `alpha` > 0. Note that `HuberLoss()` defaults 
to `alpha = 1.0`. 
"""
struct HuberLoss <: LossDiff
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
# Log Huber

struct LogHuberLoss <: LossDiff
    alpha::Float64
end

LogHuberLoss() = LogHuberLoss(1.0)

function loss(L::LogHuberLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    if length(yhat)>1
        error("Huber loss applies only to scalars")
    end
    e = yhat[1]-y[1]
    if abs(e) < L.alpha
        return e*e
    end
    y = L.alpha*L.alpha*(1-2*log(L.alpha) + 2*log(abs(e)) )
    return y
end

function derivloss(L::LogHuberLoss,  yhat::Array{Float64,1}, y::Array{Float64,1})
    e = yhat[1] - y[1]
    if abs(e) < L.alpha
        return 2*e
    end
    return  2*L.alpha*L.alpha/e

end


##############################
#  Deadzone loss
##############################

struct DeadzoneLoss <: LossNonDiff
    alpha::Float64
end

DeadzoneLoss() = DeadzoneLoss(1.0)

function loss(L::DeadzoneLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    if length(yhat)>1
        error("Deadzone loss applies only to scalars")
    end
    e = yhat[1]-y[1]
    return max(abs(e)-L.alpha, 0)
end

function cvxloss(L::DeadzoneLoss,  yhat, y)
    e = yhat - y
    return max(abs(e)- L.alpha, 0)
end

#########################################
# Hinge Loss TODO
#########################################
"`HingeLoss()` constructs the hinge loss (i.e., for SVM). Use with `Model()`."
struct HingeLoss <: LossNonDiff end

# should be different for m>1
loss(L::HingeLoss, yhat::Array{Float64,1}, y::Array{Float64,1}) = sum(max.(1 - yhat.*y, 0))

# scalars only
function cvxloss(L::HingeLoss, yhat, y)
    return max(1-yhat*y, 0)
end

#########################################
# Logistic Loss 
#########################################
"`LogisticLoss()` constructs the logistic loss for classification problems. Use with `Model()`"
struct LogisticLoss <: LossDiff end

loss(L::LogisticLoss, yhat::Array{Float64,1}, y::Array{Float64,1}) = sum(log.(1 + exp.(-yhat.*y)))

function derivloss(L::LogisticLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return -(y.*exp.(-yhat.*y))./(1 + exp.(-yhat.*y))
end


# Hubristic

struct HubristicLoss <: LossDiff
    alpha
end
function loss(L::HubristicLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    s = 0.0
    for i=1:length(yhat)
        s += hbr(-y[i]*yhat[i], L.alpha)
    end
    return s
end

function hbr(y, alpha)
    if y<-1
        return 0.0
    end
    if y<alpha-1
        return (y+1)^2
    end
    return alpha*(2-alpha+2*y)
end




#########################################
# Sigmoid Loss 
#########################################
"`SigmoidLoss()` constructs the sigmoid loss for classification problems. Use with `Model()`"
struct SigmoidLoss <: LossDiff end

loss(L::SigmoidLoss, yhat::Array{Float64,1}, y::Array{Float64,1}) = sum(1./(1 + exp.(yhat.*y)))

function derivloss(L::SigmoidLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    return -(y.*exp.(yhat.*y))./(1 + exp.(yhat.*y))./(1 + exp.(yhat.*y))
end


##############################################################################
# RocLoss
##############################################################################

struct DiffRocLoss <: LossDiff
    kappa
    ypos
    loss2
end

function loss(L::DiffRocLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    if abs(y[1] - L.ypos) < 1e-6
        return L.kappa*loss(L.loss2, yhat, y)
    end
    return loss(L.loss2, yhat, y)
end

function derivloss(L::DiffRocLoss, yhat::Array{Float64,1}, y::Array{Float64,1})
    if abs(y[1] - L.ypos) < 1e-6
        return L.kappa*derivloss(L.loss2, yhat, y)
    end
    return derivloss(L.loss2, yhat, y)
end



##############################################################################
# average loss


function loss(L::Loss, Yhat::Array{T,2}, Y::Array{Float64,2}) where {T<:Any}
    if size(Yhat,1) == 0
        return 0
    end
    n = size(Yhat, 1)
    s = 0.0
    for i=1:n
        s +=  loss(L, Yhat[i,:], Y[i,:])
    end
    return (1/n) * s
end

function addup(dtheta,X,s,i,d,m)
    for pi=1:d
        for pj=1:m
            dtheta[pi,pj] += X[i,pi]*s[pj]
        end
    end
end

# gradient of average loss wrt theta
function derivlosstheta(L::Loss, Yhat::Array{Float64,2}, Y::Array{Float64,2},
                        X::Array{Float64,2}, stochastic)
    n,d = size(X)
    m = size(Y,2)
    dtheta = zeros(d,m)
    batch = 200
    samples = rand(1:n, batch)
    if stochastic
        for i in samples
            s = derivloss(L, Yhat[i,:], Y[i,:])
            #dtheta += X[i,:]*s'
            addup(dtheta,X,s,i,d,m)
        end
        return dtheta/batch
    else
        for i in 1:n
            s = derivloss(L, Yhat[i,:], Y[i,:])
            #dtheta += X[i,:]*s'
            addup(dtheta,X,s,i,d,m)
        end
        return dtheta/n
    end
end

############################################################################## 

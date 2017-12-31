abstract type Loss end

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
    return L.weight*(2*X'*X*theta - 2*X'*y)
end

#########################################
# L1 Loss
#########################################
struct AbsoluteLoss <: Loss
    weight::Float64
end

AbsoluteLoss() = AbsoluteLoss(1.0)

function eval(L::AbsoluteLoss, X, y, theta)
    return L.weight*norm(X*theta - y, 1)
end

function deriv(L::AbsoluteLoss, X, y, theta)
    return L.weight*sum(sign.(X*theta - y) .* X, 1)'
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

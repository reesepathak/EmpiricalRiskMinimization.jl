abstract type Regularizer end

#########################################
# L1 Regularizer
#########################################
struct L1Reg <: Regularizer
    weight::Float64
end

L1Reg() = L1Reg(1.0)

eval(R::L1Reg, u) = R.weight * norm(u, 1)

function prox(R::L1Reg, grad_step, t)
    scale = R.weight*t
    return sign.(grad_step).*max.(0, abs.(grad_step) - scale)
end

#########################################
# L2 Regularizer
#########################################
struct L2Reg <: Regularizer
    weight::Float64
end

L2Reg() = L2Reg(1.0)

eval(R::L2Reg, u) = R.weight * norm(u)^2

function prox(R::L2Reg, grad_step, t)
    scale = R.weight*t
    n = norm(grad_step)
    if n == 0 return zeros(length(grad_step)) end
    return max((n - scale)/n, 0) * grad_step
end

#########################################
# Elastic net regularizer
#########################################
struct L1L2Reg <: Regularizer
    L1_weight::Float64
    L2_weight::Float64
end

eval(R::L1L2Reg, u) = R.L1_weight * norm(u, 1) + R.L2_weight * norm(u, 2)^2

function prox(R::L1L2Reg, grad_step, t)
    lambd1 = R.L1_weight*t
    lambd2 = R.L2_weight*t
    coeff = (1 + 2*lambd2)/lambd1
    thresh = lambd1^2/(1 + 2lambd2)^2
    u = (lambd1/(1 + 2lambd2)^2) * grad_step
    return coeff*(sign.(u) .* max.(0, abs.(u) - thresh))
end

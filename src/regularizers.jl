
abstract type Regularizer end

#########################################
# L1 Regularizer
#########################################
struct L1Reg <: Regularizer end

eval(R::L1Reg, u) = norm(u, 1)

function prox(R::L1Reg, grad_step, t)
    return sign.(grad_step).*max.(0, abs.(grad_step) - t)
end

#########################################
# L2 Regularizer
#########################################
struct L2Reg <: Regularizer end

eval(R::L2Reg, u) = dot(u,u)

function prox(R::L2Reg, grad_step, t)
    n = norm(grad_step)
    if n == 0
        return zeros(length(grad_step))
    end
    return max((n - t)/n, 0) * grad_step
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

abstract type Regularizer end

#########################################
# L1 Regularizer
#########################################
struct L1Reg <: Regularizer
    weight::Float64
end

L1Reg() = L1Reg(1.0)

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

function prox(R::L2Reg, grad_step, t)
    scale = R.weight*t
    norm = norm(grad_step)
    return (norm >= scale) ? (1 - scale/norm)*grad_step : zeros(length(grad_step))
end

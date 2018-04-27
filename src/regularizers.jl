
abstract type Regularizer end

# regularizers are separable
function reg(R::Regularizer, theta::Array{Float64,1})
    s = 0.0
    for i=1:length(theta)
        s += reg(R, theta[i])
    end
    return s
end

# construct prox of vector by applying prox elementwise
# include regweights
function prox(R::Regularizer, gamma::Float64,
              regweights::Array{Float64, 1},
              v::Array{Float64,1})
    n = length(v)
    p = zeros(v)
    for i=1:n
        if regweights[i] > 0
            p[i] = prox(R,  gamma/regweights[i], v[i])
        else
            p[i] = v[i]
        end
    end
    return p
end

#########################################
# L1 Regularizer
#########################################
struct L1Reg <: Regularizer end

reg(R::L1Reg, a::Float64) = abs(a)

# returns arg min ( R(x) + gamma*\norm(x-v) )
function prox(R::L1Reg, gamma::Float64, v::Float64)
    s = 0.5/gamma
    if x > s
        return x-s
    elseif x < -s
        return x+s
    end
    return 0
end




#########################################
# L2 Regularizer
#########################################

struct L2Reg <: Regularizer end
reg(R::L2Reg, a::Float64) = a*a
prox(R::L2Reg, gamma::Float64, v::Float64) = v*gamma/(1+gamma)


#########################################
# L2 Regularizer
#########################################
struct SqrtReg <: Regularizer end

regul(R::SqrtReg, u) = sqrt(abs(u))

function prox(R::SqrtReg, grad_step, t)
    return 1
end

#########################################
# Nonneg Regularizer
#########################################
struct NonnegReg <: Regularizer end

regul(R::NonnegReg, u) = 0

function prox(R::NonnegReg, grad_step, t)
    return 1
end

#########################################
# Elastic net regularizer
#########################################
struct L1L2Reg <: Regularizer
    L1_weight::Float64
    L2_weight::Float64
end

regul(R::L1L2Reg, u) = R.L1_weight * norm(u, 1) + R.L2_weight * norm(u, 2)^2

function prox(R::L1L2Reg, grad_step, t)
    lambd1 = R.L1_weight*t
    lambd2 = R.L2_weight*t
    coeff = (1 + 2*lambd2)/lambd1
    thresh = lambd1^2/(1 + 2lambd2)^2
    u = (lambd1/(1 + 2lambd2)^2) * grad_step
    return coeff*(sign.(u) .* max.(0, abs.(u) - thresh))
end


abstract type Regularizer end

# regularizers are separable
function reg(R::Regularizer, regweights::Array{Float64,1}, theta::Array{Float64,1})
    s = 0.0
    for i=1:length(theta)
        s += regweights[i]*reg(R, theta[i])
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
cvxreg(R::L1Reg, a) = abs(a)

# returns arg min ( R(x) + gamma*\norm(x-v)^2 )
function prox(R::L1Reg, gamma::Float64, v::Float64)
    s = 0.5/gamma
    if v > s
        return v-s
    elseif v < -s
        return v+s
    end
    return 0
end


#########################################
# L2 Regularizer
#########################################

struct L2Reg <: Regularizer end
reg(R::L2Reg, a::Float64) = a*a
prox(R::L2Reg, gamma::Float64, v::Float64) = v*gamma/(1+gamma)
cvxreg(R::L2Reg, a) = a*a

#########################################
# Sqrt Regularizer
#########################################
struct SqrtReg <: Regularizer end
reg(R::SqrtReg, u::Float64) = sqrt(abs(u))

# horrific approach to computing prox
function prox(R::SqrtReg, ga::Float64, vv::Float64)
    if vv<0
        return -prox(R, ga,-vv)
    end
    
    ff(x) = sqrt(abs(x)) + ga*(x-vv)*(x-vv)
    
    g = complex(ga)
    v = complex(vv)
    # roots of  4 g x^3 - 4  g v  x  + 1 = 0
    z1 = (2*g*v)/(3^(1/3)*(sqrt(3)*sqrt(27*g^4 - 64*g^6*v^3) - 9*g^2)^(1/3)) + (sqrt(3)*sqrt(27*g^4 - 64*g^6*v^3) - 9*g^2)^(1/3)/(2*3^(2/3)*g)
    z2 = -((1 + im*sqrt(3))*g*v)/(3^(1/3)*(sqrt(3)*sqrt(27*g^4 - 64*g^6*v^3) - 9*g^2)^(1/3)) - ((1 - im*sqrt(3))*(sqrt(3)*sqrt(27*g^4 - 64*g^6*v^3) - 9*g^2)^(1/3))/(4*3^(2/3)*g)
    z3 = -((1 - im*sqrt(3))*g*v)/(3^(1/3)*(sqrt(3)*sqrt(27*g^4 - 64*g^6*v^3) - 9*g^2)^(1/3)) - ((1 + im*sqrt(3))*(sqrt(3)*sqrt(27*g^4 - 64*g^6*v^3) - 9*g^2)^(1/3))/(4*3^(2/3)*g)

    x1 = z1*z1
    x2 = z2*z2
    x3 = z3*z3
    
    xmin = 0.0
    fmin = ff(xmin)
    if abs(imag(x1)) < 1e-10
        t = ff(real(x1))
        if t < fmin
            fmin = t
            xmin = real(x1)
        end
    end
    if abs(imag(x2)) < 1e-10
        t = ff(real(x2))
        if t < fmin
            fmin = t
            xmin = real(x2)
        end
    end
    if abs(imag(x3)) < 1e-10
        t = ff(real(x3))
        if t < fmin
            fmin = t
            xmin = real(x3)
        end
    end
    return xmin
end



    

#########################################
# Nonneg Regularizer
#########################################
struct NonnegReg <: Regularizer end
function reg(R::NonnegReg, a)
    if a >= 0
        return 0
    end
    return Inf
end
function prox(R::NonnegReg, gamma::Float64, v::Float64)
    if v>0
        return v
    end
    return 0
end



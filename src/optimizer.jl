
using Convex
using SCS

abstract type Solver end

struct QRSolver <: Solver end
struct DefaultSolver <: Solver end
struct CvxSolver <: Solver
    verbose
    eps
end

mutable struct ProxGradientSolver <: Solver
    verbose
    eps
    thetas
    gammas
    fgs
end

name(S::ProxGradientSolver) = "ProxGradientSolver"
name(S::CvxSolver) = "CvxSolver"

import Base.print
print(io, S::Solver) = print(io, name(S))
print(io::IO, S::Solver) = print(io, name(S))
print(io::IO, S::Solver) = print(io, name(S))

ProxGradientSolver(;verbose=true, eps=1e-12) = ProxGradientSolver(verbose, eps, nothing, nothing, nothing)
CvxSolver(;verbose=true, eps=1e-5) = CvxSolver(verbose, eps)


# we query for the solver when the user has
# asked for the default. We don't use Julia dispatch on type Solver
# since we want to use one solver for the entire
# regularization path
getsolver(L::SquareLoss, R::L2Reg) = QRSolver()
getsolver(L::LossDiff, R) = ProxGradientSolver()
getsolver(L::LossNonDiff, R) = CvxSolver()

# non convex losses or regularizers
getsolver(L::LogHuberLoss, R) = ProxGradientSolver()
getsolver(L::LossDiff, R::SqrtReg) = ProxGradientSolver()



##############################################################################
# CVX for huber

function solve(S::CvxSolver, L, R, regweights, X, Y, lambda;
                  theta_guess=nothing)
    n,d = size(X)
    theta = Convex.Variable(d)
    # we shouldn't need to specify Positive here, but
    # Convex.jl is unreliable without it
    losses = Convex.Variable(n, Positive())
    regs = Convex.Variable(d, Positive())
    problem = Convex.minimize( sum(losses)/n  + sum(regs))
    for i=1:n
        problem.constraints +=    losses[i] >= cvxloss(L, X[i,:]'*theta,  Y[i])
    end
    if isa(R, NonnegReg)
        for i=1:d
            if regweights[i] > 0
                problem.constraints +=    theta[i] >= 0
            end
            problem.constraints +=    regs[i] == 0
        end
    else
        for i=1:d
            problem.constraints +=    regs[i]   >= lambda * regweights[i] * cvxreg(R, theta[i])
        end
    end
    solve!(problem, SCSSolver(verbose=S.verbose, eps=S.eps))
    return theta.value
end



##############################################################################
# QR for quadratic case

function solve(s::QRSolver, L::SquareLoss, R::L2Reg, regweights,
               X, Y, lambda; theta_guess=nothing)
    return lsreg(X, Y, lambda, regweights)
end


# inputs:
#    X        n by d
#    y        n by 1 
#
# returns:
#    theta:   d by 1 vector
#
function ls(X,Y)
    Q,R = qr(X)
    theta = R\(Q'*Y)
    return theta
end


function lsreg(X, Y, lambda, regweights)
    d = size(X,2)
    A = sqrt(lambda) * diagm(sqrt.(regweights))
    Xh = [X; A]
    Yh = [Y; zeros(d)]
    theta = ls(Xh,Yh)
end
##############################################################################


#
# minimize (1/n) * sum loss(theta^T*x^i, y^i) + lambda*\sum_i regweights[i]*reg(theta_i)
#

function solve(S::ProxGradientSolver, L::Loss, R::Regularizer,
               regweights, X, Y, regparam; theta_guess = nothing)

    # assume Y is n by 1

    # params
    max_iters = 1000
    gamma_initial = 0.1


    d = size(X,2)
    n = size(X,1)
   
    # useful stuff
    # f = loss(theta)
    # g = sum(regparam*regweights*reg(theta))
    #
    # so we minimize f + g
    f(theta) = loss(L, matrix(X*theta), Y)
    g(theta) = reg(R, regparam*regweights, theta[:])
    gradf(theta) = dloss(L, matrix(X*theta), Y)
    proxg(gamma, v) = prox(R, gamma, regparam*regweights, v)
    function dloss(L, Yhat, Y)
        dtheta = zeros(d)
        for i=1:n
            s = derivloss(L, Yhat[i,:], Y[i,:])[1]
            dtheta += s*X[i,:]
        end
        return dtheta
    end

    # buffers
    thetas = zeros(d, max_iters)
    gammas = zeros(max_iters)
    fgs = zeros(max_iters)

    # initial conditions
    if theta_guess != nothing
        theta_initial = theta_guess
    else
        theta_initial = zeros(d)
    end

    # store first step
    thetas[:,1] = theta_initial
    gammas[1] = gamma_initial
    fgs[1] = f(theta_initial) + g(theta_initial)

    # make inner loop variables accessible
    gamma_next = 0.0
    theta_next = zeros(d)
    fg_next = 0.0
    k=1
    
    # loop
    for k=1:max_iters-1
        gamma = gammas[k]
        theta = thetas[:,k]
        fg = fgs[k]

        # printing
        if S.verbose
            @printf("%d   gamma: %f   loss: %f   reg: %f \n", k, gamma, f(theta), g(theta))
        end

        # line search
        while true
            v = theta - gradf(theta)/(2*gamma)
            theta_next = proxg(gamma, v)
            fg_next = f(theta_next) + g(theta_next)
            # should be <= not < else can get stuck if
            # g is an indicator function
            if fg_next <= fg
                # increase the step size and move to next step
                gamma_next = gamma/1.2
                break
            else
                # decrease the step size and try again
                gamma = gamma*2
            end
        end
        # save the variables
        thetas[:,k+1] = theta_next
        gammas[k+1] = gamma_next
        fgs[k+1] = fg_next
        
        # stopping criterion
        if fg - fg_next < S.eps
            break
        end
    end
    S.thetas = thetas[:,1:k+1]
    S.gammas = gammas[1:k+1]
    S.fgs = fgs[1:k+1]
    return matrix(thetas[:,k+1])
end




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
    max_iters
    min_iters
    gamma_initial
    thetas
    gammas
    gradfs
    fs
    gs
    progress
    stochastic
end

name(S::ProxGradientSolver) = "ProxGradientSolver"
name(S::CvxSolver) = "CvxSolver"
name(S::QRSolver) = "QRSolver"

import Base.print
print(io, S::Solver) = print(io, name(S))
print(io::IO, S::Solver) = print(io, name(S))


ProxGradientSolver(;verbose=false, eps=1e-6,
                   maxiters=1000, miniters=1, 
                   gamma_initial=0.1,
                   progress=nothing, stochastic=false) = ProxGradientSolver(verbose, eps, maxiters, miniters,
                                                                            gamma_initial, nothing, nothing,
                                                                            nothing, nothing, nothing, progress,
                                                                            stochastic)

CvxSolver(;verbose=false, eps=1e-5) = CvxSolver(verbose, eps)


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
    m = size(Y,2)
    theta = Convex.Variable(d,m)
    # we shouldn't need to specify Positive here, but
    # Convex.jl is unreliable without it
    losses = Convex.Variable(n, Positive())
    regs = Convex.Variable(d, m, Positive())
    problem = Convex.minimize( sum(losses)/n  + sum(regs))
    for i=1:n
        problem.constraints +=    losses[i] >= cvxloss(L, theta'*X[i,:],  Y[i,:])
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
            for j=1:m
                problem.constraints +=    regs[i]   >= lambda * regweights[i] * cvxreg(R, theta[i,j])
            end
        end
    end
    solve!(problem, SCSSolver(verbose=S.verbose, eps=S.eps))
    println("theta.value = ", theta.value)
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
    m = size(Y,2)
    A = sqrt(lambda) * diagm(sqrt.(regweights))
    Xh = [X; A]
    Yh = [Y; zeros(d,m)]
    theta = ls(Xh,Yh)
end
##############################################################################


#
# minimize (1/n) * sum loss(theta^T*x^i, y^i) + lambda*\sum_i regweights[i]*reg(theta_i)
#


function solve(S::ProxGradientSolver, L::Loss, R::Regularizer,
               regweights, X, Y, regparam; theta_guess = nothing)

    # assume Y is n by 1
    d = size(X,2)
    n = size(X,1)
    m = size(Y,2)

    # useful stuff
    # f = loss(theta)
    # g = sum(regparam*regweights*reg(theta))
    #
    # so we minimize f + g
    #
    # reshape to deal with matrix case
    #
    R1 = repmat(regweights, 1,m)
    rw = regparam*R1[:]
    f(theta) = loss(L, X*reshape(theta,d,m) , Y)
    gradf(theta) = derivlosstheta(L, X*reshape(theta,d,m), Y, X, S.stochastic)[:]
    g(theta) = reg(R, rw, theta[:])
    proxg(gamma, v) = prox(R, gamma, rw, v)

    tg = theta_guess
    if tg != nothing
        tg = tg[:]
    end
    
    theta =  proxgradient(d*m, f, gradf, g, proxg, S; theta_guess = tg)
    return reshape(theta,d,m)
end

# generic solver will work for any f and g
# gets parameters from S and stores results in S
function proxgradient(d, f, gradf, g, proxg, S; theta_guess=nothing)

    max_iters = S.max_iters
    min_iters = S.min_iters
    verbose = S.verbose
    eps = S.eps
    gamma_initial = S.gamma_initial
    
    # buffers
    thetas = zeros(d, max_iters)
    gammas = zeros(max_iters)
    gradfs = zeros(d, max_iters)
    fs = zeros(max_iters)
    gs = zeros(max_iters)

    # initial conditions
    if theta_guess != nothing
        theta_initial = theta_guess
    else
        theta_initial = zeros(d)
    end

    # store first step
    thetas[:,1] = theta_initial
    gammas[1] = gamma_initial
    fs[1] = f(theta_initial)
    gs[1] = g(theta_initial)
    gradfs[:,1] = gradf(theta_initial)
        
    # make inner loop variables accessible
    gamma_next = 0.0
    theta_next = zeros(d)
    f_next = 0.0
    g_next = 0.0
    gradf_next = zeros(d)
    k=1
    
    # loop
    for k=1:max_iters-1
        gamma = gammas[k]
        theta = thetas[:,k]
        fg = fs[k] + gs[k]


        # printing
        if verbose
            @printf("%d   gamma: %f   loss: %f   reg: %f \n", k, gamma, f(theta), g(theta))
        end

        gradfs[:,k] = gradf(theta)

        if S.stochastic
            v = theta - gradfs[:,k]/(2*gamma)
            theta_next = proxg(gamma, v)
            f_next = f(theta_next) 
            g_next = g(theta_next)
            gamma_next = gamma_initial*sqrt(k+1)

        else
            # line search
            while true
                v = theta - gradfs[:,k]/(2*gamma)
                theta_next = proxg(gamma, v)
                f_next = f(theta_next) 
                g_next = g(theta_next)

                
                # should be <= not < else can get stuck if g is an indicator function
                if f_next + g_next <= fg
                    # increase the step size and move to next step
                    gamma_next = gamma/1.2
                    break
                else
                    # decrease the step size and try again
                    gamma = gamma*2
                end
            end
        end

        
        # save the variables
        thetas[:,k+1] = theta_next
        gammas[k+1] = gamma_next
        fs[k+1] = f_next
        gs[k+1] = g_next

        if S.progress != nothing
            S.progress(k+1,theta_next)
        end

        if k > min_iters && ! S.stochastic && fg - (f_next + g_next) < eps
            break
        end

    end
    S.thetas = thetas[:,1:k+1]
    S.gammas = gammas[1:k+1]
    S.gradfs = gradfs[:,1:k+1]
    S.fs = fs[1:k+1]
    S.gs = gs[1:k+1]
    return thetas[:, k+1]
    
end





##############################################################################




using Convex

abstract type Solver end

struct QRSolver <: Solver end
struct DefaultSolver <: Solver end
struct CvxSolver <: Solver end



mutable struct FistaSolver <: Solver
    # every solve appends a dict to the list of solves
    # the dict contains solver steps and info
    solves
end
FistaSolver() = FistaSolver(Any[])

# we query for the solver when the user has
# asked for the default. We don't use Julia dispatch on type Solver
# since we want to use one solver for the entire
# regularization path
getsolver(L::SquareLoss, R::L2Reg) = QRSolver()
getsolver(L::Loss, R::Regularizer) = Fista()

getsolver(L::HuberLoss, R::L2Reg) = CvxSolver()
getsolver(L::DeadzoneLoss, R::L2Reg) = CvxSolver()
getsolver(L::AbsoluteLoss, R::L2Reg) = CvxSolver()
getsolver(L::TiltedLoss, R::L2Reg) = CvxSolver()

##############################################################################
# CVX for huber

function solve(s::CvxSolver, L::HuberLoss, R::L2Reg, regweights,
               X, Y, lambda,  theta_guess=nothing)
    n,d = size(X)
    A = diagm(regweights)
    theta = Convex.Variable(d)
    s = Convex.Variable(n)
    problem = Convex.minimize( sum(s)/n  + lambda*quadform(theta, A)  )
    for i=1:n
        problem.constraints +=    s[i] >= huber(X[i,:]'*theta - Y[i], L.alpha)
    end
    solve!(problem)
    return theta.value
end





function solve(s::CvxSolver, L::AbsoluteLoss, R::L2Reg, regweights,
               X, Y, lambda,  theta_guess=nothing)
    n,d = size(X)
    A = diagm(regweights)
    theta = Convex.Variable(d)
    s = Convex.Variable(n)
    problem = Convex.minimize( sum(s)/n + lambda*quadform(theta, A)   )
    for i=1:n
        problem.constraints +=    s[i] >= abs(X[i,:]'*theta - Y[i])
    end
    solve!(problem)
    return theta.value
end


function solve(s::CvxSolver, L::TiltedLoss, R::L2Reg, regweights,
               X, Y, lambda,  theta_guess=nothing)
    n,d = size(X)
    A = diagm(regweights)
    theta = Convex.Variable(d)
    s = Convex.Variable(n)
    e = Convex.Variable(n)
    problem = Convex.minimize( sum(s)/n  + lambda*quadform(theta, A)  )
    for i=1:n
        problem.constraints +=    e[i] == X[i,:]'*theta - Y[i]
        problem.constraints +=    s[i] >= L.tau*e[i]
        problem.constraints +=    s[i] >= (L.tau-1)*e[i]
    end
    solve!(problem)
    return theta.value
end


function solve(s::CvxSolver, L::DeadzoneLoss, R::L2Reg, regweights,
               X, Y, lambda,  theta_guess=nothing)
    n,d = size(X)
    A = diagm(regweights)
    theta = Convex.Variable(d)
    s = Convex.Variable(n)
    problem = Convex.minimize( (1/n)*sum(s)  + lambda*quadform(theta, A) )
    for i=1:n
        problem.constraints +=    s[i] >= max(abs(X[i,:]'*theta - Y[i]) - L.alpha, 0)
    end
    solve!(problem)
    return theta.value
end

##############################################################################
# QR for quadratic case

function solve(s::QRSolver, L::SquareLoss, R::L2Reg, regweights,
               X, Y, lambda, theta_guess=nothing)
    return lsreg(X, Y, lambda, regweights)
end


# inputs:
#    X        n by d
#    y        n vector
#
# returns:
#    theta:   d vector
#
function ls(X,Y)
    Q,R = qr(X)
    theta = R\(Q'*Y)
    return theta
end


#
# this doesn't regularize the first component of theta
# todo: need to make this optional
#
function lsreg(X, Y, lambda, regweights)
    d = size(X,2)
    A = diagm(regweights)
    Xh = [X; sqrt(lambda)*A]
    Yh = [Y; zeros(d)]
    theta = ls(Xh,Yh)
end

##############################################################################
# Fista

function solve(s::FistaSolver, L::Loss, R::Regularizer, regweights,
               X, Y, lambda, theta_guess=nothing)
    beta=0.8
    alpha=0.5
    init=theta_guess
    t_init=1.0
    max_iters=5000
    tol=1e-4
    optstatus = fista(M.loss, M.reg, X, y,
                beta, alpha, init, t_init;
                max_iters=max_iters, tol=tol)
    
    if optstatus == -1
        return
    end
    thetas, losses = opt
    
    solverdata = Dict{Any,Any}()
    solverdata["thetas"] = thetas
    solverdata["losses"] = losses
    push!(s.solves, solverdata)
    
    return thetas[end]
end


"""
Main function in ERM package: carries out 
accelerated proximal (sub)gradient descent. In particular, this is 
carrying out FISTA, a particular instance of Nesterov's accelerated 
gradient method.
"""
function fista(L::Loss, R::Regularizer, X, y, beta=0.8, alpha=0.5,
                  init=nothing, t_init=1.0;
                  max_iters=5000, verbose=true, tol=1e-8)
    n, d = size(X)
    decay = (typeof(L) == LossNonDiff) ? true : false
    println("Solving problem. $n samples, $d features.")
    # convenience functions
    LOSS(u) = loss(L, X, y, u);
    GRAD(u) = derivloss(L, X, y, u)
    RISK(u) = LOSS(u) + loss(R, u);
    PROX(u, t) = prox(R, u, t)
    thetas, zetas, losses = [], [], []
    if init == nothing
        init = rand(d)
    end
    assert(length(init) == d)
    push!(thetas, init)
    push!(zetas, thetas[1])
    t = t_init
    for k = 1:max_iters
        if decay
            t /= sqrt(k)
        end
        lambd = 2/(k + 1)
        phi = (1 - lambd) * thetas[end] + lambd*zetas[end]
        z = PROX(phi - t*GRAD(phi), t)
        while LOSS(z) > LOSS(phi) + dot(GRAD(phi), z - phi) + 1/(2t)*norm(z - phi)^2
            t *= beta
            z = PROX(phi - t*GRAD(phi), t)
        end 
        push!(thetas, z)
        push!(zetas, thetas[end - 1] + 1/lambd * (thetas[end] - thetas[end-1]))
        push!(losses, RISK(thetas[end]))
        if verbose
            println("Iteration: $k,  Loss: $(losses[end])")
        end
        if k > 4 && maximum(abs.(losses[end-4:end-1] - losses[end-3:end])) < tol
            println("Done.")
            return thetas, losses
        end
    end
    print("Did not converge. Loss: $(losses[end])")
    return -1
end


abstract type Solver end

struct QRSolver <: Solver end
struct DefaultSolver <: Solver end



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





##############################################################################
# QR for quadratic case

function solve(s::QRSolver, L::SquareLoss, R::L2Reg, X, Y, lambda, theta_guess=nothing)
    return lsreg(X, Y, lambda)
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
function lsreg(X, Y, lambda)
    d = size(X,2)
    A = eye(d)
    A[1,1]=0
    Xh = [X; sqrt(lambda)*A]
    Yh = [Y; zeros(d)]
    theta = ls(Xh,Yh)
end

##############################################################################
# Fista

function solve(s::FistaSolver, L::Loss, R::Regularizer, X, Y, lambda, theta_guess=nothing)
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
        M.status = "Failed"
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

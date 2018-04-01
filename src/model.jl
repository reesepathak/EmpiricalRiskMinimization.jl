abstract type AbstractModel end

mutable struct Model <: AbstractModel
    loss::Loss
    reg::Regularizer
    intercept::Bool
    param::Dict{Any, Any}
    status::String
end

"Alternative constructor for the Model class, simply requiring specification of loss and regularizer."
Model(loss, reg; fit_intercept=false) = Model(loss, reg, fit_intercept, Dict(), "No data")

# Model(;loss=:required,
#        reg=:required,
#        fit_intercept=false)
#        = Model(loss, reg, fit_intercept)

"""
Function applies changes to the model object by carrying out empirical risk minimization. Requires 
input of data
"""
function fit!(M::Model, X, y; beta=0.8, alpha=0.5, init=nothing, t_init=1.0, max_iters=5000, verbose=false, tol=1e-4)
    n, d = size(X)
    if M.intercept
        X = [X ones(n)]
        d += 1
    end
    M.param["theta"] = zeros(d)
    opt = EmpiricalRiskMinimization.minimize(M.loss, M.reg, X, y, beta, alpha, init, t_init, max_iters=max_iters, verbose=verbose, tol=tol)
    if opt == -1
        println("Model did not converge")
        M.status = "Failed"
        return
    end
    thetas, losses = opt
    M.param["theta_history"] = thetas
    M.param["theta"] = M.param["theta_history"][end]
    M.param["loss_history"] = losses
    M.status = "Converged"
end

"""
Function fits an unsupervised model. Will be deprecated in next version. 
"""
function fit_unsupervised!(M::Model, X, k; beta=0.8, alpha=0.5, init=nothing, t_init=1.0, max_iters=5000, verbose=false, tol=1e-4)
    n, d = size(X)
    if M.intercept
        X = [X ones(n)]
        d += 1
    end

    M.param["theta"] = Dict("X"=> zeros(n, k), "Y"=>zeros(d, k))

    opt = EmpiricalRiskMinimization.minimize_unsupervised(M.loss, M.reg, X, k, beta, alpha, init, t_init, max_iters=max_iters, verbose=verbose, tol=tol)
    if opt == -1
        println("Model did not converge")
        M.status = "Failed"
        return
    end

    theta_X, theta_Y, losses = opt

    M.param["theta_history"] = Dict("X"=> theta_X, "Y"=>theta_Y)
    M.param["theta"] = Dict("X"=>M.param["theta_history"]["X"][end],
                            "Y"=>M.param["theta_history"]["Y"][end])
    M.param["loss_history"] = losses

    M.status = "Converged"
end

"""
Prints and returns the status of the model.
"""
function status(M::Model)
    println("The model status is: $(M.status).")
    return M.status
end

"""
Pritns and returns the final training risk of the model.
"""
function final_risk(M::Model)
    assert(M.status == "Converged")
    final_risk = M.param["loss_history"][end]
    println("Final risk: $(final_risk)")
    return final_risk
end

"""
Returns model parameters if the mdoel has converged (i.e., 
if it has been properly trained.)
"""
function parameters(M::Model)
    assert(M.status == "Converged")
    weights = M.param["theta"]
    return weights
end

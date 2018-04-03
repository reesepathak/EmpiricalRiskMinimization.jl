

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

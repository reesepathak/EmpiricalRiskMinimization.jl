function minimize(L::Loss, R::Regularizer, X, y, beta=0.8, alpha=0.5,
                  init=nothing, t_init=1.0;
                  max_iters=5000, verbose=true, tol=1e-8)
    n, d = size(X)
    println("Solving problem. $n samples, $d features.")
    # convenience functions
    LOSS(u) = eval(L, X, y, u); GRAD(u) = deriv(L, X, y, u)
    RISK(u) = LOSS(u) + eval(R, u)
    thetas, zetas, losses = [], [], []
    if init == nothing
        init = rand(d)
    end
    assert(length(init) == d)
    push!(thetas, init)
    push!(zetas, thetas[1])
    t = t_init
    for k = 1:max_iters
        grad_step = zetas[end] - t*GRAD(zetas[end])
        while LOSS(grad_step) > LOSS(zetas[end]) - alpha*t*norm(GRAD(zetas[end]))^2
            t *= beta
            grad_step = zetas[end] - t*GRAD(zetas[end])
        end
        push!(thetas, prox(R, grad_step, t))
        FISTA_weight = k/(k+3) #(k - 1)/(k + 2)
        push!(zetas, thetas[end] + FISTA_weight *(thetas[end] - thetas[end-1]))
        push!(losses, RISK(thetas[end]))
        if verbose
            println("Iteration: $k,  Loss: $(losses[end])")
        end
        if k > 4 && maximum(abs.(losses[end-4:end-1] - losses[end-3:end])) < tol
            println("Done.")
            return losses[end]
        end
    end
    print("Did not converge. Loss: $(losses[end])")
    return -1
end

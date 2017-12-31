function minimize(L::Loss, R::Regularizer, X, y, beta=0.8, alpha=0.5,
                  init=nothing, t_init=1.0,
                  max_iters=5000; verbose=true)
    n, d = size(X)
    println("Solving problem. $n samples, $d features.")
    # convenience functions
    LOSS(u) = eval(L, X, y, u); GRAD(u) = deriv(L, X, y, u)
    thetas, zetas, losses = [], [], []
    push!(thetas, rand(d))
    push!(zetas, thetas[1])
    t = t_init
    for k = 1:max_iters
        grad_step = zetas[end] - t*GRAD(zetas[end])
        while LOSS(grad_step) > LOSS(zetas[end]) - alpha*t*norm(GRAD(zetas[end]))^2
            t *= beta
            grad_step = zetas[end] - t*GRAD(zetas[end])
        end
        push!(thetas, prox(R, grad_step, t))
        FISTA_weight = (k - 1)/(k + 2)
        push!(zetas, thetas[end] + FISTA_weight *(thetas[end] - thetas[end-1]))
        push!(losses, LOSS(thetas[end]))
        if verbose
            println("Iteration: $k,  Loss: $(LOSS(thetas[end]))")
        end
        if k > 4 && maximum(abs.(losses[end-4:end-1] - losses[end-3:end])) < 1e-4
            println("Done.")
            break
        end
    end
end

using Base.LinAlg

# Alternating minimization schemes

function eval(C, X, Y)
    return vecnorm(X*Y' - C)
end

# TODO: Find some way of more nicely enumerating these things
function deriv(C, X, Y, which="X")
    if which == "X"
        return 2*(X*Y' - C)*Y
    end
    if which == "Y"
        return 2*(Y*X' - C')*X
    end
    throw("which is defined as $(which), not one of X or Y")
end

function optimize_pca(C, k, init=nothing, 
    t_init=1.0, t_min=1e-15, alpha=.5, beta=.8, verbose=true, max_iters=5000, tol=1e-8)

    n, d = size(C)
    decay = true
    println("Solving problem. Shape is $n by $d.")

    # convenience functions
    LOSS(u, v) = eval(C, u, v);
    GRAD_X(u, v) = deriv(C, u, v, "X")
    GRAD_Y(u, v) = deriv(C, u, v, "Y")

    
    theta_X, theta_Y, losses = [], [], []
    
    if init == nothing
        init_X = rand(n, k)
        init_Y = rand(d, k)
    else 
        init_X = init["X"]
        init_Y = init["Y"]
    end
    
    if verbose
        println("The initial loss is $(LOSS(init_X, init_Y))")
        println("With gradient X $(GRAD_X(init_X, init_Y))")
        println("With gradient Y $(GRAD_Y(init_X, init_Y))")
    end

    assert(size(init_X)[1] == n)
    assert(size(init_Y)[1] == d)
    assert(size(init_X)[2] == k && size(init_Y)[2] == k)

    push!(theta_X, init_X)
    push!(theta_Y, init_Y)

    tx, ty = t_init, t_init

    progress_x, progress_y = true, true

    for k = 1:max_iters
        # Step for X
        curr_grad_x = GRAD_X(theta_X[end], theta_Y[end])
        grad_step_X = theta_X[end] - tx*curr_grad_x

        progress_x = true
        prev_loss = LOSS(theta_X[end], theta_Y[end])

        while (LOSS(grad_step_X, theta_Y[end]) >= prev_loss)
            tx *= beta
            grad_step_X = theta_X[end] - tx*curr_grad_x
            if tx < t_min
                progress_x = false
                break
            end
        end
        tx *= 2
        
        push!(theta_X, grad_step_X)
        
        # Step for Y
        curr_grad_y = GRAD_Y(theta_X[end], theta_Y[end])
        grad_step_Y = theta_Y[end] - ty*curr_grad_y

        progress_y = true
        prev_loss = LOSS(theta_X[end], theta_Y[end])

        while (LOSS(theta_X[end], grad_step_Y) >= prev_loss)
            ty *= beta
            grad_step_Y = theta_Y[end] - ty*curr_grad_y
            if ty < t_min
                progress_y = false
                break
            end
        end
        ty *= 2

        push!(theta_Y, grad_step_Y)
        push!(losses, LOSS(theta_X[end], theta_Y[end]))

        if verbose
            println("Iteration: $k,  Loss: $(losses[end])")
        end

        if (!progress_x && !progress_y)
            println("Algorithm is not making any progress... breaking.")
            return theta_X, theta_Y, losses
        end

        if k > 4 && maximum(abs.(losses[end-4:end-1] - losses[end-3:end])) < tol
            println("Done.")
            return theta_X, theta_Y, losses
        end
    end
    print("Did not converge. Loss: $(losses[end])")
    return -1
end

srand(1234)

A = randn(1000, 500)

k = 5

@time x, y, loss = optimize_pca(A, k)


println("result with loss $(loss[end])")

println("Computing exact result")
U, S, V = svd(A)

A_pca = U[:,1:k]*diagm(S)[1:k, 1:k]*V[:,1:k]'

println("Final max exact difference: $(maximum(abs.(x[end]*y[end]' - A_pca)))")



"""
Sigmoid function 
"""
sigm(u) = 1/(1 + exp(-u))


"""
Convert an n-vector to an nx1 aarrays
"""
matrix(x::Array{Float64,1}) = reshape(x, length(x),1)
matrix(x::Range) = reshape(collect(x), length(x), 1)
matrix(X::Array{Float64,2}) = X


# do things rowwise
# f should map vectors to vectors
function rowwise(f, X::Array{Float64,2})
    y1 = f(X[1,:])
    n = size(X,1)
    d = length(y1)
    Y = Array{Float64}(n,d)
    Y[1,:] = y1
    for i=2:n
        Y[i,:] = f(X[i,:])
    end
    return Y
end


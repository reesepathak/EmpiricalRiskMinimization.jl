
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


"Sigmoid function"
sigm(u) = 1/(1 + exp(-u))


"Convert an n-vector to an nx1 array"
matrix(x::Array{T,1}) where {T<:Number} = reshape(x, length(x),1)
matrix(x::Range) = reshape(collect(x), length(x), 1)
matrix(X::Array{Any,2}) = X
matrix(x::RowVector) = reshape(transpose(x), 1, length(x))

"Apply f to each row of a matrix. f should map vectors to vectors"
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

"compute the rms of a matrix or a vector"
rms(x::Array{Float64,1}) = sqrt.( (1/length(x)) * (x'*x))
rms(x::Array{Float64,2}) = rms(x[:])


function findvalue(s, lst)
    r = find(x->x==s, lst)
    if length(r) == 0
        return 0
    end
    return r[1]
end

number(x::Number) = convert(Float64,x)
number(x::String) = parse(Float64,x)

abstract type Embedding end

##############################################################################
# Embeddings


mutable struct Standardize<:Embedding
    mean
    std
end

function Standardize()
    return Standardize(0,0)
end

#
# embeddings operate on matrices 1 row at a time
# (for some embeddings, like standardize, the results depend
#  on all the entries)
#

function embed(e::Standardize, X::Array{Float64,2})
    n = size(X,1)
    e.mean = mean(X,1)
    e.std = std(X,1)
    rmean = repmat(e.mean, n,1)
    rstd  = repmat(e.std, n,1)
    Xnew = (X-rmean) ./ rstd
    return Xnew
end

struct IdentityEmbed<:Embedding end
embed(e::IdentityEmbed, X::Array{Float64,2}) = X

mutable struct PolyEmbed<:Embedding
    degree
end

function embed(e::PolyEmbed, X::Array{Float64,2})
    d,n = size(X)
    if n != 1
        println("Error: PolyEmbed only operates on scalars")
    end
    Xnew = zeros(d,e.degree+1)
    for i=1:d
        for j=0:e.degree
            Xnew[i,j+1] = X[i,1]^j
        end
    end
    return Xnew
end


# apply list of embeddings from right to left
function embed(E::Array, z)
    ne = length(E)
    for i=ne:-1:1
        z = embed(E[i], z)
    end
    return z
end

type AppendOneEmbed<:Embedding end

function embed(e::AppendOneEmbed, X::Array{Float64,2})
    return [ones(size(X,1)) X]
end


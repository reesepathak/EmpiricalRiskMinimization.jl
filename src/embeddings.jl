

##############################################################################

abstract type Embedding end

##############################################################################
# Embeddings


mutable struct Standardize<:Embedding
    mean
    std
    used::Bool
end

##############################################################################
# generic embedding

mutable struct FunctionEmbedding <: Embedding
    f
end

function embed(e::FunctionEmbedding, u::Array{Float64,1})
    x = u
end


##############################################################################
# standardize

function Standardize()
    return Standardize(0,0,false)
end

#
# embeddings operate on matrices 1 row at a time
# (for some embeddings, like standardize, the results depend
#  on all the entries)
#
function embed(e::Standardize, u::Array{Float64,1})
    if !e.used
        error("cannot standardize single data point")
    end
    x = (u - e.mean[:])./(e.std[:])
end

function unembed(e::Standardize, x::Array{Float64,1})
    if !e.used
        error("cannot standardize single data point")
    end
    return  x.*e.std + e.mean
end

function embed(e::Standardize, U::Array{Float64,2})
    n = size(U,1)
    if !e.used
        e.mean = mean(U,1)
        e.std = std(U,1)
        e.used = true
    end
    return rowwise(u->embed(e,u), U)
end



##############################################################################

struct IdentityEmbed<:Embedding end
embed(e::IdentityEmbed, u::Array{Float64,1}) = u
unembed(e::IdentityEmbed, u::Array{Float64,1}) = u

##############################################################################
# polyembed



mutable struct PolyEmbed<:Embedding
    degree
end

function embed(e::PolyEmbed, u::Array{Float64,1})
    d = length(u)
    if d != 1
        println("Error: PolyEmbed only operates on scalars")
    end
    x = zeros(e.degree+1)
    for j=0:e.degree
        x[j+1] = u[1]^j
    end
    return x
end





mutable struct PiecewiseEmbed<:Embedding
    knot
end

# this is a neat trick for allowing many knots
# we embed by appending a new feature
# TODO: we need a better API for this
#
# embeddings can be pointwise or not
# can take inputs which are x or u
# can append their output to the input or replace the input
#
function embed(e::PiecewiseEmbed, u::Array{Float64,1})
    n = length(u)
    x = zeros(n+1)
    x[1:n] = u
    x[n+1] = max(u[1]-e.knot,0)
    return x
end

##############################################################################
type AppendOneEmbed<:Embedding end

function embed(e::AppendOneEmbed, x::Array{Float64,1})
    return matrix([1 x'])
end


##############################################################################
# apply embeddings

# allow embeddings to be defined either on the whole dataset
# or per-record
# preferentially apply whole dataset embedding if available
function oneembedding(e, U::Array{Float64,2})
    # if there is a method embedding the whole dataset
    if method_exists(embed, (typeof(e), Array{Float64,2}))
        return embed(e, U)
    end
    # otherwise do one row at a time
    return rowwise(u->embed(e, u), U)
end

oneembedding(e, u::Array{Float64,1}) = embed(e,u)

# apply list of embeddings from right to left
function embed(E::Array, z)
    ne = length(E)
    for i=ne:-1:1
        z = oneembedding(E[i], z)
    end
    return z
end

##############################################################################
# apply unembeddings

function oneunembedding(e, U::Array{Float64,2})
    # if there is a method embedding the whole dataset
    if method_exists(unembed, (typeof(e), Array{Float64,2}))
        return unembed(e, U)
    end
    # otherwise do one row at a time
    return rowwise(u->unembed(e, u), U)
end

oneunembedding(e, u::Array{Float64,1}) = unembed(e,u)

function unembed(E::Array, z)
    ne = length(E)
    for i=1:ne
        z = oneunembedding(E[i], z)
    end
    return z
end


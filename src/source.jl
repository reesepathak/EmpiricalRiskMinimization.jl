
abstract type DataSource end

##############################################################################

mutable struct SimpleSource<:DataSource
    U
    V
    Xembed
    Yembed

    function SimpleSource(U, V, Xembed, Yembed)
        if Xembed == false
            Xembed = [AppendOneEmbed(), Standardize()]
        end
        if Yembed == false
        Yembed = [Standardize()]
        end
        return new(matrix(U), matrix(V), Xembed, Yembed)
    end
end


function getXY(S::SimpleSource)
    hasconstfeature = false
    if isa(S.Xembed[1], AppendOneEmbed)
        hasconstfeature = true
    end
    Y = embed(S.Yembed, S.V)
    X = embed(S.Xembed, S.U)
    return X, Y, hasconstfeature
end

getU(S::SimpleSource) = S.U
getV(S::SimpleSource) = S.V

# embed one data record
embedU(S::SimpleSource, u::Array{Float64,1}) = embed(S.Xembed, u)

# unembed one or many targets
unembedY(S::SimpleSource, y) = unembed(S.Yembed, y)


##############################################################################
# an array of strings with header info

findvalue(s, lst)  =  find(x->x==s, lst)[1]

mutable struct DFrame
    A
    names
end

function DFrame(numrows)
    A =  Array{Float64}(numrows,0)
    names = Any[]
    return DFrame(A,names)
end


colnum(DF::DFrame, c::String) = findvalue(c, DF.names)
colnum(DF::DFrame, c::Number) = c

function col2(c::Number, inframe, outframe)
    return inframe, c
end

# by default c refers to the inframe
# but can refer to the outframe if
# c is only in the outframe.names
function col2(c::String, inframe, outframe)
    if c in inframe.names
        return inframe, findvalue(c, inframe.names)
    end
    return outframe, findvalue(c, outframe.names)
end
    

function isnumeric(DF::DFrame, col)
    n = size(DF.A,1)
    j = colnum(col)
    for i=1:n
        q = tryparse(Float64, DF.A[i,j])
        if isnull(q)
            return false
        end
    end
    return true
end

number(x::Number) = convert(Float64,x)
number(x::String) = parse(Float64,x)

function numcol(DF::DFrame, col)
    j = colnum(DF, col)
    n = size(DF.A, 1)
    u = zeros(n)
    for i=1:n
        u[i] = number(DF.A[i,j])
    end
    return u
end

function appendcol(DF::DFrame, u, name)
    DF.A = [DF.A u]
    push!(DF.names, name)
end

# find all different values in column
# assign each a number 1...d
# return a dict mapping values to numbers
# and a list of values in order
function colvalues(DF::DFrame, col)
    j = colnum(DF, col)
    n = size(DF.A, 1)
    valtonum = Dict()
    vals = Any[]
    num = 1
    for val in Set(DF.A[:,j])
        valtonum[val] = num
        push!(vals, val)
        num += 1
    end
    return valtonum, vals
end

# map a column according to valtonum
function coltointegers(DF::DFrame, col, valtonum)
    j = colnum(DF, col)
    n = size(DF.A, 1)
    u = zeros(n)
    for i=1:n
        u[i] = valtonum[DF.A[i,j]]
    end
    return u 
end

# valtonum is a dict mapping strings to 1...d
function coltoonehot(DF::DFrame, col, valtonum)
    j = colnum(DF, col)
    n = size(DF.A, 1)
    d = length(valtonum)
    u = zeros(n,d)
    for i=1:n
        u[i, valtonum[DF.A[i,j]] ] = 1
    end
    return u
end


##############################################################################

abstract type FeatureMap end
mutable struct AddColumnFmap<:FeatureMap
    col  # source column name or number
    name # destination name, may be nothing
end

function applyfmap(FM::AddColumnFmap, inframe, outframe)
    sourcedf, j =  col2(FM.col, inframe, outframe)
    u  = numcol(sourcedf, j)
    appendcol(outframe, u, FM.name)
end


mutable struct OneHotFmap<:FeatureMap
    col  # source column name or number
    name # destination name, may be nothing
end

function applyfmap(FM::OneHotFmap, inframe, outframe)
    sourcedf, j =  col2(FM.col, inframe, outframe)
    valtonum, vals = colvalues(sourcedf, j)
    u = coltoonehot(sourcedf, j, valtonum)
    appendcol(outframe, u, FM.name)
end



mutable struct OrdinalFmap<:FeatureMap
    col  # source column name or number
    name # destination name, may be nothing
    categories
end

function applyfmap(FM::OrdinalFmap, inframe, outframe)
    sourcedf, j =  col2(FM.col, inframe, outframe)
    valtonum = Dict()
    for i=1:length(FM.categories)
        valtonum[FM.categories[i]] = i
    end
    u = coltointegers(sourcedf, j, valtonum)
    appendcol(outframe, u, FM.name)
end





##############################################################################
mutable struct FrameSource<:DataSource
    Uf
    Vf
    Xmaps
    Ymaps


    function FrameSource(U, V, Unames, Vnames)
        Uf = DFrame(U, Unames)
        Vf = DFrame(V, Vnames)
        n = size(U,1)
        Xf = DFrame(n)
        Yf = DFrame(n)
        return new(Uf, Vf, Any[], Any[])
    end
end


function addfeature(fmaps, col;
                    name = nothing, etype="number",
                    categories = nothing,
                    kwargs...)
    if etype == "number"
        push!(fmaps, AddColumnFmap(col, name))
    elseif etype == "onehot"
        push!(fmaps, OneHotFmap(col, name))
    elseif etype == "ordinal"
        push!(fmaps, OrdinalFmap(col, name, categories))
    end
end
    

function addfeatureU(F::FrameSource, col; kwargs...)
    addfeature(F.Xmaps, col; kwargs...)
end

function addfeatureV(F::FrameSource, col; kwargs...)
    addfeature(F.Ymaps, col; kwargs...)
end

function getXY(F::FrameSource)
    Xf = applyfmaplist(F.Xmaps, F.Uf)
    Yf = applyfmaplist(F.Ymaps, F.Vf)
    X = Xf.A
    Y = Yf.A
    hasconstfeature = false
    return X, Y, hasconstfeature
end

function applyfmaplist(fmaps, inframe)
    n = size(inframe.A,1)
    outframe = DFrame(n)
    ne = length(fmaps)
    for i=1:ne
        applyfmap(fmaps[i], inframe, outframe)
    end
    return outframe
end


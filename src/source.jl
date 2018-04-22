
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

# empty frame
function DFrame(numrows::Integer)
    A =  Array{Float64}(numrows,0)
    names = Any[]
    return DFrame(A,names)
end

# frame with no names
function DFrame(A)
    d = size(A,2)
    names = Array{Any}(d)
    fill!(names, "")
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

# appends or replaces if the name exists already
function appendcol(DF::DFrame, u, name)
    if name != nothing && name in DF.names
        j = findvalue(name, DF.names)
        DF.A[:,j] = u
    else
        DF.A = [DF.A u]
        if size(u,2) == 1
            push!(DF.names, name)
        else
            for k=1:size(u,2)
                push!(DF.names, name * "_$k")
            end
        end
    end
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
    standardize  # true if standardized
end

function applyfmap(FM::OneHotFmap, inframe, outframe)
    sourcedf, j =  col2(FM.col, inframe, outframe)
    valtonum, vals = colvalues(sourcedf, j)
    u = coltoonehot(sourcedf, j, valtonum)
    if FM.standardize
        n = size(u,1)
        m = mean(u,1)
        s = std(u,1)
        u = (u - repmat(m, n, 1))./repmat(s,n,1)
    end
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





mutable struct FunctionFmap<:FeatureMap
    col  # source column name or number
    name # destination name, may be nothing
    f
end

function applyfmap(FM::FunctionFmap, inframe, outframe)
    sourcedf, j =  col2(FM.col, inframe, outframe)
    u  = numcol(sourcedf, j)
    unew = [ FM.f(x) for x in u]
    appendcol(outframe, unew, FM.name)
end



mutable struct FunctionListFmap<:FeatureMap
    col  # list of source column names or numbers
    name # destination name, may be nothing
    f
end

function applyfmap(FM::FunctionListFmap, inframe, outframe)
    args = Any[]
    for a in FM.col
        sourcedf, j =  col2(a, inframe, outframe)
        u  = numcol(sourcedf, j)
        push!(args, u)
    end
    n = length(args[1])
    unew = zeros(n)
    for i=1:n
        a2 = Any[]
        for j=1:length(args)
            push!(a2, args[j][i])
        end
        unew[i]  = FM.f(a2...)
    end
    appendcol(outframe, unew, FM.name)
end



mutable struct StandardizeFmap<:FeatureMap
    col  # source column name or number
    name # destination name, may be nothing
    mean
    std
end

function applyfmap(FM::StandardizeFmap, inframe, outframe)
    sourcedf, j =  col2(FM.col, inframe, outframe)
    u  = numcol(sourcedf, j)
    FM.mean = mean(u)
    FM.std = std(u)
    unew = (u - FM.mean)/FM.std
    appendcol(outframe, unew, FM.name)
end

function StandardizeFmap(col, name)
    return StandardizeFmap(col, name, nothing, nothing)
end





mutable struct OneFmap<:FeatureMap
    name # destination name, may be nothing
end

function applyfmap(FM::OneFmap, inframe, outframe)
    n = size(inframe.A, 1)
    u = ones(n)
    appendcol(outframe, u, FM.name)
end





##############################################################################
mutable struct FrameSource<:DataSource
    Uf
    Vf
    Xmaps
    Ymaps
    count
end

function FrameSource(Uf::DFrame, Vf::DFrame)
    return FrameSource(Uf, Vf, Any[], Any[], [0])
end

function makeFrameSource(U, V, Unames, Vnames)
    Uf = DFrame(U, Unames)
    Vf = DFrame(V, Vnames)
    return FrameSource(Uf, Vf)
end


function makeFrameSource(U, V)
    Uf = DFrame(U)
    Vf = DFrame(V)
    return FrameSource(Uf, Vf)
end


    
function getfeature(col; name = nothing, etype="number",
                    categories = nothing,
                    f = nothing, kwargs...)
    if etype == "number"
        return AddColumnFmap(col, name)
    elseif etype == "onehot"
        return OneHotFmap(col, name, false)
    elseif etype == "onehotstd"
        return OneHotFmap(col, name, true)
    elseif etype == "ordinal"
        return OrdinalFmap(col, name, categories)
    elseif etype == "function"
        return FunctionFmap(col, name, f)
    elseif etype == "standardize"
        return StandardizeFmap(col, name)
    elseif etype == "product"
        return FunctionListFmap(col, name, (x,y)-> x*y)
    elseif etype == "one"
        return OneFmap(name)
    end
end
                    
function addfeaturex(fmaps, count, col; stand=true, name=nothing, etype="number", kwargs...)
    # make sure every feature has a name
    if name == nothing
        name = "feature$(count[1])"
        count[1] += 1
    end
    fe = getfeature(col; name=name, etype=etype, kwargs...)
    push!(fmaps, fe)
    if stand && etype != "onehot" && etype != "one" && etype != "onehotstd"
        # add a featurizer that replaces the feature with name = name
        # with a standardized version
        addfeaturex(fmaps, count, name;
                      etype="standardize", stand=false, name=name, kwargs...)
    end
end

   


function addfeatureU(F::FrameSource, col; kwargs...)
    addfeaturex(F.Xmaps, F.count, col; kwargs...)
end

function addfeatureV(F::FrameSource, col; kwargs...)
    addfeaturex(F.Ymaps, F.count, col; kwargs...)
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



##############################################################################
# an array of strings with header info



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


findcolumn(c::String, DF::DFrame) = findvalue(c, DF.names)
findcolumn(c::Number, DF::DFrame) = c



##############################################################################

abstract type DataSource end


##############################################################################
# interface

mutable struct FrameSource<:DataSource
    Uf
    Vf
    Xmaps
    Ymaps
    count
    Xnames  # saved for unembedding
    Ynames  # saved for unembedding
end

function FrameSource(Uf::DFrame, Vf::DFrame)
    return FrameSource(Uf, Vf, Any[], Any[], [0], nothing, nothing)
end

function makeFrameSource(U, V, Unames, Vnames)
    if Unames != nothing
        Uf = DFrame(matrix(U), Unames)
    else
        Uf = DFrame(matrix(U))
    end
    if Vnames != nothing
        Vf = DFrame(matrix(V), Vnames)
    else
        Vf = DFrame(matrix(V))
    end
    return FrameSource(Uf, Vf)
end




function containsone(fmaps)
    for a in fmaps
        if isa(a, OneFmap)
            return true
        end
    end
    return false
end

function getXY(F::FrameSource)
    Xf = applyfmaplist(F.Xmaps, F.Uf)
    Yf = applyfmaplist(F.Ymaps, F.Vf)
    X = Xf.A
    Y = Yf.A
    F.Xnames = Xf.names
    F.Ynames = Yf.names
    hasconstfeature = containsone(F.Xmaps)
    return X, Y, hasconstfeature
end



getU(F::FrameSource) = F.Uf.A
getV(F::FrameSource) = F.Vf.A

# embed one data record
function embedU(F::FrameSource, u::Array{Float64,1})
    U = reshape(u, 1, length(u))
    embedU(F, U)
end


function embedU(F::FrameSource, U::Array{Float64,2})
    Uf = DFrame(U, F.Uf.names)
    Xf = applyfmaplist(F.Xmaps, Uf)
    return Xf.A
end

##############################################################################
# internal code 

function findcolumn(c::Number, uvframe, xyframe)
    return uvframe, c
end

# by default c refers to the uvframe
# but can refer to the xyframe if
# c is only in the xyframe.names
function findcolumn(c::String, uvframe, xyframe)
    if c in uvframe.names
        return uvframe, findvalue(c, uvframe.names)
    end
    if c in xyframe.names
        return xyframe, findvalue(c, xyframe.names)
    end
    # we cannot resolve this column name
    # but return c so that it can be used in appendcol
    return xyframe, c
end


function columnvalues(DF::DFrame, col)
    j = findcolumn(col, DF)
    n = size(DF.A, 1)
    u = zeros(n)
    for i=1:n
        u[i] = number(DF.A[i,j])
    end
    return u
end

function appendcol(DF::DFrame, name::Int, u)
    DF.A[:,name] = u
end

# appends or replaces if the name exists already
function appendcol(DF::DFrame, name, u)
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
function uniquecolumnvalues(DF::DFrame, col)
    j = findcolumn(col, DF)
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
function columntointegers(DF::DFrame, col, valtonum)
    j = findcolumn(col, DF)
    n = size(DF.A, 1)
    u = zeros(n)
    for i=1:n
        u[i] = valtonum[DF.A[i,j]]
    end
    return u 
end

# valtonum is a dict mapping strings to 1...d
function coltoonehot(DF::DFrame, col, valtonum)
    j = findcolumn(col, DF)
    n = size(DF.A, 1)
    d = length(valtonum)
    u = zeros(n,d)
    for i=1:n
        u[i, valtonum[DF.A[i,j]] ] = 1
    end
    return u
end


#######################################################################

abstract type FeatureMap end

##############################

mutable struct AddColumnFmap<:FeatureMap
    src    # source column name or number
    dest   # destination name, may be nothing
end

function applyfmap(FM::AddColumnFmap, uvframe, xyframe)
    srcframe, srccol =  findcolumn(FM.src, uvframe, xyframe)
    u  = columnvalues(srcframe, srccol)
    appendcol(xyframe, FM.dest, u)
end

# map from dest to src
function invertfmap(FM::AddColumnFmap, uvframe, xyframe)
    destcol = findcolumn(FM.dest, xyframe)
    x = columnvalues(xyframe, destcol)
    srcframe, srccol = findcolumn(FM.src, uvframe, xyframe)
    appendcol(srcframe, FM.src, x)
end

##############################

mutable struct OneHotFmap<:FeatureMap
    src  # source column name or number
    dest # destination name, may be nothing
    standardize  # true if standardized
    used # if been standardized before
    mean 
    std 
end

OneHotFmap(c,n,s) = OneHotFmap(c,n,s,false,nothing,nothing)

function applyfmap(FM::OneHotFmap, uvframe, xyframe)
    srcframe, srccol =  findcolumn(FM.src, uvframe, xyframe)
    valtonum, vals = uniquecolumnvalues(srcframe, srccol)
    u = coltoonehot(srcframe, srccol, valtonum)
    if FM.standardize
        if !FM.used
            FM.mean = mean(u,1)
            FM.std = std(u,1)
            FM.used = true
        end
        n = size(u,1)
        u = (u - repmat(FM.mean, n, 1))./repmat(FM.std,n,1)
    end
    appendcol(xyframe, FM.dest, u)
end

##############################

mutable struct OrdinalFmap<:FeatureMap
    src   # source column name or number
    dest  # destination name, may be nothing
    categories
end

function applyfmap(FM::OrdinalFmap, uvframe, xyframe)
    srcframe, srccol =  findcolumn(FM.src, uvframe, xyframe)
    valtonum = Dict()
    for i=1:length(FM.categories)
        valtonum[FM.categories[i]] = i
    end
    u = columntointegers(srcframe, srccol, valtonum)
    appendcol(xyframe, FM.dest, u)
end

##############################

mutable struct FunctionFmap<:FeatureMap
    src   # source column name or number
    dest  # destination name, may be nothing
    f
    finv
end

function FunctionFmap(src, dest, f)
    if f == log
        finv = exp
    else
        finv = nothing
    end
    return FunctionFmap(src, dest, f, finv)
end

function applyfmap(FM::FunctionFmap, uvframe, xyframe)
    srcframe, srccol =  findcolumn(FM.src, uvframe, xyframe)
    u  = columnvalues(srcframe, srccol)
    unew = [ FM.f(x) for x in u]
    appendcol(xyframe, FM.dest, unew)
end

# maps dest to src
function invertfmap(FM::FunctionFmap, uvframe, xyframe)
    destcol = findcolumn(FM.dest, xyframe)
    x = columnvalues(xyframe, destcol)
    u = [ FM.finv(a) for a in x ]
    srcframe, srccol = findcolumn(FM.src, uvframe, xyframe)
    appendcol(srcframe, FM.src, u)
end


###########################################

mutable struct FunctionListFmap<:FeatureMap
    src  # list of source column names or numbers
    dest # destination name, may be nothing
    f
end

function applyfmap(FM::FunctionListFmap, uvframe, xyframe)
    args = Any[]
    for srci in FM.src
        srcframe, srccol =  findcolumn(srci, uvframe, xyframe)
        u  = columnvalues(srcframe, srccol)
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
    appendcol(xyframe, FM.dest, unew)
end

##############################

mutable struct StandardizeFmap<:FeatureMap
    src  # source column name or number
    dest # destination name, may be nothing
    mean
    std
    used
end

function applyfmap(FM::StandardizeFmap, uvframe, xyframe)
    srcframe, srccol =  findcolumn(FM.src, uvframe, xyframe)
    u  = columnvalues(srcframe, srccol)
    if !FM.used
        FM.mean = mean(u)
        FM.std = std(u)
        FM.used = true
    end
    unew = (u - FM.mean)/FM.std
    appendcol(xyframe, FM.dest, unew)
end

# map dest to src
function invertfmap(FM::StandardizeFmap, uvframe, xyframe)
    destcol = findcolumn(FM.dest, xyframe)
    x = columnvalues(xyframe, destcol)
    u = FM.std*x + FM.mean
    srcframe, srccol = findcolumn(FM.src, uvframe, xyframe)
    appendcol(srcframe, FM.src, u)
end
    
function StandardizeFmap(src, dest)
    return StandardizeFmap(src, dest, nothing, nothing, false)
end

##############################

mutable struct OneFmap<:FeatureMap
    dest # destination name, may be nothing
end

function applyfmap(FM::OneFmap, uvframe, xyframe)
    n = size(uvframe.A, 1)
    u = ones(n)
    appendcol(xyframe, FM.dest, u)
end

##############################################################################
    
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
                    
function addfeaturex(fmaps, count, col;
                     stand=true, name=nothing, etype="number", kwargs...)
    # make sure every feature has a name
    if name == nothing
        name = "feature$(count[1])"
        count[1] += 1
    end
    fe = getfeature(col; name=name, etype=etype, kwargs...)
    push!(fmaps, fe)
    if stand && etype != "onehot" && etype != "one" && etype != "onehotstd"
        # add a featurizer that replaces the feature with
        # with a standardized version that has the same name
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



function applyfmaplist(fmaps, uvframe)
    n = size(uvframe.A,1)
    xyframe = DFrame(n)
    ne = length(fmaps)
    for i=1:ne
        applyfmap(fmaps[i], uvframe, xyframe)
    end
    return xyframe
end

function unembedY(F::FrameSource, y::Array{Float64,1})
    Y = reshape(y, 1, length(y))
    unembedY(F, Y)
end

function unembedY(F::FrameSource, Y::Array{Float64,2})
    Yf = DFrame(Y, F.Ynames)
    d = size(F.Vf.A,2)
    n = size(Y,1)
    Vf = DFrame(zeros(n,d), F.Vf.names)
    invertfmaplist(F.Ymaps, Yf, Vf)
    V = Vf.A
    return V
end

function invertfmaplist(fmaps, xyframe, uvframe)
    n = size(xyframe.A,1)
    ne = length(fmaps)
    for i=ne:-1:1
        invertfmap(fmaps[i], uvframe, xyframe)
    end
end


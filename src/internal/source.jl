
##############################################################################
# an array of strings with header info

import Statistics

mutable struct DFrame
    A
    names
    cols
end

# empty frame
function DFrame(numrows::Integer; estnumcols=0)
    A =  Array{Float64}(undef,numrows,estnumcols)
    names = Any[]
    return DFrame(A,names,0)
end

# frame with no names
function DFrame(A)
    d = size(A,2)
    names = Array{Any}(d)
    fill!(names, "")
    return DFrame(A,names,d)
end

function DFrame(A, names)
    return DFrame(A, names, size(A,2))
end

findcolumn(c::String, DF::DFrame) = findvalue(c, DF.names)
findcolumn(c::Number, DF::DFrame) = c

function addcolumn(DF::DFrame, u::Array{Float64,2})
    newcols = size(u,2)
    if DF.cols + newcols < size(DF.A,2)
        DF.A[:,DF.cols+1:DF.cols+newcols] = u
        DF.cols += newcols
        return
    end
    DF.A = [DF.A u]
    DF.cols += newcols
end

addcolumn(DF::DFrame, u::Array{Float64,1}) = addcolumn(DF, reshape(u, length(u),1))


import Base.size
function size(DF::DFrame, x::Number)
    if x == 1
        return size(DF.A,1)
    end
    return DF.cols
end

function size(DF::DFrame)
    return size(DF.A,1), DF.cols
end
function value(DF::DFrame)
    return DF.A[:,1:DF.cols]
end

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

function makeFrameSource(U, V, Unames, Vnames; kwargs...)
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



function getXY(F::FrameSource; Uestnumcols=0, Vestnumcols=0, verbose=false)
    Xf = applyfmaplist(F.Xmaps, F.Uf; estnumcols = Uestnumcols, verbose=verbose)
    Yf = applyfmaplist(F.Ymaps, F.Vf; estnumcols = Vestnumcols, verbose=verbose)
    X = value(Xf)
    Y = value(Yf)
    F.Xnames = Xf.names
    F.Ynames = Yf.names
    return X, Y
end



getU(F::FrameSource) = value(F.Uf)
getV(F::FrameSource) = value(F.Vf)

# embed one data record
function embedU(F::FrameSource, u::Array{T,1}) where {T<:Any}
    if size(F.Uf,2) == 1
        # U has only one column, so we are embedding
        # many 1 dimensional records
        U = reshape(u, length(u),1)
    else
        # U has more than one column
        # so we must be embedding a single record
        # (and the records must have dimension equal to length(u)
        # otherwise this will fail.)
        U = reshape(u, 1, length(u))
    end
    embedU(F, U)
end


function embedU(F::FrameSource, U::Array{T,2}) where {T<:Any}
    Uf = DFrame(U, F.Uf.names)
    Xf = applyfmaplist(F.Xmaps, Uf)
    return value(Xf)
end

function embedV(F::FrameSource, U::Array{T,2}) where {T<:Any}
    Uf = DFrame(U, F.Vf.names)
    Xf = applyfmaplist(F.Ymaps, Uf)
    return value(Xf)
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
    n = size(DF, 1)
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
        if size(u,2) > 1
            println("ERROR: Cannot specify existing destination name if using onehot")
        end
        j = findvalue(name, DF.names)
        DF.A[:,j] = u
    else
        addcolumn(DF, u)
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
    n = size(DF, 1)
    valtonum = Dict()
    vals = Any[]
    num = 1
    possiblevalues = sort(collect(Set(DF.A[:,j])))
    for val in possiblevalues
        valtonum[val] = num
        push!(vals, val)
        num += 1
    end
    return valtonum, vals
end

# map a column according to valtonum
function columntointegers(DF::DFrame, col, valtonum)
    j = findcolumn(col, DF)
    n = size(DF,1)
    u = zeros(n)
    for i=1:n
        u[i] = valtonum[DF.A[i,j]]
    end
    return u 
end

# valtonum is a dict mapping strings to 1...d
function coltoonehot(DF::DFrame, col, valtonum)
    j = findcolumn(col, DF)
    n = size(DF,1)
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

status(FM::AddColumnFmap) = "Add column $(FM.src) as $(FM.dest)"

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
    valtonum  # dict mapping values to numbers
    vals  # list of the values in order (i.e. a list mapping numbers to values)
end

status(FM::OneHotFmap) = "Add one-hot column $(FM.src)"

OneHotFmap(c,n,s) = OneHotFmap(c,n,s,false,nothing,nothing,nothing,nothing)

# one hot does it's own standardization since at the moment
# each feature added only takes one input column
# we do not know at the time the feature is added to the list
# how many possible values its input column will take
# and so we don't know how many output columns will be generated
# and so we cannot generate standardizers for each of them
#


function applyfmap(FM::OneHotFmap, uvframe, xyframe)
    srcframe, srccol =  findcolumn(FM.src, uvframe, xyframe)
    valtonum, vals = uniquecolumnvalues(srcframe, srccol)
    FM.valtonum = valtonum
    FM.vals = vals
    u = coltoonehot(srcframe, srccol, valtonum)
    if FM.standardize
        if !FM.used
            FM.mean = Statistics.mean(u,dims=1)
            FM.std = Statistics.std(u,dims=1)
            FM.used = true
        end
        n = size(u,1)
        u = (u - repeat(FM.mean, n, 1))./repeat(FM.std,n,1)
    end
    appendcol(xyframe, FM.dest, u)
end


# map from dest to src
function invertfmap(FM::OneHotFmap, uvframe, xyframe)
    # K is number of classes
    K = length(FM.vals)
    n = size(xyframe, 1)
    x = Array{Any}(n)
    destcols = zeros(Int64, K)
    for k=1:K
        destcols[k] = findcolumn(FM.dest * "_$(k)", xyframe)
    end
    xi = zeros(K)
    for i=1:n
        for k=1:K
            xi[k] = xyframe.A[i,destcols[k]]
        end
        maxval, maxind = findmax(xi)
        x[i] = FM.vals[maxind]
    end
    srcframe, srccol = findcolumn(FM.src, uvframe, xyframe)
    appendcol(srcframe, FM.src, x)
end

##############################

mutable struct OrdinalFmap<:FeatureMap
    src   # source column name or number
    dest  # destination name, may be nothing
    categories
end

status(FM::OrdinalFmap) = "Add ordinal column $(FM.src) with categories $(FM.categories)"

function applyfmap(FM::OrdinalFmap, uvframe, xyframe)
    srcframe, srccol =  findcolumn(FM.src, uvframe, xyframe)
    valtonum = Dict()
    for i=1:length(FM.categories)
        valtonum[FM.categories[i]] = i
    end
    u = columntointegers(srcframe, srccol, valtonum)
    appendcol(xyframe, FM.dest, u)
end

# maps dest to src
# map back to categories by nearest neighbor
function invertfmap(FM::OrdinalFmap, uvframe, xyframe)
    destcol = findcolumn(FM.dest, xyframe)
    x = columnvalues(xyframe, destcol)
    n = length(x)
    u = zeros(n)
    for i=1:n
        closestcat = 0
        mindist = Inf
        for j=1:length(FM.categories)
            dist = abs(x[i]-j)
            if dist < mindist
                mindist = dist
                closestcat = j
            end
        end
        u[i]  = FM.categories[closestcat]
    end
    srcframe, srccol = findcolumn(FM.src, uvframe, xyframe)
    appendcol(srcframe, FM.src, u)
end




##############################

mutable struct FunctionFmap<:FeatureMap
    src   # source column name or number
    dest  # destination name, may be nothing
    f
    finv
end

status(FM::FunctionFmap) = "Add function column $(FM.src)"

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

status(FM::FunctionListFmap) = "add function to columns: $(FM.src)"

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

########################################################
mutable struct UfunctionFmap<:FeatureMap
    dest # destination name, may be nothing
    f
end

status(FM::UfunctionFmap) = "apply function to U"

function applyfmap(FM::UfunctionFmap, uvframe, xyframe)
    n = size(uvframe, 1)
    unew  = FM.f(uvframe.A)
    appendcol(xyframe, FM.dest, unew)
end

########################################################
mutable struct AllFmap<:FeatureMap
    dest # destination name, may be nothing
    stand  # boolean
    addones #bool
    mean
    std
    used
end

AllFmap(dest, stand, addones) = AllFmap(dest, stand, addones,
                                        nothing, nothing, false)


status(FM::AllFmap) = "embed all at once"

function applyfmap(FM::AllFmap, uvframe, xyframe)
    U = uvframe.A
    if !FM.used
        if FM.stand
            FM.mean = Statistics.mean(U,dims=1)
            FM.std =Statistics.std(U,dims=1)
        else
            FM.mean = zeros(1,d)
            FM.std = ones(1,d)
        end
        FM.used = true
    end
    n,d = size(U)
    unew = (U - repeat(FM.mean,n,1))./repeat(FM.std,n,1)
    if FM.addones
        unew = [ones(n) unew]
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

status(FM::StandardizeFmap) = "standardize column $(FM.src)"

function applyfmap(FM::StandardizeFmap, uvframe, xyframe)
    srcframe, srccol =  findcolumn(FM.src, uvframe, xyframe)
    u  = columnvalues(srcframe, srccol)
    if !FM.used
        FM.mean = Statistics.mean(u)
        FM.std = Statistics.std(u)
        if FM.std<1e-2
            FM.std=1
        end
        FM.used = true
    end
    unew = (u .- FM.mean)/FM.std
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

status(FM::OneFmap) = "add column of ones as $(FM.dest)"

function applyfmap(FM::OneFmap, uvframe, xyframe)
    n = size(uvframe, 1)
    u = ones(n)
    appendcol(xyframe, FM.dest, u)
end

##############################################################################
    
function getfeature(col; name = nothing, etype="number",
                    categories = nothing,
                    f = nothing, stand=true, addones=false, kwargs...)
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
    elseif etype == "functionlist"
        return FunctionListFmap(col, name, f)
    elseif etype == "ufunction"
        return UfunctionFmap(name, f)
    elseif etype == "all"
        return AllFmap(name, stand, addones)
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
    if stand && etype != "onehot" && etype != "one" && etype != "onehotstd" && etype != "ufunction" && etype != "all"
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



function applyfmaplist(fmaps, uvframe; estnumcols=0, verbose=false)
    n = size(uvframe,1)
    xyframe = DFrame(n; estnumcols=estnumcols)
    ne = length(fmaps)
    for i=1:ne
        if verbose
            println("FrameSource: Applying feature map: ", status(fmaps[i]))
        end
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
    d = size(F.Vf, 2)
    n = size(Y,1)
    Vf = DFrame(zeros(n,d), F.Vf.names)
    invertfmaplist(F.Ymaps, Yf, Vf)
    V = value(Vf)
    return V
end

function invertfmaplist(fmaps, xyframe, uvframe)
    n = size(xyframe, 1)
    ne = length(fmaps)
    for i=ne:-1:1
        invertfmap(fmaps[i], uvframe, xyframe)
    end
end


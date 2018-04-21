
abstract type DataSource end
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

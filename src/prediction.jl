##############################################################################
# predict

# if x happens to be a scalar, do we want this to work?
# predict(M::Model, x::Number,            theta=thetaopt(M))   = [x*theta]
# "predict" could be called "predict_y_from_x"
# for each record, x,y,u,v are always vectors
predict(M::Model, x::Array{Float64, 1}, theta=thetaopt(M))   = [dot(x, theta)]
# following could be defined using rowwise, but efficiency might dictate otherwise
predict(M::Model, X::Array{Float64, 2}, theta=thetaopt(M))   = X*theta

"""`predict_y_from_test(M [, theta])`

Allows you compute embedded predictions (i.e., y values) based on a trained ERM model M on test data. Option to 
specify a choice of theta. It defaults to `theta=thetaopt(M)`
"""
predict_y_from_test(M::Model,                   theta=thetaopt(M))   = predict(M, Xtest(M), theta)

"""`predict_y_from_train(M [, theta])`

Allows you compute embedded predictions (i.e., y values)  based on a trained ERM model M on train data. Option to 
specify a choice of theta. It defaults to `theta=thetaopt(M)`
"""
predict_y_from_train(M::Model,                  theta=thetaopt(M))   = predict(M, Xtrain(M), theta)

"""`predict_v_from_test(M [, theta])`

Allows you compute unembedded predictions (i.e., in V space) 
based on a trained ERM model M on test data. Option to 
specify a choice of theta. It defaults to `theta=thetaopt(M)`
"""
predict_v_from_test(M::Model,                   theta=thetaopt(M))   = unembedY(M.S, predict(M, Xtest(M), theta))

"""`predict_v_from_train(M [, theta])`

Allows you compute unembedded predictions (i.e., in V space) 
based on a trained ERM model M on train data. Option to 
specify a choice of theta. It defaults to `theta=thetaopt(M)`
"""
predict_v_from_train(M::Model,                  theta=thetaopt(M))   = unembedY(M.S, predict(M, Xtrain(M), theta))

"""`predict_y_from_u(M, U [, theta])`

Allows you compute embedded predictions (i.e., y values) 
based on a trained ERM model M on one or many raw inputs, `U`. Option to 
specify a choice of theta. It defaults to `theta=thetaopt(M)`
"""
predict_y_from_u(M::Model, u::Array{T,1}, theta=thetaopt(M)) where {T<:Any} =  predict(M::Model, embedU(M.S, u), theta)
predict_y_from_u(M::Model, U::Array{T,2}, theta=thetaopt(M)) where {T<:Any} =  rowwise(u -> predict_y_from_u(M, u, theta), U)

"""`predict_y_from_u(M, U [, theta])`

Allows you compute unembedded predictions (i.e., in V space) 
based on a trained ERM model M on one or many raw inputs, `U`. Option to 
specify a choice of theta. It defaults to `theta=thetaopt(M)`
"""
predict_v_from_u(M::Model, u::Array{T,1}, theta=thetaopt(M)) where {T<:Any} =  unembedY(M.S, predict_y_from_u(M, u, theta))
predict_v_from_u(M::Model, U::Array{T,2}, theta=thetaopt(M)) where {T<:Any} =  rowwise(u -> predict_v_from_u(M, u, theta), U)


function confusionx(vhat, v)
    rows = length(Set(vhat))
    cols = length(Set(v))
    C = zeros(rows, cols)
    n = size(v,1)
    for k=1:n
        i = convert(Int64, vhat[k])
        j = convert(Int64, v[k])
        C[i,j] += 1
    end
    return C
end

"""`confusion_train(M)`

Returns the confusion matrix based on training data
for a trained ERM model"""
function confusion_train(M::Model)
    vhat = predict_v_from_train(M)
    return confusionx(vhat, Vtrain(M))
end

"""`confusion_test(M)`

Returns the confusion matrix based on test data
for a trained ERM model"""
function confusion_test(M::Model)
    vhat = predict_v_from_test(M)
    return confusionx(vhat, Vtest(M))
end

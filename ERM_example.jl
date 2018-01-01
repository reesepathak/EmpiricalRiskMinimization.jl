#example usage EmpiricalRiskMinimization.jl (01-01-18)
using EmpiricalRiskMinimization
using Convex, SCS

# data
n = 2000; d = 350;
X = randn(n, d); y = (sign.(randn(n) - 0.5) + 1)/2;

# model
model = Model(LogisticLoss(), L2Reg(0.1), fit_intercept=true);
fit!(model, X, y)
status(model)
final_risk(model)
weights = weight(model)

# predictions
sigm(u) = 1/(1 + exp(-u))
y_tild = sigm.(X*weights[1:d] + weights[d + 1])
y_pred = 1.0 * (y_tild .>= 0.5)
accuracy = sum(1.0*(y_pred .== y))/n
println("Accuracy: $accuracy")

# CVX
theta = Variable(d + 1)
X_tild = [X ones(n)]
objective = logisticloss(-y.*(X_tild*theta))
problem = Convex.minimize(objective)
solve!(problem, SCSSolver(verbose=false, max_iters=10000), verbose=false)
weights = evaluate(theta)

# predictions
sigm(u) = 1/(1 + exp(-u))
y_tild = sigm.(X_tild*weights)
y_pred = 1.0 * (y_tild .>= 0.5)
accuracy = sum(1.0*(y_pred .== y))/n
println("Accuracy (using Convex.jl): $accuracy")

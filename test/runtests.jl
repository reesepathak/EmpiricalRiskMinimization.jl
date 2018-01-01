using EmpiricalRiskMinimization
using Convex, SCS, ECOS
using Base.Test

TOL = 1e-3


# Squared Loss tests
n = 1000; d = 200;
X = randn(n, d); y = randn(n);
loss_ERM = EmpiricalRiskMinimization.minimize(SquaredLoss(), L1Reg(2.5), X, y)
theta = Variable(d)
cvx_objective = sumsquares(X*theta - y) + 2.5*norm(theta, 1)
cvx_problem = Convex.minimize(cvx_objective)
solve!(cvx_problem, SCSSolver(verbose=false, max_iters=10000, eps=1e-9), verbose=false)
loss_CVX = cvx_problem.optval
println("Convex: $loss_CVX, ERM: $loss_ERM")
@test abs(loss_CVX - loss_ERM) < TOL

n = 1000; d = 200;
X = randn(n, d); y = randn(n);
loss_ERM = EmpiricalRiskMinimization.minimize(SquaredLoss(), L1L2Reg(0.5, 1), X, y)
theta = Variable(d)
cvx_objective = sumsquares(X*theta - y) + 0.5*norm(theta, 1) + sumsquares(theta)
cvx_problem = Convex.minimize(cvx_objective)
solve!(cvx_problem, SCSSolver(verbose=false, max_iters=10000, eps=1e-9), verbose=false)
loss_CVX = cvx_problem.optval
println("Convex: $loss_CVX, ERM: $loss_ERM")
@test abs(loss_CVX - loss_ERM) < TOL

n = 1000; d = 200;
X = randn(n, d); y = randn(n);
loss_ERM = EmpiricalRiskMinimization.minimize(SquaredLoss(), L2Reg(0.5), X, y)
cvx_objective = sumsquares(X*theta - y) + 0.5*sumsquares(theta)
cvx_problem = Convex.minimize(cvx_objective)
solve!(cvx_problem, SCSSolver(verbose=false, max_iters=10000, eps=1e-9), verbose=false)
loss_CVX = cvx_problem.optval
# Use closed form solution for ridge regression
theta_opt = inv(X'*X + 0.5*eye(d))*X'*y
loss_RR= norm(X*theta_opt - y, 2)^2 + 0.5*norm(theta_opt, 2)^2
println("Ridge regression: $loss_RR, ERM: $loss_ERM, (Convex: $loss_CVX)")
@test abs(loss_RR - loss_ERM) < TOL


# L1 loss tests
n = 300; d = 50;
X = randn(n, d); y = randn(n); lambd = rand();
loss_ERM = EmpiricalRiskMinimization.minimize(AbsoluteLoss(), L2Reg(lambd), X, y,
                                              max_iters=20000, verbose=false)
theta = Variable(d)
cvx_objective = norm(X*theta - y, 1) + lambd*norm(theta, 2)^2
cvx_problem = Convex.minimize(cvx_objective)
solve!(cvx_problem, ECOSSolver(verbose=false, max_iters=10000, eps=1e-9), verbose=false)
loss_CVX = cvx_problem.optval
println("Convex: $loss_CVX, ERM: $loss_ERM")
@test abs(loss_CVX - loss_ERM) < TOL

n = 500; d = 200;
X = randn(n, d); y = randn(n); lambd1 = rand(); lambd2 = rand();
loss_ERM = EmpiricalRiskMinimization.minimize(AbsoluteLoss(), L1L2Reg(lambd1, lambd2),
                                              X, y, max_iters=20000, verbose=false)
theta = Variable(d)
cvx_objective = norm(X*theta - y, 1) + lambd1*norm(theta, 1) + lambd2*norm(theta, 2)^2
cvx_problem = Convex.minimize(cvx_objective)
solve!(cvx_problem, SCSSolver(verbose=false, max_iters=10000, eps=1e-9), verbose=false)
loss_CVX = cvx_problem.optval
println("Convex: $loss_CVX, ERM: $loss_ERM")
@test abs(loss_CVX - loss_ERM) < TOL

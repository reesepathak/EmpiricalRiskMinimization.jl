using EmpiricalRiskMinimization
srand(123)

# data
n = 2000; k = 30;
d = k + 1;
X = randn(n, k); theta = randn(d);
y = [ones(n) X] * theta + 0.5 * randn(n);

# model
M = Model(X, y, embedall=true) 
train(M); 
status(M);
theta = thetaopt(M);

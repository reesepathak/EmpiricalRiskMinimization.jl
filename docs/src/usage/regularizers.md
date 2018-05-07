# [Regularizers](@id usage_regularizers)

Below we enumerate the regularizers implemented by ERM, and provide their mathematical definition. See the
[validation page](@ref usage_regpath) for information about regularization paths.

## Mathematical definitions

|       name                 |  ERM `Regularizer`  | mathematical definition                                                            |  notes     |
| ------------------------   | :---------------------:| :-----------------------------------------------------------------------------: | :---------:|
| L2 ($\ell_2$)              |  `L2Reg()`        | $r(\theta) = \\|\theta\\|_2 = \left(\sum_{i=1}^n (\theta_i)^{2}\right)^{\frac{1}{2}}$   |  convex   |
| L1 ($\ell_1$)              |  `L1Reg()`        | $r(\theta) = \\|\theta\\|_1 = \sum_{i=1}^n\|\theta_i\|$                      | convex, sparsifying|
| Square root ($\ell_{0.5}$) |  `SqrtReg()`      | $r(\theta) = \left(\sum_{i=1}^n \|\theta_i\|^{1/2} \right)^{2}$ |non-convex, sparsifying|
| Nonnegative   |  `NonnegReg()`    | $r(\theta) = \begin{cases} 0 & \theta_i \geq 0 \text{  for all i} \\\\ +\infty & \text{else} \end{cases}$ | convex |

A good reference for regularizers are the [EE104](http://ee104.stanford.edu) lecture slides. In particular, the
[lecture on non-quadratic regularizers](http://ee104.stanford.edu/lectures/regularizers.pdf) is helpful.


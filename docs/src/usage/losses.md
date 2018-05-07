# [Losses](@id usage_losses)

Below we enumerate the loss functions implemented by ERM, and provide their mathematical definition.
Some loss functions (e.g., `HuberLoss`) accept parameters. 

## Mathematical definitions

|       name   |     ERM `Loss`     | mathematical definition                                                         |  notes         |
| -----------  | :-----------------:| :-----------------------------------------------------------------------------: | :-------------:|
| squared      |  `SquareLoss()`    | $l^{\mathrm{sqr}}(\widehat{y}, y) = (\widehat{y} - y)^2$                        | n/a            |
| absolute     |  `AbsoluteLoss()`  | $l^{\mathrm{abs}}(\widehat y, y) =  \|\widehat y - y\|$                                  | n/a            |
| tilted       |  `TiltedLoss()`    | $l^{\mathrm{tlt}}(\widehat y, y) = \tau(\widehat y - y)_+ + (1 - \tau)(\widehat y - y)_{-}$ | $0 < \tau < 1$ |  
| deadzone     |  `DeadzoneLoss()`  | $l^{\mathrm{dz}}(\widehat y, y) = \max(\|\widehat y - y\| - \alpha, 0)$                   | $\alpha \geq 0$|
| Huber        |  `HuberLoss()`     | $l^{\mathrm{hub}}(\widehat y, y) = \begin{cases} (\widehat{y} - y)^2 & \|\widehat{y} - y\| \leq \alpha \\\\ \alpha(2\|\widehat{y}\| - \alpha) & \|\widehat{y} - y\| > \alpha \end{cases}$ | $\alpha \geq 0$  |
| log Huber    |  `LogHuberLoss()`  | $l^{\mathrm{dh}}(\widehat y, y) = \begin{cases} (\widehat{y} - y)^2 & \|\widehat{y} - y\| \leq \alpha \\\\ \alpha^2(1 + 2(\log(\widehat{y} - y) - \log(\alpha))) & \|\widehat{y} - y\| > \alpha \end{cases}$ | $\alpha \geq 0$  |
| hinge        |  `HingeLoss()`     | $l^{\mathrm{hng}}(\widehat y, y) = \max(1 - \widehat{y} y, 0)$ |  n/a         |
| logistic     |  `LogisticLoss()`  | $l^{\mathrm{lgt}}(\widehat y, y) = $                    |       n/a    |
| sigmoid      |  `SigmoidLoss()`   | $l^{\mathrm{sigm}}(\widehat y, y) = $                  |         n/a  |

A good reference for loss functions are the [EE104](http://ee104.stanford.edu) lecture slides. In particular, the
[lecture on non-quadratic losses](http://ee104.stanford.edu/lectures/losses.pdf) is helpful.

## Passing parameters

Some of the loss functions above accept parameters. To pass a parameter, simply provide it as the only argument to the `Loss` constructor.
For example, to provide $\alpha$ for $l^{\mathrm{hub}}$, simply instantiate the loss with `HuberLoss(alpha)` where `alpha >= 0`.

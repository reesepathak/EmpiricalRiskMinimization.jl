# EmpiricalRiskMinimization.jl

EmpiricalRiskMinimization.jl, abbreviated ERM or ERM.jl,
is a Julia package which bulids, trains, and tests models for use on various sources of data.
With ERM you specify a model by choosing a loss and regularizer as well as embedding for your dataset.
Then, this package carries out numerical optimization 
required to solve your (regularized) empirical risk minimization problem. The package then includes various
analysis and validation features.

Empirical risk minimization problems have numerous applications, including in
machine learning, control, computer vision, signal processing 
many more disciplines. The focus of this package is to provide easy-to-use interfaces, allowing users to
focus on model development rather than implementation.

ERM was first developed in the spring of 2018 for the course EE104, "Introduction to Machine Learning,"
at Stanford University. The course was taught by Stephen Boyd and Sanjay Lall with the help of
Reese Pathak and Guillermo Angeris. The code was developed to be educational and the notation used in the code
(mostly) follows the standard set by the course.

```@meta
CurrentModule = EmpiricalRiskMinimization
```

## Getting Started

For now, the package should be installed using the following line of code.
```
Pkg.clone("https://github.com/reesepathak/EmpiricalRiskMinimization.jl.git")
Pkg.update()
```
Although the package has been submitted to the official Julia repository, if you include it using
`Pkg.add()` or with one of the other standard installation methods, you will not recieve the most
recent version of the code. The code is very volatile right now, so it is essential that users
`Pkg.update()` prior to using the code.

Then head over to [our ERM walkthrough](examples/walkthrough.html) to begin using ERM.jl.
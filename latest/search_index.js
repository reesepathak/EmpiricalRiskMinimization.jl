var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#EmpiricalRiskMinimization.jl-1",
    "page": "Home",
    "title": "EmpiricalRiskMinimization.jl",
    "category": "section",
    "text": "EmpiricalRiskMinimization.jl, abbreviated ERM or ERM.jl, is a Julia package which bulids, trains, and tests models for use on various sources of data. With ERM you specify a model by choosing a loss and regularizer as well as embedding for your dataset. Then, this package carries out numerical optimization  required to solve your (regularized) empirical risk minimization problem. The package then includes various analysis and validation features.Empirical risk minimization problems have numerous applications, including in machine learning, control, computer vision, signal processing  many more disciplines. The focus of this package is to provide easy-to-use interfaces, allowing users to focus on model development rather than implementation.ERM was first developed in the spring of 2018 for the course EE104, \"Introduction to Machine Learning,\" at Stanford University. The course was taught by Stephen Boyd and Sanjay Lall with the help of Reese Pathak and Guillermo Angeris. The code was developed to be educational and the notation used in the code (mostly) follows the standard set by the course.CurrentModule = EmpiricalRiskMinimization"
},

{
    "location": "index.html#Getting-Started-1",
    "page": "Home",
    "title": "Getting Started",
    "category": "section",
    "text": "For now, the package should be installed using the following line of code.Pkg.clone(\"https://github.com/reesepathak/EmpiricalRiskMinimization.jl.git\")\nPkg.update()Although the package has been submitted to the official Julia repository, if you include it using Pkg.add() or with one of the other standard installation methods, you will not recieve the most recent version of the code. The code is very volatile right now, so it is essential that users Pkg.update() prior to using the code.Then head over to [usage/introduction the first examples page] to begin using ERM.jl!"
},

{
    "location": "usage/introduction.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "usage/installation.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "examples/index.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "functions/losses.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "functions/regularizers.html#",
    "page": "-",
    "title": "-",
    "category": "page",
    "text": ""
},

{
    "location": "lib/losses.html#",
    "page": "Loss functions",
    "title": "Loss functions",
    "category": "page",
    "text": ""
},

{
    "location": "lib/losses.html#EmpiricalRiskMinimization.SquareLoss",
    "page": "Loss functions",
    "title": "EmpiricalRiskMinimization.SquareLoss",
    "category": "type",
    "text": "SquareLoss()\n\nConstructs the squared loss or l2 loss.\n\n\n\n"
},

{
    "location": "lib/losses.html#Loss-functions-1",
    "page": "Loss functions",
    "title": "Loss functions",
    "category": "section",
    "text": "CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"losses.jl\"]"
},

{
    "location": "lib/modeling.html#",
    "page": "Models and Training",
    "title": "Models and Training",
    "category": "page",
    "text": ""
},

{
    "location": "lib/modeling.html#EmpiricalRiskMinimization.status-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Models and Training",
    "title": "EmpiricalRiskMinimization.status",
    "category": "method",
    "text": "status(M) Prints and returns the status of the ERM model, M. Will only  print the most recent action (e.g., training, regularization path, etc.)  performed on the model. \n\n\n\n"
},

{
    "location": "lib/modeling.html#Models-and-Training-1",
    "page": "Models and Training",
    "title": "Models and Training",
    "category": "section",
    "text": "CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"model.jl\"]"
},

]}

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
    "text": "For now, the package should be installed using the following line of code.Pkg.clone(\"https://github.com/reesepathak/EmpiricalRiskMinimization.jl.git\")\nPkg.update()Although the package has been submitted to the official Julia repository, if you include it using Pkg.add() or with one of the other standard installation methods, you will not recieve the most recent version of the code. The code is very volatile right now, so it is essential that users Pkg.update() prior to using the code.Then head over to our ERM walkthrough to begin using ERM.jl."
},

{
    "location": "examples/walkthrough.html#",
    "page": "Walkthrough",
    "title": "Walkthrough",
    "category": "page",
    "text": ""
},

{
    "location": "examples/walkthrough.html#Walkthrough-1",
    "page": "Walkthrough",
    "title": "Walkthrough",
    "category": "section",
    "text": "This page gives an overview of how to use EmpiricalRiskMinimization.jl on a basic dataset. "
},

{
    "location": "examples/additional_examples.html#",
    "page": "Usage examples",
    "title": "Usage examples",
    "category": "page",
    "text": ""
},

{
    "location": "examples/additional_examples.html#Usage-examples-1",
    "page": "Usage examples",
    "title": "Usage examples",
    "category": "section",
    "text": "The links below reference IPython notebooks where you can see how to apply EmpiricalRiskMinimization.jl for various (increasingly complex) use cases."
},

{
    "location": "usage/modelling.html#",
    "page": "ERM models",
    "title": "ERM models",
    "category": "page",
    "text": ""
},

{
    "location": "usage/modelling.html#ERM-models-1",
    "page": "ERM models",
    "title": "ERM models",
    "category": "section",
    "text": ""
},

{
    "location": "usage/losses.html#",
    "page": "Losses",
    "title": "Losses",
    "category": "page",
    "text": ""
},

{
    "location": "usage/losses.html#Losses-1",
    "page": "Losses",
    "title": "Losses",
    "category": "section",
    "text": ""
},

{
    "location": "usage/regularizers.html#",
    "page": "Regularizers",
    "title": "Regularizers",
    "category": "page",
    "text": ""
},

{
    "location": "usage/regularizers.html#Regularizers-1",
    "page": "Regularizers",
    "title": "Regularizers",
    "category": "section",
    "text": ""
},

{
    "location": "lib/losses.html#",
    "page": "Losses",
    "title": "Losses",
    "category": "page",
    "text": ""
},

{
    "location": "lib/losses.html#EmpiricalRiskMinimization.SquareLoss",
    "page": "Losses",
    "title": "EmpiricalRiskMinimization.SquareLoss",
    "category": "type",
    "text": "SquareLoss()\n\nConstructs the squared loss or l2 loss.\n\n\n\n"
},

{
    "location": "lib/losses.html#Losses-1",
    "page": "Losses",
    "title": "Losses",
    "category": "section",
    "text": "CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"losses.jl\"]"
},

{
    "location": "lib/modeling.html#",
    "page": "Modelling and training",
    "title": "Modelling and training",
    "category": "page",
    "text": ""
},

{
    "location": "lib/modeling.html#EmpiricalRiskMinimization.status-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Modelling and training",
    "title": "EmpiricalRiskMinimization.status",
    "category": "method",
    "text": "status(M) Prints and returns the status of the ERM model, M. Will only  print the most recent action (e.g., training, regularization path, etc.)  performed on the model. \n\n\n\n"
},

{
    "location": "lib/modeling.html#Modelling-and-training-1",
    "page": "Modelling and training",
    "title": "Modelling and training",
    "category": "section",
    "text": "CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"model.jl\"]"
},

]}

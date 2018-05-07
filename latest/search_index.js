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
    "text": "This page gives you a very basic example of how to use EmpiricalRiskMinimization.jl, with links to other documentation pages to learn more about advanced functionality available within the package.Suppose we want to solve a regularized least square linear regression problem. Let\'s first generate some data.using EmpiricalRiskMinimization\nsrand(123)n = 2000; k = 30;\nd = k + 1;\nU = randn(n, k); theta = randn(d);\nv = [ones(n) U] * theta + 0.5 * randn(n);So, we\'ve generated 2000 random raw data points, occuping the rows of U. These data points have 30 features. Additionally, we generated targets v, so that v[i] is the label associated with example U[i, :].Formulating and solving (regularized) least square linear regression with ERM.jl is simple. The first step is to instantiate the modelM = Model(U, v, embedall=true);The option embedall=true takes U and compiles our true training data X, by appending the constant feature to the rows of U. Additionally, it standardizes our data for us. There are many more features available for training, embedding, and modelling. Of course, to specify a different model, users must specify different losses and regularizers. Training the model and getting the output is two lines of code.train(M)\nstatus(M)This training summary is useful, and is the most basic validation tool that ERM provides; cross-validation and repeated out-of-sample validation are also available.To assess the accuracy of the model on the train and test sets, we can compute the (average) train and test losses.println(\"Training error = $(trainloss(M))\")\nprintln(\"Testing error = $(testloss(M))\")Finally, suppose we actually want to retrieve our predictions on the test data.v_test_pred = predict_v_from_test(M);There are more prediction functions available. These allow you to provide alternative model parameters, unembed predictions, and test on various other datasets. "
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
    "location": "examples/additional_examples.html#Examples-of-models-on-simple-data-1",
    "page": "Usage examples",
    "title": "Examples of models on simple data",
    "category": "section",
    "text": "These examples link to notebooks that demonstrate how to use ERM.jl for various common machine learning models. They use randomly generated data; the focus is on optional and extended features that ERM makes available to users.Linear regression\nRobust regression\nLogistic regression\nSupport vector machine"
},

{
    "location": "examples/additional_examples.html#Examples-of-models-on-real-datasets-1",
    "page": "Usage examples",
    "title": "Examples of models on real datasets",
    "category": "section",
    "text": "These examples link to notebooks demonstrating ho wto use ERM.jl on real data. Many of these examples come from the EE104 lectures.Housing prices\nDiabetes progression "
},

{
    "location": "usage/models.html#",
    "page": "Models and training",
    "title": "Models and training",
    "category": "page",
    "text": ""
},

{
    "location": "usage/models.html#usage_models-1",
    "page": "Models and training",
    "title": "Models and training",
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
    "location": "usage/losses.html#usage_losses-1",
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
    "location": "usage/regularizers.html#usage_regularizers-1",
    "page": "Regularizers",
    "title": "Regularizers",
    "category": "section",
    "text": ""
},

{
    "location": "usage/validation.html#",
    "page": "Validation and out-of-sample testing",
    "title": "Validation and out-of-sample testing",
    "category": "page",
    "text": ""
},

{
    "location": "usage/validation.html#usage_validation-1",
    "page": "Validation and out-of-sample testing",
    "title": "Validation and out-of-sample testing",
    "category": "section",
    "text": ""
},

{
    "location": "usage/prediction.html#",
    "page": "Prediction",
    "title": "Prediction",
    "category": "page",
    "text": ""
},

{
    "location": "usage/prediction.html#usage_prediction-1",
    "page": "Prediction",
    "title": "Prediction",
    "category": "section",
    "text": ""
},

{
    "location": "lib/models.html#",
    "page": "Models and training",
    "title": "Models and training",
    "category": "page",
    "text": ""
},

{
    "location": "lib/models.html#EmpiricalRiskMinimization.Model",
    "page": "Models and training",
    "title": "EmpiricalRiskMinimization.Model",
    "category": "type",
    "text": "Model(...)\n\nThe Model() function constructs an ERM model. The typical invocation is  Model(U, V, Loss(), Reg()), where U and V specify raw inputs and targets,  respectively, and Loss() specifies some type of training loss (default: SquareLoss()) and Reg() specifies some type of regularizer (default: L2Reg(1.0)).  For more details, see the description of ERM models in the usage notes. \n\n\n\n"
},

{
    "location": "lib/models.html#EmpiricalRiskMinimization.status-Tuple{IO,EmpiricalRiskMinimization.Model}",
    "page": "Models and training",
    "title": "EmpiricalRiskMinimization.status",
    "category": "method",
    "text": "Prints and returns the status of the model.\n\n\n\n"
},

{
    "location": "lib/models.html#Models-and-training-1",
    "page": "Models and training",
    "title": "Models and training",
    "category": "section",
    "text": "CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"model.jl\"]"
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
    "location": "lib/regularizers.html#",
    "page": "Regularizers",
    "title": "Regularizers",
    "category": "page",
    "text": ""
},

{
    "location": "lib/regularizers.html#Regularizers-1",
    "page": "Regularizers",
    "title": "Regularizers",
    "category": "section",
    "text": "CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"regularizers.jl\"]"
},

{
    "location": "lib/validation.html#",
    "page": "Validation and out-of-sample testing",
    "title": "Validation and out-of-sample testing",
    "category": "page",
    "text": ""
},

{
    "location": "lib/validation.html#Validation-and-out-of-sample-testing-1",
    "page": "Validation and out-of-sample testing",
    "title": "Validation and out-of-sample testing",
    "category": "section",
    "text": ""
},

{
    "location": "lib/prediction.html#",
    "page": "Prediction",
    "title": "Prediction",
    "category": "page",
    "text": ""
},

{
    "location": "lib/prediction.html#Prediction-1",
    "page": "Prediction",
    "title": "Prediction",
    "category": "section",
    "text": ""
},

]}

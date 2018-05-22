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
    "text": "The basic primitive of EmpiricalRiskMinimization.jl is the model type."
},

{
    "location": "usage/models.html#Creating-a-Model-1",
    "page": "Models and training",
    "title": "Creating a Model",
    "category": "section",
    "text": "There are many ways to instantiate a model. You use the function Model(...) do so. It has the following definitionfunction Model(U, V; loss=SquareLoss(), reg=L2Reg(),\n               Unames = nothing, Vnames = nothing,\n               embedall = false, verbose=false, kwargs...)The first two arguments specify the data: inputs U and targets V. There are many keyword arguments:loss specifies the loss (also called risk) function that will be used to train your model.\nreg specifies the regularizer on the model parameters.\nUnames and Vnames specify names for the columns of U and V, respectively. This mainly applies todatasets that are not entirely numerical.embedall determines whether all of the data in U and V will be used to train. If it is set to true, thenU and V are standardized and a constant feature is added to U, giving X and Y.verbose allows users to see more of the progress that occurs during model usage. It will automatically callstatus() at the end of model actions (see below). "
},

{
    "location": "usage/models.html#Default-parameters-1",
    "page": "Models and training",
    "title": "Default parameters",
    "category": "section",
    "text": "Suppose you create a model on U and v using the following line of code.M = Model(U, V)This specifies a squared loss, and l2 regularization. Additionally, it will create X by standardizing U and adding a constant feature, and it will create Y by standardizing V. Addditionally, it will put regularization on all the weights except for the first weight, corresponding to the constant feature. The simplest way to describe this model is a regularized least squares regression model with regularization on the non-constant parameters."
},

{
    "location": "usage/models.html#Specifying-different-models-1",
    "page": "Models and training",
    "title": "Specifying different models",
    "category": "section",
    "text": ""
},

{
    "location": "usage/models.html#Training-1",
    "page": "Models and training",
    "title": "Training",
    "category": "section",
    "text": "To train a model in ERM, you simply invoke the the train command.There are additional (hidden) parameters available to fine tune the training experience.In the example below, we adjust the regularization weight (default: 1e-10) and the fraction of data used for training (default: .60).train(M, lambda=1e-4, trainfrac=0.6)"
},

{
    "location": "usage/models.html#Status-1",
    "page": "Models and training",
    "title": "Status",
    "category": "section",
    "text": "After carrying out an ERM function (e.g., train) on a model M, you can invoke status(M) to view the outcome of the function."
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
    "text": "Below we enumerate the loss functions implemented by ERM, and provide their mathematical definition. Some loss functions (e.g., HuberLoss) accept parameters. "
},

{
    "location": "usage/losses.html#Mathematical-definitions-1",
    "page": "Losses",
    "title": "Mathematical definitions",
    "category": "section",
    "text": "name ERM Loss mathematical definition  (assuming scalar targets) notes\nsquared SquareLoss() l^mathrmsqr(widehaty y) = (widehaty - y)^2 n/a\nabsolute AbsoluteLoss() l^mathrmabs(widehat y y) =  widehat y - y n/a\ntilted TiltedLoss() l^mathrmtlt(widehat y y) = tau(widehat y - y)_+ + (1 - tau)(widehat y - y)_- 0  tau  1\ndeadzone DeadzoneLoss() l^mathrmdz(widehat y y) = max(widehat y - y - alpha 0) alpha geq 0\nHuber HuberLoss() l^mathrmhub(widehat y y) = begincases (widehaty - y)^2  widehaty - y leq alpha  alpha(2widehaty - alpha)  widehaty - y  alpha endcases alpha geq 0\nlog Huber LogHuberLoss() l^mathrmdh(widehat y y) = begincases (widehaty - y)^2  widehaty - y leq alpha  alpha^2(1 + 2(log(widehaty - y) - log(alpha)))  widehaty - y  alpha endcases alpha geq 0\nhinge HingeLoss() l^mathrmhng(widehat y y) = max(1 - widehaty y 0) n/a\nlogistic LogisticLoss() l^mathrmlgt(widehat y y) = log(1 + exp(-widehat y y) n/a\nsigmoid SigmoidLoss() l^mathrmsigm(widehat y y) = 1(1 + exp(widehat y y)) n/aA good reference for loss functions are the EE104 lecture slides. In particular, the lecture on non-quadratic losses is helpful."
},

{
    "location": "usage/losses.html#Passing-parameters-1",
    "page": "Losses",
    "title": "Passing parameters",
    "category": "section",
    "text": "Some of the loss functions above accept parameters. To pass a parameter, simply provide it as the only argument to the Loss constructor. For example, to provide alpha for l^mathrmhub, simply instantiate the loss with HuberLoss(alpha) where alpha >= 0."
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
    "text": "Below we enumerate the regularizers implemented by ERM, and provide their mathematical definition. "
},

{
    "location": "usage/regularizers.html#Mathematical-definitions-1",
    "page": "Regularizers",
    "title": "Mathematical definitions",
    "category": "section",
    "text": "name ERM Regularizer mathematical definition notes\nL2 (ell_2) L2Reg() r(theta) = theta_2 = left(sum_i=1^n (theta_i)^2right)^frac12 convex\nL1 (ell_1) L1Reg() r(theta) = theta_1 = sum_i=1^ntheta_i convex, sparsifying\nSquare root (ell_05) SqrtReg() r(theta) = left(sum_i=1^n theta_i^12 right)^2 non-convex, sparsifying\nNonnegative NonnegReg() r(theta) = begincases 0  theta_i geq 0 text  for all i  +infty  textelse endcases convexA good reference for regularizers are the EE104 lecture slides. In particular, the lecture on non-quadratic regularizers is helpful."
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
    "location": "usage/validation.html#Recovering-losses-and-optimal-parameters-1",
    "page": "Validation and out-of-sample testing",
    "title": "Recovering losses and optimal parameters",
    "category": "section",
    "text": "To view the train and test loss on a particular model M, you simply invoke trainloss and testloss on the model.To get the optimal theta and lambda recovered from training, you use thetaopt(M) and lambdaopt(M). To recover the list of lambda values used for a regularization path, you simply use lambdapath(M). Similarly, the optimal thetas are found via thetapath(M), and you can find the corresponding training and test losses through trainloss(M) and testloss(M), respectively. "
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
    "text": "After training and validating a model, we often want to use models on unseen data. In this page, we describe the prediction functions available in ERM."
},

{
    "location": "usage/prediction.html#Using-a-trained-model-on-new-data-1",
    "page": "Prediction",
    "title": "Using a trained model on new data",
    "category": "section",
    "text": "The most basic way to do prediction is to use predict.predict(M, x)\n````\nThe code above takes as input a model `M` and data point `x`, from which a prediction is\nformed according to the model\'s parameters. You can also input a different choice of `theta`,\nas in `predict(M, x, theta)`. By default, it is set to the optimal `theta` stored in `M`.\n\nOf course, in many settings you may want to predict on a certain set of data.\n\nFor example the following two lines of code will allow you to compute predictions on the training\nset. julia predict_y_from_train(M) predict_v_from_train(M)These functions allow you to compute embedded and unembedded predictions (corresponding to `y` and `v`, respectively) on the\ntrain set of `M`.\njulia predict_y_from_test(M) predict_v_from_test(M)These functions allow you to compute embedded and unembedded predictions (corresponding to `y` and `v`, respectively) on the\ntest set of `M`.\n\nAdditionally, if you would rather provide a single raw input `u`, we provide all the prediction functions\nyou could ever want. julia predict_y_from_u(M) predict_v_from_u(M)\n## Recovering losses\n\nYou can compute the train and test losses using `trainloss(M)` and `testloss(M)`, respectively. \n\nYou often also want to compute a confusion matrix when solving classification problems.julia confusion_train(M) confusion_test(M) ``These two functions compute confusion matrices on the train and test sets (respectively) forM`."
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
    "text": "Model(...)\n\nThe Model() function constructs an ERM model. The typical invocation is  Model(U, V, Loss(), Reg()), where U and V specify raw inputs and targets,  respectively, and Loss() specifies some type of training loss (default: SquareLoss()) and Reg() specifies some type of regularizer (default: L2Reg()).  For more details, see the description of ERM models in the usage notes. \n\n\n\n"
},

{
    "location": "lib/models.html#EmpiricalRiskMinimization.status-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Models and training",
    "title": "EmpiricalRiskMinimization.status",
    "category": "method",
    "text": "status(M) prints the status of the model after the most recent action performed on it.\n\n\n\n"
},

{
    "location": "lib/models.html#EmpiricalRiskMinimization.status-Tuple{IO,EmpiricalRiskMinimization.Model}",
    "page": "Models and training",
    "title": "EmpiricalRiskMinimization.status",
    "category": "method",
    "text": "Prints and returns the status of the model.\n\n\n\n"
},

{
    "location": "lib/models.html#EmpiricalRiskMinimization.train-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Models and training",
    "title": "EmpiricalRiskMinimization.train",
    "category": "method",
    "text": "train(M [, lambda=1e-10, trainfrac=nothing])  This function trains a model M. The usual invocation is  train(M). Users may choose to specify a different choice of  regularization weight lambda. For example to specify  a weight of lambda = 0.01, one invokes  train(M, lambda=0.001), and to specify a different train split,  one invokes train(M, trainfrac=0.75), which means that  75 percent of the data will be used for training and only 25 percent will  be used for test. The default parameters are  lambda = 1e-10 and trainfrac=nothing, which will result in a  80-20 train-test split.\n\n\n\n"
},

{
    "location": "lib/models.html#EmpiricalRiskMinimization.trainfolds-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Models and training",
    "title": "EmpiricalRiskMinimization.trainfolds",
    "category": "method",
    "text": "trainfolds(M [,lambda=1e-10, nfolds=5]) \n\nThe trainfolds function carries out n-fold cross validation on  a model M. Specify regularization weight through optional argument lambda, and the number of folds through nfolds. Default:  lambda=1e-10, and nfolds=5.\n\n\n\n"
},

{
    "location": "lib/models.html#EmpiricalRiskMinimization.trainpath-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Models and training",
    "title": "EmpiricalRiskMinimization.trainpath",
    "category": "method",
    "text": "trainpath(M [,lambda=logspace(-5, 5, 100), trainfrac=0.8])\n\nThe trainpath function trains a model M over a set of regularization weights.  Specify these weights by invoking the optional argument lambda, and set a train-test  ratio by using the optional argument trainfrac. \n\nDefaults: \n\nlambda=logspace(-5,5,100) so training occurs over lambda between 1e-5 and \n\n1e5. \n\ntrainfrac=0.8, so training occurs with a 80-20 train-test split.\n\nExample: trainpath(M, lambda=logspace(-1, 1, 100)) trains over lambda between 0.1 and 10. \n\nExample trainpath(M, trainfrac=0.75) trains w/ 75-25 train-test split.\n\n\n\n"
},

{
    "location": "lib/models.html#Models-and-training-1",
    "page": "Models and training",
    "title": "Models and training",
    "category": "section",
    "text": "These are the exported modelling and training functions and types made available by ERM. See the corresponding usage page to understand how to use these methods. CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nPages = [\"model.jl\"]"
},

{
    "location": "lib/losses.html#",
    "page": "Losses",
    "title": "Losses",
    "category": "page",
    "text": ""
},

{
    "location": "lib/losses.html#EmpiricalRiskMinimization.AbsoluteLoss",
    "page": "Losses",
    "title": "EmpiricalRiskMinimization.AbsoluteLoss",
    "category": "type",
    "text": "AbsoluteLoss() constructs the l1/absolute loss. Use with Model().\n\n\n\n"
},

{
    "location": "lib/losses.html#EmpiricalRiskMinimization.HingeLoss",
    "page": "Losses",
    "title": "EmpiricalRiskMinimization.HingeLoss",
    "category": "type",
    "text": "HingeLoss() constructs the hinge loss (i.e., for SVM). Use with Model().\n\n\n\n"
},

{
    "location": "lib/losses.html#EmpiricalRiskMinimization.HuberLoss",
    "page": "Losses",
    "title": "EmpiricalRiskMinimization.HuberLoss",
    "category": "type",
    "text": "HuberLoss() constructs Huber loss.  Use with Model().  Can also invoke as HuberLoss(alpha), which allows specification  of the tradeoff parameter alpha > 0. Note that HuberLoss() defaults  to alpha = 1.0. \n\n\n\n"
},

{
    "location": "lib/losses.html#EmpiricalRiskMinimization.LogisticLoss",
    "page": "Losses",
    "title": "EmpiricalRiskMinimization.LogisticLoss",
    "category": "type",
    "text": "LogisticLoss() constructs the logistic loss for classification problems. Use with Model()\n\n\n\n"
},

{
    "location": "lib/losses.html#EmpiricalRiskMinimization.SigmoidLoss",
    "page": "Losses",
    "title": "EmpiricalRiskMinimization.SigmoidLoss",
    "category": "type",
    "text": "SigmoidLoss() constructs the sigmoid loss for classification problems. Use with Model()\n\n\n\n"
},

{
    "location": "lib/losses.html#EmpiricalRiskMinimization.SquareLoss",
    "page": "Losses",
    "title": "EmpiricalRiskMinimization.SquareLoss",
    "category": "type",
    "text": "SquareLoss() constructs the l2/squared loss. Use with Model().\n\n\n\n"
},

{
    "location": "lib/losses.html#Losses-1",
    "page": "Losses",
    "title": "Losses",
    "category": "section",
    "text": "These are all the available loss functions exported by ERM. See the corresponding usage page to understand how and when to use these functions.CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"losses.jl\"]"
},

{
    "location": "lib/regularizers.html#",
    "page": "Regularizers",
    "title": "Regularizers",
    "category": "page",
    "text": ""
},

{
    "location": "lib/regularizers.html#EmpiricalRiskMinimization.L1Reg",
    "page": "Regularizers",
    "title": "EmpiricalRiskMinimization.L1Reg",
    "category": "type",
    "text": "L1Reg() constructs the L1 regularizer. Use with Model()\n\n\n\n"
},

{
    "location": "lib/regularizers.html#EmpiricalRiskMinimization.L2Reg",
    "page": "Regularizers",
    "title": "EmpiricalRiskMinimization.L2Reg",
    "category": "type",
    "text": "L2Reg() constructs the L2 regularizer. Use with Model()\n\n\n\n"
},

{
    "location": "lib/regularizers.html#EmpiricalRiskMinimization.NonnegReg",
    "page": "Regularizers",
    "title": "EmpiricalRiskMinimization.NonnegReg",
    "category": "type",
    "text": "NonnegReg() is the nonnegative regularizer. Use with Model()\n\n\n\n"
},

{
    "location": "lib/regularizers.html#EmpiricalRiskMinimization.SqrtReg",
    "page": "Regularizers",
    "title": "EmpiricalRiskMinimization.SqrtReg",
    "category": "type",
    "text": "SqrtReg() computes the square root regularizer sqrt(abs(x_i)). Use with Model()\n\n\n\n"
},

{
    "location": "lib/regularizers.html#Regularizers-1",
    "page": "Regularizers",
    "title": "Regularizers",
    "category": "section",
    "text": "These are the regularizers made available by ERM to users. See the corresponding usage page to understand how and when to use these functions.CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:function, :type]\nPages = [\"regularizers.jl\"]"
},

{
    "location": "lib/validation.html#",
    "page": "Validation and out-of-sample testing",
    "title": "Validation and out-of-sample testing",
    "category": "page",
    "text": ""
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.lambda-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.lambda",
    "category": "method",
    "text": "lambda(M) returns the lambda used during training. If multiple lambda values were specified, returns the optimal one\n\n\n\n"
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.lambdaopt-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.lambdaopt",
    "category": "method",
    "text": "lambdaopt(M) returns the optimal lambda computed during training of M\n\n\n\n"
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.lambdapath-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.lambdapath",
    "category": "method",
    "text": "lambdapath(M) returns the vector of lambdas used\n\n\n\n"
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.testloss-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.testloss",
    "category": "method",
    "text": "testloss(M) returns the average testing loss\n\n\n\n"
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.testlosspath-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.testlosspath",
    "category": "method",
    "text": "testlosspath(M) returns the vector of test losses at various lambdas\n\n\n\n"
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.thetaopt-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.thetaopt",
    "category": "method",
    "text": "thetaopt(M) returns the optimal theta chosen during training\n\n\n\n"
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.thetapath-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.thetapath",
    "category": "method",
    "text": "thetapath(M) returns the vector of optimal thetas at various lambdas\n\n\n\n"
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.trainloss-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.trainloss",
    "category": "method",
    "text": "trainloss(M) returns the average training loss\n\n\n\n"
},

{
    "location": "lib/validation.html#EmpiricalRiskMinimization.trainlosspath-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Validation and out-of-sample testing",
    "title": "EmpiricalRiskMinimization.trainlosspath",
    "category": "method",
    "text": "trainlosspath(M) returns the vector of train losses at various lambdas\n\n\n\n"
},

{
    "location": "lib/validation.html#Validation-and-out-of-sample-testing-1",
    "page": "Validation and out-of-sample testing",
    "title": "Validation and out-of-sample testing",
    "category": "section",
    "text": "These are all the available library functions that users have available for validation and out-of-sample testing. Refer to the corresponding usage page to see examples and explanations behind these functions. CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"validation.jl\"]"
},

{
    "location": "lib/prediction.html#",
    "page": "Prediction",
    "title": "Prediction",
    "category": "page",
    "text": ""
},

{
    "location": "lib/prediction.html#EmpiricalRiskMinimization.predict_v_from_test",
    "page": "Prediction",
    "title": "EmpiricalRiskMinimization.predict_v_from_test",
    "category": "function",
    "text": "predict_v_from_test(M [, theta])\n\nAllows you compute unembedded predictions (i.e., in V space)  based on a trained ERM model M on test data. Option to  specify a choice of theta. It defaults to theta=thetaopt(M)\n\n\n\n"
},

{
    "location": "lib/prediction.html#EmpiricalRiskMinimization.predict_v_from_train",
    "page": "Prediction",
    "title": "EmpiricalRiskMinimization.predict_v_from_train",
    "category": "function",
    "text": "predict_v_from_train(M [, theta])\n\nAllows you compute unembedded predictions (i.e., in V space)  based on a trained ERM model M on train data. Option to  specify a choice of theta. It defaults to theta=thetaopt(M)\n\n\n\n"
},

{
    "location": "lib/prediction.html#EmpiricalRiskMinimization.predict_v_from_u-Union{Tuple{EmpiricalRiskMinimization.Model,Array{T,1},Any}, Tuple{EmpiricalRiskMinimization.Model,Array{T,1}}, Tuple{T}} where T",
    "page": "Prediction",
    "title": "EmpiricalRiskMinimization.predict_v_from_u",
    "category": "method",
    "text": "predict_y_from_u(M, U [, theta])\n\nAllows you compute unembedded predictions (i.e., in V space)  based on a trained ERM model M on one or many raw inputs, U. Option to  specify a choice of theta. It defaults to theta=thetaopt(M)\n\n\n\n"
},

{
    "location": "lib/prediction.html#EmpiricalRiskMinimization.predict_y_from_test",
    "page": "Prediction",
    "title": "EmpiricalRiskMinimization.predict_y_from_test",
    "category": "function",
    "text": "predict_y_from_test(M [, theta])\n\nAllows you compute embedded predictions (i.e., y values) based on a trained ERM model M on test data. Option to  specify a choice of theta. It defaults to theta=thetaopt(M)\n\n\n\n"
},

{
    "location": "lib/prediction.html#EmpiricalRiskMinimization.predict_y_from_train",
    "page": "Prediction",
    "title": "EmpiricalRiskMinimization.predict_y_from_train",
    "category": "function",
    "text": "predict_y_from_train(M [, theta])\n\nAllows you compute embedded predictions (i.e., y values)  based on a trained ERM model M on train data. Option to  specify a choice of theta. It defaults to theta=thetaopt(M)\n\n\n\n"
},

{
    "location": "lib/prediction.html#EmpiricalRiskMinimization.predict_y_from_u-Union{Tuple{EmpiricalRiskMinimization.Model,Array{T,1},Any}, Tuple{EmpiricalRiskMinimization.Model,Array{T,1}}, Tuple{T}} where T",
    "page": "Prediction",
    "title": "EmpiricalRiskMinimization.predict_y_from_u",
    "category": "method",
    "text": "predict_y_from_u(M, U [, theta])\n\nAllows you compute embedded predictions (i.e., y values)  based on a trained ERM model M on one or many raw inputs, U. Option to  specify a choice of theta. It defaults to theta=thetaopt(M)\n\n\n\n"
},

{
    "location": "lib/prediction.html#EmpiricalRiskMinimization.confusion_test-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Prediction",
    "title": "EmpiricalRiskMinimization.confusion_test",
    "category": "method",
    "text": "confusion_test(M)\n\nReturns the confusion matrix based on test data for a trained ERM model\n\n\n\n"
},

{
    "location": "lib/prediction.html#EmpiricalRiskMinimization.confusion_train-Tuple{EmpiricalRiskMinimization.Model}",
    "page": "Prediction",
    "title": "EmpiricalRiskMinimization.confusion_train",
    "category": "method",
    "text": "confusion_train(M)\n\nReturns the confusion matrix based on training data for a trained ERM model\n\n\n\n"
},

{
    "location": "lib/prediction.html#Prediction-1",
    "page": "Prediction",
    "title": "Prediction",
    "category": "section",
    "text": "The functions listed below are all the methods made available by ERM for predicting y and v values on training and testing data. See the corresponding usage page to understand how to use these functions.CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nOrder = [:type, :function]\nPages = [\"prediction.jl\"]"
},

{
    "location": "lib/convenience.html#",
    "page": "Convenience functions",
    "title": "Convenience functions",
    "category": "page",
    "text": ""
},

{
    "location": "lib/convenience.html#EmpiricalRiskMinimization.matrix-Union{Tuple{Array{T,1}}, Tuple{T}} where T",
    "page": "Convenience functions",
    "title": "EmpiricalRiskMinimization.matrix",
    "category": "method",
    "text": "Convert an n-vector to an nx1 array\n\n\n\n"
},

{
    "location": "lib/convenience.html#EmpiricalRiskMinimization.sigm-Tuple{Any}",
    "page": "Convenience functions",
    "title": "EmpiricalRiskMinimization.sigm",
    "category": "method",
    "text": "Sigmoid function\n\n\n\n"
},

{
    "location": "lib/convenience.html#EmpiricalRiskMinimization.findvalue-Tuple{Any,Any}",
    "page": "Convenience functions",
    "title": "EmpiricalRiskMinimization.findvalue",
    "category": "method",
    "text": "findvalue(s, lst): Find the first index of s in a list lst\n\n\n\n"
},

{
    "location": "lib/convenience.html#EmpiricalRiskMinimization.rms-Tuple{Array{Float64,1}}",
    "page": "Convenience functions",
    "title": "EmpiricalRiskMinimization.rms",
    "category": "method",
    "text": "compute the rms of a matrix or a vector\n\n\n\n"
},

{
    "location": "lib/convenience.html#EmpiricalRiskMinimization.rowwise-Union{Tuple{Any,Array{T,2}}, Tuple{T}} where T",
    "page": "Convenience functions",
    "title": "EmpiricalRiskMinimization.rowwise",
    "category": "method",
    "text": "Apply f to each row of a matrix. f should map vectors to vectors\n\n\n\n"
},

{
    "location": "lib/convenience.html#Convenience-functions-1",
    "page": "Convenience functions",
    "title": "Convenience functions",
    "category": "section",
    "text": "ERM implements the following utility functions. CurrentModule = EmpiricalRiskMinimizationModules = [EmpiricalRiskMinimization]\nPages = [\"convenience.jl\"]"
},

]}

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
    "text": "Model(...)\n\nThe Model() function constructs an ERM model. The typical invocation is  Model(U, V, Loss(), Reg()), where U and V specify raw inputs and targets,  respectively, and Loss() specifies some type of training loss (default: SquareLoss()) and Reg() specifies some type of regularizer (default: L2Reg()).  For more details, see the description of ERM models in the usage notes. \n\n\n\n"
},

{
    "location": "lib/models.html#EmpiricalRiskMinimization.setfeatures-Tuple{EmpiricalRiskMinimization.Model,Any}",
    "page": "Models and training",
    "title": "EmpiricalRiskMinimization.setfeatures",
    "category": "method",
    "text": "setfeatures(M, lst) specifies the columns of the input data matrix U to use for training.\n\n\n\n"
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

]}

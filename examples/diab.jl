using EmpiricalRiskMinimization
using CSV, Dataframes

# data from  https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data
# There are 10 explanatory variables, including age (age), 
# sex (sex), body mass index (bmi) and mean arterial blood pressure (map) 
# of 442 patients as well as six blood serum measurements (tc, ldl, hdl,
# tch, ltg and glu). The response (y) is a continuous measure of disease
# progression one year after the baseline measurements.

DATA_URL = "https://web.stanford.edu/~hastie/Papers/LARS/diabetes.data"

function load_data()
    data = download(DATA_URL)
    D, header = CSV.read(data; delim="\t")
    V = floatarray(D[:,11])
    Vnames = Any[header[11]]
    U = floatarray(D[:,1:10])
    Unames = header[1:10]
    return U, V, Unames, Vnames
end

function individual_plots(U, V, colnames, theta, loss, yhat)
    cols = [1 2 3 4 5; 6 7 8 9 10]
    nrows, ncols = size(cols)

    # just plot data
    function f1(ax, i, j)
        col = cols[i,j]
        plot(ax, U[:,col], V, markersize=6)
             #xlabel=string(colnames[col])
    end
    arrayplot(f1, nrows, ncols, name="alldata.pdf")

    # plot regression lines
    function f2(ax, i, j)
        col = cols[i,j]
        plot(ax, U[:,col], V, xlabel=string(colnames[col]), markersize=6)
        regline(ax, theta[col])
        annotate(ax, 0.95, 0.01, @sprintf("MSE %.0f\n", loss[col]), horizontalalignment="right")
    end
    arrayplot(f2, nrows, ncols, name="alldata_reg.pdf")


    # plot performance of BMI, X column 3
    col = 3
    perfplot(yhat[col], V, xmin=0, xmax=350, name="bmi_perf.pdf")
    println("BMI loss = ", loss[col]) 
    showmatrix(theta[col], "BMI theta")


    # plot just bmi regression
    ax = plot(U[:,col], V, xlabel=string(colnames[col]))
    regline(ax, theta[col])
    plt.savefig(plotname("bmi_reg.pdf"))
    

end

function pairwise_table()
    srand(1)
    U, V, Unames, Vnames = loaddata()
    M = Model(U, V, Unames, Vnames, SquareLoss(), L2Reg();
              embedall=true, stand=false)

    
    d = size(U, 2)
    # all regressors, validated
    tests = Any[]
    train(M; trainfrac=0.8)
    push!(tests, ["all", trainloss(M), testloss(M)])

    # no regressors
    train(M, features=[1])
    push!(tests, ["NONE", trainloss(M), testloss(M)])
    
    # one regressor
    for i=1:d
        train(M, features=[1, i+1])
        push!(tests, [Unames[i], trainloss(M), testloss(M)])
    end

    # losses for pairs of regressors
    for i=2:d
        for j=1:i-1
            train(M, features=[1, i+1, j+1])
            push!(tests, [ [Unames[i], Unames[j]], trainloss(M), testloss(M)])
        end
    end
    tests = sort(tests, by = x->x[3])
    for a in tests
        if isa(a[1], Array)
            print(a[1][1], " and ", a[1][2])
        else
            print(a[1])
        end
        @printf("  &  %.4g   &  %.4g \\\\\n", a[2], a[3])
    end
end



function full_regression()
    U, V, Unames, Vnames = loaddata()
    M = Model(U, V, Unames, Vnames, SquareLoss(), L2Reg();
              embedall=true, stand=false)

    train(M, trainfrac=1)
    fulltheta = thetaopt(M)
    fullloss = trainloss(M)
    fullyhat = predict_y_from_train(M)

    ###################################################
    # plotting and display
    @printf("Loss based on all data = %.0f\n", fullloss)
    perfplot(fullyhat, V,  xmin=0, xmax=350, name="all_perf.pdf")
    showmatrix(fulltheta, "all theta")

end

function mean_predictor()
    U, V, Unames, Vnames = loaddata()

    # constant predictor
    n = length(V)
    Vmean = sum(V)/n
    meanloss = (1/n)*(norm(V-fill(Vmean,n))^2)

    println("ymean = ", Vmean)
    @printf("Loss of constant predictor = %.0f\n", meanloss)

end

function individual_regressions()
    U, V, Unames, Vnames = loaddata()
    M = Model(U, V, Unames, Vnames, SquareLoss(), L2Reg();
              embedall=true, stand=false)

    d = size(U,2)
    # compute all single variable regressions
    theta = Array{Any}(d)
    losses = zeros(d)
    yhat = Array{Any}(d)
    for i=1:10
        setfeatures(M, [1,i+1])
        train(M; trainfrac=1)
        theta[i] = thetaopt(M)
        losses[i] = trainloss(M)
        yhat[i] = predict_y_from_train(M)
    end
    individual_plots(U, V, Unames, theta, losses, yhat)
end

function repeated_train()
    U, V, Unames, Vnames = loaddata()
    M = Model(U, V, Unames, Vnames, SquareLoss(), L2Reg();
              embedall=true, stand=false)


    features = [3, 9]  # BMI and S5 in colnames
    features = features + 1 # shift to account for constant feature
    unshift!(features, 1)   # include constant feature in list
    setfeatures(M, features)

    
    trainlosses = Float64[]
    testlosses = Float64[]
    for i=1:1000
        train(M; resplit=true, trainfrac=0.8)
        push!(trainlosses, trainloss(M))
        push!(testlosses, testloss(M))
    end
    plot(testlosses; bins=20, plottype=:hist, xlabel="test loss",
         ylabel="no. of outcomes", name="repeated_test_loss.pdf")
    println("mean test loss = ", mean(testlosses))
    
    plot(trainlosses; bins=20, plottype=:hist, ylabel="training loss",
         xlim=[2000,5000])
    println("mean train loss = ", mean(trainlosses))

end

function main()
    close("all")
    
    mean_predictor()
    individual_regressions()
    full_regression()
    pairwise_table()

    srand(1)
    repeated_train()
end


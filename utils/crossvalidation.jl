# K fold cross validation for the models

function modelCrossValidation(modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})

    seed = 123
    train_set = tuple()

    Random.seed!(seed)

    N = size(inputs, 1)
    k = maximum(crossValidationIndices)
    repetitionsTraining = 1 # For deterministic models

    if modelType == :ANN
        repetitionsTraining = modelHyperparameters["repetitionsTraining"] # Non deterministic model
        targets = oneHotEncoding(targets)
    else
        targets = reshape(targets, :, 1)   

    end

    
    #Lists of metrics for the mean of each fold
    acc_metrics = zeros(Float64, k); err_rate_metrics = zeros(Float64, k);
    sensitivity_metrics = zeros(Float64, k); specificity_metrics = zeros(Float64, k);
    ppv_metrics = zeros(Float64, k); npv_metrics = zeros(Float64, k);
    f_score_metrics = zeros(Float64, k);


    for fold in 1:k
        train_indices = findall(x -> x != fold, crossValidationIndices)
        test_indices = findall(x -> x == fold, crossValidationIndices)
        
        train_set = (inputs[train_indices, :], targets[train_indices, :])
        test_set = (inputs[test_indices, :], targets[test_indices, :])

        fold_acc = Float64[]; fold_err_rate = Float64[]; fold_sensitivity = Float64[];
        fold_specificity = Float64[]; fold_PPV = Float64[]; fold_acc_NPV = Float64[];
        fold_acc_F_score = Float64[];

        for _ in 1:repetitionsTraining
            if modelType == :ANN
                hidden_layer_sizes = tuple(modelHyperparameters["architecture"]...) #Transforms a topology list into a tuple
                activation = modelHyperparameters["activation"]
                learning_rate_init = modelHyperparameters["learning_rate"]
                validation_ratio = modelHyperparameters["validation_ratio"]
                n_iter_no_change = modelHyperparameters["n_iter_no_change"]
                max_iter = modelHyperparameters["max_iter"]
                early_stopping = validation_ratio > 0 ? true : false;
    
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, 
                                      activation=activation, 
                                      learning_rate_init=learning_rate_init, 
                                      validation_fraction=validation_ratio, 
                                      n_iter_no_change=n_iter_no_change,
                                      early_stopping=early_stopping, 
                                      max_iter = max_iter,
                                      verbose = false)
            elseif modelType == :SVM
                
                kernel = modelHyperparameters["kernel"]
                degree = modelHyperparameters["degree"]
                gamma = modelHyperparameters["gamma"]
                C = modelHyperparameters["C"]

                model = SVC(kernel=kernel, degree=degree, gamma=gamma, C=C)

            elseif modelType == :DecisionTree
                
                max_depth = modelHyperparameters["max_depth"]
                model = DecisionTreeClassifier(max_depth=max_depth, random_state=seed)


            elseif modelType == :kNN
                
                n_neighbors = modelHyperparameters["n_neighbors"]
            
                model = KNeighborsClassifier(n_neighbors)

            end

            train_outs = train_set[2]
            if modelType != :ANN
                # Matrix for ann, else a vector is needed
                train_outs = vec(train_set[2])
            end
            
            # Training the model
            ScikitLearn.fit!(model, train_set[1], train_outs);

            predicted_outputs = []
            expected_outs = test_set[2]
            # Getting predictions
            if modelType == :ANN
                proba_outputs = predict_proba(model, test_set[1])
                predicted_outputs = classifyOutputs(proba_outputs)
            else
                predicted_outputs = ScikitLearn.predict(model, test_set[1])
                expected_outs = vec(test_set[2])
            end

            acc, error_rate, sensitivity, specificity, ppv, npv, f_score, _ = confusionMatrix(predicted_outputs, expected_outs);

            push!(fold_acc, acc)
            push!(fold_err_rate, error_rate)
            push!(fold_sensitivity, sensitivity)
            push!(fold_specificity, specificity)
            push!(fold_PPV, ppv)
            push!(fold_acc_NPV, npv)
            push!(fold_acc_F_score, f_score)

        end

        acc_metrics[fold] = mean(fold_acc)
        err_rate_metrics[fold] = mean(fold_err_rate)
        sensitivity_metrics[fold] = mean(fold_sensitivity)
        specificity_metrics[fold] = mean(fold_specificity)
        ppv_metrics[fold] = mean(fold_PPV)
        npv_metrics[fold] = mean(fold_acc_NPV)
        f_score_metrics[fold] = mean(fold_acc_F_score)
    end

    metrics_dict = Dict(
        "acc" => (mean(acc_metrics), std(acc_metrics)),
        "err_rate" => (mean(err_rate_metrics), std(err_rate_metrics)),
        "sensitivity" => (mean(sensitivity_metrics), std(sensitivity_metrics)),
        "specificity" => (mean(specificity_metrics), std(specificity_metrics)),
        "ppv" => (mean(ppv_metrics), std(ppv_metrics)),
        "npv" => (mean(npv_metrics), std(npv_metrics)),
        "f_score" => (mean(f_score_metrics), std(f_score_metrics))
    )

    return metrics_dict
end

function printMetricsSummary(metrics_dict::Dict{String, Tuple{Float64, Float64}})
    sorted_metrics = ["acc", "err_rate", "sensitivity", "specificity", "ppv", "npv", "f_score"]
    
    println("\n--------------- METRICS SUMMARY ---------------")
    
    for key in sorted_metrics
        if haskey(metrics_dict, key)
            println("\n----- ", key, " -----")
            println("Mean:       ", round(metrics_dict[key][1], digits=4))
            println("Std. Dev.:  ", round(metrics_dict[key][2], digits=4))
        end
    end
    
    println("\n----------------------------------------------")
end

# Collect metrics for each set of hyperparameters
function collectMetrics(model, hyperparameters_array, norm_inputs, targets, kFoldIndices)
    all_metrics = []
    for (i, hyperparameters) in enumerate(hyperparameters_array)
        println("Training with set of hyperparameters $i")
        metrics = modelCrossValidation(model, hyperparameters, norm_inputs, targets, kFoldIndices)
        push!(all_metrics, (i, metrics))
    end
    return all_metrics
end

# Sort the metrics, descending by default, ascending for error rate
function sortMetrics(all_metrics, metric_name, descending=true)
    sort!(all_metrics, by=x -> x[2][metric_name][1], rev=descending)
end

# Print the sorted metrics
function printMetricsRanking(all_metrics)
    metrics_to_print = ["acc", "sensitivity", "specificity", "ppv", "npv", "f_score"]
    for metric_name in metrics_to_print
        sortMetrics(all_metrics, metric_name)
        println("\n----- $metric_name -----")
        for (index, metrics) in all_metrics
            println("Set of hyperparameters $index -> mean: $(round(metrics[metric_name][1], digits=3)) Std. Dev.: $(round(metrics[metric_name][2], digits=3))")
        end
    end
    
    # Error rate is sorted in ascending order
    println("\n----- err_rate -----")
    sortMetrics(all_metrics, "err_rate", false)
    for (index, metrics) in all_metrics
        println("Set of hyperparameters $index -> mean: $(round(metrics["err_rate"][1], digits=3)) Std. Dev.: $(round(metrics["err_rate"][2], digits=3))")
    end
end

# Wrapper function to execute the entire process
function evaluateAndPrintMetricsRanking(model, hyperparameters_array, norm_inputs, targets, kFoldIndices)
    all_metrics = collectMetrics(model, hyperparameters_array, norm_inputs, targets, kFoldIndices)
    printMetricsRanking(all_metrics)
end
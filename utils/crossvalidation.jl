using Statistics;
using Flux
using Flux: crossentropy, binarycrossentropy, params
using Random
using Plots

function one_hot_encoding(data::Vector{T}) where T
    values = unique(data)
    numClasses = length(values)

    if length(values) == 2
        encoded = data .== values[1]
        return reshape(encoded, :, 1)
    else
        oneHot = Array{Bool,2}(undef, length(targets), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (targets.==values[numClass]);
        end
        return oneHot
    end
end
    

function oneHotEncoding(feature::AbstractArray{<:Any,1},      
        classes::AbstractArray{<:Any,1})
    # First we are going to set a line as defensive to check values
    @assert(all([in(value, classes) for value in feature]));
    
    # Second defensive statement, check the number of classes
    numClasses = length(classes);
    @assert(numClasses>1)
    
    if (numClasses==2)
        # Case with only two classes
        oneHot = reshape(feature.==classes[1], :, 1);
    else
        #Case with more than two clases
        oneHot =  BitArray{2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
    end;
    return oneHot;
end;


oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature, unique(feature));

oneHotEncoding(feature::AbstractArray{Bool,1}) = reshape(feature, :, 1);



function classifyOutputs(outputs::AbstractArray{<:Real,2}; 
                        threshold::Real=0.5) 
   numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Look for the maximum value using the findmax funtion
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Set up then boolean matrix to everything false while max values aretrue.
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Defensive check if all patterns are in a single class
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;


function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1}) 
    mean(outputs.==targets);
end;


function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}) 
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return mean(all(targets .== outputs, dims=2));
    end;
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};      
                threshold::Real=0.5)
    accuracy(outputs.>=threshold, targets);
end;


function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
                threshold::Real=0.5)
    @assert(all(size(outputs).==size(targets)));
    if (size(targets,2)==1)
        return accuracy(outputs[:,1], targets[:,1]);
    else
        return accuracy(classifyOutputs(outputs; threshold=threshold), targets);
    end;
end;




function holdOut(N::Int, P::Real)
    @assert ((P>=0.) & (P<=1.));
    indices = randperm(N)
    n_train = Int(round((1-P)*N))
    return (indices[1:n_train], indices[n_train + 1:end])
end

function holdOut(N::Int, Pval::Real, Ptest::Real) 
    @assert ((Pval>=0.) & (Pval<=1.))
    @assert ((Ptest>=0.) & (Ptest<=1.))

    train_indices, temp_indices = holdOut(N, Pval + Ptest)
    val_indices, test_indices = holdOut(length(temp_indices), Ptest / (Pval + Ptest))

    return (train_indices, temp_indices[val_indices], temp_indices[test_indices])
end

function plot_losses(losses_dict)
    plots_array = []
    
    for (topology, losses) in losses_dict
        trainingLosses, validationLosses, testLosses = losses

        p = plot(title="Losses vs Epochs for Topology: $topology", size=(900,600))
        
        plot!(p, trainingLosses, label="Training", legend=:topright)
        
        plot!(p, validationLosses, label="Validation", legend=:topright, linestyle=:dash)
        
        plot!(p, testLosses, label="Test", legend=:topright, linestyle=:dot)
        
        push!(plots_array, p)
    end

    # Show all plots
    plot(plots_array...)
end

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    @assert length(outputs) == length(targets) "Outputs and targets must have the same length."

    # Calculate confusion matrix elements
    TP = sum((outputs .== true) .& (targets .== true))
    TN = sum((outputs .== false) .& (targets .== false))
    FP = sum((outputs .== true) .& (targets .== false))
    FN = sum((outputs .== false) .& (targets .== true))

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    error_rate = 1 - accuracy

    # Handle edge cases for metrics
    sensitivity = (TP + FN == 0) ? 1.0 : TP / (TP + FN)
    specificity = (TN + FP == 0) ? 1.0 : TN / (TN + FP)
    positive_predictive_value = (TP + FP == 0) ? 1.0 : TP / (TP + FP)
    negative_predictive_value = (TN + FN == 0) ? 1.0 : TN / (TN + FN)
    denominator_f_score = (positive_predictive_value + sensitivity)
    f_score = (denominator_f_score == 0) ? 0.0 : (2 * positive_predictive_value * sensitivity) / denominator_f_score

    # Create confusion matrix
    confusion = Array{Int64,2}([TN FP; FN TP])

    return accuracy, error_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, confusion
end

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    # Binarize the outputs using the threshold
    binarized_outputs = outputs .>= threshold

    # Use the previously defined confusionMatrix function
    return confusionMatrix(binarized_outputs, targets)
end

using LinearAlgebra

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # Check if number of columns is equal for both matrices
    if size(outputs, 2) != size(targets, 2)
        throw(ArgumentError("Outputs and targets must have the same number of columns"))
    end

    numClasses = size(outputs, 2)

    # If only one column, then it's a binary classification
    if numClasses == 1
        return confusionMatrix(vec(outputs), vec(targets))
    end

    # Initialize metrics
    sensitivity = zeros(numClasses)
    specificity = zeros(numClasses)
    PPV = zeros(numClasses)
    NPV = zeros(numClasses)
    F_score = zeros(numClasses)
    confusion = zeros(Int, numClasses, numClasses)

    validClasses = 0 # Counter for classes with instances

    # Iterate over each class
    for i in 1:numClasses
        if sum(targets[:, i]) != 0 # Check if there are patterns in this class
            _, _, sens, spec, ppv, npv, f, _ = confusionMatrix(outputs[:, i], targets[:, i])
            sensitivity[i] = sens
            specificity[i] = spec
            PPV[i] = ppv
            NPV[i] = npv
            F_score[i] = f
            validClasses += 1
        end

        # Construct the confusion matrix
        for j in 1:numClasses
            confusion[i, j] = sum(outputs[:, i] .& targets[:, j])
        end
    end

    # Aggregate metrics
    if weighted
        weights = sum(targets, dims=1) / size(targets, 1)
        sensitivity = dot(sensitivity, weights)
        specificity = dot(specificity, weights)
        PPV = dot(PPV, weights)
        NPV = dot(NPV, weights)
        F_score = dot(F_score, weights)
    else
        sensitivity = sum(sensitivity) / validClasses
        specificity = sum(specificity) / validClasses
        PPV = sum(PPV) / validClasses
        NPV = sum(NPV) / validClasses
        F_score = sum(F_score) / validClasses
    end

    acc = accuracy(outputs, targets)
    error_rate = 1 - acc
    return acc, error_rate, sensitivity, specificity, PPV, NPV, F_score, confusion
end

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    bool_outputs = classifyOutputs(outputs)
    return confusionMatrix(bool_outputs, targets, weighted=weighted)
end

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Ensure all output classes are included in target classes
    @assert(all([in(output, unique(targets)) for output in outputs]))

    # Get unique classes from both outputs and targets
    classes = unique([outputs; targets])

    # Convert outputs and targets to one-hot encoded form
    encoded_outputs = oneHotEncoding(outputs, classes)
    encoded_targets = oneHotEncoding(targets, classes)

    # Call the confusionMatrix function
    return confusionMatrix(encoded_outputs, encoded_targets, weighted=weighted)
end

function printMatrix(conf_matrix)
    # prints a regular given matrix
    println("-"^50)  # Separator

    for i in 1:size(conf_matrix, 1)
        for j in 1:size(conf_matrix, 2)
            println(string(conf_matrix[i, j], "    ")[1:4])  # Four spaces for alignment
        end
        println()  #New line
    end
    
    println("-"^50) 
end

function crossvalidation(N::Int64, k::Int64)
    folds = collect(1:k) # vector with k sorted elements, from 1 to k
    reps = ceil(Int64, N / k) # Calculate the number of repetitions needed to make the length >= N
    
    repeated_vector = repeat(folds, reps) # Repeat the sorted_vector to make its length >= N
    truncated_vector = repeated_vector[1:N] # Take the first N values
    
    shuffle!(truncated_vector)
    return truncated_vector
end

function crossvalidation(targets::AbstractArray{Bool, 2}, k::Int64)
    N, n_classes = size(targets)
    index_vector = Array{Int64,1}(undef, N);
    
    for c in 1:n_classes
        index_vector[findall(x -> x, targets[:,c])] .= crossvalidation(sum(targets[:,c]), k)
    end
    
    return index_vector
end

function crossvalidation(targets::AbstractArray{<:Any, 1}, k::Int64)
    one_hot_targets = oneHotEncoding(targets) # Convert the targets to one-hot encoding
    return crossvalidation(one_hot_targets, k) # Call the second crossvalidation function
end


#####################   Unit 6   #############################

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
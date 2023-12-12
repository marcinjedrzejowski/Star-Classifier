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

function Min_Max_Scaler(data::Array{Float32, 2})
    # Dim = 1 (Each row) Dim = 2 (Each column)
    # In this case each value should be taken per column, as each column represents one atribute/feature
    
    # It is needed to convert the resultin matrix into vectors in order to use de broadcasting operations
    # Moreover, it's necessary to transpose the vectors to get the correct shape
    
    min_values = vec(minimum(inputs, dims=1))'
    max_values = vec(maximum(inputs, dims=1))'
    mean_values = vec(mean(inputs, dims=1))'
    std_dev_values = vec(std(inputs, dims=1))'
    
    println("Min Values: ", min_values)
    println("Max Values: ", max_values)
    println("Mean Values: ", mean_values)
    println("Standard Deviation Values: ", std_dev_values)

    
    min_max_ranges = max_values .- min_values
    
    # Find the columns where the minimum value is equal to the maximum value
    equal_min_max = (min_max_ranges .== 0)
    
    # Count the number of columns to be removed
    # It can be done just as a sum as equal_min_max is a vector of ones and zeros
    num_removed = sum(equal_min_max)
    println("\nNumber of attributes removed: ", num_removed)

    # Remove the uninformative columns
    informative_cols = (.!equal_min_max)'
    filtered_data = data[:, informative_cols]
    
    # Normalize the filtered data 
    normalized_data = (filtered_data .- min_values[:,informative_cols]) ./ min_max_ranges[:,informative_cols]
    
    return normalized_data
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

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return minimum(dataset, dims=1), maximum(dataset, dims=1)
end;

# Alternative more compact definition
#calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) = ( minimum(dataset, dims=1), maximum(dataset, dims=1) );


function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    return mean(dataset, dims=1), std(dataset, dims=1)
end;


function normalizeMinMax!(dataset::AbstractArray{<:Real,2},      
        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    minValues = normalizationParameters[1];
    maxValues = normalizationParameters[2];
    dataset .-= minValues;
    dataset ./= (maxValues .- minValues);
    # eliminate any atribute that do not add information
    dataset[:, vec(minValues.==maxValues)] .= 0;
    return dataset;
end;


function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizeMinMax!(dataset , calculateMinMaxNormalizationParameters(dataset));
end;


function normalizeMinMax( dataset::AbstractArray{<:Real,2},      
                normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    normalizeMinMax!(copy(dataset), normalizationParameters);
end;


function normalizeMinMax( dataset::AbstractArray{<:Real,2})
      normalizeMinMax!(copy(dataset), calculateMinMaxNormalizationParameters(dataset));
end;


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},      
                        normalizationParameters::NTuple{2, AbstractArray{<:Real,2}}) 
    avgValues = normalizationParameters[1];
    stdValues = normalizationParameters[2];
    dataset .-= avgValues;
    dataset ./= stdValues;
    # Remove any atribute that do not have information
    dataset[:, vec(stdValues.==0)] .= 0;
    return dataset; 
end;


function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizeZeroMean!(dataset , calculateZeroMeanNormalizationParameters(dataset));   
end;


function normalizeZeroMean( dataset::AbstractArray{<:Real,2},      
                            normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    normalizeZeroMean!(copy(dataset), normalizationParameters);
end;


function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
    normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));
end;


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


function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;   


function trainClassANN(topology::AbstractArray{<:Int,1},      
                    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    @assert size(dataset[1], 1) == size(dataset[2], 1) "Training inputs and targets must have the same number of rows."
    
    numEpoch = 0;trainingLoss=minLoss + 1
    inputs = dataset[1]; targets = dataset[2];
    n_patterns, n_features = size(inputs)
    n_classes = size(targets, 2)
    ann = buildClassANN(n_features, topology, n_classes, transferFunctions=transferFunctions)

    loss(m, x, y) = (n_classes == 1) ? Losses.binarycrossentropy(m(x), y) : Losses.crossentropy(m(x), y);
    
    trainingLosses = []
    opt_state = Flux.setup(Adam(learningRate), ann)
    while (numEpoch<maxEpochs) && (trainingLoss>minLoss)
    # Training. Matrixes must be transposed (each pattern in a column)
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);
        numEpoch += 1;
        # Calculate the loss values for this cycle
        trainingLoss = loss(ann, inputs', targets');
        # Store the loss values for this cycle
        push!(trainingLosses, trainingLoss);
        println("Epoch ", numEpoch, ": loss: ", trainingLoss);
    end
    return ann
end                                      


function trainClassANN(topology::AbstractArray{<:Int,1},      
                    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};      
                    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),      
                    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    @assert size(inputs, 1) == size(inputs, 1) "Training inputs and targets must have the same number of rows."

     trainClassANN(topology, (inputs, reshape(targets, length(targets), 1)); 
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate);
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

function trainClassANN(topology::AbstractArray{<:Int,1},
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
        validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
        testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
                (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
        maxEpochsVal::Int=20, showText::Bool=false)

    @assert size(trainingDataset[1], 1) == size(trainingDataset[2], 1) "Training inputs and targets must have the same number of rows."
    @assert size(validationDataset[1], 1) == size(validationDataset[2], 1) "Validation inputs and targets must have the same number of rows."
    @assert size(testDataset[1], 1) == size(testDataset[2], 1) "Test inputs and targets must have the same number of rows."


    (trainingInputs, trainingTargets) = trainingDataset
    (validationInputs, validationTargets) = validationDataset
    (testInputs, testTargets) = testDataset
    
    n_classes = size(trainingTargets, 2)

    ann = buildClassANN(size(trainingInputs, 2), topology, size(trainingTargets, 2);
                        transferFunctions=transferFunctions)

    loss(m, x, y) = (n_classes == 1) ? Losses.binarycrossentropy(m(x), y) : Losses.crossentropy(m(x), y);

    trainingLosses = Float32[]
    validationLosses = Float32[]
    testLosses = Float32[]

    bestAnn = deepcopy(ann)
    bestValidationLoss = Inf
    epochsWithoutImprovement = 0

    opt = Flux.setup(Adam(learningRate), ann)

    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(trainingInputs', trainingTargets')], opt)
        currTrainingLoss = loss(ann, trainingInputs', trainingTargets')
        currValidationLoss = length(validationInputs) > 0 ?
                             loss(ann, validationInputs', validationTargets') : Inf
        currTestLoss = length(testInputs) > 0 ?
                       loss(ann, testInputs', testTargets') : Inf
        
        #Tracking losses
        push!(trainingLosses, currTrainingLoss)
        push!(validationLosses, currValidationLoss)
        push!(testLosses, currTestLoss)

        if showText
            println("Epoch: $epoch, Training Loss: $currTrainingLoss, Validation Loss: $currValidationLoss, Test Loss: $currTestLoss")
        end
        
        # Performance checking - Deepcopy if it's the best
        if (currValidationLoss < bestValidationLoss) || (bestValidationLoss == Inf) # If is Inf, there is no val set
            bestValidationLoss = currValidationLoss
            epochsWithoutImprovement = 0
            if (bestValidationLoss == Inf) && (epoch != maxEpochs)
                continue;
            end
            bestAnn = deepcopy(ann)
        else
            epochsWithoutImprovement += 1
        end
        
        # Early stopping check
        if epochsWithoutImprovement >= maxEpochsVal
            println("Early stopping triggered at epoch $epoch.")
            break
        end

        if length(validationInputs) > 0 && epochsWithoutImprovement >= maxEpochsVal
            println("Early stopping triggered at epoch $epoch.")
            break
        end

        if currTrainingLoss <= minLoss
            break
        end
    end

    return bestAnn, trainingLosses, validationLosses, testLosses
end


function trainClassANN(
    topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
            (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20, showText::Bool=false)

    @assert size(trainingDataset[1], 1) == size(trainingDataset[2], 1) "Training inputs and targets must have the same number of rows."
    @assert size(validationDataset[1], 1) == size(validationDataset[2], 1) "Validation inputs and targets must have the same number of rows."
    @assert size(testDataset[1], 1) == size(testDataset[2], 1) "Test inputs and targets must have the same number of rows."

    # Reshape targets for training, validation, and test datasets
    reshapedTrainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
    reshapedValidationDataset = (validationDataset[1], reshape(validationDataset[2], :, 1))
    reshapedTestDataset = (testDataset[1], reshape(testDataset[2], :, 1))

    # Call the previously defined trainClassANN function
    return trainClassANN(
        topology, reshapedTrainingDataset;
        validationDataset=reshapedValidationDataset,
        testDataset=reshapedTestDataset,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        maxEpochsVal=maxEpochsVal, showText=showText
    )
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

function trainClassANN(topology::AbstractArray{<:Int,1}, 
                       trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}, 
                       kFoldIndices::Array{Int64,1};
                       transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
                       maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, 
                       repetitionsTraining::Int=1, validationRatio::Real=0.0, 
                       maxEpochsVal::Int=20)
    Random.seed!(1234)

    N = size(trainingDataset[1], 1)
    k = maximum(kFoldIndices)
    acc_metrics = zeros(Float64, k); err_rate_metrics = zeros(Float64, k);
    sensitivity_metrics = zeros(Float64, k); specificity_metrics = zeros(Float64, k);
    ppv_metrics = zeros(Float64, k); npv_metrics = zeros(Float64, k);
    f_score_metrics = zeros(Float64, k);


    for fold in 1:k
        train_indices = findall(x -> x != fold, kFoldIndices)
        test_indices = findall(x -> x == fold, kFoldIndices)
        
        train_indices, val_indices = holdOut(length(train_indices), validationRatio) #Spliting train data

        train_set = (trainingDataset[1][train_indices, :], trainingDataset[2][train_indices, :])
        val_set = (trainingDataset[1][val_indices, :], trainingDataset[2][val_indices, :])
        test_set = (trainingDataset[1][test_indices, :], trainingDataset[2][test_indices, :])

        fold_acc = Float64[]; fold_err_rate = Float64[]; fold_sensitivity = Float64[];
        fold_specificity = Float64[]; fold_PPV = Float64[]; fold_acc_NPV = Float64[];
        fold_acc_F_score = Float64[];

        for _ in 1:repetitionsTraining
            trained_model, _, _, _ = trainClassANN(topology, train_set,
                                                   validationDataset = val_set,
                                                   transferFunctions = transferFunctions,
                                                   maxEpochs = maxEpochs, minLoss = minLoss,
                                                   learningRate = learningRate, maxEpochsVal = maxEpochsVal)

            # Calculating accuracy for this iteration
            predicted_outputs = trained_model(test_set[1]')'
            #test_metric = accuracy(predicted_outputs, test_set[2])
            acc, error_rate, sensitivity, specificity, ppv, npv, f_score, _ = confusionMatrix(predicted_outputs, test_set[2]);

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

function trainClassANN(topology::AbstractArray{<:Int,1}, 
        trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}, 
        kFoldIndices::Array{Int64,1};
        transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)), 
        maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01, repetitionsTraining::Int=1, 
        validationRatio::Real=0.0, maxEpochsVal::Int=20)
    
    targets_matrix = reshape(trainingDataset[2], :, 1)
    
    new_trainingDataset = (trainingDataset[1], targets_matrix)
    
    return trainClassANN(topology, new_trainingDataset, kFoldIndices;
                         transferFunctions=transferFunctions, 
                         maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate, 
                         repetitionsTraining=repetitionsTraining, 
                         validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)
end


#####################   Unit 6   #############################

function modelCrossValidation(modelType::Symbol,
    modelHyperparameters::Dict,
    inputs::AbstractArray{<:Real,2},
    targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1})

    seed = 42
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
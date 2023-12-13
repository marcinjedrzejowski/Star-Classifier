module Preprocessing
    using Random
    using Statistics
    function preprocess_data(data::Matrix{Any}, reduction_ratio::Float64, train_ratio::Float64, norm_method::String, balance::Bool=true, indices::Vector{Int64}=4:9)
        # Reduce the data
        data = reduce_data(data, reduction_ratio)

        Random.seed!(123)
        data = data[2:end, :]

        # Balance the dataset
        if balance
            data = balance_data(data)
        end

        input_index = indices
        inputs = data[:,input_index]
        inputs = convert(Array{Float32,2}, inputs)
        targets = data[:,14]

        N=size(inputs,1)
        train_indices, test_indices = holdOut(N, train_ratio)
        train_inputs = inputs[train_indices, :]
        train_targets = targets[train_indices]
        test_inputs = inputs[test_indices, :]
        test_targets = targets[test_indices]

        if norm_method == "minmax"
            # Min-Max Normalization
            normalizeMinMax!(train_inputs)
            normalizeMinMax!(test_inputs)
        elseif norm_method == "zero_mean"
            # Zero Mean Normalization
            train_inputs = normalizeZeroMean!(train_inputs)
            test_inputs = normalizeZeroMean!(test_inputs)
        else
            error("Invalid normalization method")
        end

        @assert isa(train_inputs, Array{<:Float32,2})
        @assert isa(test_inputs, Array{<:Float32,2})
        @assert isa(train_targets, AbstractArray)
        @assert isa(test_targets, AbstractArray)

        return train_inputs, train_targets, test_inputs, test_targets
    end


    function reduce_data(dataset::Matrix, percentage_to_keep::Float64)
        # Extract data and targets from the dataset
        data = dataset[:, 1:end-1]
        targets = dataset[:, 14] 
        
        unique_classes = unique(targets)
        reduced_data = Matrix{Float64}(undef, 0, size(data, 2))
        reduced_targets = Vector{Float64}()
    
        for class in unique_classes
            # Get the data and targets for this class
            class_data = data[targets .== class, :]
            class_targets = targets[targets .== class]
    
            # Calculate the number of rows to keep
            num_rows_to_keep = Int(ceil(size(class_data, 1) * percentage_to_keep))
    
            # Randomly select the subset of rows
            indices = randperm(size(class_data, 1))[1:num_rows_to_keep]
            subset_class_data = class_data[indices, :]
            subset_class_targets = class_targets[indices]
    
            # Append the reduced data and targets for this class to the overall reduced data and targets
            reduced_data = vcat(reduced_data, subset_class_data)
            reduced_targets = vcat(reduced_targets, subset_class_targets)
        end
    
        # Combine reduced data and targets
        reduced_dataset = hcat(reduced_data, reduced_targets)
    
        return reduced_dataset
    end

    function balance_data(data)
        # Balance the dataset
        n = count(data[:,14] .== "QSO");
        # Select n random Galaxies and n random stars
        galaxies = data[data[:,14] .== "GALAXY",:];
        stars = data[data[:,14] .== "STAR",:];
        qso = data[data[:,14] .== "QSO",:];
        galaxies = galaxies[randperm(size(galaxies,1))[1:n],:];
        stars = stars[randperm(size(stars,1))[1:n],:];
        data = vcat(galaxies, stars, qso);

        @assert(count(data[:,14] .== "QSO") == n)
        @assert(count(data[:,14] .== "GALAXY") == n)
        @assert(count(data[:,14] .== "STAR") == n)

        return data
    end

    function normalize_data(inputs::Array{<:Float32,2}, normalization::String)
        # Normalize the inputs
        # Return normalized inputs
        if normalization == "minmax"
            # Min-Max Normalization
            train_inputs = normalizeMinMax!(inputs);
            # test_inputs = normalizeMinMax!(test_inputs);
        elseif normalization == "zero_mean"
            # Zero Mean Normalization
            train_inputs = normalizeZeroMean!(inputs);
            # test_inputs = normalizeZeroMean!(test_inputs);
        else
            error("Invalid normalization method")
        end;

        return train_inputs
    end

    Random.seed!(123)
    function reduce_data(dataset::Matrix, percentage_to_keep::Float64)
        # Extract data and targets from the dataset
        data = dataset[:, 1:end-1]
        targets = dataset[:, 14] 
        
        unique_classes = unique(targets)
        reduced_data = Matrix{Float64}(undef, 0, size(data, 2))
        reduced_targets = Vector{Float64}()

        for class in unique_classes
            # Get the data and targets for this class
            class_data = data[targets .== class, :]
            class_targets = targets[targets .== class]

            # Calculate the number of rows to keep
            num_rows_to_keep = Int(ceil(size(class_data, 1) * percentage_to_keep))

            # Randomly select the subset of rows
            indices = randperm(size(class_data, 1))[1:num_rows_to_keep]
            subset_class_data = class_data[indices, :]
            subset_class_targets = class_targets[indices]

            # Append the reduced data and targets for this class to the overall reduced data and targets
            reduced_data = vcat(reduced_data, subset_class_data)
            reduced_targets = vcat(reduced_targets, subset_class_targets)
        end

        # Combine reduced data and targets
        reduced_dataset = hcat(reduced_data, reduced_targets)

        return reduced_dataset
    end

    """   I used my own holdout function from my utils.  """ 

    function holdOut(N::Int, P::Real)
        @assert ((P>=0.) & (P<=1.));
        indices = randperm(N)
        n_train = Int(round((1-P)*N))
        return (indices[1:n_train], indices[n_train + 1:end])
    end


    #-------------------------------------#
    ############# LEGACY CODE #############
    #-------------------------------------#

    """ I used previous version of preprocessing
        to normalize the data."""

    # One-hot encoding
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

    # MIN-MAX NORMALIZATION
    function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
        return minimum(dataset, dims=1), maximum(dataset, dims=1)
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

    # ZERO MEAN NORMALIZATION
    function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
        return mean(dataset, dims=1), std(dataset, dims=1)
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

    function normalizeZeroMean( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
        normalizeZeroMean!(copy(dataset), normalizationParameters);
    end;

    function normalizeZeroMean( dataset::AbstractArray{<:Real,2}) 
        normalizeZeroMean!(copy(dataset), calculateZeroMeanNormalizationParameters(dataset));
    end;

    # HOLDOUT
    function holdOut(N::Int, P::Real)
        #TODO
        @assert ((P>=0.) & (P<=1.));
        permuted = randperm(N)
        n_train = Int(round((1-P)*N))
        return (permuted[1:n_train], permuted[n_train+1:end])
    end
    
    function holdOut(N::Int, Pval::Real, Ptest::Real) 
        #TODO
        P = Pval + Ptest
        @assert ((P>=0.) & (P<=1.));
        (trainingValidationIndices, testIndices) = holdOut(N, Ptest);
        (trainingIndices, validationIndices) = holdOut(length(trainingValidationIndices), Pval*N/length(trainingValidationIndices));
        return (trainingValidationIndices[trainingIndices], trainingValidationIndices[validationIndices], testIndices);
    end

    # CROSSVALIDATION
    function crossvalidation(N::Int64, k::Int64)
        #TODO
        folds = collect(1:k); # Vector with the k folds indices = repeat( );
        indices = repeat(folds, ceil(Int, N/k))[1:N]; # Select first N indexes
        indices = shuffle!(indices); # Shuffle indexes
        return indices;
    end
    
    
    function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
        #TODO
        @assert all(sum(targets, dims=1) .>= k)
    
        indices = Array{Int64,1}(undef, size(targets,1));
    
        for class in 1:size(targets, 2)
            indices[findall(targets[:,class])] = crossvalidation(sum(targets[:,class]), k);
        end
        return indices;
    end
    
    
    function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
        #TODO
        targets = oneHotEncoding(targets);
        indices = crossvalidation(targets, k);
        return indices;
    end;

    # ONE VS ALL
    using ScikitLearn;
    function oneVSall(model, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2})
        # Number of instances and classes
        numInstances, numClasses = size(targets)
    
        # Initialize the outputs matrix
        model_outputs = Array{Float32,2}(undef, numInstances, numClasses)
    
        # Train a model for each class
        for numClass in 1:numClasses
            # Consider the current class as positive and rest as negative
            currentTargets = targets[:, numClass]
    
            # Train the model using some binary classifier (this is a placeholder)
            model = fit!(model, inputs, currentTargets)

            # Store the model's output for all instances
            model_outputs[:, numClass] .= predict(model, inputs)
        end
    
        # Identify the class with the maximum output for each instance
        finalOutputs = Array{Bool,2}(undef, numInstances, numClasses)
        for i in 1:numInstances
            maxVal = maximum(model_outputs[i, :])
            maxIndices = findall(x -> x == maxVal, model_outputs[i, :])
    
            # Simple tie-breaking: choose the first maximum class
            # Our tie-breaking criterion is taking the first index in maxIndices
            finalOutputs[i, :] .= false
            finalOutputs[i, maxIndices[1]] = true
        end
    
        return finalOutputs, model
    end
    
    # CONFUSION MATRIX
    function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
        #TODO
        @assert length(outputs) == length(targets)
    
        confusion_matrix = zeros(Int64, 2, 2)
    
        TN = sum((outputs .== false) .& (targets .== false))
        FP = sum((outputs .== true) .& (targets .== false))
        FN = sum((outputs .== false) .& (targets .== true))
        TP = sum((outputs .== true) .& (targets .== true))
    
        if TN + FP + FN + TP == 0
            accuracy = 0
            error_rate = 0
        else
            accuracy = (TN + TP) / (TN + FP + FN + TP)
            error_rate = (FP + FN) / (TN + FP + FN + TP)
        end
    
        if TN == TN + FP + FN + TP
            sensitivity = 1
            ppv = 1
        else
            if TP + FN == 0
                sensitivity = 0
            else
                sensitivity = TP / (TP + FN)
            end
    
            if TP + FP == 0
                ppv = 0
            else
                ppv = TP / (TP + FP)
            end
        end
        
        if TP == (TN + FP + FN + TP)
            specificity = 1
            npv = 1
        else
            if TN + FP == 0
                specificity = 0
            else
                specificity = TN / (TN + FP)
            end
    
            if TN + FN == 0
                npv = 0
            else
                npv = TN / (TN + FN)
            end
        end
        
        if sensitivity == 0 && ppv == 0
            fscore = 0
        else
            fscore = (2 * TP) / (2 * TP + FP + FN)
        end
        
        confusion_matrix = [TN FP; FN TP]
    
        return accuracy, error_rate, sensitivity, specificity, ppv, npv, fscore, confusion_matrix
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
end
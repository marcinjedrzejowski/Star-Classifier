module Preprocessing
    using Random
    using Statistics
    function preprocess_data(data::Matrix{Any}, balance::Bool=true, indices::Vector{Int64}=4:9)
        # Implement data normalization or other preprocessing steps
        # Return preprocessed data
        Random.seed!(123)
        data = data[2:end, :];

        # Balance the dataset
        if balance
            data = balance_data(data)
        end;

        # input_index = [4,5,6,7,8,9,11,12,13,15,16,18];
        input_index = indices;
        inputs = data[:,input_index];
        println("Size :", size(inputs))
        targets = data[:,14];

        inputs = convert(Array{Float32,2}, inputs);
        targets = oneHotEncoding(targets);

        @assert isa(inputs, Array{<:Float32,2})
        @assert isa(targets, AbstractArray{Bool,2})

        return inputs, targets
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

    function normalize_data(train_inputs::Array{<:Float32,2}, normalization::String)
        # Normalize the inputs
        # Return normalized inputs
        if normalization == "minmax"
            # Min-Max Normalization
            train_inputs = normalizeMinMax!(train_inputs);
            # test_inputs = normalizeMinMax!(test_inputs);
        elseif normalization == "zero_mean"
            # Zero Mean Normalization
            train_inputs = normalizeZeroMean!(train_inputs);
            # test_inputs = normalizeZeroMean!(test_inputs);
        else
            error("Invalid normalization method")
        end;

        return train_inputs
    end

    #-------------------------------------#
    ############# LEGACY CODE #############
    #-------------------------------------#

    # ONE-HOT ENCODING
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

    function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=false)
        #TODO
        # @assert size(targets) != size(outputs)
        
        numInstances, numClasses = size(outputs)
    
        if numClasses == 1
            return confusionMatrix(outputs[:, 1], targets[:, 1])
        end
    
        Sensitivity = zeros(Float64, numClasses)
        Specificity = zeros(Float64, numClasses)
        PPV = zeros(Float64, numClasses)
        NPV = zeros(Float64, numClasses)
        F1 = zeros(Float64, numClasses)
        Accuracy = zeros(Float64, numClasses)
        ErrorRate = zeros(Float64, numClasses)
    
        for class in 1:numClasses
            outputsClass = outputs[:, class]
            targetsClass = targets[:, class]
            
            Accuracy[class], ErrorRate[class], Sensitivity[class], Specificity[class], PPV[class], NPV[class], F1[class], confusion_matrix = confusionMatrix(outputsClass, targetsClass)
        end
    
        if !weighted
            MacroSensitivity = mean(Sensitivity)
            MacroSpecificity = mean(Specificity)
            MacroPPV = mean(PPV)
            MacroNPV = mean(NPV)
            MacroF1 = mean(F1)
            MacroAccuracy = mean(Accuracy)
            return (Sensitivity, Specificity, PPV, NPV, F1, Accuracy, MacroSensitivity, MacroSpecificity, MacroPPV, MacroNPV, MacroF1, MacroAccuracy)
        else
            Weights = sum(targets, dims=1)
            WeightedSensitivity = dot(Sensitivity, Weights) / sum(Weights)
            WeightedSpecificity = dot(Specificity, Weights) / sum(Weights)
            WeightedPPV = dot(PPV, Weights) / sum(Weights)
            WeightedNPV = dot(NPV, Weights) / sum(Weights)
            WeightedF1 = dot(F1, Weights) / sum(Weights)
            WeightedAccuracy = sum(Accuracy .* Weights) / sum(Weights)
            return (Sensitivity, Specificity, PPV, NPV, F1, Accuracy, WeightedSensitivity, WeightedSpecificity, WeightedPPV, WeightedNPV, WeightedF1, WeightedAccuracy)
        end
    end
end

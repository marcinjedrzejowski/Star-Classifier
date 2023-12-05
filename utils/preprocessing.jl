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
    
end

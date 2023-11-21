module Preprocessing
    function preprocess_data(data)
        # Implement data normalization or other preprocessing steps
        # Return preprocessed data

    end




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

    # Normalization
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
end

module DataLoader    
    using DelimitedFiles 
    using DataFrames
    using CSV
    using Random
    
    Random.seed!(123)
    function load_data_for_analysis(file_path::String)
        # Load data from CSV or other formats
        # Return the dataset
        dataframe = CSV.File(file_path) |> DataFrame;

        return dataframe
    end
    
    function load_data(file_path::String)
        # Load data from CSV or other formats
        # Return the dataset
        dataset = readdlm(file_path,',');
        return dataset
    end

    function split_data(inputs::Array{<:Float32,2}, targets::AbstractArray{Bool,2}, train_ratio::Float64)
        # Split the dataset into training and testing sets
        # Return the training and testing sets
        n = size(inputs,1);
        train_size = Int(round(n*train_ratio));
        test_size = n - train_size;
        train_indices = randperm(n)[1:train_size];
        test_indices = setdiff(1:n, train_indices);
        train_inputs = inputs[train_indices,:];
        train_targets = targets[train_indices,:];
        test_inputs = inputs[test_indices,:];
        test_targets = targets[test_indices,:];

        @assert size(train_inputs,1) == size(train_targets,1)
        @assert size(test_inputs,1) == size(test_targets,1)

        return train_inputs, train_targets, test_inputs, test_targets
    end
end

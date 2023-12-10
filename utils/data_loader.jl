module DataLoader    
    using DelimitedFiles 
    using DataFrames
    using CSV
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
end

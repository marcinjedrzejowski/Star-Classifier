module DataLoader    
    using DelimitedFiles 
    using DataFrames
    using CSV
    function load_data(file_path::String)
        # Load data from CSV or other formats
        # Return the dataset
        dataframe = CSV.File(file_path) |> DataFrame;

        return dataframe
    end
end

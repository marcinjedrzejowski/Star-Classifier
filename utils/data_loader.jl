module DataLoader    
    using DelimitedFiles 
    using Random
    function load_data(file_path::String)
        # Load data from CSV or other formats
        # Return the dataset
        dataset = readdlm(file_path,',');
        return dataset
    end
end

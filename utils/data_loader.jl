module DataLoader    
    using DelimitedFiles 
    using Random
    function load_data(file_path::String)
        # Load data from CSV or other formats
        # Return the dataset
        dataset = readdlm(file_path,',');

        # # Balance the dataset
        # n = count(dataset[:,14] .== "QSO");
        # # Select n random Galaxies and n random stars
        # galaxies = dataset[dataset[:,14] .== "GALAXY",:];
        # stars = dataset[dataset[:,14] .== "STAR",:];
        # qso = dataset[dataset[:,14] .== "QSO",:];
        # galaxies = galaxies[randperm(size(galaxies,1))[1:n],:];
        # stars = stars[randperm(size(stars,1))[1:n],:];
        # dataset = vcat(galaxies, stars, qso);

        return dataset
    end
end

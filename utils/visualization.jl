module Visualization
    # import Pkg; 
    # Pkg.add("StatsBase")

    using DataFrames
    using Statistics
    using Plots
    using StatsPlots
    using StatsBase    

    function plot_results(results::Dict{String, Any})
        # Implement functions for visualizing results
        # Plot graphs, confusion matrices, etc.
    end

    function entry_visualization(data::DataFrame)
        # Analyze the data before working on it
        
        println("######### Display column names #########")
        display(names(data))  # Display column names
        println("######### Display first 5 rows #########")
        display(first(data, 5))  # Display the first 5 rows

        # Basic statistics
        println("######### Display basic statistics #########")
        display(describe(data))

        # Grouped analysis
        by_class = combine(groupby(data, :class), nrow => :count)

        # Assuming "class" is the name of the column containing the classes
        class_counts = countmap(data.class)

        # Convert the class counts to a DataFrame for easier plotting
        class_counts_df = DataFrame(Class = collect(keys(class_counts)), Count = collect(values(class_counts)))

        # Bar plot for class distribution
        plot = bar(class_counts_df.Class, class_counts_df.Count, xlabel="Stellar Class", ylabel="Count", title="Class Distribution", color=[:lightblue, :blue, :darkblue], legend=false)
        yaxis!(plot, formatter = :plain)
        
        # Display the bar plot
        display(plot)
    end
end

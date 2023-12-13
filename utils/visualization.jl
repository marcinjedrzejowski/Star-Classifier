module Visualization
    # import Pkg; 
    # Pkg.add("StatsBase")

    using DataFrames
    using Statistics
    using Plots
    using StatsPlots
    using StatsBase    

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


    function plot_confusion_metrics(acc, sensitivity, specificity, ppv, npv, f_score)
        metrics = ["Accuracy", "Sensitivity", "Specificity", "Precision (PPV)", "NPV", "F-score"]
        values = [acc*100, sensitivity*100, specificity*100, ppv*100, npv*100, f_score*100]

        bar(metrics, values, color=[:blue, :green, :orange, :purple, :red, :brown],
            xlabel="Metric", ylabel="Metric Value", title="Confusion Matrix Metrics", legend=false,
            ylims=(40, 100))  # Adjust the y-axis limit if needed
    end

    function plot_confusion_heatmap(confusionMatrix)
        p = heatmap(confusionMatrix, xticks=(1:3, ["Galaxy", "Quasar", "Star"]), yticks=(1:3, ["Galaxy", "Quasar", "Star"]),
        color=:cividis,
        c=:reds,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Heatmap",
        cbar_title="Count",
        fmt=:png,  # Format to save the plot
        size=(500, 500)  # Adjust the size as needed
        )

        # Add the actual numbers from the confusion matrix to the heatmap
        for i in 1:3
            for j in 1:3
                annotate!(p, [(i, j, text(confusionMatrix[j, i], 10, :white))])
            end
        end

        return p  # Return the plot
    end
end

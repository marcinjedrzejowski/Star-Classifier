# FILEPATH

"""
After loading the data, we apply the preprocess_data function which does the following:

1. Chooses the input and target columns from the original data.
2. Balances the data by counting the number of QSO instances and then randomly 
    selecting the same number of Galaxy and Star instances. Maybe better use SMOTE algorithm?
3. Performs one-hot encoding on the target variable, changing the target variables into binary vectors.
4. Normalizes the inputs using min-max normalization.
"""

## Why we have to vectorize the targets for crossvalidation?

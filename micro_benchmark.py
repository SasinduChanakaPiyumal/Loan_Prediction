import pandas as pd
import timeit
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load the dataset
def load_data():
    # Make sure to provide the correct path to your CSV file
    df = pd.read_csv('loan_data_set.csv')
    return df

# Original imputation logic
def original_imputation(df):
    df_copy = df.copy()
    
    # Dependents column handling
    mode_dependents = df_copy['Dependents'].mode()[0]
    df_copy['Dependents'] = df_copy['Dependents'].replace('3+', '3').fillna(mode_dependents)
    df_copy['Dependents'] = df_copy['Dependents'].astype('int32')

    # Categorical data filled with mode
    category_col = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for column in category_col:
        mode_val = df_copy[column].mode()[0]
        df_copy[column].fillna(mode_val, inplace=True)
    
    # Numeric categorical data filled with mode
    numeric_category_col = ['Credit_History', 'Loan_Amount_Term']
    for column in numeric_category_col:
        mode_val = df_copy[column].mode()[0]
        df_copy[column].fillna(mode_val, inplace=True)

    # LoanAmount filled with median
    median_loan_amount = df_copy['LoanAmount'].median()
    df_copy['LoanAmount'].fillna(median_loan_amount, inplace=True)
    
    return df_copy

# Optimized imputation logic
def optimized_imputation(df):
    df_copy = df.copy()
    
    # Dependents column handling
    mode_dependents = df_copy['Dependents'].mode()[0]
    df_copy['Dependents'] = df_copy['Dependents'].replace('3+', '3').fillna(mode_dependents)
    df_copy['Dependents'] = df_copy['Dependents'].astype('int32')

    # Categorical data filled with mode (vectorized)
    category_col = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    df_copy[category_col] = df_copy[category_col].fillna(df_copy[category_col].mode().iloc[0])
    
    # Numeric categorical data filled with mode (vectorized)
    numeric_category_col = ['Credit_History', 'Loan_Amount_Term']
    df_copy[numeric_category_col] = df_copy[numeric_category_col].fillna(df_copy[numeric_category_col].mode().iloc[0])

    # LoanAmount filled with median
    median_loan_amount = df_copy['LoanAmount'].median()
    df_copy['LoanAmount'].fillna(median_loan_amount, inplace=True)
    
    return df_copy

if __name__ == "__main__":
    print("Running micro-benchmark for imputation methods...")
    
    # Setup for timeit
    setup_code = """
from __main__ import load_data, original_imputation, optimized_imputation
df = load_data()
    """
    
    # Benchmark original imputation
    print("\nBenchmarking original imputation...")
    original_time = timeit.timeit("original_imputation(df.copy())", setup=setup_code, number=100)
    print(f"Original imputation time: {original_time:.6f} seconds")
    
    # Benchmark optimized imputation
    print("\nBenchmarking optimized imputation...")
    optimized_time = timeit.timeit("optimized_imputation(df.copy())", setup=setup_code, number=100)
    print(f"Optimized imputation time: {optimized_time:.6f} seconds")
    
    print("\n--- Comparison ---")
    print(f"Original: {original_time:.6f} seconds")
    print(f"Optimized: {optimized_time:.6f} seconds")
    if original_time > optimized_time:
        print(f"Optimized version is {original_time / optimized_time:.2f}x faster.")
    else:
        print("Optimized version is not faster or has similar performance.")

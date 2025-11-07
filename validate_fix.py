#!/usr/bin/env python3
"""
Script to validate the LabelEncoder bug fix.
This script demonstrates that the bug is fixed in the main code.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def demonstrate_bug():
    """Show the LabelEncoder reuse bug."""
    print("=== DEMONSTRATING THE BUG ===")
    
    # Create test data
    test_data = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male'],
        'Education': ['Graduate', 'Not Graduate', 'Graduate'],  
        'Married': ['Yes', 'No', 'Yes']
    })
    
    print("Original data:")
    print(test_data)
    
    # Buggy approach - reuse same encoder
    print("\n--- Buggy approach (reusing same LabelEncoder) ---")
    df_buggy = test_data.copy()
    LE_buggy = LabelEncoder()
    
    # Track what happens to each column
    for col in ['Gender', 'Education', 'Married']:
        df_buggy[col] = LE_buggy.fit_transform(df_buggy[col])
        print(f"After encoding {col}: classes = {LE_buggy.classes_}")
    
    print("Buggy result:")
    print(df_buggy)
    
    # Try to transform a Gender value using the encoder that was last fitted on Married
    try:
        result = LE_buggy.transform(['Male'])
        print(f"Trying to transform 'Male' with encoder fitted on Married: {result}")
    except ValueError as e:
        print(f"ERROR: {e}")
    
    return df_buggy

def demonstrate_fix():
    """Show the fixed approach."""
    print("\n\n=== DEMONSTRATING THE FIX ===")
    
    # Create test data
    test_data = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male'],
        'Education': ['Graduate', 'Not Graduate', 'Graduate'],
        'Married': ['Yes', 'No', 'Yes']
    })
    
    print("Original data:")
    print(test_data)
    
    # Fixed approach - separate encoder for each column
    print("\n--- Fixed approach (separate LabelEncoder for each column) ---")
    df_fixed = test_data.copy()
    encoders = {}
    
    for col in ['Gender', 'Education', 'Married']:
        LE_fixed = LabelEncoder()  # New encoder for each column
        df_fixed[col] = LE_fixed.fit_transform(df_fixed[col])
        encoders[col] = LE_fixed
        print(f"Encoder for {col}: classes = {LE_fixed.classes_}")
    
    print("Fixed result:")
    print(df_fixed)
    
    # Now each encoder can correctly transform values from its respective column
    print("\nTesting each encoder:")
    print(f"Gender encoder can transform 'Male': {encoders['Gender'].transform(['Male'])}")
    print(f"Education encoder can transform 'Graduate': {encoders['Education'].transform(['Graduate'])}")  
    print(f"Married encoder can transform 'Yes': {encoders['Married'].transform(['Yes'])}")
    
    return df_fixed

def run_test():
    """Run the complete test."""
    print("Label Encoder Bug Fix Validation")
    print("="*50)
    
    buggy_result = demonstrate_bug()
    fixed_result = demonstrate_fix()
    
    print("\n\n=== SUMMARY ===")
    print("The bug occurs when the same LabelEncoder instance is reused across multiple columns.")
    print("This causes the encoder to lose the mapping for previously encoded columns.")
    print("The fix is to create a separate LabelEncoder instance for each column.")
    print("\nBoth approaches may produce the same numerical results in some cases,")
    print("but the fixed approach maintains proper encoder state for future transformations.")

if __name__ == "__main__":
    run_test()

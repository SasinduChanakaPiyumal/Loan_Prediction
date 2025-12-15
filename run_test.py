#!/usr/bin/env python3
"""
Simple script to demonstrate the Label Encoder bug and verify the fix.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def demonstrate_bug():
    """Demonstrate the Label Encoder bug with a simple example."""
    print("=== Demonstrating Label Encoder Bug ===\n")
    
    # Create test data
    test_data = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male', 'Female'],
        'Married': ['Yes', 'No', 'Yes', 'No'],
        'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Not Graduate']
    })
    
    print("Original data:")
    print(test_data)
    print()
    
    # BUGGY APPROACH: Reusing same LabelEncoder
    print("BUGGY APPROACH: Reusing same LabelEncoder instance")
    LE_buggy = LabelEncoder()
    cate_cols = ['Gender', 'Married', 'Education']
    df_buggy = test_data.copy()
    
    for col in cate_cols:
        df_buggy[col] = LE_buggy.fit_transform(df_buggy[col])
        print(f"After encoding {col}: classes = {LE_buggy.classes_}")
    
    print("Buggy encoded result:")
    print(df_buggy)
    print()
    
    # CORRECT APPROACH: Fresh LabelEncoder for each column
    print("CORRECT APPROACH: Fresh LabelEncoder for each column")
    df_correct = test_data.copy()
    
    for col in cate_cols:
        LE = LabelEncoder()  # Fresh instance for each column
        df_correct[col] = LE.fit_transform(df_correct[col])
        print(f"Encoding {col}: classes = {LE.classes_}")
    
    print("Correct encoded result:")
    print(df_correct)
    print()
    
    # Show the difference
    print("=== COMPARISON ===")
    print("Gender encoding comparison:")
    print(f"Buggy:   {df_buggy['Gender'].tolist()}")
    print(f"Correct: {df_correct['Gender'].tolist()}")
    print()
    
    # Verify that they should be the same for Gender
    if df_buggy['Gender'].tolist() != df_correct['Gender'].tolist():
        print("❌ BUG DETECTED: Gender encoding differs between approaches!")
        print("This demonstrates that reusing LabelEncoder causes incorrect encodings.")
    else:
        print("✅ No difference detected in this case, but the bug can manifest")
        print("in other scenarios or with different data ordering.")
    
    return df_buggy, df_correct


def test_specific_bug_case():
    """Test a specific case that clearly shows the bug."""
    print("\n=== Testing Specific Bug Case ===\n")
    
    # Create data where the bug will be clearly visible
    test_data = pd.DataFrame({
        'Col1': ['A', 'B'],
        'Col2': ['X', 'Y'], 
        'Col3': ['P', 'Q']
    })
    
    print("Test data:")
    print(test_data)
    print()
    
    # Show what happens with reused encoder
    print("With REUSED LabelEncoder:")
    LE_reused = LabelEncoder()
    
    encoded_col1 = LE_reused.fit_transform(test_data['Col1'])
    print(f"Col1 encoded: {encoded_col1}, classes: {LE_reused.classes_}")
    
    encoded_col2 = LE_reused.fit_transform(test_data['Col2'])  
    print(f"Col2 encoded: {encoded_col2}, classes: {LE_reused.classes_}")
    
    encoded_col3 = LE_reused.fit_transform(test_data['Col3'])
    print(f"Col3 encoded: {encoded_col3}, classes: {LE_reused.classes_}")
    
    print("\nWith FRESH LabelEncoder for each column:")
    
    LE1 = LabelEncoder()
    encoded_col1_fresh = LE1.fit_transform(test_data['Col1'])
    print(f"Col1 encoded: {encoded_col1_fresh}, classes: {LE1.classes_}")
    
    LE2 = LabelEncoder() 
    encoded_col2_fresh = LE2.fit_transform(test_data['Col2'])
    print(f"Col2 encoded: {encoded_col2_fresh}, classes: {LE2.classes_}")
    
    LE3 = LabelEncoder()
    encoded_col3_fresh = LE3.fit_transform(test_data['Col3'])
    print(f"Col3 encoded: {encoded_col3_fresh}, classes: {LE3.classes_}")


if __name__ == '__main__':
    demonstrate_bug()
    test_specific_bug_case()
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("The bug occurs because LabelEncoder remembers all classes")
    print("from previous fit_transform() calls when reused.")
    print("This leads to inconsistent encoding across different columns.")
    print("SOLUTION: Create a fresh LabelEncoder() for each column.")
    print("="*50)

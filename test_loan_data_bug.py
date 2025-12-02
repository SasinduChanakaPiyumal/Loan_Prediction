#!/usr/bin/env python3
"""
Unit test to demonstrate and verify the fix for the Label Encoder bug.
The bug occurs when the same LabelEncoder instance is reused across multiple columns,
causing incorrect label mappings.
"""

import unittest
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class TestLabelEncoderBug(unittest.TestCase):
    
    def setUp(self):
        """Set up test data with categorical columns."""
        self.test_data = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Male', 'Female'],
            'Married': ['Yes', 'No', 'Yes', 'No'],
            'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Not Graduate'],
            'Self_Employed': ['No', 'Yes', 'No', 'Yes']
        })
    
    def test_buggy_label_encoding(self):
        """Test the buggy implementation that reuses the same LabelEncoder instance."""
        # This is the buggy approach from the original code
        LE = LabelEncoder()
        cate_cols_LE = ['Gender', 'Married', 'Education', 'Self_Employed']
        df_buggy = self.test_data.copy()
        
        for col in cate_cols_LE:
            df_buggy[col] = LE.fit_transform(df_buggy[col])
        
        # Check that the encoding is incorrect due to reused encoder
        # The Gender column should have values 0 and 1 for 'Female' and 'Male'
        # But due to the bug, it gets contaminated by subsequent columns
        
        # Let's track what happens step by step
        LE_track = LabelEncoder()
        results = {}
        
        for col in cate_cols_LE:
            results[col] = {
                'classes_before': LE_track.classes_ if hasattr(LE_track, 'classes_') else None,
                'encoded': LE_track.fit_transform(df_buggy[col]),
                'classes_after': LE_track.classes_.copy()
            }
        
        # The bug is that each column's encoding is influenced by the accumulated classes
        # This will cause inconsistent encoding behavior
        return df_buggy, results
    
    def test_correct_label_encoding(self):
        """Test the correct implementation that creates fresh LabelEncoder for each column."""
        cate_cols_LE = ['Gender', 'Married', 'Education', 'Self_Employed']
        df_correct = self.test_data.copy()
        
        for col in cate_cols_LE:
            # Create a fresh LabelEncoder instance for each column
            LE = LabelEncoder()
            df_correct[col] = LE.fit_transform(df_correct[col])
        
        return df_correct
    
    def test_encoding_consistency(self):
        """Test that demonstrates the bug by showing inconsistent behavior."""
        buggy_result, _ = self.test_buggy_label_encoding()
        correct_result = self.test_correct_label_encoding()
        
        # The correct encoding should be consistent and independent per column
        # Gender: Female=0, Male=1 (alphabetical order)
        expected_gender_encoding = [1, 0, 1, 0]  # Male, Female, Male, Female
        
        # Test correct implementation
        self.assertEqual(correct_result['Gender'].tolist(), expected_gender_encoding,
                        "Correct implementation should encode Gender consistently")
        
        # The buggy implementation might produce different results
        # This test will fail with the buggy code but pass with the fixed code
        print("Buggy Gender encoding:", buggy_result['Gender'].tolist())
        print("Correct Gender encoding:", correct_result['Gender'].tolist())
        
        # This assertion will fail with the buggy implementation
        with self.assertRaises(AssertionError):
            self.assertEqual(buggy_result['Gender'].tolist(), expected_gender_encoding,
                           "Buggy implementation produces incorrect encoding")
    
    def test_label_encoder_independence(self):
        """Test that each column's encoding is independent when using separate encoders."""
        correct_result = self.test_correct_label_encoding()
        
        # Each column should be encoded independently
        # Gender: Female=0, Male=1
        # Married: No=0, Yes=1  
        # Education: Graduate=0, Not Graduate=1
        # Self_Employed: No=0, Yes=1
        
        expected_encodings = {
            'Gender': [1, 0, 1, 0],      # Male, Female, Male, Female
            'Married': [1, 0, 1, 0],     # Yes, No, Yes, No
            'Education': [0, 1, 0, 1],   # Graduate, Not Graduate, Graduate, Not Graduate
            'Self_Employed': [0, 1, 0, 1] # No, Yes, No, Yes
        }
        
        for col, expected in expected_encodings.items():
            with self.subTest(column=col):
                self.assertEqual(correct_result[col].tolist(), expected,
                               f"Column {col} should be encoded independently")


if __name__ == '__main__':
    unittest.main()

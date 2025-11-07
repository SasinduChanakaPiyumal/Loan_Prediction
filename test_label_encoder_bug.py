import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class TestLabelEncoderBug(unittest.TestCase):
    """
    Test to demonstrate the LabelEncoder reuse bug and verify the fix.
    
    The bug: Using the same LabelEncoder instance for multiple columns
    causes inconsistent encoding because each fit_transform call overwrites
    the previous mapping.
    """
    
    def setUp(self):
        """Set up test data with different categorical values for each column."""
        self.test_data = pd.DataFrame({
            'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate'],
            'Married': ['Yes', 'No', 'Yes', 'No', 'Yes']
        })
    
    def test_buggy_label_encoder_reuse(self):
        """Test that demonstrates the bug when reusing the same LabelEncoder."""
        df_buggy = self.test_data.copy()
        
        # Buggy approach: reuse the same encoder
        LE = LabelEncoder()
        columns = ['Gender', 'Education', 'Married']
        
        encodings = {}
        for col in columns:
            df_buggy[col] = LE.fit_transform(df_buggy[col])
            # Save what the encoder thinks the classes are after each fit
            encodings[col] = dict(zip(LE.classes_, range(len(LE.classes_))))
        
        # The bug: only the last column's encoding is preserved in LE.classes_
        # All previous columns will have inconsistent encoding
        
        # For the last column (Married), the encoding should be correct
        expected_married_encoding = {'No': 0, 'Yes': 1}  # alphabetical order
        self.assertEqual(encodings['Married'], expected_married_encoding)
        
        # But if we try to use the same encoder for a previous column, it will fail
        # because LE.classes_ only contains classes from the last fit
        test_gender_value = 'Male'
        
        # This should fail because LE.classes_ now contains ['No', 'Yes'] not ['Female', 'Male']
        with self.assertRaises(ValueError):
            LE.transform([test_gender_value])
    
    def test_fixed_label_encoder_separate_instances(self):
        """Test the fixed approach using separate LabelEncoder instances."""
        df_fixed = self.test_data.copy()
        
        # Fixed approach: create separate encoder for each column
        columns = ['Gender', 'Education', 'Married']
        encoders = {}
        
        for col in columns:
            LE = LabelEncoder()  # New encoder for each column
            df_fixed[col] = LE.fit_transform(df_fixed[col])
            encoders[col] = LE
        
        # Verify that each encoder can still transform values from its respective column
        self.assertEqual(encoders['Gender'].transform(['Male'])[0], 
                        encoders['Gender'].transform(['Male'])[0])  # Should not raise error
        
        self.assertEqual(encoders['Education'].transform(['Graduate'])[0],
                        encoders['Education'].transform(['Graduate'])[0])  # Should not raise error
        
        self.assertEqual(encoders['Married'].transform(['Yes'])[0],
                        encoders['Married'].transform(['Yes'])[0])  # Should not raise error
        
        # Verify the encoded values are consistent
        # Gender: ['Female', 'Male'] -> [0, 1]
        expected_gender = [1, 0, 1, 0, 1]  # Male=1, Female=0
        np.testing.assert_array_equal(df_fixed['Gender'].values, expected_gender)
        
        # Education: ['Graduate', 'Not Graduate'] -> [0, 1]  
        expected_education = [0, 1, 0, 0, 1]  # Graduate=0, Not Graduate=1
        np.testing.assert_array_equal(df_fixed['Education'].values, expected_education)
        
        # Married: ['No', 'Yes'] -> [0, 1]
        expected_married = [1, 0, 1, 0, 1]  # Yes=1, No=0
        np.testing.assert_array_equal(df_fixed['Married'].values, expected_married)
    
    def test_encoding_consistency_comparison(self):
        """Compare buggy vs fixed approach to show the difference."""
        # Test data with specific values to make the bug more apparent
        test_df = pd.DataFrame({
            'col1': ['A', 'B', 'A', 'B'],
            'col2': ['X', 'Y', 'X', 'Y'],
            'col3': ['P', 'Q', 'P', 'Q']
        })
        
        # Buggy approach
        df_buggy = test_df.copy()
        LE_buggy = LabelEncoder()
        for col in ['col1', 'col2', 'col3']:
            df_buggy[col] = LE_buggy.fit_transform(df_buggy[col])
        
        # Fixed approach
        df_fixed = test_df.copy()
        for col in ['col1', 'col2', 'col3']:
            LE_fixed = LabelEncoder()
            df_fixed[col] = LE_fixed.fit_transform(df_fixed[col])
        
        # The encodings should be different due to the bug
        # In buggy version, all columns will have values 0,1,0,1 because
        # each column's unique values get mapped to alphabetical order
        # But the actual encoding depends on the last column fitted
        
        # With the fix, each column is encoded independently and consistently
        # col1: A=0, B=1 -> [0,1,0,1] 
        # col2: X=0, Y=1 -> [0,1,0,1]
        # col3: P=0, Q=1 -> [0,1,0,1]
        
        # Both should actually give the same result in this case since all columns
        # have the same pattern, but the bug is in the encoder state consistency
        expected = [0, 1, 0, 1]
        np.testing.assert_array_equal(df_fixed['col1'].values, expected)
        np.testing.assert_array_equal(df_fixed['col2'].values, expected)
        np.testing.assert_array_equal(df_fixed['col3'].values, expected)


if __name__ == '__main__':
    unittest.main()

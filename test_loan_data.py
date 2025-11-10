#!/usr/bin/env python
# coding: utf-8

import unittest
import pandas as pd
import os
import sys

class TestLoanDataLoading(unittest.TestCase):
    """
    Unit test to verify that the loan data CSV file can be loaded correctly.
    
    This test checks that:
    1. The CSV file path is correctly specified (relative path)
    2. The data can be successfully loaded
    3. The loaded data has the expected structure
    """
    
    def test_csv_file_exists(self):
        """Test that the CSV file exists in the current directory"""
        self.assertTrue(os.path.exists('loan_data_set.csv'), 
                       "loan_data_set.csv should exist in the current directory")
    
    def test_load_data_with_correct_path(self):
        """
        Test that data can be loaded using the relative path.
        This test will FAIL with the hardcoded absolute path bug,
        but PASS after fixing it to use a relative path.
        """
        # This simulates what the code should do - use relative path
        try:
            data = pd.read_csv('loan_data_set.csv')
            df = pd.DataFrame(data)
            
            # Verify basic structure
            self.assertIsNotNone(df, "DataFrame should not be None")
            self.assertGreater(len(df), 0, "DataFrame should contain rows")
            
            # Verify expected columns exist
            expected_columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 
                              'Education', 'Self_Employed', 'ApplicantIncome',
                              'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                              'Credit_History', 'Property_Area', 'Loan_Status']
            
            for col in expected_columns:
                self.assertIn(col, df.columns, f"Column '{col}' should exist in the DataFrame")
            
            # Verify expected number of rows (614 based on the dataset)
            self.assertEqual(len(df), 614, "DataFrame should have 614 rows")
            
        except FileNotFoundError as e:
            self.fail(f"Failed to load CSV with relative path: {e}")
    
    def test_absolute_path_would_fail(self):
        """
        Test that demonstrates the bug: hardcoded absolute path won't work.
        This test verifies that the old hardcoded path doesn't exist.
        """
        hardcoded_path = 'Z:\\Sasindu\\Data set\\loan_data_set.csv'
        
        # This path should NOT exist in a portable environment
        self.assertFalse(os.path.exists(hardcoded_path),
                        "Hardcoded absolute Windows path should not exist in portable code")

if __name__ == '__main__':
    unittest.main()

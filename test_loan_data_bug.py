#!/usr/bin/env python
# coding: utf-8
"""
Unit test for the file path bug in Loan_Data_1.py

Bug: Line 21 uses a hard-coded Windows path 'Z:\\Sasindu\\Data set\\loan_data_set.csv'
that doesn't exist in most environments, when it should use the relative path 
'loan_data_set.csv' that exists in the current directory.

This test verifies that:
1. The hard-coded path would fail (FileNotFoundError)
2. The correct relative path works properly
"""

import unittest
import pandas as pd
import os
import sys


class TestLoanDataFilePath(unittest.TestCase):
    """Test suite for verifying the file path bug and its fix."""
    
    def setUp(self):
        """Verify test prerequisites."""
        # Check that the correct CSV file exists in the current directory
        self.correct_path = 'loan_data_set.csv'
        self.assertTrue(
            os.path.exists(self.correct_path),
            f"Expected CSV file '{self.correct_path}' not found in current directory"
        )
    
    def test_hardcoded_path_fails(self):
        """Test that the hard-coded Windows path from the bug does not exist."""
        buggy_path = 'Z:\\Sasindu\\Data set\\loan_data_set.csv'
        
        # This path should not exist (unless we're on that specific machine)
        # We test that attempting to read from it would raise an error
        with self.assertRaises(FileNotFoundError):
            pd.read_csv(buggy_path)
    
    def test_relative_path_works(self):
        """Test that using the relative path works correctly."""
        correct_path = 'loan_data_set.csv'
        
        # This should work without raising an exception
        try:
            df = pd.read_csv(correct_path)
            
            # Verify we got valid data
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0, "DataFrame should not be empty")
            
            # Verify expected columns exist (from the loan dataset)
            expected_columns = ['Loan_ID', 'Gender', 'Married', 'Dependents', 
                              'Education', 'Self_Employed', 'ApplicantIncome',
                              'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                              'Credit_History', 'Property_Area', 'Loan_Status']
            
            for col in expected_columns:
                self.assertIn(col, df.columns, f"Expected column '{col}' not found")
            
            # Verify we have the expected number of rows (614 per the dataset)
            self.assertEqual(len(df), 614, "Expected 614 rows in the dataset")
            
        except FileNotFoundError:
            self.fail(f"Relative path '{correct_path}' should exist but was not found")
        except Exception as e:
            self.fail(f"Unexpected error when reading CSV: {e}")
    
    def test_load_data_function(self):
        """
        Test a fixed version of the data loading logic.
        This simulates what the corrected code should do.
        """
        # This is what the fixed code should look like
        def load_loan_data(file_path='loan_data_set.csv'):
            """Load the loan dataset from a CSV file."""
            return pd.read_csv(file_path)
        
        # Test that the fixed function works
        df = load_loan_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 614)
        self.assertEqual(len(df.columns), 13)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

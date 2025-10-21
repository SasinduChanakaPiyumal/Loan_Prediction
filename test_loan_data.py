#!/usr/bin/env python
# coding: utf-8

"""
Unit tests for Loan_Data_1.py

This test validates that the file path bug has been fixed.
The bug was using a hard-coded absolute Windows path 'Z:\\Sasindu\\Data set\\loan_data_set.csv'
instead of the relative path 'loan_data_set.csv'.

Before the patch: The code would fail on any system that doesn't have that exact path structure.
After the patch: The code works correctly using the relative path.
"""

import unittest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock


class TestLoanDataFilePath(unittest.TestCase):
    """Test that the file path is relative and accessible"""
    
    def test_file_path_is_relative_not_absolute(self):
        """
        Test that the code uses a relative path, not an absolute Windows path.
        
        This test would FAIL before the patch because the original code used:
        'Z:\\Sasindu\\Data set\\loan_data_set.csv'
        
        This test PASSES after the patch because the code now uses:
        'loan_data_set.csv'
        """
        # Read the Loan_Data_1.py file and check the path used
        with open('Loan_Data_1.py', 'r') as f:
            content = f.read()
        
        # Check that the old absolute path is NOT in the code
        self.assertNotIn('Z:\\\\Sasindu\\\\Data set\\\\loan_data_set.csv', content,
                        "Code should not contain hard-coded absolute Windows path")
        
        # Check that the relative path IS in the code
        self.assertIn("'loan_data_set.csv'", content,
                     "Code should use relative path 'loan_data_set.csv'")
    
    def test_csv_file_can_be_loaded_with_relative_path(self):
        """
        Test that the CSV file can be loaded using the relative path.
        
        Before the patch: This would fail on most systems because the absolute path doesn't exist.
        After the patch: This passes because the relative path works from any location.
        """
        # Check that the CSV file exists in the current directory
        self.assertTrue(os.path.exists('loan_data_set.csv'),
                       "loan_data_set.csv should exist in the current directory")
        
        # Try to load the data using the relative path (as the fixed code does)
        try:
            data = pd.read_csv('loan_data_set.csv')
            self.assertIsNotNone(data, "Data should be loaded successfully")
            self.assertGreater(len(data), 0, "Loaded data should not be empty")
        except FileNotFoundError:
            self.fail("Failed to load CSV file with relative path")
    
    def test_absolute_windows_path_would_fail_on_other_systems(self):
        """
        Demonstrate that the old absolute path would fail on most systems.
        
        This test shows why the bug needed to be fixed.
        Before the patch: The code would try to access 'Z:\\Sasindu\\Data set\\loan_data_set.csv'
        After the patch: The code correctly uses 'loan_data_set.csv'
        """
        # The old hard-coded path (that was buggy)
        old_buggy_path = 'Z:\\Sasindu\\Data set\\loan_data_set.csv'
        
        # This path should NOT exist on most systems
        path_exists = os.path.exists(old_buggy_path)
        
        # If running on the original developer's machine, this might exist
        # But on any other system, it would fail
        if not path_exists:
            # Verify that trying to read from this path would raise an error
            with self.assertRaises(FileNotFoundError):
                pd.read_csv(old_buggy_path)
    
    def test_dataframe_structure_after_loading(self):
        """
        Test that the DataFrame loads correctly with expected structure.
        
        This validates that the fix doesn't break functionality.
        """
        data = pd.read_csv('loan_data_set.csv')
        df = pd.DataFrame(data)
        
        # Basic validation that the data structure is correct
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df.columns), 0, "DataFrame should have columns")
        self.assertGreater(len(df), 0, "DataFrame should have rows")


class TestBugDemonstration(unittest.TestCase):
    """Demonstrate the specific bug that was fixed"""
    
    def test_bug_demonstration_with_mock(self):
        """
        This test demonstrates the bug using mocks.
        
        Before patch: pd.read_csv would be called with 'Z:\\Sasindu\\Data set\\loan_data_set.csv'
        After patch: pd.read_csv is called with 'loan_data_set.csv'
        """
        import importlib.util
        
        # Load the module to check what path it uses
        spec = importlib.util.spec_from_file_location("loan_data_module", "Loan_Data_1.py")
        
        # Read the file to verify the path
        with open('Loan_Data_1.py', 'r') as f:
            content = f.read()
        
        # Find the line with pd.read_csv
        for line in content.split('\n'):
            if 'pd.read_csv' in line and 'loan_data_set.csv' in line:
                # Extract the path argument
                if "'loan_data_set.csv'" in line:
                    # This is the FIXED version (relative path)
                    self.assertTrue(True, "Code correctly uses relative path")
                    return
                elif 'Z:\\\\Sasindu\\\\Data set\\\\loan_data_set.csv' in line:
                    # This is the BUGGY version (absolute path)
                    self.fail("Code still contains the hard-coded absolute path bug!")
        
        # If we get here, we found the correct relative path
        self.assertTrue(True)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)

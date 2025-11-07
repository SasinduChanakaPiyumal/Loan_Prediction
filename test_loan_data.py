import unittest
import pandas as pd

# Import the fix_dependents function from Loan_Data_1.py
from Loan_Data_1 import fix_dependents

class TestFixDependents(unittest.TestCase):
    def test_fix_dependents(self):
        # Create a DataFrame with 'Dependents' column that contains '3+' and NaN values
        data = {'Dependents': ['1', '2', '3+', None, '0']}
        df = pd.DataFrame(data)
        
        # Apply fix_dependents
        df_fixed = fix_dependents(df.copy())
        
        # Check that the column has no '3+' values and does not contain NaN
        self.assertFalse((df_fixed['Dependents'] == '3+').any(), "Column should not contain '3+'")
        self.assertFalse(df_fixed['Dependents'].isnull().any(), "Column should not contain NaN values")

        # Check that the column is of integer type
        self.assertTrue(pd.api.types.is_integer_dtype(df_fixed['Dependents']), "Column should be integer type")

if __name__ == '__main__':
    unittest.main()

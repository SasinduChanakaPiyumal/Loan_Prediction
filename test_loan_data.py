import unittest
import pandas as pd

class TestLoanData(unittest.TestCase):
    def test_data_loading(self):
        try:
            df = pd.read_csv('loan_data_set.csv')
            self.assertFalse(df.empty)
        except FileNotFoundError:
            self.fail("Failed to load dataset. Check the file path.")

if __name__ == '__main__':
    unittest.main()

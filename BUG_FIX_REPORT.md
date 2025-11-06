# Bug Fix Report: File Path Issue in Loan_Data_1.py

## Bug Description

**Location:** `Loan_Data_1.py`, Line 21

**Issue:** The code contained a hard-coded Windows-specific file path that would fail in most environments:
```python
df = pd.read_csv('Z:\\Sasindu\\Data set\\loan_data_set.csv')
```

This path references a specific location on a Windows machine (`Z:\Sasindu\Data set\loan_data_set.csv`) that:
1. Won't exist on other machines
2. Won't work on non-Windows operating systems (Linux, macOS)
3. Contains spaces in the directory name, which can cause additional issues
4. Is an absolute path that breaks portability

## Impact

This bug would cause the script to fail immediately with a `FileNotFoundError` when run on any machine other than the original developer's specific environment.

## Fix Applied

**Changed Line 21 from:**
```python
df = pd.read_csv('Z:\\Sasindu\\Data set\\loan_data_set.csv')
```

**To:**
```python
df = pd.read_csv('loan_data_set.csv')
```

This uses a relative path to load the CSV file from the current working directory, making the code portable across different systems and environments.

## Unit Test

A comprehensive unit test has been created in `test_loan_data_bug.py` that:

1. **Tests the buggy behavior**: Verifies that the hard-coded path doesn't exist (would raise `FileNotFoundError`)
2. **Tests the fix**: Verifies that the relative path works correctly
3. **Validates data integrity**: Confirms the loaded DataFrame has the expected structure (614 rows, 13 columns, correct column names)

### Running the Test

```bash
python test_loan_data_bug.py
```

### Expected Output

With the fix applied, all tests should pass:
```
test_hardcoded_path_fails (__main__.TestLoanDataFilePath) ... ok
test_load_data_function (__main__.TestLoanDataFilePath) ... ok
test_relative_path_works (__main__.TestLoanDataFilePath) ... ok

----------------------------------------------------------------------
Ran 3 tests in X.XXXs

OK
```

## Verification

The fix ensures:
- ✅ The script can run on any machine with the CSV file in the same directory
- ✅ Cross-platform compatibility (Windows, Linux, macOS)
- ✅ No dependency on specific directory structures
- ✅ Better code maintainability and portability

## Related Files

- **Fixed file**: `Loan_Data_1.py` (line 21)
- **Test file**: `test_loan_data_bug.py`
- **Data file**: `loan_data_set.csv` (unchanged, located in project root)

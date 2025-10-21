# Bug Fix Summary

## Bug Description

**Location:** `Loan_Data_1.py`, line 21

**Type:** Hard-coded absolute file path

**Severity:** High (prevents code from running on other systems)

### The Problem

The original code used a hard-coded absolute Windows-specific path:

```python
data = pd.read_csv('Z:\\Sasindu\\Data set\\loan_data_set.csv')
```

This path has multiple issues:
1. **Not portable:** Only works on the original developer's machine with that exact directory structure
2. **Platform-specific:** Uses Windows path separators and drive letters
3. **Fails on other systems:** Will raise `FileNotFoundError` on any other computer
4. **Unnecessary:** The CSV file `loan_data_set.csv` exists in the same directory as the Python script

### Impact

- Code cannot be executed on any system other than the original developer's machine
- Prevents collaboration and deployment
- Makes the code non-reproducible

## The Fix

**Changed line 21 from:**
```python
data = pd.read_csv('Z:\\Sasindu\\Data set\\loan_data_set.csv')
```

**To:**
```python
data = pd.read_csv('loan_data_set.csv')
```

### Why This Fixes the Bug

1. **Portable:** Works on any operating system (Windows, Linux, macOS)
2. **Relative path:** Looks for the file in the current working directory
3. **Standard practice:** Follows Python best practices for file handling
4. **Reproducible:** Anyone can run the code if they have the CSV file in the same directory

## Unit Tests

Created comprehensive unit tests in `test_loan_data.py` that verify:

1. **test_file_path_is_relative_not_absolute()**
   - Verifies the code no longer contains the hard-coded absolute path
   - Confirms the relative path is used instead
   - **Would FAIL before the patch**, PASSES after

2. **test_csv_file_can_be_loaded_with_relative_path()**
   - Tests that the CSV file can be successfully loaded with the relative path
   - Validates the loaded data is not empty
   - **Would FAIL before the patch** (on most systems), PASSES after

3. **test_absolute_windows_path_would_fail_on_other_systems()**
   - Demonstrates why the original path was problematic
   - Shows that the old path doesn't exist on most systems

4. **test_dataframe_structure_after_loading()**
   - Ensures the fix doesn't break functionality
   - Validates the DataFrame loads correctly

5. **test_bug_demonstration_with_mock()**
   - Explicitly checks which path is used in the code
   - **Would FAIL before the patch**, PASSES after

## Running the Tests

To run the unit tests:

```bash
python test_loan_data.py
```

Or with verbose output:

```bash
python -m unittest test_loan_data.py -v
```

## Test Results

- **Before the patch:** Tests would fail because the hard-coded path doesn't exist on most systems
- **After the patch:** All tests pass, confirming the bug is fixed and the code works correctly

## Verification

You can verify the fix by:
1. Checking line 21 of `Loan_Data_1.py` - it should use `'loan_data_set.csv'`
2. Running the unit tests - all should pass
3. Running the main script - it should load the data successfully

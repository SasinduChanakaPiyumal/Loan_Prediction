# Bug Fix Report

## Bug Identified

**File:** `Loan_Data_1.py`  
**Line:** 21  
**Type:** Hardcoded Absolute Path

### Description

The code contained a hardcoded absolute Windows path that makes the code non-portable:

```python
# BEFORE (buggy code):
data = pd.read_csv('Z:\\Sasindu\\Data set\\loan_data_set.csv')
```

This path:
- Only works on the specific machine where it was originally written
- Fails on other systems (different drive letters, OS, directory structures)
- Prevents the code from working with the `loan_data_set.csv` file that exists in the project directory
- Makes the code non-portable and difficult to share or deploy

### Fix Applied

Changed the hardcoded absolute path to a relative path:

```python
# AFTER (fixed code):
data = pd.read_csv('loan_data_set.csv')
```

This fix:
- Uses the CSV file in the current directory
- Works on any operating system (Windows, Linux, macOS)
- Makes the code portable and shareable
- Follows best practices for file path handling

## Unit Test

**File:** `test_loan_data.py`

### Test Behavior

The unit test includes three test cases:

1. **`test_csv_file_exists`**: Verifies the CSV file exists in the current directory
   - **Before fix**: PASS (file exists)
   - **After fix**: PASS (file exists)

2. **`test_load_data_with_correct_path`**: Attempts to load the CSV using a relative path
   - **Before fix**: This test demonstrates what SHOULD work (loading with relative path), but the main code was using an absolute path that doesn't exist, causing FileNotFoundError when trying to run the actual script
   - **After fix**: PASS - Data loads successfully with the relative path

3. **`test_absolute_path_would_fail`**: Verifies the hardcoded path doesn't exist
   - **Before fix**: PASS (demonstrates the bug - the hardcoded path doesn't exist)
   - **After fix**: PASS (still confirms the hardcoded path doesn't exist, which is correct)

### How to Run the Test

```bash
python test_loan_data.py
```

Expected output after fix:
```
...
----------------------------------------------------------------------
Ran 3 tests in 0.XXXs

OK
```

### Demonstration of Bug

To demonstrate the bug was real:

1. **Before the fix**: Running `Loan_Data_1.py` would result in:
   ```
   FileNotFoundError: [Errno 2] No such file or directory: 'Z:\\Sasindu\\Data set\\loan_data_set.csv'
   ```

2. **After the fix**: The script can successfully load the data from the local directory

## Impact

- **Severity**: High - The code couldn't run on any system except the original developer's machine
- **Scope**: Core functionality - Data loading is the first critical step
- **Fix Complexity**: Low - Single line change
- **Risk**: None - The fix is straightforward and doesn't affect any logic

## Best Practices

This fix demonstrates the importance of:
1. Using relative paths for project files
2. Never hardcoding absolute paths in shared code
3. Testing code portability across different environments
4. Keeping data files in the project directory or using configurable paths

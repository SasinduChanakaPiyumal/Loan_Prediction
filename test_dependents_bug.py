"""
Unit test to demonstrate the bug in Dependents column preprocessing.

The bug occurs when the mode of the Dependents column is '3+'.
The buggy code calculates mode before replacing '3+' with '3',
which causes NaN values to be filled with '3+' while existing '3+' 
values are replaced with '3', creating an inconsistency.
"""

import pandas as pd
import numpy as np


def test_dependents_preprocessing_bug():
    """
    Test case that demonstrates the bug in the original code.
    This test creates a scenario where '3+' is the mode.
    """
    # Create a test DataFrame where '3+' is the mode (most frequent value)
    test_data = {
        'Dependents': ['0', '1', '2', '3+', '3+', '3+', '3+', np.nan, np.nan]
    }
    df = pd.DataFrame(test_data)
    
    print("Original DataFrame:")
    print(df)
    print(f"\nMode before replacement: {df['Dependents'].mode()[0]}")
    
    # BUGGY CODE (original version)
    print("\n--- Testing BUGGY version ---")
    df_buggy = df.copy()
    mode_buggy = df_buggy['Dependents'].mode()[0]
    print(f"Mode calculated: {mode_buggy}")
    df_buggy['Dependents'] = df_buggy['Dependents'].replace('3+', '3').fillna(mode_buggy)
    print("Result after buggy code:")
    print(df_buggy)
    print(f"Unique values in Dependents: {df_buggy['Dependents'].unique()}")
    
    # Check for the bug: if mode was '3+', we should have inconsistent data
    if mode_buggy == '3+':
        has_bug = '3+' in df_buggy['Dependents'].values
        print(f"\nBUG DETECTED: Column contains '3+' after replacement: {has_bug}")
        # The buggy version will have both '3' and '3+' values
        assert has_bug, "Expected bug not found - test scenario invalid"
        assert '3' in df_buggy['Dependents'].values, "Expected '3' values from replacement"
        print("✗ BUGGY CODE: Creates inconsistency (has both '3' and '3+' values)")
    else:
        print("\nBug scenario not triggered (mode was not '3+')")
    
    # FIXED CODE (corrected version)
    print("\n--- Testing FIXED version ---")
    df_fixed = df.copy()
    df_fixed['Dependents'] = df_fixed['Dependents'].replace('3+', '3')
    mode_fixed = df_fixed['Dependents'].mode()[0]
    print(f"Mode calculated after replacement: {mode_fixed}")
    df_fixed['Dependents'] = df_fixed['Dependents'].fillna(mode_fixed)
    print("Result after fixed code:")
    print(df_fixed)
    print(f"Unique values in Dependents: {df_fixed['Dependents'].unique()}")
    
    # Check that fixed version is consistent
    has_three_plus = '3+' in df_fixed['Dependents'].values
    print(f"\nFixed version contains '3+': {has_three_plus}")
    assert not has_three_plus, "Fixed version should not contain '3+' values"
    print("✓ FIXED CODE: All '3+' values consistently replaced with '3'")
    
    # Verify the data can be converted to int32 without errors
    try:
        df_fixed['Dependents'] = df_fixed['Dependents'].astype('int32')
        print("✓ FIXED CODE: Successfully converted to int32")
    except ValueError as e:
        print(f"✗ Conversion to int32 failed: {e}")
        raise
    
    # Try the same with buggy version (will fail if bug is present)
    if mode_buggy == '3+':
        try:
            df_buggy['Dependents'] = df_buggy['Dependents'].astype('int32')
            print("✗ BUGGY CODE: Unexpectedly converted to int32")
            assert False, "Buggy code should fail int32 conversion when '3+' remains"
        except (ValueError, TypeError) as e:
            print(f"✓ BUGGY CODE: Expected failure on int32 conversion: {e}")
    
    print("\n" + "="*70)
    print("TEST PASSED: Bug demonstrated and fix verified!")
    print("="*70)


def test_dependents_preprocessing_no_bug_scenario():
    """
    Test case where '3+' is NOT the mode - both versions should work.
    This verifies that the fix doesn't break the normal case.
    """
    print("\n\n" + "="*70)
    print("Testing scenario where '3+' is NOT the mode")
    print("="*70)
    
    # Create a test DataFrame where '0' is the mode
    test_data = {
        'Dependents': ['0', '0', '0', '0', '1', '2', '3+', np.nan, np.nan]
    }
    df = pd.DataFrame(test_data)
    
    print("Original DataFrame:")
    print(df)
    print(f"\nMode: {df['Dependents'].mode()[0]}")
    
    # BUGGY CODE
    df_buggy = df.copy()
    mode_buggy = df_buggy['Dependents'].mode()[0]
    df_buggy['Dependents'] = df_buggy['Dependents'].replace('3+', '3').fillna(mode_buggy)
    
    # FIXED CODE
    df_fixed = df.copy()
    df_fixed['Dependents'] = df_fixed['Dependents'].replace('3+', '3')
    mode_fixed = df_fixed['Dependents'].mode()[0]
    df_fixed['Dependents'] = df_fixed['Dependents'].fillna(mode_fixed)
    
    print(f"\nBuggy result: {df_buggy['Dependents'].tolist()}")
    print(f"Fixed result: {df_fixed['Dependents'].tolist()}")
    
    # Both should be the same when mode is not '3+'
    assert df_buggy['Dependents'].equals(df_fixed['Dependents']), \
        "Results should be identical when mode is not '3+'"
    
    # Neither should contain '3+'
    assert '3+' not in df_buggy['Dependents'].values
    assert '3+' not in df_fixed['Dependents'].values
    
    print("✓ Both versions work correctly when '3+' is not the mode")
    print("="*70)


if __name__ == "__main__":
    test_dependents_preprocessing_bug()
    test_dependents_preprocessing_no_bug_scenario()
    print("\n✅ All tests completed successfully!")

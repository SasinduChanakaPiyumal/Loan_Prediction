import subprocess
import os
import sys

def test_script_execution():
    """
    Tests that Loan_Data_1.py runs successfully.
    Before the patch, this would fail due to a hardcoded file path.
    """
    # Set environment to use non-interactive backend for matplotlib
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    
    print("Attempting to run Loan_Data_1.py...")
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, 'Loan_Data_1.py'],
            env=env,
            capture_output=True,
            text=True,
            timeout=120 # Increased timeout
        )
        
        if result.returncode != 0:
            print(f"Script failed with return code {result.returncode}")
            print("Stderr output:")
            print(result.stderr)
            print("Stdout output (last 20 lines):")
            print('\n'.join(result.stdout.splitlines()[-20:]))
            sys.exit(1) # Test fails
        else:
            print("Script executed successfully.")
            sys.exit(0) # Test passes

    except subprocess.TimeoutExpired:
        print("Script timed out (likely stuck in plt.show() or loop)")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_script_execution()

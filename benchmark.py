import time
import cProfile
import pstats
import io
import warnings

warnings.filterwarnings('ignore')

def run_script():
    with open("Loan_Data_1.py") as f:
        code = f.read()
    
    # Execute in a global namespace to simulate script execution
    exec(code, {})

if __name__ == "__main__":
    print("Starting benchmark...")
    start_time = time.time()
    
    pr = cProfile.Profile()
    pr.enable()
    
    try:
        run_script()
    except Exception as e:
        print(f"Script failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    pr.disable()
    end_time = time.time()
    
    print(f"Total execution time: {end_time - start_time:.4f} seconds")
    
    s = io.StringIO()
    sortby = 'tottime'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(20)
    print(s.getvalue())

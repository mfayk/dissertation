import os
import time

def touch_all_files(directory):
    current_time = time.time()
    for root, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            try:
                os.utime(filepath, (current_time, current_time))
                print(f"Touched: {filepath}")
            except Exception as e:
                print(f"Failed to touch {filepath}: {e}")

# Example usage:
directory_path = "/scratch/mfaykus"
touch_all_files(directory_path)

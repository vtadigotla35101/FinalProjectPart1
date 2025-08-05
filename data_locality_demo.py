import numpy as np
import time
import sys

# --- Configuration ---
# Set the matrix size. For noticeable cache effects, it should be large enough.
# A size of 2000x2000 or larger is usually good.
MATRIX_SIZE = 2000 # N x N matrix
NUM_RUNS = 5      # Number of times to run each test for averaging

# --- Functions to demonstrate data locality ---

def sum_matrix_row_major(matrix):
    """
    Sums all elements in a matrix using row-major order traversal.
    This is generally cache-friendly for C-style (row-major) arrays like NumPy's.
    """
    total_sum = 0
    rows, cols = matrix.shape
    for r in range(rows):
        for c in range(cols):
            total_sum += matrix[r, c] # Accessing elements row by row, column by column
    return total_sum

def sum_matrix_column_major(matrix):
    """
    Sums all elements in a matrix using column-major order traversal.
    This is generally cache-unfriendly for C-style (row-major) arrays like NumPy's,
    as it jumps across memory locations for each element in a column.
    """
    total_sum = 0
    rows, cols = matrix.shape
    for c in range(cols): # Outer loop iterates through columns
        for r in range(rows): # Inner loop iterates through rows
            total_sum += matrix[r, c] # Accessing elements column by column, row by row
    return total_sum

# --- Main execution and timing ---

def run_performance_test():
    """
    Generates a large matrix and compares the performance of row-major
    and column-major summation.
    """
    print(f"--- Data Locality Performance Test ---")
    print(f"Matrix size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Number of runs per test: {NUM_RUNS}")
    print("-" * 40)

    # Create a large NumPy array filled with random integers
    # NumPy arrays are stored in row-major order by default (C-contiguous)
    print("Generating matrix...")
    data_matrix = np.random.randint(0, 100, size=(MATRIX_SIZE, MATRIX_SIZE), dtype=np.int32)
    print("Matrix generated.")
    print("-" * 40)

    # Test Row-Major Summation
    row_major_times = []
    print(f"Testing Row-Major Summation ({NUM_RUNS} runs)...")
    for i in range(NUM_RUNS):
        start_time = time.perf_counter()
        _ = sum_matrix_row_major(data_matrix)
        end_time = time.perf_counter()
        duration = end_time - start_time
        row_major_times.append(duration)
        print(f"  Run {i+1}: {duration:.6f} seconds")
    avg_row_major_time = sum(row_major_times) / NUM_RUNS
    print(f"Average Row-Major Time: {avg_row_major_time:.6f} seconds")
    print("-" * 40)

    # Test Column-Major Summation
    column_major_times = []
    print(f"Testing Column-Major Summation ({NUM_RUNS} runs)...")
    for i in range(NUM_RUNS):
        start_time = time.perf_counter()
        _ = sum_matrix_column_major(data_matrix)
        end_time = time.perf_counter()
        duration = end_time - start_time
        column_major_times.append(duration)
        print(f"  Run {i+1}: {duration:.6f} seconds")
    avg_column_major_time = sum(column_major_times) / NUM_RUNS
    print(f"Average Column-Major Time: {avg_column_major_time:.6f} seconds")
    print("-" * 40)

    # --- Analysis ---
    print("\n--- Performance Analysis ---")
    print(f"Average Row-Major Time:   {avg_row_major_time:.6f} seconds")
    print(f"Average Column-Major Time: {avg_column_major_time:.6f} seconds")

    if avg_row_major_time > 0:
        speedup = avg_column_major_time / avg_row_major_time
        print(f"Column-Major is approximately {speedup:.2f}x slower than Row-Major.")
    else:
        print("Row-major time was too small to calculate speedup.")

    print("\nObservation: The row-major traversal is significantly faster because NumPy arrays")
    print("are stored in row-major (C-contiguous) order. Accessing elements in row-major")
    print("order ensures that consecutive memory accesses hit CPU cache lines, reducing")
    print("expensive main memory fetches. Column-major access, on the other hand, results")
    print("in frequent cache misses as it jumps across non-contiguous memory locations.")
    print("-" * 40)

if __name__ == "__main__":
    # It's recommended to run this script in a clean environment
    # to minimize interference from other processes.
    # The actual speedup can vary based on CPU architecture, cache sizes,
    # and operating system activity.
    run_performance_test()
"""Main script to run the sleep characterization pipeline."""
import numpy as np

# Import our custom functions from batch_data.py
from batch_data import batch_load_data, split_by_subject

if __name__ == "__main__":
    print("Starting pipeline...")
    
    # 1. Batch load data from the first 5 subjects
    X_all, y_all, subjects = batch_load_data(max_recordings=5)
    
    if X_all is not None:
        print("\n--- Batch Loading Complete ---")
        print(f"Combined X shape: {X_all.shape}")
        print(f"Combined y shape: {y_all.shape}")
        
        # 2. Split data safely by subject (80% train, 20% test)
        X_train, y_train, X_test, y_test = split_by_subject(X_all, y_all, subjects, test_size=0.2)
        
        print("\n--- Split Complete ---")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape:  {X_test.shape}")
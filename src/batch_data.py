"""Main script to batch-load and process multiple sleep recordings."""
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Import our custom functions from load_data.py
from load_data import download_and_extract, find_recordings, load_epochs


def batch_load_data(max_recordings=5):
    """Load and concatenate epochs from multiple recordings with error handling."""
    # 1. Ensure data is downloaded and extracted
    download_and_extract()
    
    # 2. Get the list of all available recording pairs
    recs = find_recordings()
    
    # Limit to max_recordings for testing (change or remove later for full dataset)
    selected_recs = recs[:max_recordings]
    print(f"Processing {len(selected_recs)} out of {len(recs)} total recordings...\n")
    
    X_list = []
    y_list = []
    subject_ids = []
    
    for idx, (psg_path, hyp_path) in enumerate(selected_recs):
        # Extract unique subject ID from filename (e.g., "SC4001E0" -> "SC4001")
        subject_id = psg_path.name[:7]
        print(f"[{idx+1}/{len(selected_recs)}] Loading subject {subject_id}...")
        
        try:
            # TRY to load epochs for this recording
            X_rec, y_rec = load_epochs(psg_path, hyp_path)
            
            # If successful, store them in our lists
            X_list.append(X_rec)
            y_list.append(y_rec)
            
            # Track patient ID for each epoch (crucial for train/test splitting)
            subject_ids.extend([subject_id] * len(y_rec))
            
        except Exception as e:
            # EXCEPT block: catches errors so the script doesn't crash
            print(f"  ⚠️ Error loading {subject_id}: {e}")
            print("  Skipping this recording and moving to the next one...")
            continue
            
    # Combine all patient data into master NumPy arrays
    if X_list:
        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)
        subject_ids = np.array(subject_ids)
        return X_all, y_all, subject_ids
    else:
        print("No recordings were successfully loaded.")
        return None, None, None


if __name__ == "__main__":
    # Run the batch loader for the first 5 recordings
    X_all, y_all, subjects = batch_load_data(max_recordings=5)
    
    if X_all is not None:
        print("\n--- Batch Loading Complete ---")
        print(f"Combined X shape (Epochs, Channels, Time): {X_all.shape}")
        print(f"Combined y shape (Labels): {y_all.shape}")
        print(f"Total unique subjects successfully loaded: {len(np.unique(subjects))}")
        print(f"Overall class distribution (bincount): {np.bincount(y_all)}")

def split_by_subject(X, y, subjects, test_size=0.2, random_state=42):
    """Split data by subject to prevent data leakage."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=subjects))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_subjects = np.unique(subjects[train_idx])
    test_subjects = np.unique(subjects[test_idx])
    
    print(f"\nTraining subjects: {train_subjects} ({len(X_train)} epochs)")
    print(f"Testing subjects:  {test_subjects} ({len(X_test)} epochs)")
    
    return X_train, y_train, X_test, y_test
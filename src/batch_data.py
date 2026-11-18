"""Data access for the whole group.

Most people only need load_epoch_table(): the shared table of features and
labels (built once by features.py), so nobody has to touch the raw EDF files.

batch_load_raw() is for anyone who needs the raw signal itself (for example a
neural network on raw EEG). Raw arrays are large, so use max_recordings.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from constants import EPOCH_TABLE_PATH, SEED


def load_epoch_table(sleep_only=True, rebuild=False):
    """Load the shared epoch table (building it first if it doesn't exist).

    sleep_only=True keeps only epochs inside the sleep period (sleep onset to the
    first epoch of the final wake bout). That includes mid-night wake, which is
    the awakening data. Use sleep_only=False for the whole recording.
    """
    if rebuild or not EPOCH_TABLE_PATH.exists():
        from features import build_epoch_table  # imported here so mne is only needed to build
        build_epoch_table(rebuild=rebuild)
    df = pd.read_csv(EPOCH_TABLE_PATH)
    if sleep_only:
        df = df[df["in_sleep_period"]].reset_index(drop=True)
    return df


def split_by_subject(subjects, test_size=0.2, seed=SEED):
    """Split by PERSON so no one appears in both train and test.

    subjects: one subject ID per row (e.g. df["subject"].to_numpy()).
    Returns (train_idx, test_idx); use df.iloc[train_idx] or X[train_idx].
    Both of a person's nights share the same subject number, so they always
    land on the same side of the split.
    """
    subjects = np.asarray(subjects)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(np.zeros(len(subjects)), groups=subjects))
    print(f"Train: {len(np.unique(subjects[train_idx]))} subjects, {len(train_idx):,} epochs")
    print(f"Test:  {len(np.unique(subjects[test_idx]))} subjects, {len(test_idx):,} epochs")
    return train_idx, test_idx


def batch_load_raw(max_recordings=5, sleep_only=True, channels=("EEG Fpz-Cz",)):
    """Load raw 30 s epochs from several recordings into NumPy arrays.

    Returns X (epochs, channels, time) as float32, y (stage codes), subjects,
    nights. Returns (None, None, None, None) if nothing could be loaded.
    Memory: roughly 12 KB per epoch per channel, so the full dataset is several GB.
    """
    # imported here so that people who only use the table don't need these
    from sleep_period import annotate_night
    from load_data import (download_and_extract, find_recordings, load_recording,
                           parse_recording_name)

    download_and_extract()
    recs = find_recordings()
    selected = recs[:max_recordings] if max_recordings else recs
    print(f"Processing {len(selected)} out of {len(recs)} total recordings...\n")

    X_list, y_list, subj_list, night_list = [], [], [], []
    for i, (psg, hyp) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] Loading {psg.name}...")
        try:
            subject, night = parse_recording_name(psg)
            rec = load_recording(psg, hyp, channels=channels)
            X, y = rec["X"], rec["stage"]
            if sleep_only:
                night_df = pd.DataFrame({"epoch_idx": rec["epoch_idx"], "stage": y})
                keep = annotate_night(night_df)["in_sleep_period"].to_numpy()
                X, y = X[keep], y[keep]
            X_list.append(X.astype(np.float32))
            y_list.append(y)
            subj_list.append(np.full(len(y), subject))
            night_list.append(np.full(len(y), night))
        except Exception as e:
            print(f"  WARNING: error loading {psg.name}: {e}\n  Skipping this recording.")

    if not X_list:
        print("No recordings were successfully loaded.")
        return None, None, None, None
    return (np.concatenate(X_list), np.concatenate(y_list),
            np.concatenate(subj_list), np.concatenate(night_list))


if __name__ == "__main__":
    X, y, subjects, nights = batch_load_raw(max_recordings=5)
    if X is not None:
        print("\n--- Raw batch loading complete ---")
        print(f"X shape (epochs, channels, time): {X.shape}")
        print(f"Unique subjects: {len(np.unique(subjects))}")
        print(f"Class distribution (bincount): {np.bincount(y)}")

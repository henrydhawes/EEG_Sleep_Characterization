"""Compute spectral features for every recording and write the shared epoch table.

Output: data/epochs.csv, one row per 30 s epoch, with columns
  subject, night, epoch_idx, stage, t_min, clock_hour,
  <feature columns>,
  in_sleep_period, wake_bout_len, wake_bout_start, final_bout, awakening, final_awakening
(see sleep_period.annotate_night for what the sleep/awakening columns mean).

Run once with:  python src/features.py        (add --rebuild to force a rebuild)
Everyone else reads the table with batch_data.load_epoch_table().
"""
import sys

import numpy as np
import pandas as pd
from scipy.signal import welch

from constants import (BANDS, CHANNEL_TAGS, EPOCH_SEC, EPOCH_TABLE_PATH,
                       STAGE_NAMES)
from load_data import (download_and_extract, find_recordings, load_recording,
                       parse_recording_name)
from sleep_period import annotate_night


def spectral_features(x, fs, tag):
    """x: (n_epochs, n_samples) for one channel -> dict of feature columns.

    For each band: <tag>_<band>_logpow (log10 absolute power) and
    <tag>_<band>_relpow (fraction of 0.5-30 Hz power).
    """
    freqs, psd = welch(x, fs=fs, nperseg=256, axis=-1)
    total = psd[:, (freqs >= 0.5) & (freqs <= 30)].sum(axis=1)
    feats = {}
    for name, (lo, hi) in BANDS.items():
        band = psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
        feats[f"{tag}_{name}_logpow"] = np.log10(band + 1e-20)
        feats[f"{tag}_{name}_relpow"] = band / total
    return feats


def night_table(rec, subject, night):
    """Build the annotated table for one loaded recording."""
    feats = {}
    for i, tag in enumerate(CHANNEL_TAGS.values()):
        feats.update(spectral_features(rec["X"][:, i, :], rec["sfreq"], tag))

    epoch_idx = rec["epoch_idx"]
    df = pd.DataFrame({
        "subject": subject,
        "night": night,
        "epoch_idx": epoch_idx,
        "stage": rec["stage"],
        # clock time at the start of each epoch, wrapped past midnight
        "clock_hour": (rec["start_clock_hour"] + epoch_idx * EPOCH_SEC / 3600) % 24,
        **feats,
    })
    out = annotate_night(df)

    # put the identifying columns first, t_min right after stage
    cols = ["subject", "night", "epoch_idx", "stage", "t_min", "clock_hour"]
    return out[cols + [c for c in out.columns if c not in cols]]


def build_epoch_table(rebuild=False):
    """Build (or load) the shared epoch table."""
    if EPOCH_TABLE_PATH.exists() and not rebuild:
        print(f"{EPOCH_TABLE_PATH} already exists (use --rebuild to recompute).")
        return pd.read_csv(EPOCH_TABLE_PATH)

    download_and_extract()
    recs = find_recordings()
    frames = []
    for i, (psg, hyp) in enumerate(recs, 1):
        print(f"[{i}/{len(recs)}] {psg.name}")
        try:
            subject, night = parse_recording_name(psg)
            rec = load_recording(psg, hyp, channels=list(CHANNEL_TAGS))
            frames.append(night_table(rec, subject, night))
        except Exception as e:  # keep going; one bad file shouldn't stop the batch
            print(f"  WARNING: skipped {psg.name}: {e}")

    if not frames:
        raise RuntimeError("No recordings could be processed.")
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(EPOCH_TABLE_PATH, index=False, float_format="%.6g")
    print(f"\nWrote {len(table):,} epochs from "
          f"{table.groupby(['subject', 'night']).ngroups} nights to {EPOCH_TABLE_PATH}")
    return table


if __name__ == "__main__":
    table = build_epoch_table(rebuild="--rebuild" in sys.argv)
    sleep = table[table["in_sleep_period"]]
    print(f"{len(sleep):,} epochs fall inside the sleep period")
    print("Stage counts in sleep period:",
          dict(zip(STAGE_NAMES, np.bincount(sleep["stage"], minlength=5))))

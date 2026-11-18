"""Sleep-period and awakening rules (see constants.py for the parameters).

Pure NumPy/pandas, so it can be tested without any EEG files.

Epochs are placed on a regular 30 s grid (`epoch_idx`). Epochs the scorers left
unlabeled are simply absent from the table; they count as -1 here and never
count as sleep or wake.
"""
import numpy as np
import pandas as pd

from constants import EPOCH_SEC, MIN_WAKE_EPOCHS, SLEEP_ONSET_RUN, STAGE_WAKE


def find_sleep_onset(raster, run=SLEEP_ONSET_RUN):
    """Index of the first epoch that starts `run` consecutive non-wake epochs.

    raster: stage per grid position (0 = wake, 1-4 = sleep, -1 = unscored).
    Returns None if the recording never contains such a run.
    """
    is_sleep = raster > STAGE_WAKE
    if len(is_sleep) < run:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(is_sleep, run)
    hits = np.flatnonzero(windows.all(axis=1))
    return int(hits[0]) if hits.size else None


def wake_bouts(raster):
    """Runs of consecutive wake epochs. Returns (start_indices, lengths)."""
    padded = np.concatenate([[False], raster == STAGE_WAKE, [False]]).astype(int)
    change = np.diff(padded)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    return starts, ends - starts


def annotate_night(df, onset_run=SLEEP_ONSET_RUN, min_wake_epochs=MIN_WAKE_EPOCHS):
    """Add sleep-period and awakening columns to one subject-night.

    df needs `epoch_idx` (position on the 30 s grid) and `stage`.
    Added columns:
      t_min            minutes since sleep onset (negative before onset; NaN if no onset)
      in_sleep_period  onset <= epoch <= first epoch of the final wake bout
      wake_bout_len    length in epochs of the wake bout containing this epoch (0 if not wake)
      wake_bout_start  True on the first epoch of each wake bout
      final_bout       True on every epoch of the final wake bout (the one right after
                       the last sleep epoch)
      awakening        first epoch of a qualifying mid-night wake bout
      final_awakening  first epoch of the final wake bout, if it qualifies
    The final awakening is flagged separately and is NOT also flagged as `awakening`.
    """
    out = df.sort_values("epoch_idx").reset_index(drop=True)
    idx = out["epoch_idx"].to_numpy()
    n = int(idx.max()) + 1

    raster = np.full(n, -1, dtype=int)
    raster[idx] = out["stage"].to_numpy()

    # defaults for a night with no sleep onset
    out["t_min"] = np.nan
    out["in_sleep_period"] = False
    out["wake_bout_len"] = 0
    out["wake_bout_start"] = False
    out["final_bout"] = False
    out["awakening"] = False
    out["final_awakening"] = False

    onset = find_sleep_onset(raster, onset_run)
    if onset is None:
        return out

    out["t_min"] = (idx - onset) * EPOCH_SEC / 60.0

    # wake bouts on the full grid
    starts, lengths = wake_bouts(raster)
    bout_len = np.zeros(n, dtype=int)
    bout_start = np.zeros(n, dtype=bool)
    for s, length in zip(starts, lengths):
        bout_len[s:s + length] = length
        bout_start[s] = True

    # last sleep epoch, and the final wake bout (if wake follows it directly)
    last_sleep = int(np.flatnonzero(raster > STAGE_WAKE)[-1])
    final_bout = np.zeros(n, dtype=bool)
    end = last_sleep
    if last_sleep + 1 < n and raster[last_sleep + 1] == STAGE_WAKE:
        final_bout[last_sleep + 1:last_sleep + 1 + bout_len[last_sleep + 1]] = True
        end = last_sleep + 1

    out["in_sleep_period"] = (idx >= onset) & (idx <= end)
    out["wake_bout_len"] = bout_len[idx]
    out["wake_bout_start"] = bout_start[idx]
    out["final_bout"] = final_bout[idx]
    out["awakening"], out["final_awakening"] = flag_awakenings(out, min_wake_epochs)
    return out


def flag_awakenings(df, min_wake_epochs=MIN_WAKE_EPOCHS):
    """Recompute (awakening, final_awakening) for any k = min_wake_epochs.

    Works on a table produced by annotate_night, so robustness checks can vary k
    without rebuilding the epoch table.
    """
    event = (df["wake_bout_start"] & df["in_sleep_period"]
             & (df["wake_bout_len"] >= min_wake_epochs))
    final = event & df["final_bout"]
    return event & ~df["final_bout"], final

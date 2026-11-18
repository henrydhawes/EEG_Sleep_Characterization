"""Project-wide constants and rules.

Everything the group must agree on lives here. If a rule changes, change it
here (and rebuild the epoch table) so everyone's results stay consistent.
"""
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
# Resolve from this file's location, so scripts work from any directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"          # raw EDFs + epoch table (git-ignored)
RESULTS_DIR = REPO_ROOT / "results"    # small CSVs + figures (committed)
EPOCH_TABLE_PATH = DATA_DIR / "epochs.csv"

# ---------------------------------------------------------------- randomness
SEED = 42


def seed_everything(seed=SEED):
    """Seed Python and NumPy. Pass `random_state=SEED` to scikit-learn models."""
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------- recording
EPOCH_SEC = 30
EEG_FS = 100  # Sleep-EDF EEG sampling rate (Hz)

# Stage codes used everywhere. N3 and N4 are merged into 3.
STAGE_WAKE, STAGE_N1, STAGE_N2, STAGE_N3, STAGE_REM = 0, 1, 2, 3, 4
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
ANNOT_MAP = {
    "Sleep stage W": STAGE_WAKE,
    "Sleep stage 1": STAGE_N1,
    "Sleep stage 2": STAGE_N2,
    "Sleep stage 3": STAGE_N3,
    "Sleep stage 4": STAGE_N3,
    "Sleep stage R": STAGE_REM,
}

# ---------------------------------------------------------------- spectral features
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12),
         "sigma": (12, 16), "beta": (16, 30)}

# Channels to load, and the short tag used in feature column names.
# Phase 1 will add Pz-Oz, EOG and EMG here.
CHANNEL_TAGS = {"EEG Fpz-Cz": "fpz"}

# ---------------------------------------------------------------- sleep-period rules
# Sleep onset = first run of this many consecutive non-wake epochs.
SLEEP_ONSET_RUN = 3

# The sleep period runs from onset to the first epoch of the final wake bout
# (or to the last sleep epoch if the recording ends during sleep).
# Wake before onset and after the final wake bout starts is outside the period.

# A sleep-to-wake transition counts as an "awakening" event only if the wake
# bout lasts at least this many epochs (2 epochs = 1 minute). Vary in robustness checks.
MIN_WAKE_EPOCHS = 2

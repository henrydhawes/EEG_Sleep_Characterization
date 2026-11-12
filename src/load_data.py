"""Download/unpack Sleep-EDF Expanded from the PhysioNet zip and load epochs."""
import urllib.request
import zipfile
from pathlib import Path

import mne
import numpy as np

# Always resolves to <repo>/data, no matter where you run the script from
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ZIP_PATH = DATA_DIR / "sleep-edf.zip"
ZIP_URL = ("https://physionet.org/static/published-projects/sleep-edfx/"
           "sleep-edf-database-expanded-1.0.0.zip")

ANNOT_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3,  # merge N3/N4
    "Sleep stage R": 4,
}


def _progress(blocks, block_size, total):
    if total > 0:
        pct = min(100, blocks * block_size * 100 / total)
        print(f"\rDownloading: {pct:5.1f}%", end="", flush=True)


def download_and_extract(keep_zip=False):
    """Download the zip (if needed) and extract only the Sleep Cassette files."""
    DATA_DIR.mkdir(exist_ok=True)
    if any(DATA_DIR.rglob("*-PSG.edf")):
        print("Data already extracted, skipping.")
        return
    if not ZIP_PATH.exists():
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, _progress)
        print()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = [m for m in zf.namelist() if "sleep-cassette/" in m]
        zf.extractall(DATA_DIR, members=members)
    print(f"Extracted {len(members)} files to {DATA_DIR}")
    if not keep_zip:
        ZIP_PATH.unlink()


def find_recordings():
    """Pair each PSG file with its hypnogram. Returns list of (psg, hyp) paths."""
    pairs = []
    for psg in sorted(DATA_DIR.rglob("*-PSG.edf")):
        # SC4001E0-PSG.edf  <->  SC4001EC-Hypnogram.edf (first 7 chars match)
        hyp = next(psg.parent.glob(f"{psg.name[:7]}*-Hypnogram.edf"), None)
        if hyp is not None:
            pairs.append((psg, hyp))
    return pairs


def load_epochs(psg_path, hyp_path, channel="EEG Fpz-Cz"):
    raw = mne.io.read_raw_edf(psg_path, stim_channel="marker",
                              misc=["rectal"], preload=True, verbose=False)
    raw.set_annotations(mne.read_annotations(hyp_path))
    raw.pick(channel)
    events, _ = mne.events_from_annotations(
        raw, event_id=ANNOT_MAP, chunk_duration=30.0, verbose=False)
    tmax = 30.0 - 1.0 / raw.info["sfreq"]
    epochs = mne.Epochs(raw, events, event_id=list(set(ANNOT_MAP.values())),
                        tmin=0.0, tmax=tmax, baseline=None, verbose=False)
    return epochs.get_data(), epochs.events[:, 2]


if __name__ == "__main__":
    download_and_extract()
    recs = find_recordings()
    print(f"Found {len(recs)} recordings")
    X, y = load_epochs(*recs[0])
    print(X.shape, np.bincount(y))
"""Download/unpack Sleep-EDF from the PhysioNet zip file and load epochs."""
import re
import urllib.request
import zipfile

import mne
import numpy as np

from constants import ANNOT_MAP, DATA_DIR, EPOCH_SEC

ZIP_PATH = DATA_DIR / "sleep-edf.zip"
ZIP_URL = ("https://physionet.org/static/published-projects/sleep-edfx/"
           "sleep-edf-database-expanded-1.0.0.zip")


def _progress(blocks, block_size, total):
    if total > 0:
        pct = min(100, blocks * block_size * 100 / total)
        print(f"\rDownloading: {pct:5.1f}%", end="", flush=True)


def download_and_extract(keep_zip=False):
    """Download the zip (if needed) and extract only the Sleep Cassette files."""
    DATA_DIR.mkdir(exist_ok=True)
    if any(DATA_DIR.rglob("*-PSG.edf")):  # PSG files exist, so assume data is extracted
        print("Data already extracted, skipping.")
        return
    if not ZIP_PATH.exists():
        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, _progress)
        print()
    with zipfile.ZipFile(ZIP_PATH) as zf:
        # change to include "sleep-telemetry/" if we ever want those recordings too
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


def parse_recording_name(psg_path):
    """'SC4001E0-PSG.edf' -> (subject=0, night=1).

    File names are SC4 + 2-digit subject + 1-digit night + letters. The same
    person's two nights share a subject number, so always group by SUBJECT,
    not by file name, when splitting train/test data.
    """
    m = re.match(r"SC4(\d{2})(\d)", psg_path.name)
    if m is None:
        raise ValueError(f"Unexpected file name (not a Sleep Cassette recording): {psg_path.name}")
    return int(m.group(1)), int(m.group(2))


def load_recording(psg_path, hyp_path, channels=("EEG Fpz-Cz",)):
    """Load one night as 30 s epochs plus the timing information.

    Returns a dict:
      X                 (n_epochs, n_channels, n_samples) signal
      stage             (n_epochs,) stage codes (see constants.py)
      epoch_idx         (n_epochs,) position of each epoch on the 30 s grid since recording start
      start_clock_hour  clock time of recording start in hours (e.g. 22.5 = 10:30 pm)
      sfreq             sampling rate (Hz)
    Epochs the scorers left unlabeled ("?" or movement) are absent.
    """
    raw = mne.io.read_raw_edf(psg_path, stim_channel="marker",
                              misc=["rectal"], preload=True, verbose=False)
    raw.set_annotations(mne.read_annotations(hyp_path))

    start = raw.info["meas_date"]
    start_clock_hour = (start.hour + start.minute / 60 + start.second / 3600
                        if start is not None else np.nan)

    raw.pick(list(channels))
    events, _ = mne.events_from_annotations(
        raw, event_id=ANNOT_MAP, chunk_duration=float(EPOCH_SEC), verbose=False)
    sfreq = raw.info["sfreq"]
    tmax = EPOCH_SEC - 1.0 / sfreq
    epochs = mne.Epochs(raw, events, event_id=list(set(ANNOT_MAP.values())),
                        tmin=0.0, tmax=tmax, baseline=None, verbose=False)

    # use epochs.events (not `events`): MNE may drop epochs that run past the end
    samples = epochs.events[:, 0] - raw.first_samp
    return {
        "X": epochs.get_data(),
        "stage": epochs.events[:, 2],
        "epoch_idx": np.rint(samples / sfreq / EPOCH_SEC).astype(int),
        "start_clock_hour": start_clock_hour,
        "sfreq": sfreq,
    }


def load_epochs(psg_path, hyp_path, channel="EEG Fpz-Cz"):
    """Kept for backward compatibility: returns (X, stage) for one channel."""
    rec = load_recording(psg_path, hyp_path, channels=(channel,))
    return rec["X"], rec["stage"]


if __name__ == "__main__":
    download_and_extract()
    recs = find_recordings()
    print(f"Found {len(recs)} recordings")
    rec = load_recording(*recs[0])
    print(rec["X"].shape, np.bincount(rec["stage"]), f"starts at {rec['start_clock_hour']:.2f} h")

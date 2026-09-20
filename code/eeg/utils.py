"""Helpers shared by the scripts in this folder."""
import glob
import os
import re
import sys
import warnings

import mne

mne.set_log_level("ERROR")
for msg in ("The unit for channel", "No coordinate information found", "Not setting positions"):
    warnings.filterwarnings("ignore", message=msg)

STIM_CODES = list(range(21, 39))  # word onset
PROD_CODES = list(range(41, 59))  # go cue


def vhdr_files(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.vhdr")))
    if not files:
        sys.exit(f"no .vhdr files in {folder}")
    # one file per run, otherwise runs get averaged twice
    runs = [run_id(f) for f in files]
    dup = sorted({r for r in runs if runs.count(r) > 1})
    if dup:
        sys.exit(f"{folder} has several files for {dup}; keep one version per run")
    return files


def run_id(path):
    # '251121_2_scanneron_phonated_2_Pulse Artifact Correction.vhdr' -> '251121_2_scanneron_phonated_2'
    name = os.path.basename(path)[:-5]
    m = re.match(r"\d+_\d+_[A-Za-z]+_[A-Za-z]+_\d+", name)
    return m.group(0) if m else name


def set_types(raw):
    types = {}
    for ch in raw.ch_names:
        up = ch.upper()
        if "EOG" in up:
            types[ch] = "eog"
        elif "EMG" in up:
            types[ch] = "emg"
        elif "ECG" in up or "EKG" in up:
            types[ch] = "ecg"
    raw.set_channel_types(types)
    return raw


def event_ids(event_id, codes):
    return {desc: i for desc, i in event_id.items()
            if any(f"S {c}" in desc or f"S{c}" in desc for c in codes)}

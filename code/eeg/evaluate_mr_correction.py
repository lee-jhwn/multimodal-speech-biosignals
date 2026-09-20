"""Evaluate MR artifact correction against the outside-scanner recording.

Compares EEG around word onset in three conditions: (a) raw inside the scanner,
(b) inside after gradient + pulse correction, (c) outside the scanner.
Saves the time courses (-1 to 1 s) and power spectra (0-200 Hz) of the grand-average
ERPs, and the per-channel correlation between (b) and (c).
Only stimuli shared by both protocols are used (outside runs have 9 of the 18).

    python evaluate_mr_correction.py RAW_IN MR_CORRECTED_IN RAW_OUT [--out out/mr_correction]
"""
import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import resample

from utils import STIM_CODES, event_ids, run_id, set_types, vhdr_files

TMIN, TMAX = -1.0, 1.0
TITLES = ["(a) Raw EEG inside running scanner",
          "(b) Inside scanner after magnetic artifact correction",
          "(c) Outside scanner (reference)"]


def load(path):
    raw = mne.io.read_raw_brainvision(path, preload=True)
    set_types(raw)
    raw.filter(0.1, None)
    return raw


def codes_in(files):
    found = set()
    for f in files:
        _, eid = mne.events_from_annotations(mne.io.read_raw_brainvision(f))
        found |= {c for c in STIM_CODES if event_ids(eid, [c])}
    return found


def grand_average(files, codes):
    evokeds, counts = [], {}
    for f in files:
        raw = load(f)
        events, eid = mne.events_from_annotations(raw)
        ids = event_ids(eid, codes)
        if not ids:
            continue
        epochs = mne.Epochs(raw, events, ids, TMIN, TMAX, baseline=(TMIN, 0), picks="eeg", preload=True)
        if len(epochs):
            evokeds.append(epochs.average())
            counts[run_id(f)] = len(epochs)
    print(f"  {len(counts)} runs, {sum(counts.values())} epochs")
    return (evokeds[0] if len(evokeds) == 1 else mne.grand_average(evokeds)), counts


def single_trial(files, codes, run):
    path = next((f for f in files if run_id(f) == run), None)
    if path is None:
        sys.exit(f"run {run} not found")
    raw = load(path)
    events, eid = mne.events_from_annotations(raw)
    onset = events[np.isin(events[:, 2], list(event_ids(eid, codes).values()))][0, 0]
    sf = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])
    data = raw.get_data(picks=picks, start=max(0, int(onset + TMIN * sf)),
                        stop=min(raw.n_times, int(onset + TMAX * sf))) * 1e6
    return data, np.linspace(TMIN, TMAX, data.shape[1]), [raw.ch_names[i] for i in picks]


def to_250hz(data, times):
    n = int(np.round((times[-1] - times[0]) * 250)) + 1
    return np.stack([resample(ch, n) for ch in data]), np.linspace(times[0], times[-1], n)


def spectrum(data, sf):
    freqs = rfftfreq(data.shape[1], 1 / sf)
    keep = freqs <= 200
    return freqs[keep], (np.abs(rfft(data, axis=1)) ** 2)[:, keep]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_in")
    p.add_argument("corrected_in")
    p.add_argument("raw_out")
    p.add_argument("--out", default="out/mr_correction")
    p.add_argument("--single-trial-run", default="251121_2_scanneron_phonated_10")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    files = [vhdr_files(d) for d in (args.raw_in, args.corrected_in, args.raw_out)]
    codes = sorted(set.intersection(*(codes_in(f) for f in files)))
    print(f"stimuli used: {codes}")

    # row (a) shows a single trial with the raw gradient artifact (run 10 in the paper)
    trial, t, names = single_trial(files[0], codes, args.single_trial_run)
    gas, counts = [], {}
    for label, f in zip("abc", files):
        print(f"({label})")
        ga, counts[label] = grand_average(f, codes)
        gas.append(ga)

    rows = [(*to_250hz(trial, t), names, *spectrum(gas[0].data * 1e6, gas[0].info["sfreq"]))]
    for ga in gas[1:]:
        rows.append((*to_250hz(ga.data * 1e6, ga.times), ga.ch_names, *spectrum(ga.data * 1e6, ga.info["sfreq"])))

    colors = plt.cm.jet(np.linspace(0, 1, len(names)))
    power = np.concatenate([r[4].ravel() for r in rows])
    power = power[power > 0]
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    plt.subplots_adjust(hspace=0.38, wspace=0.28)
    for i, (d, tt, chs, freqs, pw) in enumerate(rows):
        left, right = axes[i]
        for j, ch in enumerate(chs):
            left.plot(tt, d[j], color=colors[j], lw=0.8, label=ch)
            right.plot(freqs, pw[j], color=colors[j], lw=0.8)
        left.axvline(0, color="k", ls="--", alpha=0.5, lw=0.8)
        left.set_title(TITLES[i], fontsize=12, fontweight="bold", loc="left")
        left.set_xlabel("Time (s)", fontsize=14, fontweight="bold")
        left.set_ylabel("Amplitude (µV)", fontsize=14, fontweight="bold")
        left.grid(alpha=0.3)
        if i == 0:
            left.legend(loc="upper right", fontsize="x-small", ncol=2)
        else:
            left.set_ylim(-16, 41)  # same range for (b) and (c)
        right.set_yscale("log")  # shared log axis so the rows are comparable
        right.set_ylim(np.percentile(power, 0.5), power.max() * 2)
        right.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
        right.set_ylabel("Power (µV²)", fontsize=14, fontweight="bold")
        right.grid(alpha=0.3, which="both")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"erp_spectra.{ext}"), dpi=300, bbox_inches="tight")

    # similarity of the corrected in-scanner ERP to the outside-scanner ERP, per channel
    b, c = rows[1], rows[2]
    r = {ch: float(np.corrcoef(b[0][i], c[0][c[2].index(ch)])[0, 1]) for i, ch in enumerate(b[2]) if ch in c[2]}
    vals = np.array(list(r.values()))
    with open(os.path.join(args.out, "erp_correlation.csv"), "w", newline="") as fh:
        csv.writer(fh).writerows([["channel", "r"]] + [[ch, round(v, 3)] for ch, v in r.items()])
    with open(os.path.join(args.out, "counts.json"), "w") as fh:
        json.dump({"stimuli": codes, "epochs_per_run": counts}, fh, indent=2)
    print(f"correlation (b) vs (c): mean {vals.mean():.2f}, sd {vals.std():.2f}; "
          + ", ".join(f"{ch} {v:.2f}" for ch, v in r.items()))


if __name__ == "__main__":
    main()

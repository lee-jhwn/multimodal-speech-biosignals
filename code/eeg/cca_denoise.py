"""Remove EMG/EOG artifacts from MR-corrected EEG with reference-based CCA.

Each run is resampled to 250 Hz, notch filtered at 60 Hz and band-passed 0.1-45 Hz.
CCA is fit between the 9 EEG channels and EMG1-3 + EOG1-2, and the EEG-side components
listed for that run in cca_components.json are regressed out of the EEG.

    python cca_denoise.py IN_DIR OUT_DIR                    # clean all runs
    python cca_denoise.py IN_DIR OUT_DIR --verify REF_DIR   # compare with released files
    python cca_denoise.py IN_DIR OUT_DIR --inspect RUN_ID   # plot used to pick components
    python cca_denoise.py IN_DIR --match CLEAN_DIR          # rebuild cca_components.json
"""
import argparse
import csv
import itertools
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.export import export_raw
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

from utils import run_id, set_types, vhdr_files

CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cca_components.json")
THRESHOLD = 0.4


def load(path):
    raw = mne.io.read_raw_brainvision(path, preload=True)
    if raw.info["sfreq"] > 250:
        raw.resample(250, npad="auto")
    set_types(raw)
    raw.notch_filter([60], fir_design="firwin")
    raw.filter(0.1, 45.0, fir_design="firwin")
    return raw


def fit(raw):
    eeg = mne.pick_types(raw.info, eeg=True)
    ref = mne.pick_types(raw.info, eeg=False, emg=True, eog=True)
    data = raw.get_data().T
    X, Y = data[:, eeg], data[:, ref]
    n = min(X.shape[1], Y.shape[1])
    Xs, Ys = CCA(n_components=n, scale=True, max_iter=2000).fit(X, Y).transform(X, Y)
    rho = np.array([np.corrcoef(Xs[:, i], Ys[:, i])[0, 1] for i in range(n)])
    return eeg, ref, X, Y, Xs, rho


def clean(X, Xs, comps):
    # least-squares regression of the selected components out of every EEG channel
    if not comps:
        return X.copy()
    S = Xs[:, list(comps)]
    return X - S @ np.linalg.lstsq(S, X, rcond=None)[0]


def plot(raw, eeg, ref, X, Y, Xs, rho, comps, start, dur, path):
    sf = raw.info["sfreq"]
    a, b = int(start * sf), int((start + dur) * sf)
    t = np.linspace(start, start + dur, b - a)
    fig, ax = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    for k, (dat, title) in zip((0, 3), ((X, "EEG before"), (clean(X, Xs, comps), "EEG after"))):
        for i in range(len(eeg)):
            ax[k].plot(t, dat[a:b, i] + i * 60e-6, "k", lw=0.7)
            ax[k].text(t[0] - 0.1, i * 60e-6, raw.ch_names[eeg[i]], ha="right", va="center", fontsize=9)
        ax[k].plot([t[0], t[0]], [-50e-6, 0], "k", lw=3)
        ax[k].text(t[0] + 0.05, -25e-6, "50 µV", va="center", fontsize=9)
        ax[k].set_title(title)
        ax[k].set_yticks([])
    refs = StandardScaler().fit_transform(Y[a:b])
    for i in range(len(ref)):
        name = raw.ch_names[ref[i]]
        color = "r" if "EMG" in name else "b"
        ax[1].plot(t, refs[:, i] + i * 2.5, color, lw=0.9)
        ax[1].text(t[0] - 0.1, i * 2.5, name, color=color, ha="right", va="center", fontsize=9)
    ax[1].set_title("References (z-scored)")
    ax[1].set_yticks([])
    comp = StandardScaler().fit_transform(Xs[a:b])
    for c in comps:
        ax[2].plot(t, comp[:, c], lw=1.5, label=f"comp {c + 1} (rho = {rho[c]:.2f})")
    ax[2].legend(loc="upper right", fontsize=9)
    ax[2].set_title("Removed components")
    ax[3].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def match(files, clean_dir):
    """Find which components were removed, by comparing with existing cleaned files."""
    targets = {run_id(f): f for f in vhdr_files(clean_dir)}
    found = {}
    for f in files:
        run = run_id(f)
        if run not in targets:
            print(f"{run}: no cleaned file, skipped")
            continue
        eeg, _, X, _, Xs, rho = fit(load(f))
        T = mne.io.read_raw_brainvision(targets[run], preload=True).get_data()[eeg].T
        subsets = [s for k in range(Xs.shape[1] + 1) for s in itertools.combinations(range(Xs.shape[1]), k)]
        errs = sorted((np.linalg.norm(clean(X, Xs, s) - T) / np.linalg.norm(T), s) for s in subsets)
        # the right subset reproduces the file almost exactly; every other subset is far off
        (e1, best), (e2, _) = errs[:2]
        ok = e1 < 1e-3 and e2 > 100 * e1
        print(f"{'OK' if ok else '??'} {run}: removed {list(best)}, rho {np.round(rho, 2).tolist()}, "
              f"error {e1:.0e} (next best {e2:.0e})")
        found[run] = list(best)
    with open(CONFIG, "w") as fh:
        json.dump(dict(sorted(found.items())), fh, indent=2)
    print(f"saved {CONFIG}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("in_dir")
    p.add_argument("out_dir", nargs="?", default="out/cca")
    p.add_argument("--verify", metavar="REF_DIR")
    p.add_argument("--inspect", metavar="RUN_ID")
    p.add_argument("--start", type=float, default=32.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--match", metavar="CLEAN_DIR")
    p.add_argument("--auto", action="store_true", help=f"remove all components with rho > {THRESHOLD}")
    args = p.parse_args()

    files = vhdr_files(args.in_dir)
    if args.match:
        return match(files, args.match)

    chosen = json.load(open(CONFIG)) if os.path.exists(CONFIG) else {}
    refs = {run_id(f): f for f in vhdr_files(args.verify)} if args.verify else {}
    os.makedirs(args.out_dir, exist_ok=True)
    log, worst = [], 0.0

    for f in files:
        run = run_id(f)
        if args.inspect and run != args.inspect:
            continue
        raw = load(f)
        eeg, ref, X, Y, Xs, rho = fit(raw)
        if args.auto:
            comps = [i for i, r in enumerate(rho) if r > THRESHOLD]
        elif run in chosen:
            comps = chosen[run]
        else:
            sys.exit(f"{run} is missing from cca_components.json (use --match or --auto)")

        if args.inspect:
            path = os.path.join(args.out_dir, f"{run}_inspect.png")
            plot(raw, eeg, ref, X, Y, Xs, rho, comps, args.start, args.duration, path)
            return print(f"saved {path}")

        raw._data[eeg] = clean(X, Xs, comps).T
        out = os.path.join(args.out_dir, f"{run}_CCA_Cleaned.vhdr")
        export_raw(out, raw, fmt="brainvision", overwrite=True)
        log.append([run, comps] + np.round(rho, 4).tolist())
        line = f"{run}: removed {comps}, rho {np.round(rho, 2).tolist()}"

        if run in refs:
            new = mne.io.read_raw_brainvision(out, preload=True).get_data()[eeg]
            old = mne.io.read_raw_brainvision(refs[run], preload=True).get_data()[eeg]
            diff = np.abs(new - old).max() / np.abs(old).max() if new.shape == old.shape else np.inf
            worst = max(worst, diff)
            line += f", max diff vs released {diff:.0e}"
        print(line)

    with open(os.path.join(args.out_dir, "cca_log.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "removed"] + [f"rho{i + 1}" for i in range(len(log[0]) - 2)])
        w.writerows(log)
    if args.verify:
        print("verify: OK" if worst < 1e-4 else f"verify: FAILED (max diff {worst:.0e})")


if __name__ == "__main__":
    main()

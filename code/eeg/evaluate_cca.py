"""Evaluate EMG/EOG removal: grand-average ERP and scalp maps before and after CCA.

Epochs are locked to the go cue (-1.2 to 2 s, baseline -0.2 to 0 s), filtered 0.1-30 Hz
and re-referenced to linked mastoids. One average per run, then the grand average over
runs. Maps at 0, 0.3, 0.6, 0.9 and 1.2 s. Also saves the peak ERP amplitude per channel.

    python evaluate_cca.py BEFORE_DIR AFTER_DIR [--out out/cca_evaluation]
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np

from utils import PROD_CODES, event_ids, set_types, vhdr_files


def grand_average(folder):
    evokeds = []
    for f in vhdr_files(folder):
        raw = mne.io.read_raw_brainvision(f, preload=True)
        set_types(raw)
        if "FPz" in raw.ch_names:
            raw.rename_channels({"FPz": "Fpz"})  # name used by the standard_1020 montage
        raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
        raw.filter(0.1, 30.0)
        events, eid = mne.events_from_annotations(raw)
        ids = event_ids(eid, PROD_CODES)
        if not ids:
            print(f"  no go-cue triggers in {os.path.basename(f)}, skipped")
            continue
        raw.set_eeg_reference(["M1", "M2"])  # linked mastoids
        evokeds.append(mne.Epochs(raw, events, ids, -1.2, 2.0, baseline=(-0.2, 0), preload=True).average())
    ga = mne.grand_average(evokeds)
    if not np.isfinite(ga.data).all():
        raise ValueError(f"NaN/Inf in grand average of {folder}")
    return ga


def plot(ga, path):
    fig = ga.plot_joint(times=[0.0, 0.3, 0.6, 0.9, 1.2], title="", picks="eeg", show=False)
    fig.suptitle("")
    fig.axes[-1].set_title("         (µV)", fontsize=8)
    ax = next(a for a in fig.axes if "time" in a.get_xlabel().lower())
    lo, hi = ax.get_ylim()
    step = (hi - lo) * 0.1
    style = dict(arrowprops=dict(facecolor="#333333", arrowstyle="->"),
                 ha="center", va="bottom", fontsize=9, color="#333333")
    ax.annotate("Stimulus\nPresentation", xy=(-1.0, lo + 0.1 * step), xytext=(-0.8, lo + step), **style)
    ax.annotate("Speech\nProduction\nOnset", xy=(0.0, lo + 0.1 * step), xytext=(-0.2, lo + step), **style)
    ax.axvline(-1.0, color="grey", ls="--", alpha=0.5, lw=1)
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    # MNE's interactive callbacks break PDF export ('copy_from_bbox' error)
    for cid in list(fig.canvas.callbacks.callbacks.get("draw_event", {})):
        fig.canvas.mpl_disconnect(cid)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--out", default="out/cca_evaluation")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    peaks = {}
    for name, folder in (("erp_before_cca", args.before), ("erp_after_cca", args.after)):
        ga = grand_average(folder)
        ga.save(os.path.join(args.out, f"{name}-ave.fif"), overwrite=True)
        plot(ga, os.path.join(args.out, name))
        amp = np.abs(ga.copy().pick("eeg").crop(0, None).data).max(axis=1) * 1e6
        peaks[name] = dict(zip(ga.copy().pick("eeg").ch_names, np.round(amp, 1).tolist()))
        print(f"{name}: N = {ga.nave} runs, max |ERP| after go cue {amp.max():.1f} µV")
    with open(os.path.join(args.out, "peak_amplitude.json"), "w") as fh:
        json.dump(peaks, fh, indent=2)


if __name__ == "__main__":
    main()

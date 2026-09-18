#!/usr/bin/env python3
"""Split this release into one EEG array (and one rtMRI video clip) per stimulus.

Nothing here ships pre-segmented: run this to generate the segments yourself.

For every recording the script epochs the EEG on the production triggers
and writes each epoch as a .npy array. For in-scanner recordings it also cuts the
matching window out of the rtMRI video, so clip and array cover the same interval.

    python split_per_stimulus.py --out-dir ./segments

Outputs, under --out-dir:

    eeg/<context>/<condition>/<recording>_<stimulus>.npy   float64 (n_channels, n_times)
    video/<context>/<condition>/<recording>_<stimulus>.avi (in-scanner only)
    segments.csv                                           one row per segment
    channels.txt                                           channel order for every .npy

EEG processing: EOG/EMG/ECG channels typed, standard_1020 montage, 0.1-30 Hz band-pass,
mastoid (M1/M2) reference, epochs at tmin/tmax with a (-0.2, 0) baseline. These defaults
reproduce the epoching used in the paper.

EEG-to-video alignment is END-ALIGNED:
the scanner writes one 'Volume' marker per acquired MRI volume, and the video ends with
the last volume, so a trigger sitting `delta` seconds before the last Volume marker sits
`video_duration - delta` seconds into the video. Clock drift between the EEG amplifier
and the scanner makes start-aligning wrong by up to ~100 ms by the end of a run; this
avoids that.

Outside-scanner recordings have no MRI, hence no video and no Volume markers, so they
produce EEG arrays only. Those runs used a shorter 9-syllable protocol (codes 41-49),
so they yield 9 segments per recording rather than 18. They are released at the raw
stage only, so segmenting them needs `--stage raw`; the default `denoised` stage
covers in-scanner recordings and skips outside-scanner ones.

Note that `--stage raw` data is 5000 Hz and unreferenced, so segments from it are
(15, 16001) at the default window rather than (15, 801). Both released stages are
trigger-recovered, so segment counts match the protocol either way.

Checked against the release (see code/README.md): the EEG segments reproduce the
project's own epoching to float64 noise, and across all 216 in-scanner phonated trials
the clip audio peaks 0.997 +/- 0.159 s after the production trigger, which is where the
spoken syllable belongs.

One caveat on the video end-alignment: the video stream runs ~8 frames (~0.08 s) longer
than the rtMRI articulator ROI traces derived from the same acquisition, so clips sit
~0.08 s later than windows cut against those traces. That is well inside the spread of
production onsets and does not affect ordinary use, but it matters if you align clips
against ROI traces sample-for-sample.

Requires: mne, numpy, pandas, and ffmpeg on PATH for video/audio cutting.
"""

import argparse
import csv
import os
import subprocess
import sys

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

# Production triggers. In-scanner runs use the full 18-syllable set (41-58);
# outside-scanner runs use the shorter 9-syllable protocol (41-49). Codes absent
# from a recording are simply skipped, so one range covers both.
PRODUCTION_TRIGGERS = list(range(41, 59))

CONTEXTS = {
    "in_scanner": ["phonated", "silent", "imagined"],
    "outside_scanner": ["phonated", "silent", "imagined"],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Split released EEG and rtMRI video into per-stimulus segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument("--release-root", default=os.path.dirname(here),
                   help="Release root (the directory holding eeg/, video_with_audio/).")
    p.add_argument("--out-dir", default=os.path.join(os.getcwd(), "segments"),
                   help="Where to write segments. Created if absent.")
    p.add_argument("--stage", default="denoised",
                   choices=["denoised", "raw"],
                   help="Which EEG processing stage to segment.")
    p.add_argument("--contexts", nargs="+", default=list(CONTEXTS),
                   choices=list(CONTEXTS))
    p.add_argument("--conditions", nargs="+", default=["phonated", "silent", "imagined"])
    p.add_argument("--tmin", type=float, default=-1.2, help="Epoch start, s (relative to trigger).")
    p.add_argument("--tmax", type=float, default=2.0, help="Epoch end, s.")
    p.add_argument("--baseline-start", type=float, default=-0.2)
    p.add_argument("--baseline-end", type=float, default=0.0)
    p.add_argument("--l-freq", type=float, default=0.1)
    p.add_argument("--h-freq", type=float, default=30.0)
    p.add_argument("--reference", nargs="+", default=["M1", "M2"],
                   help="Reference channels. Pass 'none' to skip re-referencing.")
    p.add_argument("--no-video", action="store_true", help="Skip video cutting.")
    p.add_argument("--no-filter", action="store_true", help="Skip band-pass filtering.")
    p.add_argument("--video-codec", default="mpeg4",
                   help="ffmpeg video codec for clips. 'copy' is fast but not frame-accurate.")
    p.add_argument("--dry-run", action="store_true", help="Report what would be written.")
    return p.parse_args()


def load_labels(release_root):
    """produce-trigger code -> phonetic label dict, from phoneme_class.txt."""
    path = os.path.join(release_root, "phoneme_class.txt")
    df = pd.read_csv(path, skipinitialspace=True)
    return {
        int(r["produce_idx"]): {
            "stimuli": r["stimuli"],
            "voicing": r["voicing"],
            "place": r["place"],
            "manner": r["manner"],
        }
        for _, r in df.iterrows()
    }


def stage_dir(release_root, stage, context, condition):
    d = os.path.join(release_root, "eeg", stage, context, condition)
    return d if os.path.isdir(d) else None


def recording_id(vhdr_name):
    """Strip the BrainVision extension and any processing suffix."""
    base = os.path.splitext(os.path.basename(vhdr_name))[0]
    for suffix in ("_CCA_Cleaned",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base


def annotation_times(raw, code):
    """Onsets (s) of Stimulus S <code> annotations."""
    want = f"S {code:>3d}".replace("  ", " ")
    out = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        # BrainVision descriptions look like 'Stimulus/S 41' with variable padding.
        if "S" not in desc:
            continue
        tail = desc.split("S")[-1].strip()
        if tail.isdigit() and int(tail) == code:
            out.append(float(onset))
    return sorted(out)


def last_volume_time(raw):
    """Onset (s) of the final MRI Volume marker, or None outside the scanner."""
    vols = [float(o) for o, d in zip(raw.annotations.onset, raw.annotations.description)
            if "Volume" in d]
    return max(vols) if vols else None


def video_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def cut_clip(src, dst, start, duration, codec):
    """Cut [start, start+duration) from src. Re-encodes for frame accuracy."""
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.6f}", "-i", src,
           "-t", f"{duration:.6f}"]
    cmd += ["-c", "copy"] if codec == "copy" else ["-c:v", codec, "-q:v", "3", "-c:a", "copy"]
    cmd.append(dst)
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    root = os.path.abspath(args.release_root)
    labels = load_labels(root)
    baseline = (args.baseline_start, args.baseline_end)
    ref = None if [r.lower() for r in args.reference] == ["none"] else args.reference

    if not args.no_video and not args.dry_run:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            sys.exit("ffmpeg not found on PATH. Install it, or pass --no-video.")

    rows = []
    channel_order = None
    n_eeg = n_vid = n_skip = 0

    for context in args.contexts:
        for condition in args.conditions:
            d = stage_dir(root, args.stage, context, condition)
            if d is None:
                print(f"[skip] no {args.stage} data for {context}/{condition}")
                continue

            vhdrs = sorted(f for f in os.listdir(d)
                           if f.endswith(".vhdr") and not f.startswith("."))
            for vhdr in vhdrs:
                rec = recording_id(vhdr)
                raw = mne.io.read_raw_brainvision(os.path.join(d, vhdr),
                                                  preload=True, verbose=False)

                types = {}
                for ch in raw.ch_names:
                    up = ch.upper()
                    if "EOG" in up:
                        types[ch] = "eog"
                    elif "EMG" in up:
                        types[ch] = "emg"
                    elif "ECG" in up or "EKG" in up:
                        types[ch] = "ecg"
                if types:
                    raw.set_channel_types(types)
                raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                                on_missing="ignore")

                if not args.no_filter:
                    raw.filter(args.l_freq, args.h_freq, verbose=False)
                if ref and all(c in raw.ch_names for c in ref):
                    raw.set_eeg_reference(ref_channels=ref, verbose=False)

                sfreq = float(raw.info["sfreq"])
                dur_eeg = raw.n_times / sfreq
                last_vol = last_volume_time(raw)

                vid_path = os.path.join(root, "video_with_audio", context,
                                        condition, f"{rec}.avi")
                has_video = (not args.no_video and last_vol is not None
                             and os.path.isfile(vid_path))
                vid_dur = video_duration(vid_path) if has_video else None

                for code in PRODUCTION_TRIGGERS:
                    onsets = annotation_times(raw, code)
                    if not onsets:
                        continue
                    t_trig = onsets[-1]
                    t0, t1 = t_trig + args.tmin, t_trig + args.tmax
                    if t0 < 0 or t1 > dur_eeg:
                        print(f"  [skip] {rec} S{code}: window outside recording")
                        n_skip += 1
                        continue

                    stim = labels[code]["stimuli"]
                    i0 = int(round(t0 * sfreq))
                    i1 = i0 + int(round((args.tmax - args.tmin) * sfreq)) + 1
                    seg = raw.get_data(start=i0, stop=i1)

                    # Baseline-correct against the pre-trigger window. The window is
                    # closed at both ends, matching mne.Epochs, which includes the
                    # sample at baseline[1] -- dropping it shifts every channel by a
                    # small DC offset.
                    b0 = int(round((baseline[0] - args.tmin) * sfreq))
                    b1 = int(round((baseline[1] - args.tmin) * sfreq)) + 1
                    if b1 > b0:
                        seg = seg - seg[:, b0:b1].mean(axis=1, keepdims=True)

                    if channel_order is None:
                        channel_order = list(raw.ch_names)

                    eeg_rel = os.path.join("eeg", context, condition, f"{rec}_{stim}.npy")
                    vid_rel = ""
                    if not args.dry_run:
                        p = os.path.join(args.out_dir, eeg_rel)
                        os.makedirs(os.path.dirname(p), exist_ok=True)
                        np.save(p, seg)
                    n_eeg += 1

                    if has_video:
                        # End-aligned: place the trigger relative to the video's end.
                        t_vid = vid_dur - (last_vol - t_trig)
                        c0 = t_vid + args.tmin
                        if c0 < 0 or t_vid + args.tmax > vid_dur:
                            print(f"  [skip video] {rec} S{code}: clip outside video")
                        else:
                            vid_rel = os.path.join("video", context, condition,
                                                   f"{rec}_{stim}.avi")
                            if not args.dry_run:
                                p = os.path.join(args.out_dir, vid_rel)
                                os.makedirs(os.path.dirname(p), exist_ok=True)
                                cut_clip(vid_path, p, c0, args.tmax - args.tmin,
                                         args.video_codec)
                            n_vid += 1

                    lab = labels[code]
                    rows.append({
                        "eeg_npy": eeg_rel, "video_clip": vid_rel,
                        "context": context, "condition": condition, "recording": rec,
                        "stimulus": stim, "trigger_code": code,
                        "voicing": lab["voicing"], "place": lab["place"],
                        "manner": lab["manner"],
                        "eeg_sfreq": sfreq, "n_channels": seg.shape[0],
                        "n_times": seg.shape[1],
                        "tmin": args.tmin, "tmax": args.tmax,
                        "trigger_time_eeg_s": round(t_trig, 6),
                        "clip_start_video_s": round(t_vid + args.tmin, 6) if vid_rel else "",
                    })

                print(f"  {context}/{condition}/{rec}: "
                      f"{sum(1 for r in rows if r['recording'] == rec)} segments"
                      f"{'' if has_video else '  (no video)'}")

    if not args.dry_run and rows:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "segments.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(args.out_dir, "channels.txt"), "w") as f:
            f.write("\n".join(channel_order) + "\n")

    print(f"\nEEG segments: {n_eeg}   video clips: {n_vid}   skipped: {n_skip}")
    if not args.dry_run:
        print(f"Written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

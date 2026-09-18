#!/usr/bin/env python3
"""Split this release into one EEG array, and one rtMRI video clip, per stimulus.

No segmented files ship with the release; run this to generate them.

Each recording is epoched on the production triggers and each epoch written as a .npy
array. For in-scanner recordings the matching window is also cut out of the rtMRI video,
so clip and array cover the same interval.

    python split_per_stimulus.py --out-dir ./segments

Outputs, under --out-dir:

    eeg/<context>/<condition>/<recording>_<stimulus>.npy    float64 (n_channels, n_times)
    video/<context>/<condition>/<recording>_<stimulus>.avi  in-scanner only
    segments.csv    one row per segment: labels, timings, shapes, paths
    channels.txt    channel order for every .npy

Defaults reproduce the epoching used in the paper: production triggers, -1.2 to +2.0 s,
baseline (-0.2, 0), 0.1-30 Hz band-pass, mastoid reference, standard_1020 montage,
EOG/EMG/ECG channels typed.

EEG-to-video alignment is END-ALIGNED. The scanner writes one Volume marker per acquired
MRI volume and the video ends with the last volume, so a trigger `delta` seconds before
the final Volume marker sits `video_duration - delta` into the video. The EEG amplifier
and scanner clocks drift apart, which would make start-aligning wrong by up to ~100 ms by
the end of a run.

Outside-scanner recordings have no MRI, so no video and no Volume markers: they produce
EEG arrays only, 9 per run rather than 18. They are released at the raw stage only, so
use --stage raw for them.

See code/README.md for verification results and caveats.

Requires mne and numpy, plus ffmpeg on PATH unless --no-video.
"""

import argparse
import csv
import os
import subprocess
import sys

import mne
import numpy as np

mne.set_log_level("ERROR")

# Production-cue triggers. In-scanner runs use all 18 (41-58); outside-scanner runs use
# the first 9 (41-49). Codes absent from a recording are skipped, so one range covers both.
PRODUCTION_TRIGGERS = range(41, 59)

CONTEXTS = ("in_scanner", "outside_scanner")
CONDITIONS = ("phonated", "silent", "imagined")


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Split released EEG and rtMRI video into per-stimulus segments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--release-root", default=os.path.dirname(here),
                   help="Release root: the directory holding eeg/ and video_with_audio/.")
    p.add_argument("--out-dir", default="./segments",
                   help="Where to write segments. Created if absent.")
    p.add_argument("--stage", default="denoised", choices=["denoised", "raw"],
                   help="EEG processing stage to segment.")
    p.add_argument("--contexts", nargs="+", default=list(CONTEXTS), choices=list(CONTEXTS))
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    p.add_argument("--tmin", type=float, default=-1.2, help="Epoch start, s, relative to trigger.")
    p.add_argument("--tmax", type=float, default=2.0, help="Epoch end, s.")
    p.add_argument("--baseline", type=float, nargs=2, default=(-0.2, 0.0),
                   metavar=("START", "END"), help="Baseline window, s. Pass 0 0 to skip.")
    p.add_argument("--band", type=float, nargs=2, default=(0.1, 30.0),
                   metavar=("LOW", "HIGH"), help="Band-pass, Hz. Pass 0 0 to skip.")
    p.add_argument("--reference", nargs="+", default=["M1", "M2"],
                   help="Reference channels, or 'none' to leave unreferenced.")
    p.add_argument("--no-video", action="store_true", help="Skip video cutting.")
    p.add_argument("--dry-run", action="store_true", help="Report without writing.")
    return p.parse_args()


def load_labels(release_root):
    """produce-trigger code -> {stimuli, voicing, place, manner}, from phoneme_class.txt."""
    path = os.path.join(release_root, "phoneme_class.txt")
    with open(path, newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return {int(r["produce_idx"]): {k: r[k] for k in
                                        ("stimuli", "voicing", "place", "manner")}
                for r in reader}


def prepare(vhdr_path, band, reference):
    """Load one recording and apply the standard preprocessing."""
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

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
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")

    if band[0] or band[1]:
        raw.filter(band[0] or None, band[1] or None, verbose=False)
    if reference and all(c in raw.ch_names for c in reference):
        raw.set_eeg_reference(ref_channels=reference, verbose=False)
    return raw


def marker_times(raw, code):
    """Onsets, s, of Stimulus markers for `code`. Descriptions look like 'Stimulus/S 41'."""
    out = []
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        tail = desc.rsplit("S", 1)[-1].strip() if "S" in desc else ""
        if tail.isdigit() and int(tail) == code:
            out.append(float(onset))
    return sorted(out)


def last_volume_time(raw):
    """Onset, s, of the final MRI Volume marker, or None outside the scanner."""
    vols = [float(o) for o, d in zip(raw.annotations.onset, raw.annotations.description)
            if "Volume" in d]
    return max(vols) if vols else None


def video_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", path],
                         capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def cut_clip(src, dst, start, duration):
    """Cut [start, start+duration) from src, re-encoding so the window is frame-accurate."""
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.6f}", "-i", src,
                    "-t", f"{duration:.6f}", "-c:v", "mpeg4", "-q:v", "3", "-c:a", "copy",
                    dst], check=True)


def epoch(raw, t_trig, tmin, tmax, baseline):
    """Cut one epoch around t_trig and baseline-correct it."""
    sfreq = float(raw.info["sfreq"])
    i0 = int(round((t_trig + tmin) * sfreq))
    seg = raw.get_data(start=i0, stop=i0 + int(round((tmax - tmin) * sfreq)) + 1)

    # Baseline window is closed at both ends, matching mne.Epochs. Dropping the sample at
    # baseline[1] would shift every channel by a small DC offset.
    b0 = int(round((baseline[0] - tmin) * sfreq))
    b1 = int(round((baseline[1] - tmin) * sfreq)) + 1
    if b1 > b0:
        seg = seg - seg[:, b0:b1].mean(axis=1, keepdims=True)
    return seg


def write(out_dir, rel_path, save):
    path = os.path.join(out_dir, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save(path)


def main():
    args = parse_args()
    root = os.path.abspath(args.release_root)
    labels = load_labels(root)
    reference = None if [r.lower() for r in args.reference] == ["none"] else args.reference

    if not args.no_video and not args.dry_run:
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            sys.exit("ffmpeg not found on PATH. Install it, or pass --no-video.")

    rows, channels = [], None
    n_eeg = n_vid = n_skip = 0

    for context in args.contexts:
        for condition in args.conditions:
            d = os.path.join(root, "eeg", args.stage, context, condition)
            if not os.path.isdir(d):
                print(f"[skip] no {args.stage} data for {context}/{condition}")
                continue

            for vhdr in sorted(f for f in os.listdir(d)
                               if f.endswith(".vhdr") and not f.startswith(".")):
                rec = os.path.basename(vhdr)[:-5].removesuffix("_CCA_Cleaned")
                raw = prepare(os.path.join(d, vhdr), args.band, reference)
                channels = channels or list(raw.ch_names)

                eeg_end = raw.n_times / float(raw.info["sfreq"])
                last_vol = last_volume_time(raw)
                vid_path = os.path.join(root, "video_with_audio", context, condition, rec + ".avi")
                has_video = (not args.no_video and last_vol is not None
                             and os.path.isfile(vid_path))
                vid_end = video_duration(vid_path) if has_video else None

                n_rec = 0
                for code in PRODUCTION_TRIGGERS:
                    times = marker_times(raw, code)
                    if not times:
                        continue
                    t_trig = times[-1]
                    if t_trig + args.tmin < 0 or t_trig + args.tmax > eeg_end:
                        print(f"  [skip] {rec} S{code}: window outside recording")
                        n_skip += 1
                        continue

                    label = labels[code]
                    stim = label["stimuli"]
                    seg = epoch(raw, t_trig, args.tmin, args.tmax, args.baseline)

                    eeg_rel = os.path.join("eeg", context, condition, f"{rec}_{stim}.npy")
                    if not args.dry_run:
                        write(args.out_dir, eeg_rel, lambda p, s=seg: np.save(p, s))
                    n_eeg += 1
                    n_rec += 1

                    # End-aligned: place the trigger relative to the video's end.
                    vid_rel, clip_start = "", ""
                    if has_video:
                        t_vid = vid_end - (last_vol - t_trig)
                        if t_vid + args.tmin < 0 or t_vid + args.tmax > vid_end:
                            print(f"  [skip video] {rec} S{code}: clip outside video")
                        else:
                            clip_start = round(t_vid + args.tmin, 6)
                            vid_rel = os.path.join("video", context, condition, f"{rec}_{stim}.avi")
                            if not args.dry_run:
                                write(args.out_dir, vid_rel,
                                      lambda p, c=clip_start: cut_clip(
                                          vid_path, p, c, args.tmax - args.tmin))
                            n_vid += 1

                    rows.append({
                        "eeg_npy": eeg_rel, "video_clip": vid_rel,
                        "context": context, "condition": condition, "recording": rec,
                        "stimulus": stim, "trigger_code": code,
                        "voicing": label["voicing"], "place": label["place"],
                        "manner": label["manner"],
                        "eeg_sfreq": raw.info["sfreq"],
                        "n_channels": seg.shape[0], "n_times": seg.shape[1],
                        "tmin": args.tmin, "tmax": args.tmax,
                        "trigger_time_eeg_s": round(t_trig, 6),
                        "clip_start_video_s": clip_start,
                    })

                print(f"  {context}/{condition}/{rec}: {n_rec} segments"
                      f"{'' if has_video else '  (no video)'}")

    if rows and not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "segments.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with open(os.path.join(args.out_dir, "channels.txt"), "w") as f:
            f.write("\n".join(channels) + "\n")

    print(f"\nEEG segments: {n_eeg}   video clips: {n_vid}   skipped: {n_skip}")
    if not args.dry_run:
        print(f"Written to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()

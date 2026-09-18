#!/usr/bin/env python3
"""Recover stimulus triggers lost at acquisition, writing ONLY the marker files.

Some stimulus triggers were dropped by the acquisition system. This reconstructs them
from the presentation logs in `stimulus_logs/` and writes them back into the BrainVision
`.vmrk` marker files.

    python recover_raw_triggers.py --release-root ..            # apply
    python recover_raw_triggers.py --release-root .. --dry-run  # report only

Unlike `recover_from_log.py`, which re-exports the whole recording through MNE, this
touches nothing but the marker file. The `.eeg` binary and the `.vhdr` header are left
byte-for-byte identical. That matters for the `raw` stage: MNE's BrainVision writer emits
IEEE_FLOAT_32, so re-exporting would silently convert the INT_16 recordings, double their
size and rescale their values. Data labelled "raw" should stay exactly as recorded.

Method, per recording:

  1. read the existing stimulus markers in position order;
  2. match them against the recording's own presentation log by walking both sequences in
     order, which works because the syllable order is randomised per run;
  3. least-squares fit of log time (s) onto marker position (samples);
  4. predict a position for every log event with no marker, and insert it;
  5. renumber all markers and rewrite the file, preserving CRLF line endings.

Everything else in the marker file — New Segment, Volume, SyncStatus, Pulse Artifact
entries — is passed through untouched.
"""

import argparse
import os
import re
import sys

MARKER_RE = re.compile(r"^Mk(\d+)=(?P<rest>.*)$")
STIM_RE = re.compile(r"^Stimulus,S\s*(\d+),(\d+),")


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--release-root", default=os.path.dirname(here))
    p.add_argument("--stage", default="raw", help="EEG stage to operate on.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change without writing.")
    p.add_argument("--min-r2", type=float, default=0.9999,
                   help="Refuse to write if the log-to-EEG fit is worse than this.")
    return p.parse_args()


def read_log(path):
    """Presentation log -> [(relative_time_s, trigger_code)] in order."""
    out = []
    with open(path, errors="ignore") as f:
        for line in f:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) == 2:
                try:
                    out.append((float(parts[0]), int(parts[1])))
                except ValueError:
                    pass
    return out


def read_vmrk(path):
    with open(path, "rb") as f:
        raw = f.read()
    crlf = b"\r\n" in raw
    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\r\n" if crlf else "\n")
    return lines, crlf


def n_samples(vhdr_path, eeg_path):
    """Sample count, from the header's channel count and binary width."""
    nchan, fmt = None, None
    with open(vhdr_path, errors="ignore") as f:
        for line in f:
            if line.startswith("NumberOfChannels="):
                nchan = int(line.split("=")[1])
            elif line.startswith("BinaryFormat="):
                fmt = line.split("=")[1].strip()
    width = {"INT_16": 2, "IEEE_FLOAT_32": 4, "INT_32": 4}.get(fmt, 2)
    return os.path.getsize(eeg_path) // (nchan * width)


def fit(xs, ys):
    """Least-squares y = a*x + b, returning (a, b, r2)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1.0 - ss_res / ss_tot if ss_tot else 1.0)


def recover_one(vmrk_path, log_path, vhdr_path, eeg_path, min_r2, dry_run):
    lines, crlf = read_vmrk(vmrk_path)
    log = read_log(log_path)
    if not log:
        return None, "empty log"

    # existing markers, in file order
    entries = []           # (index_in_lines, type_and_rest)
    stim = []              # (position, code, index_in_entries)
    for i, line in enumerate(lines):
        m = MARKER_RE.match(line)
        if not m:
            continue
        rest = m.group("rest")
        entries.append(rest)
        s = STIM_RE.match(rest)
        if s:
            stim.append((int(s.group(2)), int(s.group(1)), len(entries) - 1))
    if not entries:
        return None, "no markers"
    stim.sort()

    # walk both sequences in order
    eeg_seq = [(pos, code) for pos, code, _ in stim]
    matched, ei = [], 0
    for li, (t, code) in enumerate(log):
        if ei < len(eeg_seq) and eeg_seq[ei][1] == code:
            matched.append((li, t, eeg_seq[ei][0]))
            ei += 1
    if len(matched) < 3:
        return None, f"only {len(matched)} matched markers"

    a, b, r2 = fit([t for _, t, _ in matched], [float(p) for _, _, p in matched])
    if r2 < min_r2:
        return None, f"poor fit r2={r2:.6f}"

    total = n_samples(vhdr_path, eeg_path)
    have = {li for li, _, _ in matched}
    fmt = "S {:>2}"
    additions = []
    for li, (t, code) in enumerate(log):
        if li in have:
            continue
        pos = int(round(a * t + b))
        if not (1 <= pos <= total):
            return None, f"predicted position {pos} outside recording (1..{total})"
        additions.append((pos, f"Stimulus,{fmt.format(code)},{pos},1,0", code))

    if not additions:
        return {"added": 0, "r2": r2, "codes": []}, None

    # splice in, keeping markers ordered by position
    def pos_of(rest):
        p = rest.split(",")
        try:
            return int(p[2])
        except (IndexError, ValueError):
            return -1

    merged = list(entries)
    for pos, text, _ in sorted(additions):
        at = len(merged)
        for j, rest in enumerate(merged):
            pj = pos_of(rest)
            if pj > pos:
                at = j
                break
        merged.insert(at, text)

    if not dry_run:
        first = next(i for i, l in enumerate(lines) if MARKER_RE.match(l))
        last = max(i for i, l in enumerate(lines) if MARKER_RE.match(l))
        out = lines[:first] + [f"Mk{i+1}={r}" for i, r in enumerate(merged)] + lines[last+1:]
        sep = "\r\n" if crlf else "\n"
        tmp = vmrk_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            f.write(sep.join(out))
        os.replace(tmp, vmrk_path)

    return {"added": len(additions), "r2": r2,
            "codes": sorted(c for _, _, c in additions)}, None


def main():
    args = parse_args()
    root = os.path.abspath(args.release_root)
    total_added, worst_r2, failures, touched = 0, 1.0, [], 0

    for ctx in sorted(os.listdir(os.path.join(root, "eeg", args.stage))):
        for cond in sorted(os.listdir(os.path.join(root, "eeg", args.stage, ctx))):
            d = os.path.join(root, "eeg", args.stage, ctx, cond)
            if not os.path.isdir(d):
                continue
            for fn in sorted(f for f in os.listdir(d)
                             if f.endswith(".vmrk") and not f.startswith(".")):
                rec = fn[:-5]
                log = os.path.join(root, "stimulus_logs", ctx, cond, rec + ".txt")
                if not os.path.isfile(log):
                    failures.append((rec, "no stimulus log")); continue
                res, err = recover_one(os.path.join(d, fn), log,
                                       os.path.join(d, rec + ".vhdr"),
                                       os.path.join(d, rec + ".eeg"),
                                       args.min_r2, args.dry_run)
                if err:
                    failures.append((rec, err)); continue
                if res["added"]:
                    touched += 1
                    total_added += res["added"]
                    worst_r2 = min(worst_r2, res["r2"])
                    print(f"  {rec:<36} +{res['added']:>2}  r2={res['r2']:.6f}  {res['codes']}")

    print(f"\n{'would add' if args.dry_run else 'added'} {total_added} triggers "
          f"across {touched} recordings; worst r2 = {worst_r2:.6f}")
    if failures:
        print("\nnot processed:")
        for rec, why in failures:
            print(f"  {rec}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    main()

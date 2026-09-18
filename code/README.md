# Processing code

Two kinds of script live here.

**`split_per_phoneme.py`** was written for this release and is the one to run.
Everything else is the project's own code, copied **verbatim** and unmodified for
provenance — it is the code that produced the released data, written against the original
working tree rather than this layout, and several parts will not run as-is.

| Script | |
|---|---|
| `split_per_phoneme.py` | **Splits EEG into per-phoneme `.npy` arrays and cuts matching rtMRI video clips.** Written for this release; runs against this layout |
| `recover_raw_triggers.py` | Recovers lost stimulus triggers into the `.vmrk` marker files, leaving signal files byte-identical. Written for this release; already applied to the shipped data |
| `align_eeg_mri.py` | verbatim — produced the released epoch pickles (EEG + rtMRI ROI traces) |
| `recover_from_log.py` | verbatim — reconstructs stimulus triggers from the presentation logs |
| `map_eeg_mri.py` | verbatim — recording-to-scanner-video mapping, EEG/video drift QC |
| `compose_trigger57_topomap_video.py` | verbatim — figure script; source of the video timing logic |

## `split_per_phoneme.py`

No segmented files ship with this release — only this script, so you can generate them
with whatever window and preprocessing you want.

```sh
python split_per_phoneme.py --out-dir ./segments          # everything
python split_per_phoneme.py --out-dir ./segments --contexts in_scanner --no-video
python split_per_phoneme.py --out-dir ./segments --tmin -0.5 --tmax 1.5 --dry-run
```

Writes, under `--out-dir`:

```
eeg/<context>/<condition>/<recording>_<stimulus>.npy    float64 (n_channels, n_times)
video/<context>/<condition>/<recording>_<stimulus>.avi  in-scanner only
segments.csv     one row per segment: labels, timings, paths, shapes
channels.txt     channel order for every .npy
```

Defaults reproduce the released epoch pickles: production triggers (41-58), `tmin=-1.2`,
`tmax=+2.0`, baseline `(-0.2, 0)`, 0.1-30 Hz band-pass, mastoid reference, `standard_1020`
montage, EOG/EMG/ECG channels typed. At those defaults each EEG segment is
`(15, 801)` at 250 Hz and each clip is 317 frames at 99.01 fps — the same 3.204 s window.

Requires `mne`, `numpy`, `pandas`, and `ffmpeg` on `PATH` (only for video; `--no-video`
drops that dependency).

### How EEG and video are aligned

End-aligned, following `compose_trigger57_topomap_video.py`. The scanner writes a
`Volume` marker per acquired MRI volume and the video ends with the last volume, so a
trigger `delta` seconds before the final `Volume` marker sits `video_duration - delta`
into the video. Start-aligning instead would be wrong by up to ~100 ms by the end of a
run, because the EEG amplifier and scanner clocks drift apart.

### What was checked

- **EEG matches the project's own epoching.** At default settings all 216 in-scanner
  phonated segments reproduce the released `epoch_pkl` arrays to float64 noise
  (max relative difference 1.8e-16).
- **Video lands on the speech.** Across those same 216 trials the clip audio peaks
  0.997 +/- 0.159 s after the production trigger — 100% of trials after the trigger and
  within 1.5 s of it, which is where the produced syllable belongs.
- **Outside-scanner path.** Runs on `--stage raw` (the only stage released for that
  context) and produces no video, skipping conditions with no data for the requested
  stage instead of failing.

### Caveats

1. **The video runs ~8 frames (~0.08 s) longer than the mview ROI traces** the released
   pickles were cut against, so clips sit ~0.08 s later than those pickled ROI windows.
   Well inside the spread of production onsets, but it matters if you are aligning clips
   against the ROI traces sample-for-sample.
2. Clips are re-encoded (`mpeg4`, `-q:v 3`) for frame accuracy. `--video-codec copy` is
   much faster but cuts only on keyframes, so the window will be off.
3. `--stage raw` data is 5000 Hz and unreferenced, so its segments are `(15, 16001)` at
   the default window, not `(15, 801)`. Outside-scanner EEG is released at the raw stage
   only, so use `--stage raw` for it.
4. Both released stages are trigger-recovered, so segment counts match the protocol:
   18 per in-scanner run, 9 per outside-scanner run.
5. Trigger codes absent from a recording are skipped silently; windows that would run off
   the end of a recording are skipped with a printed notice.

## `align_eeg_mri.py` (verbatim)

The original per-phoneme splitter, which produced the released epoch pickles: it epochs
the EEG exactly as above and pairs each epoch with the matching window of the six rtMRI
articulator ROI traces (LA, TT, TB, VL, LX, TR), writing
`{recording}_{stimulus}.pkl` with `eeg`, `eeg_sr`, `roi`, `label`, `trigger_code`,
`filename`, `epoch_idx`.

**It cannot run as committed.** Line 101 calls `loadmat(...)` but nothing imports it;
there is no `scipy` import in the file, so it raises
`NameError: name 'loadmat' is not defined` at the rtMRI step. Both copies of this script
on the original drive are byte-identical and both carry the bug, so the released pickles
were produced from an interactive session where `from scipy.io import loadmat` had
already been run. It also needs the `mviews/` `.mat` files (~1.9 GB), which are not part
of this release, and expects `phoneme_class.txt`, `eeg_phonated/` and `mviews/` in the
working directory. `split_per_phoneme.py` covers the EEG side without any of that.

## `recover_from_log.py` (verbatim)

Matches each recording's surviving trigger sequence against the presentation logs, fits a
linear regression from log time to EEG time, and writes predicted onsets for the missing
triggers. This produced the recovered triggers in `eeg/denoised/`.

**Caveat:** it has a live `save_recovered_files(...)` call at module level, so importing
it executes that call and overwrites an output directory. Strip the usage calls at the
bottom before importing. It also needs the presentation logs from
`data/vol1395_timestamps/`, which are not part of this release.

## `map_eeg_mri.py` (verbatim)

Holds the authoritative mapping from each recording id to its raw scanner AVI, and reports
drift between EEG duration (first-to-last `Volume` trigger) and video duration. Provenance
for where each released video came from. The raw scanner AVIs are not in this release.

## `compose_trigger57_topomap_video.py` (verbatim)

A talk-figure script: cuts one hardcoded trigger's window from one recording's video and
composites it beside ERP topomaps, needing a precomputed grand-average `.fif`. Included
because its `parse_vmrk_positions` and end-aligned timing logic is where
`split_per_phoneme.py`'s video alignment comes from.

## `recover_raw_triggers.py`

Reconstructs stimulus triggers dropped by the acquisition system and writes them into the
BrainVision `.vmrk` marker files. **It has already been run on the data in this release**;
it ships so the procedure is inspectable and repeatable, not because you need to run it.

```sh
python recover_raw_triggers.py --dry-run     # report what it would add
python recover_raw_triggers.py               # apply
```

Per recording it reads the existing stimulus markers in position order, walks them against
that recording's own presentation log (the syllable order is randomised per run, so the
sequences align unambiguously), least-squares fits log time onto marker position, and
inserts a marker at the predicted position for every log event with no marker. Markers are
then renumbered and the file rewritten with its original CRLF line endings. It refuses to
write if the fit is worse than `--min-r2` (default 0.9999) or if a predicted position
falls outside the recording.

The important difference from `recover_from_log.py` is that this touches **only** the
marker file. `recover_from_log.py` re-exports the whole recording through MNE, whose
BrainVision writer emits IEEE_FLOAT_32 — fine for the already-float `denoised` stage, but
it would have converted the INT_16 raw recordings, doubled their size and rescaled their
values. Data labelled raw should stay exactly as recorded, so the `.eeg` and `.vhdr` files
are never opened for writing; this was verified by checksumming all 104 of them before and
after.

Applied to this release: 158 markers added across 44 of 52 recordings, every fit at
r^2 = 1.000000. See the release README for the validation results.

# Code

One script ships with this release: `split_per_stimulus.py`. It is licensed under the MIT
License (see `LICENSE` in this directory), separately and more permissively than the data.
Running it on the data does not exempt you from the data licence in `../LICENSE`.

## `split_per_stimulus.py`

Splits the release into per-stimulus segments: one `.npy` EEG array per VCV syllable, plus
the matching rtMRI video clip for in-scanner recordings. **No segmented files ship with
this release** — run this to generate them with whatever window and preprocessing you
want.

```sh
python split_per_stimulus.py --out-dir ./segments          # everything
python split_per_stimulus.py --out-dir ./segments --contexts in_scanner --no-video
python split_per_stimulus.py --out-dir ./segments --tmin -0.5 --tmax 1.5 --dry-run
python split_per_stimulus.py --out-dir ./segments --band 1 40 --reference none
```

Writes, under `--out-dir`:

```
eeg/<context>/<condition>/<recording>_<stimulus>.npy    float64 (n_channels, n_times)
video/<context>/<condition>/<recording>_<stimulus>.avi  in-scanner only
segments.csv     one row per segment: labels, timings, paths, shapes
channels.txt     channel order for every .npy
```

Defaults reproduce the epoching used in the paper: production triggers (41-58),
`tmin=-1.2`, `tmax=+2.0`, baseline `(-0.2, 0)`, 0.1-30 Hz band-pass, mastoid reference,
`standard_1020` montage, EOG/EMG/ECG channels typed. At those defaults each EEG segment is
`(15, 801)` at 250 Hz and each clip is 317 frames at 99.01 fps — the same 3.204 s window.

Requires `mne` and `numpy`, plus `ffmpeg` on `PATH` for video cutting (`--no-video`
drops that dependency).

Window and preprocessing are all flags: `--tmin/--tmax`, `--baseline START END`,
`--band LOW HIGH` (`0 0` to skip filtering), `--reference CH... | none`, `--stage`,
`--contexts`, `--conditions`, `--no-video`, `--dry-run`.

### How EEG and video are aligned

End-aligned. The scanner writes a `Volume` marker per acquired MRI volume and the video
ends with the last volume, so a trigger `delta` seconds before the final `Volume` marker
sits `video_duration - delta` into the video. Start-aligning instead would be wrong by up
to ~100 ms by the end of a run, because the EEG amplifier and scanner clocks drift apart.

### What was checked

- **EEG matches the project's own epoching.** At default settings all 216 in-scanner
  phonated segments reproduce the reference epochs to float64 noise (max relative
  difference 1.8e-16).
- **Video lands on the speech.** Across those same 216 trials the clip audio peaks
  0.997 +/- 0.159 s after the production trigger — 100% of trials after the trigger and
  within 1.5 s of it, which is where the produced syllable belongs.
- **Both contexts and both stages run.** In-scanner `--stage denoised` yields 468 segments
  and 450 clips; `--stage raw` yields 468 in-scanner and 234 outside-scanner segments,
  matching the protocol (18 syllables per in-scanner run, 9 per outside-scanner run).

### Caveats

1. **The video runs ~8 frames (~0.08 s) longer** than the rtMRI articulator ROI traces
   derived from the same acquisition, so clips sit ~0.08 s later than windows cut against
   those traces. Well inside the spread of production onsets, but it matters if you are
   aligning clips against ROI traces sample-for-sample.
2. Clips are re-encoded (`mpeg4`, `-q:v 3`) so the cut is frame-accurate. Stream-copying
   instead would be faster but cuts only on keyframes, putting the window off by up to a
   keyframe interval.
3. `--stage raw` data is 5000 Hz and unreferenced, so its segments are `(15, 16001)` at
   the default window, not `(15, 801)`. Outside-scanner EEG is released at the raw stage
   only, so use `--stage raw` for it.
4. Trigger codes absent from a recording are skipped silently; windows that would run off
   the end of a recording are skipped with a printed notice.

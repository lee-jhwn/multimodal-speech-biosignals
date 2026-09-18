# split_per_stimulus.py

Cuts the recordings into per-stimulus segments: one NumPy array per syllable, plus the
matching rtMRI video clip for in-scanner recordings. Nothing in the release is
pre-segmented, so you can choose your own window and preprocessing.

```sh
python split_per_stimulus.py --out-dir ./segments
python split_per_stimulus.py --out-dir ./segments --stage raw --contexts outside_scanner
python split_per_stimulus.py --out-dir ./segments --tmin -0.5 --tmax 1.5 --dry-run
```

Output:

```
eeg/<context>/<condition>/<recording>_<stimulus>.npy      float64 (n_channels, n_times)
video/<context>/<condition>/<recording>_<stimulus>.avi    in-scanner only
segments.csv    one row per segment, with labels and timings
channels.txt    channel order for the .npy files
```

Needs `mne` and `numpy`, plus `ffmpeg` on your PATH for the video (`--no-video` if you
don't want it).

## Options

Defaults reproduce the epoching used in the paper: production triggers, −1.2 to +2.0 s,
baseline (−0.2, 0), 0.1–30 Hz, mastoid reference. Each segment comes out `(15, 801)` at
250 Hz, and each clip 317 frames at 99 fps — the same 3.204 s window.

| | |
|---|---|
| `--tmin` `--tmax` | epoch window, s, relative to the trigger |
| `--baseline START END` | baseline window, s; `0 0` to skip |
| `--band LOW HIGH` | band-pass, Hz; `0 0` to skip |
| `--reference CH...` | reference channels, or `none` |
| `--stage` | `denoised` (default) or `raw` |
| `--contexts` `--conditions` | subset what gets processed |
| `--no-video` `--dry-run` | |

## How the video is aligned

End-aligned. The scanner writes a marker per acquired MRI volume and the video ends with
the last volume, so a trigger *d* seconds before the final volume marker sits
`video_duration − d` into the video. The amplifier and scanner clocks drift apart over a
run, so aligning from the start instead would be off by up to ~100 ms by the end.

## Notes

- Outside-scanner recordings have no MRI, so no video, and they're released at the raw
  stage only — use `--stage raw` for them. They also use a 9-syllable protocol, so you get
  9 segments per run instead of 18.
- `--stage raw` is 5000 Hz and unreferenced, so segments come out `(15, 16001)` at the
  default window.
- Clips are re-encoded rather than stream-copied, so the cut lands on the right frame.
- The video runs about 8 frames longer than the articulator traces derived from the same
  acquisition. It doesn't matter for normal use, but it does if you're lining clips up
  against those traces sample by sample.

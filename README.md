# Simultaneous real-time MRI, EEG and surface EMG during speech production

Official repository for:

> **An Approach to Simultaneous Acquisition of Real-Time MRI Video, EEG, and Surface EMG
> for Articulatory, Brain, and Muscle Activity During Speech Production**
> Interspeech 2026

This repository holds the **processing code**. The **dataset** is distributed separately
through a gated Hugging Face repository — see [Data access](#data-access).

## What is in the dataset

One participant, one session, 18 VCV syllables, recorded in two contexts.

| Context | Conditions | Runs | Syllables/run | Modalities |
|---|---|---:|---:|---|
| In scanner | phonated, silent, imagined | 12 / 12 / 2 | 18 | EEG (3 stages), rtMRI video, audio, logs |
| Outside scanner | phonated, silent, imagined | 12 / 12 / 2 | 9 | EEG (raw), audio, logs |

EEG is 15 channels — 9 EEG, 2 EOG, **3 surface EMG**, 1 ECG — in BrainVision format.
rtMRI video is 104x104 at 99.01 fps with a 16 kHz audio track. In-scanner EEG ships at
three processing stages: raw, MR-gradient/pulse-artifact corrected, and CCA-cleaned.

## Data access

The dataset is **gated**. You will be asked to accept a responsible-use licence before
download, because the data is single-subject and contains the participant's voice and
real-time MRI of their head and vocal tract, and so is not anonymous.

```
https://huggingface.co/datasets/<ORG_OR_USER>/multimodal-speech-biosignals
```

> **TODO before publication:** create the Hugging Face dataset repository, enable gating,
> and replace the URL above.

## Code

```
code/
├── split_per_phoneme.py                 split EEG into per-phoneme .npy + cut rtMRI clips
├── align_eeg_mri.py                     (verbatim) produced the released epoch pickles
├── recover_from_log.py                  (verbatim) reconstructs lost stimulus triggers
├── map_eeg_mri.py                       (verbatim) recording↔scanner-video map, drift QC
├── compose_trigger57_topomap_video.py   (verbatim) figure script; source of video timing
└── README.md                            what each script does, and what will not run as-is
```

`split_per_phoneme.py` is the one to run. The four scripts marked *verbatim* are the
original analysis scripts, copied unmodified for provenance; two of them have known traps
(a missing import, and a module-level call that overwrites a directory). All of this is
documented in [`code/README.md`](code/README.md) — read it before running anything.

### Quick start

```sh
pip install mne numpy pandas          # plus ffmpeg on PATH for video cutting
# from inside the downloaded dataset directory:
python code/split_per_phoneme.py --out-dir ./segments
```

This writes one `.npy` EEG array and one rtMRI video clip per phoneme, plus a
`segments.csv` of labels and timings. Defaults reproduce the epoching used in the paper:
production triggers, -1.2 to +2.0 s, baseline (-0.2, 0), 0.1-30 Hz, mastoid reference.

Add `--dry-run` to see what would be produced, `--no-video` to skip ffmpeg, and
`--stage raw` for the outside-scanner recordings (released at the raw stage only).

## Licence

Code and data are licensed **differently**:

- **Code** in this repository — MIT, see [`LICENSE`](LICENSE).
- **Data** — Responsible Use Data License, distributed with the dataset. It imposes
  use-based restrictions, including no re-identification, no speaker-identification or
  voice-biometric use, no voice or likeness synthesis, and no commercial use without
  permission. Models trained on the data are derivative works and carry the same
  restrictions.

Running this MIT-licensed code on the data does not exempt you from the data licence.

## Citation

```bibtex
@inproceedings{lee2026multimodal,
  title     = {An Approach to Simultaneous Acquisition of Real-Time MRI Video, EEG,
               and Surface EMG for Articulatory, Brain, and Muscle Activity During
               Speech Production},
  author    = {The authors},
  booktitle = {Interspeech 2026},
  year      = {2026}
}
```

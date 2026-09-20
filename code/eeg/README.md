# EEG/EMG preprocessing

Artifact removal for the in-scanner EEG, and the evaluation reported in the paper.

## Processing

**1. MR artifacts (BrainVision Analyzer 2.3).**
Gradient and pulse (BCG) artifacts were removed in BrainVision Analyzer 2.3 using the standard template subtraction methods from EEG-fMRI.

- Gradient artifact: average artifact subtraction (Allen et al., 2000). Artifact onsets were detected from the EEG with a gradient threshold, and a sliding-window template was subtracted.
- Pulse artifact: ECG-based template subtraction (Allen et al., 1998). R peaks were found on the ECG channel after a 15 Hz low-pass and verified semi-automatically.

Analyzer's default settings were used, adapted to the rtMRI sequence (TR 5.05 ms). The output of this step is released in the dataset under `eeg/mr_corrected/`, so the rest of the pipeline runs without Analyzer.

**2. EMG/EOG artifacts (`cca_denoise.py`).**

- Each run is resampled to 250 Hz, notch filtered at 60 Hz and band-passed 0.1–45 Hz.
- CCA is fit between the 9 EEG channels and 5 references (EMG1–3, EOG1–2).
- Components with canonical correlation above 0.4 were candidates. The final choice was made by looking at their time courses (`--inspect`).
- The components removed in each run are listed in `cca_components.json`, and the selected components are regressed out of the EEG.
- The output is `eeg/denoised/` in the dataset.

## Setup

```sh
cd code/eeg
pip install -r requirements.txt huggingface_hub   # Python 3.12
hf auth login                  # accept the dataset terms on Hugging Face first
hf download lee-jhwn/multimodal-speech-biosignals --repo-type dataset \
    --local-dir ~/mmsb --include "eeg/*" "phoneme_class.txt"
```

## Running

Run from `code/eeg/`:

```sh
D=~/mmsb/eeg

# EMG/EOG removal; reproduces eeg/denoised/ and checks it against the released files
python cca_denoise.py $D/mr_corrected/in_scanner/phonated out/cca \
    --verify $D/denoised/in_scanner/phonated

# evaluation
python evaluate_mr_correction.py $D/raw/in_scanner/phonated \
    $D/mr_corrected/in_scanner/phonated $D/raw/outside_scanner/phonated
python evaluate_cca.py $D/mr_corrected/in_scanner/phonated $D/denoised/in_scanner/phonated
```

| Script | Output | Paper |
|---|---|---|
| `cca_denoise.py` | `out/cca/`: cleaned runs, `cca_log.csv` | Sec. 2.4 |
| `evaluate_mr_correction.py` | `out/mr_correction/`: ERPs and spectra, `erp_correlation.csv` | Sec. 3.3, Fig. 4 |
| `evaluate_cca.py` | `out/cca_evaluation/`: ERPs and scalp maps, `peak_amplitude.json` | Sec. 3.4, Fig. 5 |

Each script runs in a few minutes on a laptop.

## Notes

- Trigger codes: 21–38 mark word onset, 41–58 the go cue. phoneme_class.txt maps each go-cue code (produce_idx) to its syllable and phonetic class.
- `evaluate_mr_correction.py` uses epochs from −1 to 1 s around word onset, baseline −1 to 0 s, a 0.1 Hz high-pass and the recording reference. Its spectra are computed from the grand-average ERP.
- `evaluate_cca.py` uses epochs from −1.2 to 2 s around the go cue, baseline −0.2 to 0 s, a 0.1–30 Hz band-pass and the linked-mastoid reference.
- `cca_denoise.py --auto` applies the 0.4 threshold without the manual choices.

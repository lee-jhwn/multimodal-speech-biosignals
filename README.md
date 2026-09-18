# An Approach to Simultaneous Acquisition of Real-Time MRI Video, EEG, and Surface EMG for Articulatory, Brain, and Muscle Activity During Speech Production (Interspeech 2026)

Official repository for our Interspeech 2026 paper ([arXiv](https://arxiv.org/abs/2603.04840)).

## Data

The dataset is on the Hugging Face Hub:

https://huggingface.co/datasets/lee-jhwn/multimodal-speech-biosignals

It covers 18 VCV stimuli produced three ways — spoken, silently articulated, and
imagined — recorded both inside the MRI scanner (with simultaneous rtMRI, EEG, EMG, and audio)
and outside it as an EEG reference. EEG is 15 channels including three surface EMG.

## Code

`code/split_per_stimulus.py` cuts the recordings into per-stimulus segments: one NumPy
array per syllable, plus the matching rtMRI video clip.

```sh
pip install mne numpy       # ffmpeg also needed for the video
python code/split_per_stimulus.py --out-dir ./segments
```

Run it from inside the downloaded dataset directory. Defaults match the paper. See
[`code/README.md`](code/README.md) for the options.

## License

The code here is MIT (`LICENSE`). The data has its own, more restrictive license, which
ships with the dataset.

## Citation

```bibtex
@article{lee2026approach,
  title={An Approach to Simultaneous Acquisition of Real-Time MRI Video, EEG, and Surface EMG for Articulatory, Brain, and Muscle Activity During Speech Production},
  author={Lee, Jihwan and Razmara, Parsa and Huang, Kevin and Foley, Sean and Kommineni, Aditya and Hsu, Haley and Jeong, Woojae and Kumar, Prakash and Shi, Xuan and Lee, Yoonjeong and Feng, Tiantian and Medani, Takfarinas and Tian, Ye and Kadiri, Sudarsana Reddy and Nayak, Krishna S. and Byrd, Dani and Goldstein, Louis and Leahy, Richard M. and Narayanan, Shrikanth},
  journal={Interspeech 2026},
  year={2026}
}
```

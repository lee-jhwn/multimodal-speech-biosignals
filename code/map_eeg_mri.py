import mne
import os
import glob
import cv2
import numpy as np
import warnings

# --- 1. IGNORE WARNINGS ---
warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

# --- CONFIGURATION ---
EEG_DIR = '/Volumes/T9/lambda43_backup/jihwan/projects/2601_eegmri/data/251121_speech_eeg_emg_rtmri'
AVI_DIR = '/Volumes/T9/lambda43_backup/jihwan/projects/2601_eegmri/data/vol1395_20251121/video/stcr/video_with_audio'

file_mapping = {
    # '251121_0_soundcheck': ['usc_disc_20251121_095350_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes1381_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    # '251121_1_scanneron_rest_fix': ['usc_disc_20251121_095608_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes1353_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_1_fixfixfix': ['usc_disc_20251121_101605_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3323_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_2': ['usc_disc_20251121_101811_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3219_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_3': ['usc_disc_20251121_102000_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3355_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_4': ['usc_disc_20251121_102154_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3368_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_5': ['usc_disc_20251121_102418_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3466_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_6': ['usc_disc_20251121_102747_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3459_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_7': ['usc_disc_20251121_102939_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3440_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_8': ['usc_disc_20251121_103132_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3544_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_9': ['usc_disc_20251121_103352_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3433_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_10': ['usc_disc_20251121_103606_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3492_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_11': ['usc_disc_20251121_103759_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3349_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_2_scanneron_phonated_12': ['usc_disc_20251121_103951_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3531_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_1_fix': ['usc_disc_20251121_105117_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3817_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_2': ['usc_disc_20251121_105315_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3440_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_3': ['usc_disc_20251121_105511_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3381_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_4': ['usc_disc_20251121_105658_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3446_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_5': ['usc_disc_20251121_105849_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3446_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_6': ['usc_disc_20251121_110038_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3505_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_7': ['usc_disc_20251121_110327_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3381_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_8': ['usc_disc_20251121_110542_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3531_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_9': ['usc_disc_20251121_110735_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3583_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_10': ['usc_disc_20251121_110928_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3511_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_11': ['usc_disc_20251121_111128_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3485_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_3_scanneron_slient_12': ['usc_disc_20251121_111318_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3485_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_4_scanneron_imagine_1': ['usc_disc_20251121_114153_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3511_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi'],
    '251121_4_scanneron_imagine_2': ['usc_disc_20251121_114835_pk_speech_rt_ssfp_fov24_res24_n13_vieworder_bitr_recon_narms02_13_sl06mm_nframes3648_lt0.080_ls0.000_lw0.000_FOV240mm_sRes2.31_tRes10.10_GIRF_acq_delay-6.0_demcor_TRTrim0150.avi']
}

def get_video_duration_metadata(avi_path):
    """Opens the AVI and reads the header FPS and Frame Count directly."""
    if not os.path.exists(avi_path):
        return 0, 0, 0
    
    cap = cv2.VideoCapture(avi_path)
    if not cap.isOpened():
        return 0, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0
    
    return duration, frame_count, fps

def calculate_stats(eeg_dir, mapping):
    print(f"{'Key (Short)':<30} | {'EEG Dur':<9} | {'Vid Dur':<9} | {'Diff(s)':<8}")
    print("-" * 65)

    all_diffs = []
    all_diffs_per_sec = []

    for eeg_name, avi_list in mapping.items():
        avi_filename = avi_list[0]
        avi_full_path = os.path.join(AVI_DIR, avi_filename)
        
        # 1. Video Duration
        vid_dur, vid_frames, vid_fps = get_video_duration_metadata(avi_full_path)
        
        # 2. EEG Duration
        vhdr_pattern = os.path.join(eeg_dir, f"{eeg_name}.vhdr")
        eeg_files = glob.glob(vhdr_pattern)
        
        if eeg_files and vid_dur > 0:
            try:
                raw = mne.io.read_raw_brainvision(eeg_files[0], verbose=False)
                events, event_id = mne.events_from_annotations(raw, verbose=False)
                fs = raw.info['sfreq']
                id_to_desc = {v: k for k, v in event_id.items()}
                
                # Filter for Volume Triggers
                vol_samples = [s for s, _, eid in events if "Volume/V  1" in id_to_desc[eid]]
                
                if len(vol_samples) > 1:
                    first_samp = vol_samples[0]
                    last_samp = vol_samples[-1]
                    
                    # Duration from First to Last Trigger
                    eeg_dur_sec = (last_samp - first_samp) / fs
                    
                    # Difference
                    diff = eeg_dur_sec - vid_dur
                    diff_per_sec = diff / eeg_dur_sec
                    all_diffs.append(diff)
                    all_diffs_per_sec.append(diff_per_sec)
                    
                    print(f"{eeg_name[:30]:<30} | {eeg_dur_sec:<9.3f} | {vid_dur:<9.3f} | {diff:<8.3f} | {diff_per_sec}")
                else:
                    print(f"{eeg_name[:30]:<30} | {'No Trigs':<9} | {vid_dur:<9.3f} | {'-':<8}")

            except Exception as e:
                print(f"Error {eeg_name}: {e}")
        else:
            print(f"Missing File: {eeg_name}")

    print("-" * 65)
    if all_diffs:
        avg_diff = np.mean(all_diffs)
        std_diff = np.std(all_diffs)
        print(f"Summary Statistics:")
        print(f"  Count:              {len(all_diffs)}")
        print(f"  Average Difference: {avg_diff:.4f} s")
        print(f"  Standard Deviation: {std_diff:.4f} s")
    else:
        print("No valid comparisons found.")

    if all_diffs_per_sec:
        avg_diff = np.mean(all_diffs_per_sec)
        std_diff = np.std(all_diffs_per_sec)
        print(f"Summary Statistics:")
        print(f"  Count:              {len(all_diffs_per_sec)}")
        print(f"  Average Difference: {avg_diff:.4f} s")
        print(f"  Standard Deviation: {std_diff:.4f} s")
    else:
        print("No valid comparisons found.")

calculate_stats(EEG_DIR, file_mapping)
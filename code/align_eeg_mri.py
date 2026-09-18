import pickle
import pandas as pd
import mne
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# Load phoneme class mapping
phoneme_df = pd.read_csv('phoneme_class.txt', skipinitialspace=True)
produce_idx_to_label = {}
for _, row in phoneme_df.iterrows():
    produce_idx_to_label[int(row['produce_idx'])] = {
        'stimuli': row['stimuli'],
        'voicing': row['voicing'],
        'place': row['place'],
        'manner': row['manner'],
    }

# Create output directory
pkl_output_dir = 'epoch_pkl'
if not os.path.exists(pkl_output_dir):
    os.makedirs(pkl_output_dir)

eeg_dir = 'eeg_phonated'
art_dir = 'mviews'

target_triggers = list(range(41, 59))
tmin, tmax = -1.2, 2.0
baseline = (-0.2, 0.0)

epoch_count = 0

for file in sorted(os.listdir(eeg_dir)):
    if not file.endswith('.vhdr'):
        continue
    
    filename = file[:-17]  # Remove .vhdr extension
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")
    
    # Load raw EEG data
    raw = mne.io.read_raw_brainvision(os.path.join(eeg_dir, file), preload=True, verbose=False)
    
    # Set channel types
    channel_mapping = {}
    for ch in raw.ch_names:
        ch_upper = ch.upper()
        if 'EOG' in ch_upper:
            channel_mapping[ch] = 'eog'
        elif 'EMG' in ch_upper:
            channel_mapping[ch] = 'emg'
        elif 'ECG' in ch_upper or 'EKG' in ch_upper:
            channel_mapping[ch] = 'ecg'
    
    if channel_mapping:
        raw.set_channel_types(channel_mapping)
    
    # Set montage
    try:
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"  Montage warning: {e}")
    
    # Filter
    raw.filter(0.1, 30.0, verbose=False)
    
    # Get events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # Map target triggers
    found_ids = {}
    id_to_desc = {v: k for k, v in event_id.items()}
    for code in target_triggers:
        for eid_code, eid_desc in id_to_desc.items():
            if f"S {code}" in eid_desc or f"S{code}" in eid_desc:
                found_ids[eid_desc] = eid_code
    
    if not found_ids:
        print(f"  Skipping: No production triggers (41-58) found.")
        continue
    
    # Set reference
    raw.set_eeg_reference(ref_channels=['M1', 'M2'])
    
    # Create epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id=found_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=False
    )
    
    # Load artifact data
    art_mat = loadmat(os.path.join(art_dir, f'{filename}_mview.mat'))['data']
    art_data_dict = {}
    for i in range(2, 8):
        d = art_mat[0, i]
        label = d[0] if isinstance(d[0], str) else d[0].item() if hasattr(d[0], 'item') else str(d[0])
        sr = float(d[1].item()) if hasattr(d[1], 'item') else float(d[1])
        y = d[2].flatten()
        art_data_dict[label] = {'y': y, 'sr': sr}
    
    # Get EEG signal parameters
    eeg_sr = raw.info['sfreq']
    eeg_total_duration = raw.times[-1] - raw.times[0]
    
    # Get total recording duration from artifact
    first_art_label = list(art_data_dict.keys())[0]
    art_total_samples = art_data_dict[first_art_label]['y'].shape[0]
    art_sr = art_data_dict[first_art_label]['sr']
    art_total_duration = art_total_samples / art_sr
    
    start_time = events[events[:, 2] == 10003, 0][0] / eeg_sr
    end_time = events[events[:, 2] == 10008, 0][-1] / eeg_sr
    offset = art_total_duration - end_time
    print(f"  EEG times: {end_time:.2f} - {start_time:.2f} = {end_time-start_time:.2f}, mri = {art_total_duration:.2f}")
    
    # Process each epoch
    event_ids_reversed = {v: k for k, v in found_ids.items()}
    
    for epoch_idx in range(len(epochs)):
        epoch = epochs[epoch_idx]
        epoch_label_desc = epoch.event_id
        trigger_code = list(epoch_label_desc.values())[0]
        
        # Get epoch timing
        ep_time = events[events[:, 2] == trigger_code, 0][-1] / eeg_sr
        
        # Store as dictionary mapping channel names to signal arrays
        eeg_data = epoch.get_data()[0]  # shape: (n_channels, n_times)
        eeg_signal = {ch_name: eeg_data[ch_idx] for ch_idx, ch_name in enumerate(epochs.ch_names)}
        
        # Extract corresponding artifact signals
        art_window = {}
        for art_label, art_info in art_data_dict.items():
            art_signal = art_info['y']
            art_sr_val = art_info['sr']
            
            # Convert epoch time window to artifact sample indices
            offset_tmin = ep_time + epoch.times[0] + offset
            offset_tmax = ep_time + epoch.times[-1] + offset
            
            art_idx_start = max(0, int(offset_tmin * art_sr_val))
            art_idx_end = min(len(art_signal), int(offset_tmax * art_sr_val))
            
            art_window[art_label] = {
                'y': art_signal[art_idx_start:art_idx_end],
                'sr': art_sr_val
            }
        label_data = produce_idx_to_label[trigger_code]
        
        # Create epoch dictionary
        epoch_dict = {
            'eeg': eeg_signal,
            'eeg_sr': eeg_sr,
            'roi': art_window,
            'label': label_data,
            'trigger_code': trigger_code,
            'filename': filename,
            'epoch_idx': epoch_idx,
        }
        
        pkl_filename = f"{filename}_{label_data['stimuli']}.pkl"
        pkl_path = os.path.join(pkl_output_dir, pkl_filename)
        with open(pkl_path, 'wb') as f:
            pickle.dump(epoch_dict, f)
        
        #print(f"  Epoch {epoch_count}: {label_data['stimuli']} (trigger={trigger_code}) -> {pkl_filename}")
        with open(pkl_path, 'wb') as f:
            pickle.dump(epoch_dict, f)
        
        print(f"  Epoch {epoch_count}: {label_data['stimuli']} (trigger={trigger_code}) -> {pkl_filename}")
        epoch_count += 1
    
    #break

print(f"\n{'='*60}")
print(f"Total epochs saved: {epoch_count}")
print(f"Output directory: {pkl_output_dir}")
print(f"{'='*60}")
import mne
import pandas as pd
import numpy as np
import os
import glob
from sklearn.linear_model import LinearRegression
import re

# --- 1. LOG PARSER (Keep as is) ---
def parse_log_file(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(',')]
                if len(parts) == 2:
                    try:
                        t = float(parts[0])
                        c = int(parts[1])
                        data.append({'relative_time': t, 'trigger_code': c})
                    except ValueError:
                        continue 
        return pd.DataFrame(data)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

# --- 2. MATCHING HELPER (Keep as is) ---
def find_best_log_match(eeg_codes, log_library):
    best_file = None
    best_score = -1
    best_log_df = None
    
    if not eeg_codes:
        return None, None, 0

    for log_name, log_df in log_library.items():
        log_seq = log_df['trigger_code'].tolist()
        eeg_idx = 0
        matches = 0
        for log_code in log_seq:
            if eeg_idx < len(eeg_codes) and log_code == eeg_codes[eeg_idx]:
                matches += 1
                eeg_idx += 1
        
        score = matches / len(eeg_codes) if len(eeg_codes) > 0 else 0
        if score > best_score:
            best_score = score
            best_file = log_name
            best_log_df = log_df
            
    return best_file, best_log_df, best_score

# --- 3. MAIN FUNCTION (Updated) ---
def save_recovered_files(eeg_dir, log_dir, output_dir, eeg_pattern='*.vhdr', log_pattern='*.txt'):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Logs
    log_files = glob.glob(os.path.join(log_dir, log_pattern))
    log_library = {} 
    print(f"Loading {len(log_files)} log files...")
    for lf in log_files:
        df = parse_log_file(lf)
        if not df.empty:
            log_library[os.path.basename(lf)] = df

    # Process EEG
    eeg_files = glob.glob(os.path.join(eeg_dir, eeg_pattern))
    print(f"Processing {len(eeg_files)} EEG files...\n")

    eeg_files = sorted(eeg_files, key=lambda x: os.path.basename(x))  # Sort for consistent order
    
    for file_path in eeg_files:
        filename = os.path.basename(file_path)
        if filename.startswith('.'): continue
        
        try:
            # 1. Read Raw Data
            raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
            sfreq = raw.info['sfreq']
            
            # Extract events for alignment
            events, event_id = mne.events_from_annotations(raw, verbose=False)
            id_to_desc = {v: k for k, v in event_id.items()}
            
            eeg_codes = []
            eeg_events_full = []
            
            for ev in events:
                desc = id_to_desc[ev[2]]
                match = re.search(r'S\s*(\d+)', desc)
                if match:
                    code = int(match.group(1))
                    eeg_codes.append(code)
                    eeg_events_full.append({'code': code, 'eeg_time': ev[0] / sfreq})
            
            # 2. Match
            best_log_name, best_log_df, score = find_best_log_match(eeg_codes, log_library)
            if score < 0.8:
                print(f"Skipping {filename}: Low match score ({score:.2f})")
                continue

            # 3. Align
            eeg_df_full = pd.DataFrame(eeg_events_full)
            matches = []
            eeg_idx = 0
            for log_idx, log_row in best_log_df.iterrows():
                if eeg_idx < len(eeg_df_full) and log_row['trigger_code'] == eeg_df_full.iloc[eeg_idx]['code']:
                    matches.append({
                        'log_idx': log_idx,
                        'log_time': log_row['relative_time'],
                        'eeg_time': eeg_df_full.iloc[eeg_idx]['eeg_time']
                    })
                    eeg_idx += 1
            
            if len(matches) < 2:
                print(f"  -> {filename}: Alignment failed.")
                continue
                
            match_df = pd.DataFrame(matches)
            reg = LinearRegression().fit(match_df[['log_time']], match_df['eeg_time'])
            slope = reg.coef_[0]
            intercept = reg.intercept_
            r2 = reg.score(match_df[['log_time']], match_df['eeg_time'])

            print(f"✔ {filename} aligned (R2={r2:.4f})")

            # 4. CREATE COMBINED ANNOTATIONS (The Fix)
            # Instead of adding objects (orig + new), we merge lists.
            
            # Start with existing annotations
            final_onsets = list(raw.annotations.onset)
            final_durations = list(raw.annotations.duration)
            final_descriptions = list(raw.annotations.description)
            
            matched_indices = set(match_df['log_idx'])
            recovered_count = 0
            
            for log_idx, log_row in best_log_df.iterrows():
                if log_idx not in matched_indices:
                    # Predict time
                    pred_time = (slope * log_row['relative_time']) + intercept
                    
                    # Append to lists
                    final_onsets.append(pred_time)
                    final_durations.append(0.0)
                    final_descriptions.append(f"Stimulus/S {int(log_row['trigger_code'])}")
                    
                    recovered_count += 1
            
            if recovered_count > 0:
                print(f"  -> Recovered {recovered_count} triggers. Saving...")
                
                # Create ONE SINGLE object
                # We use raw.annotations.orig_time to ensure the reference date is kept
                new_annots = mne.Annotations(
                    onset=final_onsets,
                    duration=final_durations,
                    description=final_descriptions,
                    orig_time=raw.annotations.orig_time
                )
                
                raw.set_annotations(new_annots)
                
                save_path = os.path.join(output_dir, filename)
                mne.export.export_raw(save_path, raw, fmt='brainvision', overwrite=True)
                print(f"  -> SAVED: {save_path}")
            else:
                print("  -> No missing triggers. Copying file.")
                save_path = os.path.join(output_dir, filename)
                mne.export.export_raw(save_path, raw, fmt='brainvision', overwrite=True)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# --- USAGE ---

# --- USAGE ---
# 1. Folder with original broken files
# 2. Folder with log text files
# 3. New folder where you want the fixed files
# save_recovered_files('./x_cleaned_noscanner_phonated', './vol1395_timestamps', './recovered_triggers_noscanner_phonated')
# save_recovered_files('./x_cleaned_silent', './vol1395_timestamps', './recovered_triggers_silent')
# save_recovered_files('./x_cleaned_4components_noscanner_phonated', './vol1395_timestamps', './recovered_triggers_noscanner_phonated_4c')
# save_recovered_files('./x_cleaned_2components_noscanner_phonated', './vol1395_timestamps', './recovered_triggers_noscanner_phonated_2c')
# save_recovered_files('./noscanner_beforeEMG', './vol1395_timestamps', './recovered_triggers_noscanner_beforeEMG')
save_recovered_files('./phonated_denoised_beforeEMGEOGremoval_insidescanner', './vol1395_timestamps', './recovered_triggers_phonated_denoised_beforeEMGEOGremoval_insidescanner')
# save_recovered_files('./linear_regression_noscanner_phonated_cleaned', './vol1395_timestamps', './recovered_triggers_linear_regression_noscanner_phonated_cleaned')
# save_recovered_files('./x_cleaned_silent_noscanner', './vol1395_timestamps', './recovered_triggers_x_cleaned_silent_noscanner')

# --- USAGE ---
# Point to the folder with .vhdr files, and the folder with .txt files
# auto_recover_triggers('./x_cleaned', './vol1395_timestamps')
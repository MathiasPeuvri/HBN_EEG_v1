import pickle
from pathlib import Path
import pandas as pd
import numpy as np

def convert_maximv2_format_with_window_augmentation(maxime_df, sampling_rate=100):
    """
    obj: créer 4 fenêtres de 200 sp avec sample_idx comme nouvelle target 
    (idx du moment de la fenêtre correspondant au reaction time)
    !! need to keep window_start_sample to find back the original reaction time
    """
    rows = []
    window_size = 200

    for _, row_maxime in maxime_df.iterrows():
        for i in range(len(row_maxime['winwdows'])):
            augmentations = [
                ('challenge1_original', np.nan),
                ('early', np.random.randint(20, 70)),
                ('middle', np.random.randint(71, 130)),
                ('late', np.random.randint(131, 185))]
            
            full_sig = row_maxime['winwdows'][i]
            original_reaction_time = row_maxime['vals'][i][0]
            reaction_idx_global = 200
            trigger_idx = reaction_idx_global - int(original_reaction_time * sampling_rate)

            for aug_type, target_reaction_idx in augmentations:
                if aug_type == "challenge1_original":
                    if not (0.5 < original_reaction_time < 2.5):
                        continue
                    window_start_sample = 50
                    start_idx = trigger_idx + window_start_sample
                    rt_idx = int(original_reaction_time * sampling_rate) - window_start_sample
                else:
                    rt_idx = target_reaction_idx
                    start_idx = reaction_idx_global - rt_idx
                    window_start_sample = start_idx - trigger_idx

                stop_idx = start_idx + window_size
                new_signal = full_sig[:, start_idx:stop_idx]

                rows.append({'signal': new_signal, 'rt_idx': rt_idx, 'window_start_sample': window_start_sample})

    return pd.DataFrame(rows)

# # original challenge 1 format for training but similar to real challenge task

# def convert_maximv1_format(df):
#     """Convert eval format (winwdows/vals) to standard format (signal/response_time)."""
#     if 'signal' in df.columns or 'winwdows' not in df.columns:
#         return df

#     rows = [{'signal': row['winwdows'][i], 'response_time': row['vals'][i, 0],
#              'p_factor': row['vals'][i, 1], 'attention': row['vals'][i, 2],
#              'internalizing': row['vals'][i, 3], 'externalizing': row['vals'][i, 4],
#              'subject': row['subject']}
#             for _, row in df.iterrows() for i in range(len(row['winwdows']))]

#     return pd.DataFrame(rows)

# pkl_file = Path("datasets/eval/R1.pkl")
# with open(pkl_file, "rb") as f: data = pickle.load(f)
# data = pd.DataFrame(data)

# print(f"Original: {data.shape}, columns: {data.columns.tolist()}")
# print(f"winwdows[0] shape: {data['winwdows'].iloc[0].shape}, vals[0] shape: {data['vals'].iloc[0].shape}")

# converted = convert_maximv1_format(data)
# print(f"\nConverted: {converted.shape}, columns: {converted.columns.tolist()}")
# print(f"signal[0] shape: {converted['signal'].iloc[0].shape}, response_time sample: {converted['response_time'].head(3).tolist()}")



# print(f"\n test implementing new approach for chall 1\n \n")
# # New chall 1 taks : reaction_time -> samplepoint_idx
# pkl_file = Path("datasets/eval/R1_clickcentered.pkl")
# with open(pkl_file, "rb") as f: data = pickle.load(f)
# data = pd.DataFrame(data)

# print(f"Original: {data.shape}, columns: {data.columns.tolist()}")
# # Original: (293, 3), columns: ['winwdows', 'vals', 'subject']
# #-> windows = dict avec 293 entrées (chaque sujet/run unique); chaque entrée contient un nombre de windows Nwindows de 129 channels x 400 samplepoints
# print(f"winwdows[0] shape: {data['winwdows'].iloc[0].shape}, vals[0] shape: {data['vals'].iloc[0].shape} (ex: {data['vals'].iloc[0]})")
# # winwdows[0] shape: (22, 129, 400), vals[0] shape: (22, 5)
# # vals[0] shape: (22, 5) -> 22 windows x 5 features, feature 0 = reaction_time

# def convert_maximv2_format_with_window_augmentation_v0(maxime_df, sampling_rate = 100):
#     """
#     obj: créer 4 fenêtres de 200 sp avec sample_idx comme nouvelle target 
#     (idx du moment de la fenêtre correspondant au reaction time)
#     !! need to keep window_start_time to find back the original reaction time
#     """

#     rows = []
#     window_duration = 2.0 # sec
#     window_size = int(window_duration * sampling_rate)

#     # Définir les 4 décalages pour créer les sous-fenêtres
#     augmentations = [
#         ('challenge1_original', np.nan),  # window based according to trigger time; 0.5s - 2.5s after trigger
#         ('early', np.random.randint(20, 71)),   # start_idx = reaction_global - random[20,70]
#         ('middle', np.random.randint(70, 131)),  # start_idx = reaction_global - random[70,130]
#         ('late', np.random.randint(130, 181))    # start_idx = reaction_global - random[130,180]
#         ]

#     for _, row_maxime in maxime_df.iterrows():
#         for i in range(len(row_maxime['winwdows'])):
#             full_sig = row_maxime['winwdows'][i]
#             original_reaction_time = row_maxime['vals'][i][0] # value in seconds
#             reaction_idx_global = 200 # we build the windows to be centered on the reaction time

#             # trigger_time_relative_to_large_window = int(original_reaction_time * sampling_rate) -reaction_idx_global # in samplepoints
#             trigger_time_relative_to_large_window = reaction_idx_global - int(original_reaction_time * sampling_rate)

#             # print(f"trigger_time_relative_to_large_window: {trigger_time_relative_to_large_window}, original_reaction_time: {original_reaction_time}")
#             #fullsig_relative_rt = 
#             for aug_type, target_reaction_idx in augmentations:
#                 if aug_type == "challenge1_original":
#                     # confirm the original rt is > 0.5 and < 2.5sec
#                     if original_reaction_time > 0.5 and original_reaction_time < 2.5:
#                         #print(f"Challenge 1 original window test")
#                         # compute the new window
#                         window_start_sample = 50 # window start sample relatif au trigger time
#                         start_idx = trigger_time_relative_to_large_window + window_start_sample 
#                         stop_idx = start_idx + window_size
#                         new_signal = full_sig[:, start_idx:stop_idx]
#                         rt_idx = int(original_reaction_time * sampling_rate) - window_start_sample 
#                         trig_to_start = trigger_time_relative_to_large_window 
#                         # reconstructed_rt =  (window_start_sample + rt_idx) / sampling_rate
#                         # print(f"start_idx: {start_idx}, stop_idx: {stop_idx}, window_start_sample: {window_start_sample}")
#                         # # check reaction time stuffs
#                         # print(f"original_reaction_time: {original_reaction_time}, reconstructed reaction time: {reconstructed_rt}")
#                         # print(f" rt_in_window = {rt_idx}")

#                         # do the row append
#                         rows.append({
#                             'signal': new_signal,
#                             'rt_idx': rt_idx,
#                             'window_start_sample': window_start_sample
#                         })
                        
#                     else: # skip this window
#                         continue 
#                 else:
#                     # print(f"augmentation type: {aug_type}, target_reaction_idx: {target_reaction_idx}")
#                     rt_idx = target_reaction_idx
#                     start_idx = reaction_idx_global - rt_idx 
#                     stop_idx = start_idx + window_size
#                     new_signal = full_sig[:, start_idx:stop_idx]
#                     window_start_sample = (start_idx - trigger_time_relative_to_large_window)
#                     # reconstructed_rt = (window_start_sample + rt_idx ) / sampling_rate
#                     trig_to_start = 200- trigger_time_relative_to_large_window 
#                     # print(f"start_idx: {start_idx}, stop_idx: {stop_idx}, window_start_sample: {window_start_sample}")
#                     # # check reaction time stuffs
#                     # print(f"original_reaction_time: {original_reaction_time}, reconstructed reaction time: {reconstructed_rt}")
#                     # print(f" rt_in_window = {rt_idx}")      
#                     # # do the row append
#                     rows.append({
#                         'signal': new_signal,
#                         'rt_idx': rt_idx,
#                         'window_start_sample': window_start_sample
#                     })
#     converted = pd.DataFrame(rows)
#     return converted
    
# maximv2_converted = convert_maximv2_format_with_window_augmentation_v0(data)
# print(f"\nConverted: {maximv2_converted.shape}, columns: {maximv2_converted.columns.tolist()}")
# print(maximv2_converted.head())




# maximv2_converted = convert_maximv2_format_with_window_augmentation(data)
# print(f"\nConverted: {maximv2_converted.shape}, columns: {maximv2_converted.columns.tolist()}")
# print(maximv2_converted.head(8))

# converted = convert_maximv1_format(data)
# print(f"\nConverted: {converted.shape}, columns: {converted.columns.tolist()}")
# print(f"signal[0] shape: {converted['signal'].iloc[0].shape}, response_time sample: {converted['response_time'].head(3).tolist()}")

# %%

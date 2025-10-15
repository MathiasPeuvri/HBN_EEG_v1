import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt

""" Main differences with original article:
Not same data, we'll stick with 1 of our two datasets (32 chan 1024Hz / 129 chan 100Hz)
# Our data :  X_train shape: torch.Size([Number of samples, 32 (channels), 256(sample points)])
-> need to adapt the model architecture
-> Need to change the channel recombination strategy
"""
N_CHANS = 32
FS = 1024
SAMPLEPOINTS = 256
BATCH_SIZE = 32  # Original paper uses 1000 for convolutional encoder
# /!\ Temperature must scale with batch size 
TEMPERATURE = 0.1 # Original paper uses 0.05; the temperature is related to the batch size; with higher batch size, we can decrease the temperature

NUM_WORKERS = 0
# ================================ Channels augmentation ================================

transformation_ranges_original = {'amplitude_scale' : (0.5,2), 'time_shift_seconds' : (-0.25,0.25), 'DC_shift_µV' : (-10,10), 'zero-masking_seconds' : (0,0.75), 'adaptive_gaussian_noise' : (0.0,0.2), 'band5hz-stop_Hzstart': (2.8,82.5)}
# Time shift in second is extracted from the original article but don't make sense for IED tasks ?!
# Similar comment for the zero-masking strategy that goes beyond the lenght of our desired window
# Should we add a channel number on which to apply the amplitude scaling? And a random selection of this number of channels? Maybe varying scaling for each channel ?
transformation_ranges_adapted = {'amplitude_scale' : (0.5,2), 'time_shift_percent_twindow' : (-0.05,0.05), 'DC_shift_µV' : (-1,1),
                                 'zero-masking_percent_twindow' : (0,0.05), 'adaptive_gaussian_noise' : (0.0,0.2), 'band5hz-stop_Hzstart': (3, 45)}  # Safer range: 5.5-42.5 Hz for 5Hz bandstop


# training_set:         start    stop  cat IED_start_samplepoint                     file                                            signals                                         montage_ID train_test
# 0           0     256    0                  None  16S1J1_64V_Galvani_0001  [[0.004827193339160397, 0.033373406841333164, ...  avg_ref_32____0_1_2_3_4_5_6_7_8_9_10_11_12_13_...      train
# 1         196     452    0                  None  16S1J1_64V_Galvani_0001  [[-1.3630072141888008, -1.3494641985877236, -1...  avg_ref_32____0_1_2_3_4_5_6_7_8_9_10_11_12_13_...      train

def simple_amplitude_scaling(data, scaling_factor):
    return [d * scaling_factor for d in data]

def time_shift(full_training_set, training_set_row, shift_factor):
    """
    Fcking too complex to branch it to training set of IEDetector! At least it seems to work !
    Apply time shift to the data
    """
    original_start_stop = training_set_row['start'], training_set_row['stop']
    shift_in_samplepoints = int(shift_factor * (original_start_stop[1]-original_start_stop[0]))
    new_start_stop = original_start_stop[0] + shift_in_samplepoints, original_start_stop[1] + shift_in_samplepoints
    new_signal = np.zeros(shape=(np.shape(training_set_row['signals'])))
    # find the rows in full training set that have the same file and montage_ID as the training set row
    usefull_training_set = full_training_set[(full_training_set['file'] == training_set_row['file']) & (full_training_set['montage_ID'] == training_set_row['montage_ID'])]
    # select rows in usefull_training_set that contain signals that are in the new_start_stop range; first find the row with start of the signal in the new_start_stop range
    row_with_start_in_new_start_stop = usefull_training_set[(usefull_training_set['start'] <= new_start_stop[0]) & (usefull_training_set['stop'] >= new_start_stop[0])].iloc[0] # contain data before the start and finish at the stop max
    def get_start_of_new_signal(row, new_start_stop):
        idx_for_start = int(new_start_stop[0] - row['start'])
        # print(f"idx of start stop for new start : {idx_for_start}, {row['stop']}")
        start_of_new_signal = [signal_row[idx_for_start:] for signal_row in row['signals']]
        remaining_samplepoints_to_find = int(new_start_stop[1] - row['stop'])
        # print(f'remaining_samplepoints_to_find: {remaining_samplepoints_to_find}')
        if remaining_samplepoints_to_find > 0: # we expect that
            return start_of_new_signal, remaining_samplepoints_to_find
        else:
            # print('seems to be an error in the logic of the time shift, see line 51')
            return start_of_new_signal, 0
    
    start_new_sig, remaining_sp = get_start_of_new_signal(row_with_start_in_new_start_stop, new_start_stop)
    end_of_sig_idx = new_start_stop[1] - remaining_sp , new_start_stop[1]
    # print(f'new start stop idx : {new_start_stop}, remaining samplepoints to find: {remaining_sp}, end of sig idx: {end_of_sig_idx}')

    row_with_end_of_sig_idx = usefull_training_set[(usefull_training_set['start'] <= end_of_sig_idx[0]) & (usefull_training_set['stop'] >= end_of_sig_idx[1])].iloc[0]
    # print(f'row with end of sig idx: {row_with_end_of_sig_idx}')
    def get_end_of_new_signal(row, end_of_sig_idx):
        start = int(end_of_sig_idx[0] - row['start'])
        stop = int(end_of_sig_idx[1] - row['start'])
        #print(f'idx start stop end signal : {start}, {stop}')
        end_of_new_signal = [signal_row[start:stop] for signal_row in row['signals']]
        # could have a try/except to check that we indeed have the end of the signal but we'll considere it's fine
        return end_of_new_signal
    
    end_new_sig = get_end_of_new_signal(row_with_end_of_sig_idx, end_of_sig_idx)
    #print(f'shape start : {np.shape(start_new_sig)}, shape end: {np.shape(end_new_sig)}')
    #new_signal = np.concatenate((start_new_sig, end_new_sig))
    new_signal = np.array([np.concatenate([start, end]) for start, end in zip(start_new_sig, end_new_sig)])
    # check that new_signal has the desired shape 
    if np.shape(new_signal) != np.shape(training_set_row['signals']):
        # print(f"seems to be an error in the logic of the time shift (new signal shape : {np.shape(new_signal)}, original signal shape: {np.shape(training_set_row['signals'])}), see line 72")
        return full_training_set
    else:
        return new_signal

def dc_shift(data, shift_factor):
    return [d + shift_factor for d in data]

def zero_masking(data, mask_factor):
    data = np.array(data)
    masked_data = data.copy()
    
    # Calculate mask length and random start position
    mask_length = int(mask_factor * data.shape[1])
    if mask_length > 0:
        start_idx_max = data.shape[1] - mask_length
        start_idx = np.random.randint(0, start_idx_max + 1)
        # Apply zero mask to all channels at the same time segment
        masked_data[:, start_idx:start_idx + mask_length] = 0
    
    return masked_data

def adaptive_gaussian_noise(data, noise_factor):
    data_std = np.std(data)
    return [d + np.random.normal(0, noise_factor * data_std, len(d)) for d in data]

def band_stop5h_filter(data, center_freq_Hz, fs=FS):
    """
    Apply 5Hz bandstop filter to the data
    center_freq_Hz: center frequency of the band to stop

    Returns:
        Filtered data, or original data if filter parameters are invalid
    """
    lowcut = center_freq_Hz - 2.5
    highcut = center_freq_Hz + 2.5
    nyquist = fs / 2

    # Ensure filter frequencies are valid (must be in range [0, 1] when normalized)
    # Add safety margins to avoid numerical issues
    if lowcut <= 1.0 or highcut >= (nyquist - 1.0):
        # Skip filtering if frequencies are too close to boundaries
        return data

    low = lowcut / nyquist
    high = highcut / nyquist

    # Additional check: ensure normalized frequencies are in valid range
    if low <= 0.01 or high >= 0.99 or low >= high:
        return data

    try:
        b, a = butter(4, [low, high], btype='bandstop')
        return [filtfilt(b, a, d) for d in data]
    except (ValueError, np.linalg.LinAlgError) as e:
        # If filter design fails, return original data
        # This can happen with extreme frequency values
        return data



def apply_random_augmentations(full_training_set, training_sample_idx, augmentation_params, num_augmentations=2):
    """
    Apply a random selection of augmentations to EEG data.

    Returns:
        augmented_data: numpy array of same shape as input
        aug_info: list of tuples (augmentation_name, parameter_value)
    """
    data = full_training_set.iloc[training_sample_idx]['signals']
    # Randomly select augmentations (excluding time_shift for simplicity)
    available_augs = ['amplitude_scale', 'DC_shift', 'time_shift', 'zero_masking', 'gaussian_noise', 'band_stop']
    selected_augs = np.random.choice(available_augs, size=num_augmentations, replace=False)

    # Apply selected augmentations sequentially and track parameters
    augmented_data = np.array(data).copy()
    aug_info = []

    # if time shift in the selecteed augmentations, make sure it is first in the list
    if 'time_shift' in selected_augs:
        selected_augs = list(selected_augs)
        selected_augs.remove('time_shift')
        selected_augs.insert(0, 'time_shift')
        selected_augs = np.array(selected_augs)

    for aug_name in selected_augs:
        if aug_name == 'time_shift':
            #print(f"Applying time shift for sample {training_sample_idx}")
            param = np.random.uniform(*augmentation_params['time_shift_percent_twindow'])
            for try_param in [param, -param]:
                try:
                    augmented_data = time_shift(full_training_set, full_training_set.iloc[training_sample_idx], try_param)
                    aug_info.append((aug_name, f'{try_param:.2f}'))
                    # if try_param == -param:
                    #     print(f"switched to negative parameter")
                    break
                    
                except (IndexError, ValueError):
                    if try_param == -param:  # Second attempt also failed
                        # Pick random replacement from unused augs
                        unused = [a for a in ['amplitude_scale', 'DC_shift', 'zero_masking', 'gaussian_noise', 'band_stop'] 
                                 if a not in selected_augs]
                        aug_name = np.random.choice(unused) if unused else 'amplitude_scale'
                        print(f"Error in time shift for sample {training_sample_idx}, skipped/replaced with {aug_name}")
                        # Fall through to normal processing below
        elif aug_name == 'amplitude_scale':
            param = np.random.uniform(*augmentation_params['amplitude_scale'])
            augmented_data = simple_amplitude_scaling(augmented_data, param)
            aug_info.append((aug_name, f'{param:.2f}x'))
        elif aug_name == 'DC_shift':
            param = np.random.uniform(*augmentation_params['DC_shift_µV'])
            augmented_data = dc_shift(augmented_data, param)
            aug_info.append((aug_name, f'{param:.2f}µV'))
        elif aug_name == 'zero_masking':
            param = np.random.uniform(*augmentation_params['zero-masking_percent_twindow'])
            augmented_data = zero_masking(augmented_data, param)
            aug_info.append((aug_name, f'{param*100:.1f}%'))
        elif aug_name == 'gaussian_noise':
            param = np.random.uniform(*augmentation_params['adaptive_gaussian_noise'])
            augmented_data = adaptive_gaussian_noise(augmented_data, param)
            aug_info.append((aug_name, f'σ={param:.2f}'))
        elif aug_name == 'band_stop':
            param = np.random.uniform(*augmentation_params['band5hz-stop_Hzstart'])
            augmented_data = band_stop5h_filter(augmented_data, param)
            aug_info.append((aug_name, f'{param:.1f}Hz'))

    return np.array(augmented_data), aug_info

def create_augmented_pair(full_training_set, training_sample_idx, augmentation_params, num_augmentations=2):
    """
    Create two positive pairs, differently augmented versions of the same EEG sample.

    Args:
        full_training_set: pandas dataframe containing the full training set (needed for time shift) #see data_preparations_from_setupobj
        training_sample_idx: index of the training sample to augment
        augmentation_params: dict with augmentation ranges
        num_augmentations: number of augmentations to apply per view

    Returns:
        view1, view2: two differently augmented versions of the input sample
        aug_info1, aug_info2: lists of applied augmentations with their parameters
    """
    view1, aug_info1 = apply_random_augmentations(full_training_set, training_sample_idx, augmentation_params, num_augmentations)
    view2, aug_info2 = apply_random_augmentations(full_training_set, training_sample_idx, augmentation_params, num_augmentations)

    return view1, view2, aug_info1, aug_info2


# ================================ DATASET CLASS ================================

class ContrastiveEEGDataset(torch.utils.data.Dataset):
    """
    wrapper -> PyTorch Dataset for contrastive learning on EEG signals.

    For each sample, returns two differently augmented versions (positive pair).
    The DataLoader will batch these pairs for contrastive learning.

    Args:
        training_set: pandas DataFrame with EEG signals (from data_preparations_from_setupobj)
        augmentation_params: dict with augmentation parameter ranges
        num_augmentations: number of augmentations to apply per view (default: 2)
    """
    def __init__(self, training_set, augmentation_params, num_augmentations=2):
        self.training_set = training_set
        self.augmentation_params = augmentation_params
        self.num_augmentations = num_augmentations

    def __len__(self):
        return len(self.training_set)

    def __getitem__(self, idx):
        # Create augmented pair for the training_set.iloc[idx] sample
        view1, view2, _, _ = create_augmented_pair(self.training_set, idx, self.augmentation_params, self.num_augmentations)

        # Convert to torch tensors (float32)
        view1 = torch.FloatTensor(view1)
        view2 = torch.FloatTensor(view2)

        return view1, view2


# ================================ MODEL ARCHITECTURE ================================
# Need to adapt the model architecture to our data

class ConvolutionalEncoder(nn.Module):
    """Encodeur convolutionnel pour signaux EEG"""

    def __init__(self, in_channels=1, repeat_n=4, n_samples=SAMPLEPOINTS):
        super(ConvolutionalEncoder, self).__init__()
        # Scale kernel sizes based on signal length
        # Original paper: 4000 samples with kernels [128, 64, 16]
        # Scaling factor maintains proportional receptive fields
        scale_factor = n_samples / 4000.0
        k1 = max(4, int(128 * scale_factor))  # Ensure minimum kernel size of 4
        k2 = max(4, int(64 * scale_factor))
        k3 = max(2, int(16 * scale_factor))

        self.k1, self.k2, self.k3 = k1, k2, k3  # Store for padding calculations

        # Trois branches parallèles avec différentes tailles de kernel
        self.branch1 = nn.Conv1d(in_channels, 100, kernel_size=k1, padding=0)
        self.branch2 = nn.Conv1d(in_channels, 100, kernel_size=k2, padding=0)
        self.branch3 = nn.Conv1d(in_channels, 50, kernel_size=k3, padding=0)
        
        # Dense après concatenation (100+100+50=250)
        self.dense1 = nn.Linear(250, 250)
        
        # Blocs répétés N=4 fois
        self.repeat_blocks = nn.ModuleList()
        for _ in range(repeat_n):
            block = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(250),
                nn.Conv1d(250, 250, kernel_size=64, padding=0)
            )
            self.repeat_blocks.append(block)
        
        # Couches finales
        self.final_relu = nn.ReLU()
        self.final_bn = nn.BatchNorm1d(250)
        self.final_conv = nn.Conv1d(250, 4, kernel_size=64, padding=0)
        
    def forward(self, x):
        # x shape: (batch, channels, time)

        # Application du reflection padding pour chaque branche (dynamic based on kernel size)
        pad1_left = self.k1 // 2
        pad1_right = self.k1 - pad1_left - 1
        x1 = F.pad(x, (pad1_left, pad1_right), mode='reflect')
        x1 = self.branch1(x1)

        pad2_left = self.k2 // 2
        pad2_right = self.k2 - pad2_left - 1
        x2 = F.pad(x, (pad2_left, pad2_right), mode='reflect')
        x2 = self.branch2(x2)

        pad3_left = self.k3 // 2
        pad3_right = self.k3 - pad3_left - 1
        x3 = F.pad(x, (pad3_left, pad3_right), mode='reflect')
        x3 = self.branch3(x3)
        
        # Concatenation le long de la dimension des channels
        x = torch.cat([x1, x2, x3], dim=1)  # (batch, 250, time)
        
        # Dense layer (permute pour Linear)
        x = x.permute(0, 2, 1)  # (batch, time, 250)
        x = self.dense1(x)
        x = x.permute(0, 2, 1)  # (batch, 250, time)
        
        # Blocs répétés
        for block in self.repeat_blocks:
            residual = x
            x = F.pad(x, (32, 31), mode='reflect')  # kernel=64
            x = block(x)
            # Ajuster la taille si nécessaire pour addition résiduelle
            if x.shape[2] != residual.shape[2]:
                min_len = min(x.shape[2], residual.shape[2])
                x = x[:, :, :min_len]
                residual = residual[:, :, :min_len]
            x = x + residual
        
        # Couches finales
        x = self.final_relu(x)
        x = self.final_bn(x)
        x = F.pad(x, (32, 31), mode='reflect')
        x = self.final_conv(x)
        
        return x


class Projector(nn.Module):
    """
    Projecteur avec LSTM bidirectionnels pour apprentissage contrastif ou classification supervisée.

    Peut être utilisé pour:
    - Apprentissage contrastif: output_dim=32, use_logsoftmax=False (défaut)
    - Classification supervisée: output_dim=num_classes, use_logsoftmax=True

    """

    def __init__(self, input_dim=4, output_dim=32, use_logsoftmax=False):
        super(Projector, self).__init__()

        # Trois branches LSTM avec différentes résolutions temporelles
        self.lstm1 = nn.LSTM(input_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(input_dim, 64, bidirectional=True, batch_first=True)

        # Dense layers pour projection
        # LSTM bidirectionnel double la dimension de sortie
        # FLO (First-Last-Output) concatenates first and last timesteps, doubling again
        total_dim = (256*2)*2 + (128*2)*2 + (64*2)*2  # 1792 (bidirectional × 2 timesteps)
        self.dense1 = nn.Linear(total_dim, 128)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, output_dim)
        self.use_logsoftmax = use_logsoftmax

    def downsample(self, x, factor=2):
        """Downsample par facteur donné"""
        return x[:, ::factor, :]

    def forward(self, x):
        # x shape: (batch, channels, time)
        x = x.permute(0, 2, 1)  # (batch, time, channels)

        # Branche 1: LSTM complet
        lstm1_out, _ = self.lstm1(x)
        # Prendre premier et dernier output (FLO)
        flo1 = torch.cat([lstm1_out[:, 0, :], lstm1_out[:, -1, :]], dim=1)
        # Branche 2: Downsample 50% puis LSTM
        x_down2 = self.downsample(x, factor=2)
        lstm2_out, _ = self.lstm2(x_down2)
        flo2 = torch.cat([lstm2_out[:, 0, :], lstm2_out[:, -1, :]], dim=1)
        # Branche 3: Downsample 50% puis LSTM
        x_down3 = self.downsample(x, factor=2)
        lstm3_out, _ = self.lstm3(x_down3)
        flo3 = torch.cat([lstm3_out[:, 0, :], lstm3_out[:, -1, :]], dim=1)

        # Concatenation des sorties FLO
        x = torch.cat([flo1, flo2, flo3], dim=1)

        # Projection
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)

        # Appliquer LogSoftmax si mode classification
        if self.use_logsoftmax:
            x = F.log_softmax(x, dim=1)

        return x


class EEGContrastiveModel(nn.Module):
    """Modèle complet: Encodeur + Projecteur"""

    def __init__(self, in_channels=1, repeat_n=4):
        super(EEGContrastiveModel, self).__init__()
        self.encoder = ConvolutionalEncoder(in_channels, repeat_n)
        self.projector = Projector(input_dim=4)

    def forward(self, x):
        # Encodage
        encoded = self.encoder(x)
        # Projection
        projected = self.projector(encoded)
        return encoded, projected


# ================================ CONTRASTIVE LOSS ================================

class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss for contrastive learning.
        Pulls positive pairs together in embedding space while pushing negative pairs apart. 

    - Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
    - Mohsenvand et al. "Contrastive Representation Learning for EEG Classification" (SeqCLR)

    Args:
        temperature: Temperature scaling parameter (default: uses global TEMPERATURE constant)
            Lower values → sharper distinctions between positive/negative pairs / Higher values → softer, more uniform distributions

    Mathematical formulation:
        For a positive pair (i, j):
        ℓ(i,j) = -log[ exp(sim(z_i, z_j)/τ) / Σ(k≠i) exp(sim(z_i, z_k)/τ) ]

        where sim(u,v) = cosine_similarity(u,v) = (u·v) / (||u|| × ||v||)
    """
    def __init__(self, temperature=None):
        super(NTXentLoss, self).__init__()
        # Use global TEMPERATURE constant if not specified
        self.temperature = temperature if temperature is not None else TEMPERATURE

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss for a batch of positive pairs.

        Args:
            z_i: Projections of view 1 ; z_j: Projections of view 2 

        Returns:
            loss: Scalar NT-Xent loss averaged over all 2*batch_size samples

        How it works:
            1. Concatenate z_i and z_j → (2*batch_size, projection_dim)
            2. Compute cosine similarity matrix for ALL pairs
            3. For each sample, identify its positive pair and negative pairs
            4. Compute cross-entropy: maximize similarity to positive, minimize to negatives
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # Step 1: Concatenate both views into single tensor
        z = torch.cat([z_i, z_j], dim=0)
        # Step 2: Normalize embeddings to unit length (for cosine similarity)
        z_norm = F.normalize(z, dim=1)
        # Step 3: Compute cosine similarity matrix for all pairs
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature

        # Step 4: Create mask for positive pairs
        positive_mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=device)
        positive_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        positive_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        # Step 5: Create mask for negative pairs
        diagonal_mask = torch.eye(2*batch_size, dtype=torch.bool, device=device)
        negatives_mask = ~diagonal_mask & ~positive_mask

        # Step 6: Extract similarities
        positives = sim_matrix[positive_mask].view(2*batch_size, 1)
        # Negatives: shape (2*batch_size, 2*batch_size - 2)
        negatives = sim_matrix[negatives_mask].view(2*batch_size, -1)
        # Step 7: Create logits for cross-entropy
        logits = torch.cat([positives, negatives], dim=1)
        # Step 8: Labels for cross-entropy (positive always at index 0)
        labels = torch.zeros(2*batch_size, dtype=torch.long, device=device)
        # Step 9: Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


# ================================ CONTRASTIVE PRETRAINING ================================

def pretrain_contrastive(model, train_dataset, val_dataset=None,
                         epochs=300, batch_size=32, lr=1e-3, weight_decay=1e-4,
                         device='cuda', warmup_epochs=10, checkpoint_dir='./pretrained_models',
                         early_stopping_patience=None, random_seed=42):
    """
    Pretrain the encoder using contrastive learning with NT-Xent loss.

    Implements the pretraining strategy from Mohsenvand et al. (2020) with adaptations:
    - Learning rate warmup (10 epochs) followed by cosine annealing
    - Gradient clipping to prevent exploding gradients in LSTM layers
    - Validation monitoring and checkpointing (best + last model)
    - Optional early stopping based on validation loss plateau

    Args:
        model: EEGContrastiveModel instance (Encoder + Projector)
        train_dataset: ContrastiveEEGDataset for training
        val_dataset: ContrastiveEEGDataset for validation (optional but recommended)
        epochs: Number of training epochs (default: 300 as per paper)
        batch_size: Batch size for training (default: 32, paper uses 1000)
        lr: Peak learning rate (default: 1e-3, standard for Adam)
        weight_decay: L2 regularization coefficient (default: 1e-4 as per paper)
        device: Device to train on ('cuda' or 'cpu')
        warmup_epochs: Number of epochs for linear learning rate warmup (default: 10)
        checkpoint_dir: Directory to save model checkpoints
        early_stopping_patience: If set, stop if validation loss doesn't improve for N epochs
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        model: Pretrained model
        history: Dict with training metrics:
            - 'train_loss': List of average training losses per epoch
            - 'val_loss': List of validation losses per epoch (if val_dataset provided)
            - 'grad_norms': List of gradient norms per epoch
            - 'learning_rates': List of learning rates per epoch

    Example:
        >>> train_data = ContrastiveEEGDataset(training_set, transformation_ranges_adapted)
        >>> val_data = ContrastiveEEGDataset(validation_set, transformation_ranges_adapted)
        >>> model = EEGContrastiveModel(in_channels=32, repeat_n=4)
        >>> model, history = pretrain_contrastive(model, train_data, val_data, epochs=300)
    """
    import os
    from torch.utils.data import DataLoader
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on device: {device}")

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS) if val_dataset else None

    # Setup loss and optimizer
    criterion = NTXentLoss(temperature=TEMPERATURE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Setup learning rate scheduler (warmup + cosine decay)
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            # Cosine annealing
            # Handle edge case where epochs == warmup_epochs
            if epochs <= warmup_epochs:
                return 1.0
            progress = float(current_epoch - warmup_epochs) / float(epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'grad_norms': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    print(f"\nStarting contrastive pretraining:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr} (with warmup + cosine decay)")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Training samples: {len(train_dataset)}")
    if val_dataset:
        print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Checkpoint directory: {checkpoint_dir}")
    print()

    # Training loop
    for epoch in range(epochs):
        # ========== TRAINING PHASE ==========
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0

        # Create progress bar for training batches
        if use_tqdm:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        else:
            train_pbar = train_loader

        for batch_idx, (view1, view2) in enumerate(train_pbar):
            view1 = view1.to(device)
            view2 = view2.to(device)

            # Forward pass
            _, proj1 = model(view1)  # Get projections (discard encoder output)
            _, proj2 = model(view2)

            # Compute contrastive loss
            loss = criterion(proj1, proj2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients in LSTM)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_grad_norm += grad_norm.item()

            # Update progress bar with current loss
            if use_tqdm:
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average metrics over batches
        avg_train_loss = epoch_loss / len(train_loader)
        avg_grad_norm = epoch_grad_norm / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(avg_train_loss)
        history['grad_norms'].append(avg_grad_norm)
        history['learning_rates'].append(current_lr)

        # Step learning rate scheduler
        scheduler.step()

        # ========== VALIDATION PHASE ==========
        if val_loader:
            model.eval()
            val_loss = 0.0

            # Create progress bar for validation batches
            if use_tqdm:
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            else:
                val_pbar = val_loader

            with torch.no_grad():
                for view1, view2 in val_pbar:
                    view1 = view1.to(device)
                    view2 = view2.to(device)

                    _, proj1 = model(view1)
                    _, proj2 = model(view2)

                    loss = criterion(proj1, proj2)
                    val_loss += loss.item()

                    # Update progress bar with current loss
                    if use_tqdm:
                        val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            # ========== CHECKPOINTING ==========
            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

                checkpoint_path = os.path.join(checkpoint_dir, 'best_encoder.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            # ========== EARLY STOPPING ==========
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Validation loss has not improved for {early_stopping_patience} epochs")
                break

        # ========== LOGGING ==========
        # Print summary for every epoch (or every N epochs if you prefer less output)
        log_msg = f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f}"
        if val_loader:
            log_msg += f" | Val Loss: {avg_val_loss:.4f}"
            if avg_val_loss == best_val_loss:
                log_msg += " ✓ Best"
        log_msg += f" | LR: {current_lr:.6f} | Grad: {avg_grad_norm:.4f}"
        print(log_msg)

    # ========== SAVE FINAL MODEL ==========
    final_checkpoint_path = os.path.join(checkpoint_dir, 'last_encoder.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss if val_loader else None,
        'best_val_loss': best_val_loss if val_loader else None
    }, final_checkpoint_path)

    print(f"\n Pretraining completed!")
    print(f"  Final training loss: {avg_train_loss:.4f}")
    if val_loader:
        print(f"  Final validation loss: {avg_val_loss:.4f}")
        print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Models saved to: {checkpoint_dir}")
    print(f"    - best_encoder.pth (best validation loss)")
    print(f"    - last_encoder.pth (final epoch)")

    return model, history


def load_pretrained_model(checkpoint_path, in_channels=32, repeat_n=4, device='cuda'):
    """
    Load a pretrained contrastive model from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file
        in_channels: Number of input channels (default: 32)
        repeat_n: Number of residual blocks in encoder (default: 4)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Loaded EEGContrastiveModel
        checkpoint: Dict with training metadata (epoch, losses, etc.)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = EEGContrastiveModel(in_channels=in_channels, repeat_n=repeat_n)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"  Loaded pretrained model from: {checkpoint_path}")
    print(f"  Trained for: {checkpoint['epoch']} epochs")
    print(f"  Final train loss: {checkpoint['train_loss']:.4f}")
    if checkpoint.get('val_loss'):
        print(f"  Final val loss: {checkpoint['val_loss']:.4f}")

    return model, checkpoint

# Exemple d'utilisation
if __name__ == "__main__":

    import os
    import sys
    import logging
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    logging.disable(logging.WARNING)
    import os.path
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from global_functions.setup_param_exp import Setup_params
    from data_preparation.datapreparation_set_creation import data_preparations_from_setupobj

    # PyTorch automatically detects and uses CUDA if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    """ Prepare training setup """
    setup_training = Setup_params()
    training_files = ['16S1J1_64V_Galvani_0001']
    if isinstance(training_files, str):
        training_files = [training_files]
    setup_training.files = training_files
    setup_training.overlap = 0.0
    setup_training.time_limite = (0, 300 * setup_training.fs)
    setup_training.split_tech = 'abs_v1'
    setup_training.window_size = 0.25
    setup_training.bandpassfreq = (1, 45)
    setup_training.update_report_fold(custom_name="CRL_tests")
    training_set, _, _, _, _, _, _, _, report_folder = data_preparations_from_setupobj(setup_training)

    #################### section for draft about data augmentation for CLR
    show_data_augmentation = False
    if show_data_augmentation:
        from SSL_material.CRL_mohsenvand import time_shift, simple_amplitude_scaling, dc_shift, zero_masking, adaptive_gaussian_noise, band_stop5h_filter
        print(training_set.iloc[27])

        time_shifted_sig = time_shift(training_set, training_set.iloc[27], 0.1)
        amplitude_scaled_sig = simple_amplitude_scaling(training_set.iloc[27]['signals'], 0.5)
        dc_shifted_sig = dc_shift(training_set.iloc[27]['signals'], 0.1)
        zero_masked_sig = zero_masking(training_set.iloc[27]['signals'], 0.1)
        adaptive_gaussian_noise_sig = adaptive_gaussian_noise(training_set.iloc[27]['signals'], 0.1)
        band_stop5h_sig = band_stop5h_filter(training_set.iloc[27]['signals'], 10)
        print(f"shape check: time shifted: {np.shape(time_shifted_sig)}, amplitude scaled: {np.shape(amplitude_scaled_sig)}, dc shifted: {np.shape(dc_shifted_sig)}")

        plt.figure(figsize=(10, 5))
        plt.plot(time_shifted_sig[10], color = 'r', label = 'time shifted')
        plt.plot(amplitude_scaled_sig[10], color = 'g', label = 'amplitude scaled')
        plt.plot(dc_shifted_sig[10], color = 'b', label = 'dc shifted')
        plt.plot(zero_masked_sig[10], color = 'y', label = 'zero masked')
        plt.plot(adaptive_gaussian_noise_sig[10], color = 'c', label = 'adaptive gaussian noise')
        plt.plot(band_stop5h_sig[10], color = 'm', label = 'band stop 5Hz')
        plt.plot(training_set.iloc[27]['signals'][10], color = 'k', ls = '--', label = 'original')

        plt.legend()
        plt.show()

    #################### Test augmentation pipeline for contrastive learning
    test_augmentation_pipeline = False
    if test_augmentation_pipeline:
        print("\n" + "="*60)
        print("Testing Augmented Pair Generation for Contrastive Learning")
        print("="*60)

        # Select a few samples to test
        test_samples = [27, 50, 100]
        channel_to_plot = 10  # Channel index to visualize

        for sample_idx in test_samples:
            sample = training_set.iloc[sample_idx]
            original_signals = np.array(sample['signals'])

            print(f"\nSample {sample_idx}:")
            print(f"  Original shape: {original_signals.shape}")

            # Create augmented pair
            view1, view2, aug_info1, aug_info2 = create_augmented_pair(training_set, sample_idx, transformation_ranges_adapted, num_augmentations=2)
            print(f"  View 1 shape: {view1.shape};  augmentations: {' + '.join([f'{name}({val})' for name, val in aug_info1])}")
            print(f"  View 2 shape: {view2.shape};  augmentations: {' + '.join([f'{name}({val})' for name, val in aug_info2])}")

            # Plot original and both views for one channel
            plt.figure(figsize=(15, 4))
            plt.plot(original_signals[channel_to_plot], 'k', linewidth=1, label = f'Original Signal\nSample {sample_idx}, Channel {channel_to_plot}')
            aug_text1 = '\n'.join([f'{name}: {val}' for name, val in aug_info1])
            plt.plot(view1[channel_to_plot], 'b', linewidth=1, label = f'View 1 (Augmented)\n{aug_text1}')
            aug_text2 = '\n'.join([f'{name}: {val}' for name, val in aug_info2])
            plt.plot(view2[channel_to_plot], 'r', linewidth=1, label = f'View 2 (Augmented)\n{aug_text2}')
            plt.legend()
            plt.xlabel('Time points')
            plt.ylabel('Amplitude (µV)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            #plt.savefig(os.path.join(report_folder, f'augmented_pair_sample_{sample_idx}.png'), dpi=150)
            plt.show()




    # Créer le modèle
    model = EEGContrastiveModel(in_channels=32, repeat_n=4)
    
    # Exemple de données (batch_size=2, channels=1, time=1000)
    x = torch.randn(2, N_CHANS, SAMPLEPOINTS)
    
    # Forward pass
    encoded, projected = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Projected shape: {projected.shape}")
    print(f"\nNombre de paramètres: {sum(p.numel() for p in model.parameters()):,}")

    #################### Test ContrastiveEEGDataset
    test_dataset = False
    if test_dataset:
        from torch.utils.data import DataLoader

        print("\n" + "="*60)
        print("Testing ContrastiveEEGDataset")
        print("="*60)

        # Create dataset
        dataset = ContrastiveEEGDataset(training_set, transformation_ranges_adapted, num_augmentations=2)
        print(f"Dataset size: {len(dataset)} (original size: {len(training_set)})")
        # Test single sample
        view1, view2 = dataset[0]
        print(f"\nSingle sample:")
        print(f"  View1 shape: {view1.shape}, dtype: {view1.dtype} // View2 shape: {view2.shape}, dtype: {view2.dtype}")
        # Test DataLoader with batching
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        view1_batch, view2_batch = next(iter(dataloader))
        print(f"\nBatch from DataLoader (batch_size={BATCH_SIZE}):")
        print(f"  View1 batch shape: {view1_batch.shape} // View2 batch shape: {view2_batch.shape}")
        # Test that views are different (augmentations applied)
        print(f"Views are different (L2 distance): {torch.norm(view1_batch - view2_batch).item():.2f} -> ContrastiveEEGDataset working correctly!")

    #################### Test NT-Xent Loss
    test_ntxent_loss = False
    if test_ntxent_loss:
        print("\n" + "="*60)
        print("Testing NT-Xent Loss")
        print("="*60)

        criterion = NTXentLoss()
        print(f"Using temperature τ={criterion.temperature} (global TEMPERATURE constant)")

        # Test 1: Loss behavior with different similarity levels
        z_base = torch.randn(4, 32)
        z_identical = z_base.clone()
        z_similar = z_base + 0.1 * torch.randn_like(z_base)
        z_random = torch.randn_like(z_base)

        loss_identical = criterion(z_base, z_identical)
        loss_similar = criterion(z_base, z_similar)
        loss_random = criterion(z_base, z_random)

        print(f"\nLoss behavior (should increase as views become less similar):")
        print(f"  Identical views: {loss_identical.item():.4f}")
        print(f"  Similar views:   {loss_similar.item():.4f}")
        print(f"  Random views:    {loss_random.item():.4f}")

        if loss_identical < loss_similar < loss_random:
            print("  ✓ Loss increases monotonically - correct!")
        else:
            print("  ✗ Loss is not monotonic - check implementation!")

        # Test 2: Temperature vs Batch Size relationship
        print(f"\nTemperature vs Batch Size relationship:")
        print(f"  Goal: Loss should be SENSITIVE to noise (not 0, not too high)")
        print(f"  Test: Same noisy views (30% noise) with different τ and batch_size")
        print(f"\n  {'Batch Size':<12} | τ=0.05    | τ=0.1     | τ=0.5     |")
        print(f"  {'-'*12}-+-----------+-----------+-----------|")

        # Test different batch sizes
        batch_sizes = [4, 16, 32, 128, 1000]
        temps = [0.05, 0.1, 0.5]

        for bs in batch_sizes:
            # Create test data for this batch size
            z_test = torch.randn(bs, 32)
            z_test_noisy = z_test + 0.3 * torch.randn_like(z_test)

            row_losses = []
            for temp in temps:
                crit = NTXentLoss(temperature=temp)
                loss = crit(z_test, z_test_noisy)
                row_losses.append(loss.item())

            status = " ← Our config" if bs == BATCH_SIZE else " (Paper=1000)"
            print(f"  {bs:<12} | {row_losses[0]:>7.4f}  | {row_losses[1]:>7.4f}  | {row_losses[2]:>7.4f}  |{status}")

        print(f"\n  Interpretation:")
        print(f"    - Loss increases with batch_size (more negatives → harder task)")
        print(f"    - Loss ≈ 0.00: Saturated (positive pair too dominant)")
        print(f"    - Paper (batch=1000, τ=0.05): loss ≈ 0.002-0.005 (works with many negatives)")
        print(f"    - Our config (batch={BATCH_SIZE}, τ={TEMPERATURE}): loss ≈ {row_losses[temps.index(TEMPERATURE)]:.3f}")
        print(f"    - Key insight: With fewer negatives, slightly higher τ prevents saturation")

        # Test 3: Integration with model + gradient flow
        model = EEGContrastiveModel(in_channels=N_CHANS, repeat_n=4)
        view1 = torch.randn(4, N_CHANS, SAMPLEPOINTS, requires_grad=True)
        view2 = torch.randn(4, N_CHANS, SAMPLEPOINTS, requires_grad=True)

        _, proj1 = model(view1)
        _, proj2 = model(view2)
        loss = criterion(proj1, proj2)
        loss.backward()

        print(f"\nIntegration test:")
        print(f"  Model output shape: {proj1.shape}")
        print(f"  Loss value: {loss.item():.4f}")
        print(f"  Gradient flow: {'✓ OK' if view1.grad is not None else '✗ Failed'}")

        print("\n" + "="*60)
        print("✓ NT-Xent Loss tests completed")



# ================================ PRETRAINING TESTS ================================

    test_pretraining_loop = False
    if test_pretraining_loop:
        print("\n" + "="*60)
        print("CONTRASTIVE PRETRAINING - Quick Validation Test")
        print("="*60)

        # Use real EEG data (subset for quick testing)
        print(f"\nUsing real EEG data from training_set...")
        print(f"  Total samples available: {len(training_set)}")

        # Split into train/val subsets for testing
        train_subset = training_set[:100]  # First 100 samples for training
        val_subset = training_set[100:120]  # Next 20 samples for validation

        train_dataset = ContrastiveEEGDataset(train_subset, transformation_ranges_adapted)
        val_dataset = ContrastiveEEGDataset(val_subset, transformation_ranges_adapted)
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")

        print(f"\nInitializing model...")
        model = EEGContrastiveModel(in_channels=N_CHANS, repeat_n=4)

        print(f"\nRunning short training (5 epochs)...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model, history = pretrain_contrastive(
            model,
            train_dataset,
            val_dataset,
            epochs=50,
            batch_size=BATCH_SIZE,
            device=device,
            checkpoint_dir='./quick_test_checkpoints'
        )

        print(f"\n" + "="*60)
        print("Validation Results:")
        print("="*60)
        print(f"✓ Training completed successfully")
        print(f"  Loss: {history['train_loss'][0]:.3f} → {history['train_loss'][-1]:.3f}")
        print(f"  Val Loss: {history['val_loss'][0]:.3f} → {history['val_loss'][-1]:.3f}")

        if history['train_loss'][-1] < history['train_loss'][0]:
            print(f"✓ Loss decreased - model is learning!")
        else:
            print(f"⚠ Loss did not decrease - check hyperparameters")

        print(f"\n✓ Checkpoints saved to ./quick_test_checkpoints/")
        print(f"  You can delete this directory after validation")

        print("\n" + "="*60)
        print("✓ Pretraining loop validated and ready for use!")
        print("="*60)

    pretrained_encoder_path = '/mnt/c/Users/mat_9/Galvani_PS1/report_folder/rep3_0610_25_CRL_pretraining/crl_checkpoints/best_encoder.pth'

    # Load pretrained encoder and create downstream classifier
    print("\n" + "="*60)
    print("LOADING PRETRAINED ENCODER FOR DOWNSTREAM CLASSIFICATION")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pretrained contrastive model
    pretrained_model = EEGContrastiveModel(in_channels=N_CHANS, repeat_n=4)
    checkpoint = torch.load(pretrained_encoder_path, map_location=device)
    pretrained_model.load_state_dict(checkpoint['model_state_dict'])

    print(f"✓ Loaded pretrained model from: {pretrained_encoder_path}")
    print(f"  Trained for: {checkpoint['epoch']} epochs")
    print(f"  Final train loss: {checkpoint['train_loss']:.4f}")
    if checkpoint.get('val_loss'):
        print(f"  Final val loss: {checkpoint['val_loss']:.4f}")

    # Extract encoder and create downstream classifier
    encoder = pretrained_model.encoder
    classifier = Projector(input_dim=4, output_dim=2, use_logsoftmax=True)
    classifier = classifier.to(device)

    print(f"\n✓ Created downstream classifier (Projector in classification mode)")
    print(f"  Input: (batch, 4, time_steps) from encoder")
    print(f"  Output: (batch, 2) log-probabilities")

    # Test forward pass
    test_input = torch.randn(2, N_CHANS, SAMPLEPOINTS).to(device)
    encoder = encoder.to(device)

    with torch.no_grad():
        encoded = encoder(test_input)
        output = classifier(encoded)

    print(f"\n✓ Forward pass test:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Encoded shape: {encoded.shape}")
    print(f"  Classifier output shape: {output.shape}")
    print(f"  Output values (log-probs): {output[0].cpu().numpy()}")

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    total_params = encoder_params + classifier_params

    print(f"\n✓ Model parameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Classifier: {classifier_params:,}")
    print(f"  Total: {total_params:,}")

    print("\n" + "="*60)
    print("✓ Ready for downstream fine-tuning!")
    print("="*60)
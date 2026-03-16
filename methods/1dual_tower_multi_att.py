"""


Hierarchical Dual-Tower Architecture gait classification template =====================================
Function: Hierarchical double-tower classifier based on original signal (300Hz) + TS features
Dataset: Gait dataset (.pkl format)
Task: Four-classification (als=0, control=1, hunt=2, park=3)

Instructions for use:
1. Modify the dataset path
2. Adjust the hyperparameters as needed
3. Replace the plan_model class with any other model is acceptable
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import random
import os
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import random_split
# LeaveOneOut
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== Global Settings ====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use the equipment: {device}")


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# ==================== Unify minimum ts feature period length ====================
def unify_min_num_cycles(ts_features):

    # num_cycle_length array
    num_cyc_length = np.zeros((len(ts_features)))
    # select shortest
    for i, ts_feature in enumerate(ts_features):
        num_cyc_length[i] = ts_features[i].shape[0]
    min_num_cycle = int(num_cyc_length.min())
    # unify min_num_cycles
    ts_features_unify = [ts_feature[:min_num_cycle,:] for ts_feature in ts_features]

    return ts_features_unify, min_num_cycle



# ==================== Normalization  ====================

def max_min_global(features):
    """
    Global maximum-minimum normalization

    Parameter:
    features: ndarray, shape [B, C, L] or [B, L, C]

    Return:
    Normalized features

    """
    min_val = np.min(features)
    max_val = np.max(features)


    if max_val - min_val < 1e-8:
        return features - min_val

    normalized = (features - min_val) / (max_val - min_val)
    return normalized



def z_score_global(features):
"""
Global Z-score normalization

Parameter:
features: ndarray, shape [B, L, C] or any shape

Return:
Normalized features

"""
    mean_val = np.mean(features)
    std_val = np.std(features)

    if std_val < 1e-8:
        return features - mean_val

    normalized = (features - mean_val) / std_val
    return normalized



def ndstrarr2ndarray(str_nd_arr):
"""
String array to numeric array

Parameter:
str_nd_arr: String type ndarray

Return:
Numeric type ndarray
"""
    float_nd_arr = np.zeros(str_nd_arr.shape, dtype=np.float32)
    for i in range(str_nd_arr.shape[0]):
        for j in range(str_nd_arr.shape[1]):

            # try:
            float_nd_arr[i][j] = float(str_nd_arr[i][j])
            # except (ValueError, TypeError):
                # float_nd_arr[i][j] = 0.0
    return float_nd_arr


# ==================== Model Definition ====================
class plan_model(nn.Module):

"""
The general operation of the hierarchical double-tower architecture model

Architecture:
Original signal tower (300Hz): 1D-CNN -> Global Pooling
TS feature tower (gait level): LSTM/Transformer -> Final hidden state
Concatenation -> Fully Connected -> Classification (4 categories)
"""
    def __init__(self,
                 # Original signal tower parameters
                 original_in_channels=2,      # Left and Right Foot Dual Channels
                 original_seq_len=None,       # Length of the original signal (determined during runtime)
                 cnn_out_channels=64,
                 cnn_kernel_sizes=[3, 5, 7],

                 # TS Feature Tower Parameters
                 ts_feature_dim=12,           # TS Feature Dimension (Without Time Column)
                 ts_hidden_dim=128,
                 ts_num_layers=2,
                 ts_dropout=0.3,
                 # *** It is recommended to use LSTM first.
                 # use_transformer=True,
                 use_transformer=False,

                 # Classifier Parameters
                 num_classes=4,               # Four Categories: als/control/hunt/park
                 fusion_hidden_dim=256,
                 dropout=0.3):

        super().__init__()




        self.original_tower = nn.ModuleList()

        # Multi-scale 1D-CNN Layer
        in_channels = original_in_channels
        for i, kernel_size in enumerate(cnn_kernel_sizes):
            padding = kernel_size // 2
            self.original_tower.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, cnn_out_channels,
                             kernel_size = kernel_size, padding = padding),
                    nn.BatchNorm1d(cnn_out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size = 2, stride = 2)
                )
            )
            in_channels = cnn_out_channels

        #### It is not recommended to use global pooling here. It should be placed after the fusion.
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # [B, C, L] -> [B, C, 1]


        self.original_output_dim = cnn_out_channels * len(cnn_kernel_sizes)




        # === 2. TS Feature Tower, ===
        self.use_transformer = use_transformer

        if use_transformer:
            ## input: [L,B,C]
            ### If batch_first is set, it will be [B, L, C]
            encoder_layer = nn.TransformerEncoderLayer(
                d_model = ts_feature_dim,
                nhead = 4,
                dim_feedforward = ts_hidden_dim,
                dropout = ts_dropout,
                batch_first = True
            )
            ## TransformerEncoder is a stack of N encoder layers
            self.ts_encoder = nn.TransformerEncoder(encoder_layer, num_layers = ts_num_layers)
            self.ts_output_dim = ts_feature_dim


        else:
            # LSTM Encoder
            ## Outputs: output(N,L,D∗H_out), (h_n, c_n)
            ## inputs:(N,L,H_in)
            self.lstm = nn.LSTM(
                input_size = ts_feature_dim,
                hidden_size = ts_hidden_dim,
                num_layers = ts_num_layers,
                batch_first = True,
                dropout = ts_dropout if ts_num_layers > 1 else 0.0,
                bidirectional=True
            )
            self.ts_output_dim = ts_hidden_dim * 2



        self.fusion_layer = nn.Sequential(
            nn.Linear(self.original_output_dim + self.ts_output_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )


        self._initialize_weights()




    def _initialize_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:

                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, original_signal, ts_features):
        """
        Forward propagation

        Parameters:
        original_signal: Tensor, shape [B, C, L] (C=2: left and right feet, L=original length)
        ts_features: Tensor, shape [B, num_cycles, feature_dim], which is [B, L, C]

        Return: logits: Tensor, shape [B, num_classes]
        """

        batch_size = original_signal.size(0)

        # === 1. Original Signal Tower ===
        cnn_outputs = []
        flag = 0
        for cnn_layer in self.original_tower:
            if flag==0:
                # print ('original_signal_shape:', original_signal.shape)
                out = cnn_layer(original_signal)  # [B, cnn_out_channels, L/2/2/...]
                # print ('out0_shape:', out.shape)
            else:
                # print ('out1_shape:', out.shape)
                out = cnn_layer(out)


            out_pool = self.global_pool(out)       # [B, cnn_out_channels, 1]
            out_sq = out_pool.squeeze(-1)             # [B, cnn_out_channels]
            cnn_outputs.append(out_sq)

            flag +=1

        original_features = torch.cat(cnn_outputs, dim=1)  # [B, original_output_dim]
        # original_features = torch.tensor(cnn_outputs)



        # === 2. TS Feature Tower ===
        if self.use_transformer:
            # Transformer Encoding
            ts_encoded = self.ts_encoder(ts_features)  # [B, num_cycles, feature_dim]
            # Global average pooling
            ts_features_global = ts_encoded.mean(dim=1)  # [B, feature_dim]
        else:
            # LSTM Encoding
            lstm_out, (h_n, c_n) = self.lstm(ts_features)  # [B, num_cycles, hidden*2]
         # Use the output of the last time step and multiply it by2
             in both directions.
            # *** Actually, it is not recommended to consider
             ## using -1 or avg here. It is better to keep them until
               ### the fusion extraction process is completed. ***
            ts_features_global = lstm_out[:, -1, :]  # [B, hidden*2]

        # === Integration & Classification ===
        fused_features = torch.cat([original_features, ts_features_global], dim=1)
        logits = self.fusion_layer(fused_features)

        return logits







# ==================== Dataset Class ====================
class DoganGaitDataset(Dataset):
    """
    Gait Data Set Class

    Encapsulate the original signal and TS features
    """
    def __init__(self, original_signals, ts_features, labels):
        """
        Initialize the dataset

        Parameters:
        original_signals: ndarray, shape [N, C, L] Original signals
        ts_features: ndarray, shape [N, num_cycles, feature_dim] TS features
        labels: ndarray, shape [N] Labels

        """
        #
        self.original_signals = torch.FloatTensor(original_signals)
        self.ts_features = torch.FloatTensor(ts_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        return (
            self.original_signals[idx],  # [C, L]
            self.ts_features[idx],        # [num_cycles, feature_dim] [L,C]
            self.labels[idx]              # scalar
        )






# ==================== Data-Loading Function ====================

def load_dogen_data(dogen_path, train_ratio=0.5):
    """
    Load the gait data and perform preprocessing

    Parameters:
    dogen_path: Dataset path (directory where the .pkl file is located)
    train_ratio: Training set ratio (used for initial partitioning, and will be used for leave-one-out cross-validation later)

    Return: train_original_signals: ndarray, shape [N_train, C, L]
    train_ts_features: ndarray, shape [N_train, num_cycles, feature_dim], that is [N_train, L, C] train_labels: ndarray, shape [N_train]


    test_original_signals: ndarray, shape [N_test, C, L]
    test_ts_features: ndarray, shape [N_test, num_cycles, feature_dim], that is [N_test, L, C] test_labels: ndarray, shape [N_test]


    """

    print("=" * 80)
    print("Load gait data")
    print("=" * 80)

    # Concatenate all the data from the pkl files to form a batch (with the B dimension)
    lr_data_features = []
    ts_features = []
    basenames_l = []

    pkl_files = [f for f in os.listdir(dogen_path) if f.endswith('.pkl')]

    print(f"Found {len(pkl_files)} pkl files")

    for pkl_file in tqdm(pkl_files, desc="Load the pkl file"):
        with open(os.path.join(dogen_path, pkl_file), 'rb') as f:
            current_dogen_dict = pickle.load(f)



        if pkl_file.startswith('a'):
            backup_pkl_file = pkl_file[:3]
        elif pkl_file.startswith('c'):
            backup_pkl_file = pkl_file[:7]
        elif pkl_file.startswith('h') and pkl_file.startswith('p'):
            backup_pkl_file = pkl_file[:4]
        # else:
        #     pass

        # base_name = current_dogen_dict.get('subject', pkl_file.split('_')[0])
        base_name = current_dogen_dict.get('subject', backup_pkl_file)


        left_data = current_dogen_dict['left_data'].astype(np.float32).reshape(1, -1)

        right_data = current_dogen_dict['right_data'].astype(np.float32).reshape(1, -1)

        # [2, L]
        left_right_data = np.concatenate((left_data, right_data), axis=0)

        # Read the ts data and remove the time dimension (the header),
        # keeping only 12 dimensions.
        # The shape of ts_array is: [num_cycles, 13], and the 0th column
        # represents the time.
        ts_data = current_dogen_dict['ts13_array'] # [num_cycles, 13]


        ts_data = ndstrarr2ndarray(ts_data).astype(np.float32)
        # # Percentage Conversion, note that the time dimension
        ## has been removed and a -1 operation is required
        ts_percent2_float_indice = [(5-1),(6-1),(9-1),(10-1),(12-1)]


        for _, k  in enumerate(ts_percent2_float_indice):
            ts_data[:,k] = ts_data[:,k]/100


        #
        lr_data_features.append(left_right_data)

        #
        ts_features.append(ts_data)

        #
        basenames_l.append(base_name.lower())  #

    # Add a method to determine the minimum
     ## number of cycles (min_num_cycles) in
       ### the ts_features and print the result.
       #### Additionally, return and print the
         ##### shortest ts_features.
    print("\n Unify ts feature length...")
    ts_features, uni_num_cycles = unify_min_num_cycles(ts_features)
    # print ("The unified feature length is:", uni_num_cycles)

    #
    print("\n Convert to numpy array...")
    lr_data_features = np.array(lr_data_features)  # [N, 2, L]
    ts_features = np.array(ts_features)           # [N, num_cycles, 12]

    print(f"Shape of the original signal:{lr_data_features.shape}")
    print(f"TS features shape: {ts_features.shape}")

    print("\n Perform normalization...")
    print("  - Original signal: Max-Min normalization...")
    lr_data_features = max_min_global(lr_data_features)
    # lr_data_features = z_score_global(lr_data_features)

    print("  - TS features: Z-score normalization...")
    ts_features = z_score_global(ts_features)
    # ts_features = max_min_global(ts_features)



    print("\n Generate label...")
    labels_l = []
    for base_name in basenames_l:
        if 'als' in base_name:
            curr_label = 0
        elif 'control' in base_name or 'con' in base_name:
            curr_label = 1
        elif 'hunt' in base_name:
            curr_label = 2
        elif 'park' in base_name or 'pd' in base_name:
            curr_label = 3
        else:
            print(f"Warning: Unknown sample type {base_name}, skipping...")
            continue

        labels_l.append(curr_label)

    labels = np.array(labels_l).astype(np.int64)

    print(f"Label shape: {labels.shape}")




    ### Purely few-shot sample learning
    unique, counts = np.unique(labels, return_counts=True)
    label_names = ['ALS', 'Control', 'Huntington', 'Parkinson']
    print("\nCategory Distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label_names[label]}: {count} samples")

    # Divide the training set and the test set
    print(f"\nDivide the data into training and testing sets according to {train_ratio*100:.0f}:{(1-train_ratio)*100:.0f}...")

    n_samples = len(labels)
    # shuffle
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_original_signals = lr_data_features[train_indices]
    train_ts_features = ts_features[train_indices]
    train_labels = labels[train_indices]

    test_original_signals = lr_data_features[test_indices]
    test_ts_features = ts_features[test_indices]
    test_labels = labels[test_indices]

    print(f"\nTraining set: {len(train_labels)} samples")
    print(f"Test set: {len(test_labels)} samples")

    print("=" * 80)

    return (train_original_signals, train_ts_features, train_labels,
            test_original_signals, test_ts_features, test_labels)







# ==================== Training ====================

def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3,
                patience=10, device='cpu'):
    """
    Train the model

    Parameters:
    model: Model instance
    train_loader: Training data loader
    val_loader: Validation data loader
    epochs: Number of training epochs
    lr: Learning rate
    *** patience: Early stopping patience value, first time heard of it ***
    device: Device

    Return:
    train_losses: List of training losses
    val_losses: List of validation losses
    train_accs: List of training accuracies
    val_accs: List of validation accuracies
    best_model_state_dict: Best model parameters

    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=0.5, patience=5)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state_dict = None
    patience_counter = 0

    print("\n" + "=" * 80)
    print(f"Start training (for a total of {epochs} rounds)")
    print("=" * 80)

    #
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (original_signal, ts_features, labels) in enumerate(pbar):
            original_signal = original_signal.to(device)
            ts_features = ts_features.to(device)
            labels = labels.to(device)

            logits = model(original_signal, ts_features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            if (batch_idx + 1) % 10 == 0:
                acc = 100. * correct / total
                # Set/modify postfix (additional stats) with automatic formatting
                # based on datatype
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.2f}%'})

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # Verification stage
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for original_signal, ts_features, labels in val_loader:
                original_signal = original_signal.to(device)
                ts_features = ts_features.to(device)
                labels = labels.to(device)

                # Start the validation test for val.
                logits = model(original_signal, ts_features)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        '''

        '''
        scheduler.step(avg_val_loss)


        print(f'Epoch {epoch+1:3d}/{epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict().copy()
            patience_counter = 0
            print(f' ✓ The best model has been saved (Validation Accuracy: {val_acc:.2f}%)')
        else:
            patience_counter += 1




    # Restore the best model
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        print(f'\nBest validation accuracy: {best_val_acc:.2f}%')

    print("=" * 80)

    return train_losses, val_losses, train_accs, val_accs, best_model_state_dict





# ==================== Leave-One-Out Cross Validation ====================

def cross_validation_loo(model_class, original_signals, ts_features, labels,
                        model_kwargs, epochs=30, lr=1e-3, device='cpu'):
    """

    Parameters:
    model_class: Model class (e.g. plan_model)
    original_signals: Original signals [N, C, L]
    ts_features: TS features [N, num_cycles, feature_dim]
    labels: Labels [N]
    model_kwargs: Dictionary of model initialization parameters
    epochs: Number of training epochs per fold
    lr: Learning rate
    device: Device

    # First, divide each k-fold, and then instantiate the Dataset to create the dataset and the instantiation method



    Return:
    cv_results: Dictionary of cross-validation results

    """
    print("\n" + "=" * 80)
    print("Start with Leave-One-Out cross-validation")
    print("=" * 80)


    n_samples = len(labels)
    logo = LeaveOneGroupOut()

    # Create groups (one group for each sample)
    groups = np.arange(n_samples)

    # Store the result of each fold.
    fold_accs = []
    fold_losses = []
    all_train_losses = []
    all_val_losses = []


    fold_idx = 0
    n_splits = logo.get_n_splits(groups = groups)

    #
    print(f"Total discount: {n_splits}")
    print(f"Size of training set: {n_samples - 1} / Size of validation set: 1")

    for train_idx, val_idx in logo.split(original_signals, labels, groups=groups):
        fold_idx += 1
        print(f"\n{'='*80}")
        print(f"Fold {fold_idx} of {n_splits}")
        print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        print(f"{'='*80}")

        #
        '''
        ~
        '''
        train_original = original_signals[train_idx]
        train_ts = ts_features[train_idx]
        train_labels = labels[train_idx]

        val_original = original_signals[val_idx]
        val_ts = ts_features[val_idx]
        val_labels = labels[val_idx]




        # Based on each k-fold data, encapsulate and create the dataset in real time
        train_dataset = DoganGaitDataset(train_original, train_ts, train_labels)
        val_dataset = DoganGaitDataset(val_original, val_ts, val_labels)

        # Create each data loader
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)




        # Initialize the model
        model = model_class(**model_kwargs).to(device)


        '''
        After encapsulating the k-fold data divided by CV
        and then using train_model for training
        '''
        train_losses, val_losses, train_accs, val_accs, _ = train_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr,
            patience=5, device=device
        )

        #
        #
        # Validation set evaluation
        model.eval()
        with torch.no_grad():
            # The k-fold data divided by ndarray → torch.tensor()
            val_original_tensor = torch.FloatTensor(val_original).to(device)
            val_ts_tensor = torch.FloatTensor(val_ts).to(device)
            val_labels_tensor = torch.LongTensor(val_labels).to(device)

            logits = model(val_original_tensor, val_ts_tensor)
            val_loss = F.cross_entropy(logits, val_labels_tensor)
            _, predicted = logits.max(1)
            val_acc = accuracy_score(val_labels, predicted.cpu().numpy())

        fold_accs.append(val_acc)
        fold_losses.append(val_loss.item())
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

        print(f"Fold {fold_idx} result: Accuracy = {val_acc * 100:.2f}%, Loss = {val_loss:.4f}")

        del model
        # torch.cuda.empty_cache()

    # summary
    cv_results = {
        'fold_accuracies': fold_accs,
        'fold_losses': fold_losses,
        'mean_accuracy': np.mean(fold_accs),
        'std_accuracy': np.std(fold_accs),
        'mean_loss': np.mean(fold_losses),
        'std_loss': np.std(fold_losses),
        'all_train_losses': all_train_losses,
        'all_val_losses': all_val_losses
    }

    # Print the cross-validation results
    print("\n" + "=" * 80)
    print("Summary of Leave-One-Out Cross-Validation Results")
    print("=" * 80)
    print(f"Average accuracy: {cv_results['mean_accuracy'] * 100:.2f}% ± {cv_results['std_accuracy'] * 100:.2f}%")
    print(f"Average loss: {cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}")
    print(f"Accuracy of each fold: {[f'{acc * 100:.2f}%' for acc in fold_accs]}")
    print("=" * 80)

    return cv_results





# ==================== Test  ====================
def test_model(model, test_loader, device='cpu'):
    """
    Testing model

    Parameters:
    model: Trained model
    test_loader: Test data loader
    device: Device

    Return:
    test_results: Dictionary of test results
    """
    print("\n" + "=" * 80)
    print("Model testing")
    print("=" * 80)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()


    with torch.no_grad():
        for original_signal, ts_features, labels in test_loader:
            original_signal = original_signal.to(device)
            ts_features = ts_features.to(device)
            labels = labels.to(device)

            logits = model(original_signal, ts_features)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    test_accuracy = accuracy_score(all_labels, all_preds)
    avg_test_loss = total_loss / len(test_loader)

    cm = confusion_matrix(all_labels, all_preds)

    class_names = ['ALS', 'Control', 'Huntington', 'Parkinson']
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, output_dict=True)

    print(f"Test set accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test set average loss: {avg_test_loss:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    #
    test_results = {
        'accuracy': test_accuracy,
        'loss': avg_test_loss,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

    print("=" * 80)

    return test_results





# ==================== Visualization ====================

def plot_training_results(train_losses, val_losses, train_accs, val_accs,
                         cv_results=None, test_results=None):


    """
    Visualized training results

    Parameters:
    train_losses: List of training losses
    val_losses: List of validation losses
    train_accs: List of training accuracies
    val_accs: List of validation accuracies
    cv_results: Cross-validation results (optional)
    test_results: Test results (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss Curve
    axes[0, 0].plot(train_losses, label='Training loss', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(val_losses, label='Verification loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('Training and validation loss curves', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy curve
    axes[0, 1].plot(train_accs, label='Training accuracy rate', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(val_accs, label='Verification accuracy rate', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Training and val accuracy curves', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)



    # 3. Cross-validation results
    if cv_results is not None:
        fold_nums = np.arange(1, len(cv_results['fold_accuracies']) + 1)
        axes[1, 0].bar(fold_nums, cv_results['fold_accuracies'],
                      color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(y=cv_results['mean_accuracy'], color='red',
                          linestyle='--', linewidth=2, label=f'Average value: {cv_results["mean_accuracy"]*100:.2f}%')
        axes[1, 0].set_title('The accuracy rate of leave-one-out cross-validation', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'No cross-validation was conducted.',
                        ha='center', va='center', fontsize=14,
                        transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Cross-validation results', fontsize=12, fontweight='bold')



    # 4. Summary of Test Results
    if test_results is not None:
        # Confusion matrix heatmap
        cm = test_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                    xticklabels=['ALS', 'Control', 'Hunt', 'Park'],
                    yticklabels=['ALS', 'Control', 'Hunt', 'Park'])
        axes[1, 1].set_title(f'Test set confusion matrix (Acc: {test_results["accuracy"]*100:.2f}%)',
                           fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Prediction label')
        axes[1, 1].set_ylabel('True label')
    else:
        axes[1, 1].text(0.5, 0.5, 'Not tested',
                        ha='center', va='center', fontsize=14,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Test results', fontsize=12, fontweight='bold')

    plt.tight_layout()

    plt.savefig('D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_boards_model_pred_results\dual_tower_dogen_results.png', dpi=300, bbox_inches='tight')
    # print ('---1---')
    # plt.show()
    plt.cla()
    # print ('---2---')
    print("The visualization results have been saved to: dual_tower_dogen_results.png")
    plt.close()
    # print ('---3---')




# ==================== Main function ====================

def main():
    """
    Main function - Complete training-validation-testing process
    """
    print("\n" + "=" * 80)
    print("Hierarchical dual-tower architecture for gait classification")
    print("=" * 80)

    # ==================== 1. Load data ====================
    print("\n[Step 1/8] Load data...")
    print("-" * 80)

    #
    ## *** Modify the path
    dogen_path = r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\gait64_neodegen_pkls'

    # If the dataset does not exist, use simulated data for demonstration.
    if not os.path.exists(dogen_path):
        print(f"Warning: The dataset path does not exist.: {dogen_path}")
        print("Carry out the demonstration using simulated data...")
        # Create simulated data
        num_samples = 100
        seq_len = 300
        num_cycles = 50

        # Simulate the original signal [N, 2, L]
        train_original_signals = np.random.randn(num_samples, 2, seq_len).astype(np.float32)
        test_original_signals = np.random.randn(20, 2, seq_len).astype(np.float32)

        # Simulated TS characteristics [N, num_cycles, 12]
        train_ts_features = np.random.randn(num_samples, num_cycles, 12).astype(np.float32)
        test_ts_features = np.random.randn(20, num_cycles, 12).astype(np.float32)

        # Simulation tag
        train_labels = np.random.randint(0, 4, num_samples)
        test_labels = np.random.randint(0, 4, 20)

        print("Simulation data has been generated!")

    else:

        #
        (train_original_signals, train_ts_features, train_labels,
         test_original_signals, test_ts_features, test_labels) = load_dogen_data(
            dogen_path, train_ratio=0.5
        )


    # ==================== 2. Encapsulated dataset ====================
    print("\n [Step 2/8] Package the dataset...")
    print("-" * 80)

    # Create the training dataset
    train_dataset = DoganGaitDataset(train_original_signals, train_ts_features, train_labels)
    test_dataset = DoganGaitDataset(test_original_signals, test_ts_features, test_labels)

    # Divide the validation set from the training set
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Create Data Loader
    batch_size = 8
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training set: {len(train_subset)} samples")
    print(f"Validation set: {len(val_subset)} samples")
    print(f"Test set: {len(test_dataset)} samples")



    # ==================== 3. Initialize the model ====================
    print("\n [Step 3/8] Initialize the model...")
    print("-" * 80)



    # Model Parameters
    # *** Parameters Modified by Different Methods
    model_kwargs = {
        'original_in_channels': 2,
        'cnn_out_channels': 64,
        'cnn_kernel_sizes': [3, 5, 7],
        'ts_feature_dim': 12,
        'ts_hidden_dim': 128,
        'ts_num_layers': 2,
        'use_transformer': True,  # True: Transformer, False: LSTM
        'num_classes': 4,
        'fusion_hidden_dim': 256,
        'dropout': 0.3
    }

    model = plan_model(**model_kwargs).to(device)

    # Parameter information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters of the model: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


    # ==================== 4. Train the model ====================
    print("\n [Step 4/8] Training the model...")
    print("-" * 80)
    epoch_train_num = 10

    train_losses, val_losses, train_accs, val_accs, best_model_state = train_model(
        model, train_loader, val_loader, epochs=epoch_train_num, lr=1e-3,
        patience=10, device=device
    )




    # ==================== 5. Leave-one-out cross-validation ====================
    print("\n [Step 5/8] Keep one-fold cross-validation...")
    print("-" * 80)
    # epoch_train_val = 8
    epoch_train_val = 3


    if len(train_labels) <= 44:
        cv_results = cross_validation_loo(
            plan_model, train_original_signals, train_ts_features, train_labels,
            model_kwargs, epochs=20, lr=1e-3, device=device
        )
    else:
        print(f"The number of samples ({len(train_labels)}) is too large, so the holdout cross-validation method is skipped.")
        cv_results = None





    # ==================== 6. Testing model ====================
    print("\n [Step 6/8] Test the model...")
    print("-" * 80)

    # Restore the optimal model parameters
    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    test_results = test_model(model, test_loader, device=device)




    # ==================== 7. Visualized results====================
    print("\n [Step 7/8] Visualize the results...")
    print("-" * 80)

    plot_training_results(train_losses, val_losses, train_accs, val_accs,
                          cv_results, test_results)

    # print ('---4---')



    # ==================== 8. Save the model ====================
    print("\n [Step 8/8] Save the model...")
    print("-" * 80)

    # *** Modify the path
    save_path = r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\neodogen_models\two_tower_dogen_gait_classifier.pth'
    print ('***best_model_state***:', model.state_dict(best_model_state))
    torch.save({
        'model_state_dict:':model.state_dict(best_model_state),
        # 'model_state_dict': best_model_state,
        # 'model_state_dict': model.state_dict().copy(),
        'model_kwargs': model_kwargs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_results': test_results,
        'cv_results': cv_results
    }, save_path)

    print(f"The model has been saved to: {save_path}")




    # ==================== Summary ====================
print("\n" + "=" * 80)
print("Training process completed!" )
print("=" * 80)
print(f"\n📊 Training set accuracy: {train_accs[-1]:.2f}%")
print(f"📊 Validation set accuracy: {val_accs[-1]:.2f}%")
print(f"📊 Test set accuracy: {test_results['accuracy']*100:.2f}%")

    if cv_results is not None:
        print(f"📊 Cross-validation accuracy: {cv_results['mean_accuracy']*100:.2f}% ± {cv_results['std_accuracy']*100:.2f}%")
    print("\n" + "=" * 80)

    return model, train_losses, val_losses, train_accs, val_accs, test_results




if __name__ == "__main__":
    #
    model, train_losses, val_losses, train_accs, val_accs, test_results = main()

    #
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n GPU memory has been cleared.")

    #

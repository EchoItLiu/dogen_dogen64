
"""
****  Two types of cross-attention methods ****
## I. One approach is to use the original signal as the
query (Q) [L], and the time-series signal as the key/value [Source_length]
, aiming to obtain highly discriminative and strongly correlated
information~~

# II. The key idea of our approach on degen-based operations is naive, that
is to say, enable the model to learn "at what time points, which gait features
should be focused on":
  #### For example: At the moment of foot react (GRF_signal), focus on
  "stance" and "double support" phases (stance/support). ####

 interpolate VS info_entropy

@、Interpolate is used first.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.interpolate as interpolate
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

    """
    'previous' or 'linear'
    """

def interpolate_ts_features(ts_samples, elaspsed_time, signal_length, sample_rate=300):
    """
    'previous' or 'linear'
    """

    singal_time = np.arange(signal_length) / sample_rate
    ts_interpolate = np.zeros((signal_length,12))

    # print ('elas_type:',  elaspsed_time.dtype)
    # print ('ts_sample_type:', ts_samples[:,0].dtype)
    # print ('ts_samples_before:', ts_samples)
    # print ('ts_samples_before_shape:', ts_samples.shape)
    for col in range(ts_interpolate.shape[1]):
        interp_func = interpolate.interp1d(
        elaspsed_time,
        ts_samples[:,col],
        # kind = 'previous',
        kind = 'linear',
        bounds_error = False,
        fill_value = 'extrapolate'
        )
        ts_interpolate[:,col] = interp_func(singal_time)

    #
    # print (":", ts_interpolate[:,-1])

    # print ('ts_interpolate_shape:', ts_interpolate.shape)
    # ts_interploate_mask_nan = np.isnan(ts_interpolate)
    # stats_nan = np.sum(ts_interploate_mask_nan)
    # print ("stats_nan:", stats_nan)
    return ts_interpolate


def max_min_global(features):

    min_val = np.min(features)
    max_val = np.max(features)

    if max_val - min_val < 1e-8:
        return features - min_val

    normalized = (features - min_val) / (max_val - min_val)
    return normalized



def z_score_global(features):
    #
    mean_val = np.mean(features)
    std_val = np.std(features)

    if std_val < 1e-8:
        return features - mean_val

    normalized = (features - mean_val) / std_val
    return normalized



def ndstrarr2ndarray(str_nd_arr):
    #
    float_nd_arr = np.zeros(str_nd_arr.shape, dtype=np.float32)
    for i in range(str_nd_arr.shape[0]):
        for j in range(str_nd_arr.shape[1]):
            float_nd_arr[i][j] = float(str_nd_arr[i][j])
    return float_nd_arr



class plan_model(nn.Module):
    """
    #
    1、strategy1(long-short |  info_entropy to be continued)

    2、 strategy2
    #

    """



    def __init__(self,
                 use_cross_attention_strategy_serial='A40',
                 dim_single=2,
                 dim_ts=12,

                 mutHeaAtt_embeded_dim=32,
                 mutHeadAtt_head_num=8,
                 # window_size = 128,
                 # down_target_seq_len = 6000,
                 # down_target_seq_len = 3000,
                 down_target_seq_len = 300,

                 d_model=128,

                 dropout=0.3,
                 num_classes=4,
                 ):

        super().__init__()
        '''
        '''
        self.use_crossAtt_strategy_se = use_cross_attention_strategy_serial
        # self.windos_size = window_size
        self.down_target_seq_len = down_target_seq_len

        # A40 strategy: The original signal serves as the query,
         ##  while the ts time-series signal acts as the key and value.
        if self.use_crossAtt_strategy_se=='A40':

            # query
            self.proj_singnal = nn.Linear(dim_single,mutHeaAtt_embeded_dim)
            # key | value，share weights
            self.proj_ts = nn.Linear(dim_ts,mutHeaAtt_embeded_dim)

            self.attention = nn.MultiheadAttention(
            embed_dim = mutHeaAtt_embeded_dim,
            num_heads = mutHeadAtt_head_num,
            batch_first = True
            )


            # transition layer
            self._transition_layer_A40 = nn.Conv1d(
            in_channels=mutHeaAtt_embeded_dim,
            out_channels=64,
            kernel_size=7
            )


        elif self.use_crossAtt_strategy_se=='B100':

            # The TS feature (length L) is projected onto
             ## the **Query.
            self.ts_query = nn.Linear(dim_ts, d_model)

            '''
            Obtain Value/Key without sharing weights
            '''
            # The signal (source length S) is mapped to **Key，
            self.singal_key = nn.Linear(dim_ts, d_model)

            # The signal (source length S) is mapped to **Value,
            self.singal_value = nn.Linear(dim_ts,d_model)

            # transition layer
            self._transition_layer_B100 = nn.Conv1d(
            in_channels=d_model + dim_single,
            out_channels=64,
            kernel_size=7
            )

        # [N,C,L] → [N,C,L]
        self.bn = nn.BatchNorm1d(64)

        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64,num_classes)
        )
        #
        #
        #
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


    # def create_sparse_attention_mask(self, seq_len, window_size, device):
    #     """
    #     Create sparse attention mask
    #
    #     Args:
    #         seq_len: Sequence length
    #         window_size: Attention window size
    #         device: Tensor device
    #
    #     Returns:
    #         mask: [seq_len, seq_len]，True Indicates that it needs to be masked (not involved in attention calculation)
    #     """
    #     mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=device)
    #
    #     for i in range(seq_len):
    #         start = max(0, i - window_size // 2)
    #         end = min(seq_len, i + window_size // 2 + 1)
    #         mask[i, start:end] = False
    #
    #     return mask


    '''

    ... Here, an high efficient chunking method is
    introduced (which can save memory by writing
    to disk and calculate the score)...
    to be continued...
    def chunk_strategy():
        return 0
        pass
    '''

    def forward(self, original_signal, ts_features):

        batch_size = original_signal.size(0)
        # print ('original_signal_shape:', original_signal.shape)
        # (N,C,L) → (N,L,C)
        original_signal_input = torch.permute(original_signal,(0,2,1))

        # print ('original_signal_input:', original_signal_input)
        # print ('original_signal_input_shape:', original_signal_input.shape)

        if self.use_crossAtt_strategy_se=='A40':
            # [N,L,C]
            print ('--A40--')
            query = self.proj_singnal(original_signal_input)
            # print ('query_A40:', query)
            # [N,S,C]
            key = value = self.proj_ts(ts_features)
            '''
            '''

            # print ('query_dtype:', query.dtype)
            # print ('key_dtype:', key.dtype)
            # print ('value_dtype:', value.dtype)

            # print ('query_shape:', query.shape)
            # print ('key_shape:', key.shape)
            # print ('value_shape:', value.shape)

            query_down = F.interpolate(query.permute(0, 2, 1), size = self.down_target_seq_len, mode='linear').permute(0, 2, 1)
            key_down = F.interpolate(key.permute(0, 2, 1), size = self.down_target_seq_len, mode='linear').permute(0, 2, 1)
            value_down = F.interpolate(value.permute(0, 2, 1), size = self.down_target_seq_len, mode='linear').permute(0, 2, 1)


            # print ('querydown_shape:', query_down.shape)
            # print ('keydown_shape:', key_down.shape)
            # print ('valuedown_shape:', value_down.shape)

            # Use mask_attn to generate sparse attention masks
            # attn_mask  = self.create_sparse_attention_mask(query.shape[1], self.windos_size, query.device)
            # attn_out, _ = self.attention(query, key, value)
            # attn_out, _ = self.attention(query, key, value, attn_mask = attn_mask)
            attn_out, _ = self.attention(query_down, key_down, value_down)

            # Upsample to original length
            attn_out = F.interpolate(attn_out.permute(0, 2, 1), size=query.shape[1], mode='linear').permute(0, 2, 1)

            # Conv1d
            out = self._transition_layer_A40(torch.permute(attn_out, (0,2,1)))
            # bn
            out = self.bn(out)
            # cls
            logit_A = self.classifier(out)

            return logit_A


        elif self.use_crossAtt_strategy_se=='B100':
            """
            signal: (B, 2, T) [Left foot, Right foot]
            ts_interploted: (B, T, 12) interpolated and expanded TS features
            """
            print ('--B100--')
            # Calculate the query&key
            query = self.ts_query(ts_features) # [B,L,C]
            key = self.singal_key(torch.permute(original_signal,(0,2,1))) # [B,C,L] → [B,L,C]

            # Weighted signal
            attn_scores = torch.matmul(query, key.transpose(-2,-1)) # [B,L,L]
            attn_weights = F.softmax(attn_scores, dim=-1)

            # Weighted signal
            value = self.singal_value(torch.permute(original_signal,(0,2,1))) # [B,C,L] → [B,L,C]
            # (K*Q) @ V
            attended_singal = torch.matmul(attn_weights, value)  # (B, T, 128)

            # fusion
            signal_enhanced = torch.cat([
            original_signal,
            torch.permute(attended_singal, (0,2,1))],
            dim = 1
            )

            # 中间层
            features = self._transition_layer_B100(signal_enhanced)
            out = self.bn(features)
            # 最终分类
            logit_B = self.classifier(out)

            return logit_B



class DoganGaitDataset(Dataset):

    def __init__(self, original_signals, ts_features, labels):
        #
        self.original_signals = torch.FloatTensor(original_signals)
        self.ts_features = torch.FloatTensor(ts_features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        return (
            self.original_signals[idx],  # [C, L]
            self.ts_features[idx],        # [num_cycles, feature_dim]
            self.labels[idx]              # scalar
        )



def load_dogen_data(dogen_path, train_ratio=0.5):

    print("=" * 80)
    print("=" * 80)

    lr_data_features = []
    ts_features = []
    basenames_l = []

    pkl_files = [f for f in os.listdir(dogen_path) if f.endswith('.pkl')]


    for pkl_file in tqdm(pkl_files, desc="Load the pkl file..."):
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
        left_right_data = np.concatenate((left_data, right_data), axis=0)


        left_right_data_length = left_right_data.shape[1]


        ts_data = current_dogen_dict['ts13_array'][:, 1:]  # [num_cycles, 12]
        elapsed_time = current_dogen_dict['ts13_array'][:,0] # Elapsed Time (sec)
        sample_rate = current_dogen_dict['sample_rate']

        ts_data = ndstrarr2ndarray(ts_data).astype(np.float32)
        ts_percent2_float_indice = [(5-1),(6-1),(9-1),(10-1),(12-1)]

        for _, k  in enumerate(ts_percent2_float_indice):
            ts_data[:,k] = ts_data[:,k]/100


        ts_data_interp = interpolate_ts_features(ts_data, elapsed_time, left_right_data_length, sample_rate)

        #
        lr_data_features.append(left_right_data)
        ts_features.append(ts_data_interp)
        basenames_l.append(base_name.lower())


    lr_data_features = np.array(lr_data_features)  # [N, 2, L]
    ts_features = np.array(ts_features)           # [N, num_cycles, 12]


    lr_data_features = max_min_global(lr_data_features)
    # lr_data_features = z_score_global(lr_data_features)

    ts_features = z_score_global(ts_features)
    # ts_features = max_min_global(ts_features)


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
        # else:
            # pass
            # continue

        labels_l.append(curr_label)

    labels = np.array(labels_l).astype(np.int64)


    unique, counts = np.unique(labels, return_counts=True)
    label_names = ['ALS', 'Control', 'Huntington', 'Parkinson']
    for label, count in zip(unique, counts):
        print(f"  {label_names[label]}: {count} samples")

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

    print("=" * 80)

    return (train_original_signals, train_ts_features, train_labels,
            test_original_signals, test_ts_features, test_labels)







def train_model(model, train_loader, val_loader, epochs=50, lr=1e-3,
                patience=10, device='cpu'):

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
    print(f"Start training (total {epochs} epochs))")
    print("=" * 80)


    #
    for epoch in range(epochs):
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

                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.2f}%'})

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0


        with torch.no_grad():
            for original_signal, ts_features, labels in val_loader:
                original_signal = original_signal.to(device)
                ts_features = ts_features.to(device)
                labels = labels.to(device)

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
            print(f'  ✓ The best model has been saved (Val Acc: {val_acc:.2f}%)')
        else:
            patience_counter += 1
        '''
        '''



    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        print(f'\n Optimal verification accuracy rate {best_val_acc:.2f}%')
    print("=" * 80)

    return train_losses, val_losses, train_accs, val_accs, best_model_state_dict





def cross_validation_loo(model_class, original_signals, ts_features, labels,
                        model_kwargs, epochs=30, lr=1e-3, device='cpu'):

    print("\n" + "=" * 80)
    print("start cross-validation... (Leave-One-Out)")
    print("=" * 80)


    n_samples = len(labels)
    logo = LeaveOneGroupOut()

    groups = np.arange(n_samples)

    fold_accs = []
    fold_losses = []
    all_train_losses = []
    all_val_losses = []


    fold_idx = 0
    n_splits = logo.get_n_splits(groups = groups)

    #
    print(f"fold_num: {n_splits}")
    print(f"Training set size: {n_samples-1} / Validation set size: 1")

    for train_idx, val_idx in logo.split(original_signals, labels, groups=groups):
        fold_idx += 1
        print(f"\n{'='*80}")
        print(f"{fold_idx}/{n_splits} fold")
        print(f"Train sample num: {len(train_idx)}, Validation sample num: {len(val_idx)}")
        print(f"{'='*80}")


        train_original = original_signals[train_idx]
        train_ts = ts_features[train_idx]
        train_labels = labels[train_idx]

        val_original = original_signals[val_idx]
        val_ts = ts_features[val_idx]
        val_labels = labels[val_idx]


        train_dataset = DoganGaitDataset(train_original, train_ts, train_labels)
        val_dataset = DoganGaitDataset(val_original, val_ts, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


        model = model_class(**model_kwargs).to(device)


        train_losses, val_losses, train_accs, val_accs, _ = train_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr,
            patience=5, device=device
        )

        #
        #
        model.eval()
        with torch.no_grad():
            #
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

        print(f"The results of {fold_idx} fold: Acc = {val_acc*100:.2f}%, Loss = {val_loss:.4f}")

        #
        del model
        # torch.cuda.empty_cache()

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

    print("\n" + "=" * 80)
    print("Summary of Leave-One-Out Cross-Validation Results")
    print("=" * 80)
    print(f"avg accuracy: {cv_results['mean_accuracy']*100:.2f}% ± {cv_results['std_accuracy']*100:.2f}%")
    print(f"avg loss: {cv_results['mean_loss']:.4f} ± {cv_results['std_loss']:.4f}")
    print(f"acc for every fold: {[f'{acc*100:.2f}%' for acc in fold_accs]}")
    print("=" * 80)

    return cv_results





def test_model(model, test_loader, device='cpu'):

    print("\n" + "=" * 80)
    print("Testing")
    print("=" * 80)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    '''

    '''
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


    #
    print(f"acc for test: {test_accuracy*100:.2f}%")
    print(f"acc for avg loss: {avg_test_loss:.4f}")
    print("\n Cconfusion Matrix:")
    print(cm)
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names = class_names))
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





def plot_training_results(train_losses, val_losses, train_accs, val_accs,
                         cv_results=None, test_results=None):


    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss Curve
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(val_losses, label='Validation Loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('Loss Curve for Train and Val', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Curve of Acc
    axes[0, 1].plot(train_accs, label='Train Acc', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(val_accs, label='Validation Acc', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Acc Curve for Train and Val', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)



    # 3. The results of cross-validation(CV)
    if cv_results is not None:
        fold_nums = np.arange(1, len(cv_results['fold_accuracies']) + 1)
        axes[1, 0].bar(fold_nums, cv_results['fold_accuracies'],
                      color='steelblue', alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(y=cv_results['mean_accuracy'], color='red',
                          linestyle='--', linewidth=2, label=f'AVG: {cv_results["mean_accuracy"]*100:.2f}%')
        axes[1, 0].set_title('Accuracy of leave-one-out-based cross-validation', fontsize=12, fontweight='bold')
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
        # Confusion Matrix Heatmap
        cm = test_results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                    xticklabels=['ALS', 'Control', 'Hunt', 'Park'],
                    yticklabels=['ALS', 'Control', 'Hunt', 'Park'])
        axes[1, 1].set_title(f'Confusion matrix of the test set (Acc: {test_results["accuracy"]*100:.2f}%)',
                           fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Prediction label')
        axes[1, 1].set_ylabel('Ground-Truth label')
    else:
        axes[1, 1].text(0.5, 0.5, 'Not tested',
                        ha='center', va='center', fontsize=14,
                        transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Test results', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_boards_model_pred_results\cross_att_st2_dogen_results.png', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.cla()
    plt.close()
    #
    print("The visualization results have been saved to: cross_att_st2_dogen_results.png")





def main():

    print("\n" + "=" * 80)
    print("=" * 80)

    # ==================== 1. Loading data ====================
    print("\n[Step 1/8] Load gait data...")
    print("-" *80)

    dogen_path = r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\gait64_neodegen_pkls'

    if not os.path.exists(dogen_path):
        print("Carrying out the demonstration using simulated data...")
        num_samples = 100
        seq_len = 300
        num_cycles = 50

        # [N, 2, L]
        train_original_signals = np.random.randn(num_samples, 2, seq_len).astype(np.float32)
        test_original_signals = np.random.randn(20, 2, seq_len).astype(np.float32)

        #  [N, num_cycles, 12]
        train_ts_features = np.random.randn(num_samples, num_cycles, 12).astype(np.float32)
        test_ts_features = np.random.randn(20, num_cycles, 12).astype(np.float32)

        # simulated label
        train_labels = np.random.randint(0, 4, num_samples)
        test_labels = np.random.randint(0, 4, 20)

    else:
        (train_original_signals, train_ts_features, train_labels,
         test_original_signals, test_ts_features, test_labels) = load_dogen_data(
            dogen_path, train_ratio=0.5
        )



    # ==================== 2. Encapsulated dataset ====================
    print("\n [Step 2/8] Package the dataset...")
    print("-" * 80)


    train_dataset = DoganGaitDataset(train_original_signals, train_ts_features, train_labels)
    test_dataset = DoganGaitDataset(test_original_signals, test_ts_features, test_labels)


    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    #
    #
    #
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )


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

    model_kwargs = {
        'use_cross_attention_strategy_serial': 'A40',
        # 'use_cross_attention_strategy_serial': 'A100',
        'dim_single': 2,
        # 'dim_single': 2,
        'dim_ts': 12,
        'mutHeaAtt_embeded_dim': 256,
        # 'mutHeaAtt_embeded_dim': 128,
        # 'mutHeaAtt_embeded_dim': 64,
        # 'mutHeaAtt_embeded_dim': 8,

        'mutHeadAtt_head_num': 8,
        # 'window_size': 128,
        # 'down_target_seq_len': 6000,
        # 'down_target_seq_len': 3000,
        'down_target_seq_len': 300,

        'd_model': 128,
        'dropout': 0.3,
        'num_classes': 4,
    }

    # select on/off strategy for 'A40' or 'B100'
    model = plan_model(**model_kwargs).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {total_params:,}")
    print(f"The number of trainable parameters: {trainable_params:,}")



    # ==================== 4. Training ====================
    print("\n [Step 4/8] Train the model...")
    print("-" * 80)

    train_losses, val_losses, train_accs, val_accs, best_model_state = train_model(
        model, train_loader, val_loader, epochs=50, lr=1e-3,
        patience=10, device=device
    )



    # ==================== 5. Leave-one-out cross-validation ====================
    print("\n [Step 5/8] Keep one-fold cross-validation...")
    print("-" * 80)


    if len(train_labels) <= 20:
        cv_results = cross_validation_loo(
            plan_model, train_original_signals, train_ts_features, train_labels,
            model_kwargs, epochs=20, lr=1e-3, device=device
        )
    else:
        print(f"The number of samples ({len(train_labels)}) is too large, so the holdout method cross-validation is skipped.")
        cv_results = None



    # ==================== 6. Testing ====================
    print("[Step 6/8] Test the model...")
    print("-" * 80)

    if best_model_state is not None:
        model.load_state_dict(best_model_state)


    test_results = test_model(model, test_loader, device=device)



    # ==================== 7. Visualized results ====================
    print("[Step 7/8] Visualized results...")
    print("-" * 80)

    plot_training_results(train_losses, val_losses, train_accs, val_accs,
                          cv_results, test_results)



    # ==================== 8. Save the model ====================
    print("[Step 8/8] Save the model...")
    print("-" * 80)

    save_path = r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\neodogen_models\cross_att_st2_dogen_gait_classifier.pth'
    torch.save({
        'model_state_dict': model.state_dict(best_model_state),
        'model_kwargs': model_kwargs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_results': test_results,
        'cv_results': cv_results
    }, save_path)

    print(f"The model has been saved to: {save_path}")



    print("\n" + "=" * 80)
    print("Training process completed!")
    print("=" * 80)
    print(f"\n📊 Training set accuracy rate: {train_accs[-1]:.2f}%")
    print(f"📊 Validation set accuracy rate: {val_accs[-1]:.2f}%")
    print(f"📊 Accuracy of the test set: {test_results['accuracy']*100:.2f}%")

    if cv_results is not None:
        print(f"📊 Cross-validation accuracy rate: {cv_results['mean_accuracy']*100:.2f}% ± {cv_results['std_accuracy']*100:.2f}%")

    print("\n" + "=" * 80)

    return model, train_losses, val_losses, train_accs, val_accs, test_results




if __name__ == "__main__":
    model, train_losses, val_losses, train_accs, val_accs, test_results = main()

    #
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n nGPU memory has been cleared")

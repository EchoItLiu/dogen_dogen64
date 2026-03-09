
'''

1 Graph Convolution；

2 The construction and then initialization of 
edge weights are not included here.

3 ...

'''

'''
# Core parameters: Signal sampling ratio
# Number of control nodes, balance speed and accuracy

signal_subsample_ratio = 0.05  # 5% sampling, 4171 nodes ()
# signal_subsample_ratio = 0.02  # 2% sampling, 1668 nodes (faster)
# signal_subsample_ratio = 0.01  # 1% sampling, 834 nodes (fastest)

'''





"""
Graph Neural Network (GNN) Gait Classification - Real Dataset Version
 =======================================
Fully compatible with the structure of your actual data set:
- Original signal: (83421,) Combined left and right foot pressure signals
- TS features: (310, 13) 310 gait cycles × 13 features
 (time + 12 gait features) 
"""

"""
Graph structure design:
- Signal nodes: Pressure values at each time point (83421 nodes)
- TS feature nodes: 12 gait features (13 nodes, excluding the time column) 

(MS)Edge types:
1. Time edge: Signals connecting adjacent time points (after downsampling)
2. Signal-TS feature edge: Connected based on gait cycle alignment
3. TS feature-to-feature edge: Connecting related TS features 
"""

'''
num_gnn_layers = 2  # It is recommended to have 2-3 layers.
# A very deep structure is prone to overfitting. GNNs usually
 do not require extremely deep structures.
'''

"""
**************************
# Define the relevant feature pairs (used for constructing the edges between features) '''
This position is of considerable research value. 

When writing a paper there is indeed a strong theoretical basis to 
explain~~~. the interpretability of graph networks
This might be demonstrated here. 

Here, the TS features are combined and classified only into 
left-foot-left-foot and right-foot-right-foot.
If the left-right feet are mixed and different (support, phase, 
stance, swing) are adopted, it might be another study, possibly.
It might increase somewhat and be more explanatory.
"""


"""
Subsequently, the Graph Attention Network, GAT... will be added... 
(to be continued...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
from torch_geometric.data import Data, Batch
## 
# from torch_geometric.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader as GeoDataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ==================== Global Settings ====================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Use the equipment: {device}")

# Set the random seed
torch.manual_seed(42)
np.random.seed(42)


# ==================== Normalization  ====================

def max_min_global(features):
    """Global maximum-minimum normalization"""
    min_val = np.min(features)
    max_val = np.max(features)

    if max_val - min_val < 1e-8:
        return features - min_val

    normalized = (features - min_val) / (max_val - min_val)
    return normalized


def z_score_global(features):
    """Global Z-score normalization"""
    mean_val = np.mean(features)
    std_val = np.std(features)

    if std_val < 1e-8:
        return features - mean_val

    normalized = (features - mean_val) / std_val
    return normalized


# ==================== Graph Construction ====================

class GaitGraphBuilder:
    """
    Gait Graph Builder - Real Dataset Version 

    Optimizing for your dataset: 

    Here is a single piece of data (Per data): 

    - Original Signal: (83421,) Left and Right Foot Pressure Signals
    - TS Feature: (310, 13) 310 Gait Cycles × 13 Features

    """

    '''
    Currently, all three methods for establishing the edge
    policy have added "on/off" options.
    '''

    def __init__(self,
                 signal_subsample_ratio=0.05,  # Signal sampling ratio (control chart scale)
                 ts_feature_dim=12,            # TS feature dimension (excluding the time column)
                 add_time_edges=True,         # Does the graph network add or not add the time edge connection?
                 add_signal_ts_edges=True,     # Whether to add the signal-TS feature edge in the graph network
                 add_ts_feature_edges=True):   # Should the edges between TS features be added?
        """
        Initializing graph builder
        """

        self.signal_subsample_ratio = signal_subsample_ratio
        self.ts_feature_dim = ts_feature_dim
        self.add_time_edges = add_time_edges
        self.add_signal_ts_edges = add_signal_ts_edges
        self.add_ts_feature_edges = add_ts_feature_edges



        # Define the feature names of the TS data 
        # (based on your dataset)
        # TS data structure: (310, 13)
        # The 0th column: Time (in seconds)
        # The 1st - 12th columns: 12 gait features

        ### This does not include the elapsed time
        self.ts_feature_names = [
            'stride_time_1',      # Column 1: Left step duration
            'stride_time_2',      # Column 2: Right step duration
            'swing_1',            # Column 3: Left Swing Phase
            'swing_2',            # Column 4: Right Swing Phase
            'swing_pct_1',        # Column 5: Percentage of Left Swing Phase
            'swing_pct_2',        # Column 6: Percentage of Right Swing Phase
            'stance_1',           # Column 7: Left stance 
            'stance_2',           # Column 8: Right stance
            'stance_pct_1',       # Column 9: Percentage of Left Stance
            'stance_pct_2',       # Column 10: Percentage of Right Stance
            'double_support',    # Column 11: Double Support Phase
            'double_support_pct'  # Column 12: Percentage of Double Support Phase
        ]

        # Define the relevant feature pairs (used to construct the edges between features)
    
        self.ts_feature_pairs = [
            ('stride_time_1', 'stride_time_2'),      # Left and Right Strides for a Long Time
            ('swing_1', 'swing_2'),                  # Left-right swing phase
            ('swing_pct_1', 'swing_pct_2'),          # Percentage of left-right swing phase
            ('stance_1', 'stance_2'),                # Left and right support phase
            ('stance_pct_1', 'stance_pct_2'),        # Percentage of left-right stance
            ('stride_time_1', 'swing_1'),            # Left stride length - swing phase,indeed the left foot - left-left foot association may be the greatest. 
            ('stride_time_2', 'swing_2'),            # Right stride length - swing phase,indeed the right foot - right-right foot association may be the greatest.
            ('swing_1', 'stance_1'),                 # Left swing phase - stance phase
            ('swing_2', 'stance_2'),                 # Right swing phase - stance phase
            ('double_support', 'double_support_pct') # Double support phase - percentage
        ]


    def build_graph(self, original_signal, ts_features, label):
        """
        Construct the graph data for a single sample (each piece of data) 

        Parameters:
        original_signal: ndarray, shape [T] Left and right foot pressure signals
        - For example: (83421,) Combined pressure signal
        ts_features: ndarray, shape [num_cycles, 13] TS feature matrix
        - For example: (310, 13) 310 gait cycles × 13 features
        - The 0th column: Time (seconds)
        - The 1st - 12th columns: 12 gait features
        label: int, Sample label (0 = ALS, 1 = Control, 2 = Hunt, 3 = Park) 

        Return:
        Data: torch_geometric.data.Data object
        """


        # ==================== 1. Signal nodex ====================
        T = original_signal.shape[0]  
        subsample_T = max(1, int(T * self.signal_subsample_ratio))
        step = max(1, T // subsample_T)
        # 
        signal_nodes = original_signal[::step]  # [subsample_T,]
        num_signal_nodes = len(signal_nodes)

        
        # ==================== 2. TS features nodes ====================
        ts_feature_nodes = ts_features[:, 1:]  # [num_cycles, 12]
    
        # e.g.: (310, 12) -> mean -> (12,)
        ts_feature_nodes = ts_feature_nodes.mean(axis=0)  # [12,]
        num_ts_nodes = len(ts_feature_nodes)  # 12
        
        # ==================== 3. Node Feature Matrix ==========
        
        # Signal node features: [num_signal_nodes, 1]
        signal_node_features = signal_nodes.reshape(-1, 1).astype(np.float32)
        # TS node features: [12, 1]
        ts_node_features = ts_feature_nodes.reshape(-1, 1).astype(np.float32)
        #  [total_nodes, 1]
        x = np.concatenate([signal_node_features, ts_node_features], axis=0)
        x = torch.FloatTensor(x)  # [total_nodes, 1]



        # ==================== 4. Build edge index ====================
        edge_list = []
        total_nodes = num_signal_nodes + num_ts_nodes

        
        '''
        # This is a bidirectional edge. Only the edge_index 
        of the edge is obtained.
        ## Only connect the adjacent time points of the original 
        signal (of course, the interval here is 'step') in both
         directions.
        '''
        #
        # ---- Type 1: Time Edge (signal connecting adjacent time points) ----
        if self.add_time_edges:
            # Connect the adjacent signal nodes
            for i in range(num_signal_nodes - 1):
                node1 = i
                node2 = i + 1
                edge_list.append([node1, node2])
                edge_list.append([node2, node1])  # Bidirectional edge




        # # ---- MS-Type 2: Signal-TS Feature Edge ----
        if self.add_signal_ts_edges:

            num_ts_feature_nodes = num_ts_nodes  # 12
            #
            segment_size = num_signal_nodes // num_ts_feature_nodes  # Signal node segmentation

            for ts_idx in range(num_ts_feature_nodes):
    
                # Each TS feature node is connected to the corresponding 
                ## signal node segment.
                start_idx = ts_idx * segment_size
                end_idx = min((ts_idx + 1) * segment_size, num_signal_nodes)

                ts_node_global_idx = num_signal_nodes + ts_idx


                for signal_node_idx in range(start_idx, end_idx):
                    edge_list.append([signal_node_idx, ts_node_global_idx])
                    edge_list.append([ts_node_global_idx, signal_node_idx])

        
        # ---- MS-Type 3: Edges between TS features ----
        if self.add_ts_feature_edges:
            # Build edges based on the predefined features
            for feature_name_1, feature_name_2 in self.ts_feature_pairs:
                # Search for the feature index
                idx_1 = self.ts_feature_names.index(feature_name_1)
                idx_2 = self.ts_feature_names.index(feature_name_2)

                # Convert to global node index
                node_1 = num_signal_nodes + idx_1
                node_2 = num_signal_nodes + idx_2

                edge_list.append([node_1, node_2])
                edge_list.append([node_2, node_1])



        '''
        Data type: The indices in the edge_index
        must be of long integer type (torch.long)
        '''
        # === 5. Convert to torch.Tensor of LongTensor integer type to
        # obtain the edge index tensor and then transpose it ======
        #==============        
        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).T  # [2, num_edges]
        else:
            edge_index = torch.LongTensor([[0], [0]])  # At least one of the edges




        # ==================== 6. Create a Data object ====================
        graph_data = Data(
            x = x,                  # [total_nodes, 1] Node features
            edge_index=edge_index,  # [2, num_edges] Edge Indice
            y=torch.LongTensor([label])  # [1] Label
        )
        
        # Add additional attributes
        graph_data.num_signal_nodes = num_signal_nodes
        graph_data.num_ts_nodes = num_ts_nodes
        graph_data.total_nodes = total_nodes
        graph_data.original_signal_length = T  # Record the length of the original signal

        return graph_data






# ==================== 
# Import the doge64 data and divide it into training and
 # testing sets (note that this is for single-legged DJ here; 
   # you can choose double-legged later)
# ====================

def load_dogen_data(dogen_path, train_ratio=0.5, select_foot = 'left'):

    print("=" * 80)
    print("=" * 80)

    origin_data_features = []
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

        if select_foot=="left":
            originalSingal_data = current_dogen_dict['left_data'].astype(np.float32).reshape(1, -1)
        elif select_foot=="right":
            originalSingal_data = current_dogen_dict['right_data'].astype(np.float32).reshape(1, -1)
        elif select_foot=="left_right":
            left_data = current_dogen_dict['left_data'].astype(np.float32).reshape(1, -1)
            right_data =current_dogen_dict['right_data'].astype(np.float32).reshape(1, -1)
            originalSingal_data = np.concatenate((left_data, right_data), axis=0)

            
        originalSingal_length = originalSingal_data.shape[1]


        ts_data = current_dogen_dict['ts_array'][:, 1:]  # [num_cycles, 13]
        elapsed_time = current_dogen_dict['ts_array'][0] # Elapsed Time (sec)

        ts_data = ndstrarr2ndarray(ts_data).astype(np.float32)
        ts_percent2_float_indice = [(5-1),(6-1),(9-1),(10-1),(12-1)]

        for _, k  in enumerate(ts_percent2_float_indice):
            ts_data[:,k] = ts_data[:,k]/100

        # concat elapsed time dim [L,13]
        ts_data_ = np.concatenate((elapsed_time.reshape(len(elaspsed_time),1),ts_data), axis=1)
        #
        origin_data_features.append(originalSingal_data)
        ts_features.append(ts_data_)
        basenames_l.append(base_name.lower())


    origin_data_features = np.array(origin_data_features)  # [N, L]/[N,2,L]
    ts_features = np.array(ts_features)           # [N, num_cycles, 13]

    origin_data_features = max_min_global(origin_data_features)
    # origin_data_features = z_score_global(origin_data_features)

    # w/o elaspsed_time
    elased_0 = ts_features[:,:,0]
    ts_features = z_score_global(ts_features)
    # cat again
    ts_features = np.concatenate((elased_0.reshape(elased_0.shape[0], elased_0.shape[1], 1), ts_features),axis=2)
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
            continue

        labels_l.append(curr_label)

    labels = np.array(labels_l).astype(np.int64)



    # 
    ## → Parkinson
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




# ==================== GNN model====================

class GaitGNN(nn.Module):
    """
    Gait Graph Neural Network Model - Real Dataset Version 

    Fully compatible with your dataset:
    - Input: Graph data (signal nodes + TS feature nodes)
    - Output: 4 classifications (ALS/Control/Hunt/Park)
     """

    def __init__(self,
                 signal_node_dim=1,           # Signal node feature dimension (fixed at 1)
                 ts_node_dim=1,               # The feature dimension of the TS  node (fixed at 1)
                 hidden_dim=128,              # Hidden layer dimension
                 num_gnn_layers=2,            # Number of GNN layers
                 num_classes=4,               # Classification number
                 dropout=0.3,
                 pooling_type='mean'):        # Pooling type: 'mean', 'max', 'both'
        super().__init__()

        self.signal_node_dim = signal_node_dim
        self.ts_node_dim = ts_node_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.num_classes = num_classes
        self.pooling_type = pooling_type


        # ==================== Node feature encoding ====================
        # Signal node encoder: [1] -> [hidden_dim]
        self.signal_encoder = nn.Sequential(
            nn.Linear(signal_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        # TS feature node encoder: [1] -> [hidden_dim]
        self.ts_encoder = nn.Sequential(
            nn.Linear(ts_node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )


        # ==================== GNN layer ====================
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))

        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))


        # ==================== Classifier ====================
        if pooling_type == 'both':
            classifier_input_dim = hidden_dim * 2
        else:
            classifier_input_dim = hidden_dim
        #
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Weight Initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        """
        Forward propagation 

        Parameters:
        data: torch_geometric.data.Data object or Batch object
        - data.x: [total_nodes, 1] Node features
        - data.edge_index: [2, num_edges] Edge indices
        - data.batch: [total_nodes] Batch indices
        - data.num_signal_nodes: Number of signal nodes (for each graph)
        - data.num_ts_nodes: Number of TS feature nodes (for each graph) 

        Return: logits: [batch_size, num_classes]
        """

        x = data.x  # [total_nodes, 1]
        edge_index = data.edge_index  # [2, num_edges]
        '''

        '''
        
        '''
        ## It is likely that the attribute of data.batch needs
        to be pre-defined earlier. ##

        '''
        batch = data.batch  # The batch index of [total_nodes]

        
       # ==================== Node Feature Encoding ====================


        if hasattr(data, 'num_signal_nodes'):
            # The case of a single graph
            num_signal_nodes = data.num_signal_nodes
            num_ts_nodes = data.num_ts_nodes
        else:
            
            total_nodes = x.size(0)
            num_graphs = batch.max().item() + 1
            # Estimate the number of signal nodes
            num_signal_nodes = int(total_nodes * 0.9 // num_graphs)
            # Total number of [individual] graph nodes - 
            ## Number of signal nodes
            num_ts_nodes = total_nodes // num_graphs - num_signal_nodes


        # Encoded signal node -  
        '''
        ###
        **There is also a batch dimension, so I understand that
        **there needs to be a determination here.
        signal_nodes = x[:,:num_signal_nodes,:]
        ...
        ts_nodes = x[:,num_signal_nodes:,:]
        ###
        ...
        '''
        signal_nodes = x[:num_signal_nodes, :]  # [num_signal_nodes, 1]
        signal_encoded = self.signal_encoder(signal_nodes)  # [num_signal_nodes, hidden_dim]

        # Encoding TS feature nodes - 
        ts_nodes = x[num_signal_nodes:, :]  # [num_ts_nodes, 1]
        ts_encoded = self.ts_encoder(ts_nodes)  # [num_ts_nodes, hidden_dim]

        # The merged encoded node features
        '''
        Similarly, if "batch" is included, then:
        x = torch.cat([signal_encoded, ts_encoded],dim=1)
        '''
        x = torch.cat([signal_encoded, ts_encoded], dim=0)  # [total_nodes, hidden_dim]




        # ==================== GNN propagation ====================
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)

            if i < len(self.gnn_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.3, training=self.training)



        # ==================== Pooling4Graph, 
        # which means expanding along the node dimension 
        # and averaging. ====================
        if self.pooling_type == 'mean':
            x_pooled = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        elif self.pooling_type == 'max':
            x_pooled = global_max_pool(x, batch)  # [batch_size, hidden_dim]
        elif self.pooling_type == 'both':
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x_pooled = torch.cat([x_mean, x_max], dim=1)  # [batch_size, hidden_dim*2]
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        # ==================== Classification ====================
        logits = self.classifier(x_pooled)  # [batch_size, num_classes]

        return logits





# ==================== Dataset(torch_geometric.data.*) ====================
class GaitGraphDataset:
    """
    Gait Graph Dataset Class - Real Dataset Version
    """

    def __init__(self, original_signals, ts_features, labels,
                 signal_subsample_ratio=0.05):
        """
        Initialize the dataset

        Parameters:
        original_signals: ndarray, shape [N, T] Original signal (or [N, 2, T])
        - N: Number of samples
        - T: Number of time points (e.g., 83421)
        ts_features: ndarray, shape [N, num_cycles, 13] TS features
        - N: Number of samples
        - num_cycles: Number of gait cycles (e.g., 310)
        - 13: Number of features (time + 12 gait features)
        labels: ndarray, shape [N] Labels
        signal_subsample_ratio: Signal sampling ratio (controls the scale of the plot)
        """

        self.original_signals = original_signals
        self.ts_features = ts_features
        self.labels = labels


        # Initialize Graph Builder
        self.graph_builder = GaitGraphBuilder(
            signal_subsample_ratio=signal_subsample_ratio
        )

        # Pre-build all graphs
        print(f"Build graph data (signal_subsample_ratio={signal_subsample_ratio})...")
        self.graph_list = []

        '''
        Note that each building map ultimately corresponds to one label.
        '''
        for i in tqdm(range(len(labels)), desc="Construct a graph"):
            graph_data = self.graph_builder.build_graph(
                self.original_signals[i],
                self.ts_features[i],
                self.labels[i]
            )
            self.graph_list.append(graph_data)

        print(f"he graph data has been constructed, with a total of {len(self.graph_list)} graphs.")

        # Statistical graph information
        self._print_graph_stats()


    def _print_graph_stats(self):
        "Print graph statistics information"
        if len(self.graph_list) > 0:

            # Only the first image was used for statistics
            sample_graph = self.graph_list[0]
            print(f"\n Graph statistics information:")
            print(f"  Number of signal nodes: {sample_graph.num_signal_nodes}")
            print(f"  Number of TS feature nodes: {sample_graph.num_ts_nodes}")
            print(f"  Total number of nodes: {sample_graph.total_nodes}")
            print(f"  Number of edges: {sample_graph.edge_index.shape[1]}")
            print(f"  Original signal length: {sample_graph.original_signal_length}")

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]


# ==================== Training  ====================

def train_gnn_model(model, train_loader, val_loader, epochs=50, lr=1e-3,
                    patience=10, device='cpu'):
    """
    Train the GNN model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5, patience=5, verbose=True)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_model_state_dict = None
    patience_counter = 0

    print("\n" + "=" * 80)
    print(f"Start training the GNN model (for a total of {epochs} epochs))")
    print("=" * 80)

    for epoch in range(epochs):
        # training
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, data in enumerate(pbar):

            # ✔Here, a training batch size should be set as an 
              ## attribute of the batch in the data.:
                ### data.batch = 4


            data = data.to(device)

            
            logits = model(data)
            loss = criterion(logits, data.y)

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            
            _, predicted = logits.max(1)
            total += data.y.size(0)
            correct += predicted.eq(data.y).sum().item()

            if (batch_idx + 1) % 5 == 0:
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
            for data in val_loader:
                data = data.to(device)


                # ✔ Set the batch size in the same way✔
                # data.batch = 4
                logits = model(data)
                loss = criterion(logits, data.y)

                val_loss += loss.item()
                _, predicted = logits.max(1)
                val_total += data.y.size(0)
                val_correct += predicted.eq(data.y).sum().item()

        # indicate indicators
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Print the results of the current epochs
        print(f'Epoch {epoch+1:3d}/{epochs}: '
              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')

        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict().copy()
            patience_counter = 0
            print(f'   The best model has been saved (Val Accuracy: {val_acc:.2f}%)')
        else:
            patience_counter += 1

    # optimize
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        print(f'\n Best validation accuracy: {best_val_acc:.2f}%')

    print("=" * 80)

    return train_losses, val_losses, train_accs, val_accs, best_model_state_dict

'''
This GCN does not require validation~~
'''
# ==================== Test ====================

def test_gnn_model(model, test_loader, device='cpu'):
    """Test the GNN model"""
    print("\n" + "=" * 80)
    print("Testing of GNN model")
    print("=" * 80)

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_loader:

            # data.batch = 4
            data = data.to(device)

            logits = model(data)
            loss = criterion(logits, data.y)

            total_loss += loss.item()

            probs = F.softmax(logits, dim=1)
            _, predicted = logits.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy array
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculation Indicators
    test_accuracy = accuracy_score(all_labels, all_preds)
    avg_test_loss = total_loss / len(test_loader)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification Report
    class_names = ['ALS', 'Control', 'Huntington', 'Parkinson']
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, output_dict=True)

    # Results
    print(f"Accuracy of the test set: {test_accuracy*100:.2f}%")
    print(f"Average loss of the test set: {avg_test_loss:.4f}")
    print("\n Confusion matrix:")
    print(cm)
    print("\n Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

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


# ==================== Visual function ====================

def plot_gnn_results(train_losses, val_losses, train_accs, val_accs, test_results):
    """Visualized training results of GNN"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss curve
    axes[0, 0].plot(train_losses, label='Training loss', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(val_losses, label='Val loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('Training and validation loss curves', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Accuracy curve
    axes[0, 1].plot(train_accs, label='Training accuracy rate', linewidth=2, marker='o', markersize=4)
    axes[0, 1].plot(val_accs, label='Val accuracy rate', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Training and validation accuracy curves', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Confusion Matrix Heatmap
    cm = test_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['ALS', 'Control', 'Hunt', 'Park'],
                yticklabels=['ALS', 'Control', 'Hunt', 'Park'])
    axes[1, 0].set_title(f'Test set confusion matrix (Acc: {test_results["accuracy"]*100:.2f}%)',
                       fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Prediction label')
    axes[1, 0].set_ylabel('True label')

    # Indicators by Categories
    class_names = ['ALS', 'Control', 'Huntington', 'Parkinson']
    metrics = ['precision', 'recall', 'f1-score']

    x = np.arange(len(class_names))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [test_results['classification_report'][name][metric] for name in class_names]
        axes[1, 1].bar(x + i*width, values, width, label=metric)

    axes[1, 1].set_title('Comparison of indicators in various categories', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Category')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_xticks(x + width)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_boards_model_pred_results\gnn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("The visualization results have been saved to: gnn_training_results.png")


# ==================== Main Function ====================

def main():
    """
    Main Function - Complete Process of GNN Gait Classification 
    (Real Dataset Version)
    """
    print("\n" + "=" * 80)
    print("Graph Neural Network (GNN) gait classification system - Real dataset version")
    print("=" * 80)


    # ==================== 1. Generate simulated data (based on the structure of your dataset) ====================
    print("\n [Step 1/5] Prepare data (simulate the structure of a real data set)")
    print("-" * 80)

    dogen_path = r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\gait64_neodegen_pkls'


    if not os.path.exists(dogen_path):

        # Generate simulated data based on the structure of your dataset
        num_samples = 80
        original_signal_length = 83421  # The actual length of your signal
        ts_num_cycles = 310  # The actual number of TS gait cycles you have

        print(f"Generate simulated data: {num_samples} samples")
        print(f" Original signal length: {original_signal_length}")
        print(f" TS gait cycle number: {ts_num_cycles}")

        # Simulate the original signal [N, T]
        train_original_signals = np.random.randn(num_samples, original_signal_length).astype(np.float32) * 100 + 500
        test_original_signals = np.random.randn(20, original_signal_length).astype(np.float32) * 100 + 500

        # Simulated TS feature [N, num_cycles, 13]
        train_ts_features = np.random.randn(num_samples, ts_num_cycles, 13).astype(np.float32)
        test_ts_features = np.random.randn(20, ts_num_cycles, 13).astype(np.float32)

        # Simulation label
        train_labels = np.random.randint(0, 4, num_samples)
        test_labels = np.random.randint(0, 4, 20)

        # Normalization
        train_original_signals = max_min_global(train_original_signals)
        test_original_signals = max_min_global(test_original_signals)
        train_ts_features = z_score_global(train_ts_features)
        test_ts_features = z_score_global(test_ts_features)

        print("Simulation data generation completed!")
        print(f"  Training set: {train_original_signals.shape[0]} samples")
        print(f"  Testing set: {test_original_signals.shape[0]} samples")


    else:
        (train_original_signals, train_ts_features, train_labels,
         test_original_signals, test_ts_features, test_labels) = load_dogen_data(
            dogen_path, train_ratio=0.5, select_foot = 'left'
        )




    # ==================== 2. Build graph data ====================
    print("\n [Step 2/5] Build graph data")
    print("-" * 80)

    signal_subsample_ratio = 0.01  # 


    train_graph_dataset = GaitGraphDataset(
        train_original_signals, train_ts_features, train_labels,
        signal_subsample_ratio=signal_subsample_ratio
    )

    test_graph_dataset = GaitGraphDataset(
        test_original_signals, test_ts_features, test_labels,
        signal_subsample_ratio=signal_subsample_ratio
    )

    # Divide the training/validation set
    train_size = int(0.8 * len(train_graph_dataset))
    val_size = len(train_graph_dataset) - train_size

    train_subset, val_subset = torch.utils.data.random_split(
        train_graph_dataset, [train_size, val_size]
    )

    # Create a data loader
    batch_size = 4  # GNN usually requires a smaller batch size.
    train_loader = GeoDataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = GeoDataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = GeoDataLoader(test_graph_dataset, batch_size=batch_size, shuffle=False)


    print("\nInformation of the data loader:")
    print("  Training set: " + str(len(train_subset)) + " graphs")
    print("  Validation set: " + str(len(val_subset)) + " graphs")
    print("  Test set: " + str(len(test_graph_dataset)) + " graphs")
    print("  Batch size: " + str(batch_size))




    # ==================== 3. Initialize the GNN model ====================
    print("\n[Step 3/5] Initialize the GNN model")
    print("-" * 80)

    model = GaitGNN(
        signal_node_dim=1,
        ts_node_dim=1,
        hidden_dim=128,
        num_gnn_layers=2,
        num_classes=4,
        dropout=0.3,
        pooling_type='mean'
    ).to(device)

    # model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {total_params:,}")
    print(f"The number of trainable parameters: {trainable_params:,}")

    # ==================== 4. Train the model ====================
    print("\n [Step 4/5] Train the GNN model")
    print("-" * 80)

    train_losses, val_losses, train_accs, val_accs, best_model_state = train_gnn_model(
        model, train_loader, val_loader, epochs=30, lr=1e-3,
        patience=10, device=device
    )

    # ==================== 5. Testing model ====================
    print("\n [Step 5/5] Test the GNN model")
    print("-" * 80)

    # Restore the best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    test_results = test_gnn_model(model, test_loader, device=device)


    # ==================== 6. Visualized results ====================
    print("\n [Step 6/6] Visualized result")
    print("-" * 80)

    plot_gnn_results(train_losses, val_losses, train_accs, val_accs, test_results)

    # ==================== Save the model ====================
    save_path = 'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_boards_model_pred_results/gait_gnn_classifier_real.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_results': test_results,
        'model_config': {
            'signal_subsample_ratio': signal_subsample_ratio,
            'hidden_dim': 128,
            'num_gnn_layers': 2
        }
    }, save_path)

    print(f"\n The model has been saved to: {save_path}")

    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("The GNN training process has been completed!")
    print("=" * 80)
    print(f"\n📊 Training set accuracy rate: {train_accs[-1]:.2f}%")
    print(f"📊 Accuracy of the validation set: {val_accs[-1]:.2f}%")
    print(f"📊 Accuracy of the test set: {test_results['accuracy']*100:.2f}%")
    print("\n" + "=" * 80)

    return model, train_losses, val_losses, train_accs, val_accs, test_results


if __name__ == "__main__":
    # Run the main process
    model, train_losses, val_losses, train_accs, val_accs, test_results = main()

    # Clear GPU Memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n GPU memory has been cleared.")

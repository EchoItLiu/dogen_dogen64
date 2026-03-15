


"""
Traditional Machine Learning Methods for Gait Classification
=========================================
This code implements 11 traditional ML methods for 4-class gait classification task.
All methods replace the original deep learning approach (plan_model) with sklearn-based algorithms.

Author: Modified from original entropy_select_k.py
Date: 2026-03-09

Classification Labels:
    0 - ALS (Amyotrophic Lateral Sclerosis)
    1 - Control (Healthy)
    2 - Huntington's Disease
    3 - Parkinson's Disease

Features:
    1. Information Entropy-based Feature Selection
    2. 11 Traditional ML Algorithms
    3. Cross-validation Support (when available)
    4. Comprehensive Evaluation Metrics
"""

import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Data Processing and Splitting
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Traditional ML Algorithms
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import AdaBoostClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# For Apriori algorithm (association rule mining)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

'''
Add feature selection library
'''
from sklearn.feature_selection import SelectKBest, f_classif

# For PageRank (network analysis)
import networkx as nx

# For EM algorithm (Gaussian Mixture Model)
from sklearn.mixture import GaussianMixture

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Random seed for reproducibility
np.random.seed(42)

# ==================== Global Configuration ====================
CLASS_NAMES = ['ALS', 'Control', 'Huntington', 'Parkinson']
NUM_CLASSES = 4

# ==================== 0. Unify minimum ts feature period length ====================
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



# ==================== 1. Information Entropy Calculation ====================
def calculate_entropy_importance(left_right_data, top_k_entropy_th):
    """
    Calculate information entropy importance for feature selection.

    Args:
        left_right_data (np.ndarray): Input feature map [C, L] where C is channel,
                                      L is sequence length
        top_k_entropy_th (int): Number of top-K features to select based on entropy

    Returns:
        np.ndarray: Selected features [C, top_k_entropy_th]


    If the information entropy has high
    efficiency and performance, it can be
    given priority for use to select
    min_num_cycles(Top-k) of ts sequences with the
    highest information content.

    """
    try:
        # Convert to numpy array if needed
        if not isinstance(left_right_data, np.ndarray):
            left_right_data = np.array(left_right_data)

        # Validate input dimension
        if left_right_data.ndim != 2:
            raise ValueError(f'Input must be 2D array [C, L], got {left_right_data.ndim}D')

        # Softmax calculation along C dimension (axis=0)
        # For numerical stability, subtract max value before exponentiation
        max_vals = np.max(left_right_data, axis=0, keepdims=True)
        exp_vals = np.exp(left_right_data - max_vals)
        probs = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)

        # Information entropy calculation: H = -sum(p * log(p))
        entropy = -np.sum(probs * np.log(probs + 1e-8), axis=0)

        # Min-max normalization to [0, 1]
        entropy_min = np.min(entropy)
        entropy_max = np.max(entropy)

        if entropy_max - entropy_min > 1e-8:
            entropy_normalized = (entropy - entropy_min) / (entropy_max - entropy_min)
        else:
            entropy_normalized = np.zeros_like(entropy)

        # Sort by entropy (descending) and select top-K indices
        # High entropy = more information
        sorted_entropy_normalized_idx = np.argsort(entropy_normalized)[::-1][:top_k_entropy_th]

        # Slice original data based on selected indices
        left_right_data_k = left_right_data[:, sorted_entropy_normalized_idx]

        return left_right_data_k

    except Exception as e:
        print(f"Error in calculating entropy importance: {e}")
        # Return zeros as fallback
        if hasattr(left_right_data, 'shape'):
            return np.zeros((left_right_data.shape[0], min(top_k_entropy_th, 110)))
        else:
            return np.zeros((2, min(top_k_entropy_th, 110)))


# ==================== 2. Data Normalization Functions ====================
def max_min_global(features):
    """
    Min-Max normalization (scale to [0, 1])

    Args:
        features (np.ndarray): Input features

    Returns:
        np.ndarray: Normalized features
    """
    min_val = np.min(features)
    max_val = np.max(features)

    if max_val - min_val < 1e-8:
        return features - min_val

    normalized = (features - min_val) / (max_val - min_val)
    return normalized


def z_score_global(features):
    """
    Z-score normalization (standardization)

    Args:
        features (np.ndarray): Input features

    Returns:
        np.ndarray: Standardized features
    """
    mean_val = np.mean(features)
    std_val = np.std(features)

    if std_val < 1e-8:
        return features - mean_val

    normalized = (features - mean_val) / std_val
    return normalized


def ndstrarr2ndarray(str_nd_arr):
    """
    Convert string numpy array to float array

    Args:
        str_nd_arr: String array

    Returns:
        np.ndarray: Float array
    """
    float_nd_arr = np.zeros(str_nd_arr.shape, dtype=np.float32)
    for i in range(str_nd_arr.shape[0]):
        for j in range(str_nd_arr.shape[1]):
            float_nd_arr[i][j] = float(str_nd_arr[i][j])
    return float_nd_arr


# ==================== 3. Data Loading Function ====================
def load_dogen_data(dogen_path, train_ratio=0.5, top_k_entropy_th=200):
    """
    Load DOGEN gait data from pickle files.

    Args:
        dogen_path (str): Path to directory containing .pkl files
        train_ratio (float): Ratio of training data (default: 0.5)
        top_k_entropy_th (int): Number of top-K features to select

    Returns:
        tuple: (train_X, train_y, test_X, test_y) where X is feature matrix, y is labels
    """
    print("=" * 80)
    print("=" * 80)
    print("Loading DOGEN Gait Data...")
    print("=" * 80)

    lr_data_features = []
    ts_features = []
    basenames_l = []

    # Get all pickle files
    pkl_files = [f for f in os.listdir(dogen_path) if f.endswith('.pkl')]

    # Load data from pickle files
    for pkl_file in tqdm(pkl_files, desc="Loading pickle files..."):
        with open(os.path.join(dogen_path, pkl_file), 'rb') as f:
            current_dogen_dict = pickle.load(f)

        # Extract base name (subject ID)
        if pkl_file.startswith('a'):
            backup_pkl_file = pkl_file[:3]
        elif pkl_file.startswith('c'):
            backup_pkl_file = pkl_file[:7]
        elif pkl_file.startswith('h') and pkl_file.startswith('p'):
            backup_pkl_file = pkl_file[:4]
        else:
            backup_pkl_file = pkl_file.split('_')[0]

        base_name = current_dogen_dict.get('subject', backup_pkl_file)

        # Load left and right foot data
        left_data = current_dogen_dict['left_data'].astype(np.float32).reshape(1, -1)
        right_data = current_dogen_dict['right_data'].astype(np.float32).reshape(1, -1)
        left_right_data = np.concatenate((left_data, right_data), axis=0)

        '''
        Shorten the original signal and unify
        the length of the time ts signal
        for concatenation:
        ① Here, calculate_entropy_importance is
        applied to the original signal.
        ② For the ts time sequence signal,
        unify_min_num_cycles is used to calculate
         the minimum ts length L
         (Default return value is 122).
        '''
        # Apply information entropy feature selection
        left_right_data_ied = calculate_entropy_importance(left_right_data, top_k_entropy_th)

        # Load time-series features
        ts_data = current_dogen_dict['ts13_array'][:, 1:]  # [num_cycles, 12]
        elapsed_time = current_dogen_dict['ts13_array'][:, 0]  # Elapsed Time (sec)
        sample_rate = current_dogen_dict['sample_rate']

        ts_data = ndstrarr2ndarray(ts_data).astype(np.float32)

        # Convert percentage columns to float (columns 5, 6, 9, 10, 12)
        ts_percent2_float_indices = [4, 5, 8, 9, 11]  # 0-indexed
        for k in ts_percent2_float_indices:
            ts_data[:, k] = ts_data[:, k] / 100

        # Store features
        lr_data_features.append(left_right_data_ied)
        ts_features.append(ts_data)
        basenames_l.append(base_name.lower())

    # Add a method to determine the minimum
     ## number of cycles (min_num_cycles) in
       ### the ts_features and print the result.
       #### Additionally, return and print the
         ##### shortest ts_features.
    print("\n Unify ts feature length...")
    ts_features, uni_num_cycles = unify_min_num_cycles(ts_features)
    # print ("The unified feature length is:", uni_num_cycles)

    # Convert to numpy arrays
    lr_data_features = np.array(lr_data_features)  # [N, 2, K]
    ts_features = np.array(ts_features)  # [N, num_cycles, 12]

    # Normalize features
    lr_data_features = max_min_global(lr_data_features)
    ts_features = z_score_global(ts_features)

    # Generate labels based on subject name
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
            continue  # Skip unknown labels
        labels_l.append(curr_label)

    labels = np.array(labels_l).astype(np.int64)

    # Print dataset statistics
    print("\nDataset Statistics:")
    print("-" * 80)
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {CLASS_NAMES[label]}: {count} samples")
    print(f"Total samples: {len(labels)}")
    print(f"Feature shape: lr_data={lr_data_features.shape}, ts_data={ts_features.shape}")

    # Flatten 3D features to 2D for traditional ML
    # lr_data_features: [N, 2, K] -> [N, 2*K]
    # ts_features: [N, num_cycles, 12] -> [N, num_cycles*12]
    # The flattening feature is used in ML calculations.
    lr_flat = lr_data_features.reshape(lr_data_features.shape[0], -1)
    ts_flat = ts_features.reshape(ts_features.shape[0], -1)

    # Concatenate all features
    # Merging Feature Layers
    all_features = np.concatenate([lr_flat, ts_flat], axis=1)  # [N, 2*K + num_cycles*12]

    print(f"Flattened feature shape: {all_features.shape}")
    print("-" * 80)

    # Split into train and test sets
    n_samples = len(labels)
    indices = np.random.permutation(n_samples)

    n_train = int(n_samples * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    train_X = all_features[train_indices]
    train_y = labels[train_indices]
    test_X = all_features[test_indices]
    test_y = labels[test_indices]

    print(f"Train set: {len(train_y)} samples")
    print(f"Test set: {len(test_y)} samples")
    print("=" * 80)

    return train_X, train_y, test_X, test_y




# ==================== 4. Traditional ML Model Definitions ====================
class TraditionalMLModels:
    """
    Wrapper class for 11 traditional machine learning algorithms.
    All methods follow sklearn's fit/predict interface.
    """

    def __init__(self, method_name, random_state=42):
        """
        Initialize a traditional ML model.

        Args:
            method_name (str): Name of the algorithm
            random_state (int): Random seed for reproducibility
        """
        self.method_name = method_name
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the specific model based on method_name."""

        if self.method_name == 'C4.5':
            # C4.5 Decision Tree (use DecisionTreeClassifier as C4.5 equivalent)
            # Parameters:
            #   - criterion: 'entropy' for information gain (C4.5 style)
            #   - max_depth: Maximum depth of the tree (None = unlimited)
            #   - min_samples_split: Minimum samples required to split a node
            #   - min_samples_leaf: Minimum samples required at a leaf node
            #   - random_state: Random seed
            self.model = DecisionTreeClassifier(
                criterion='entropy',  # Information gain (C4.5)
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state
            )


        elif self.method_name == 'k-Means':
            # k-Means Clustering (adapted for classification via nearest centroid)
            # Parameters:
            #   - n_clusters: Number of clusters (should match number of classes)
            #   - max_iter: Maximum number of iterations
            #   - random_state: Random seed
            #   - n_init: Number of times to run with different centroids
            self.model = KMeans(
                n_clusters=NUM_CLASSES,
                max_iter=300,
                random_state=self.random_state,
                n_init=10
            )

        elif self.method_name == 'SVM':
            # Support Vector Machine with RBF kernel
            # Parameters:
            #   - C: Regularization parameter (higher = less regularization)
            #   - kernel: Kernel type ('rbf', 'linear', 'poly', 'sigmoid')
            #   - gamma: Kernel coefficient ('scale', 'auto', or float)
            #   - probability: Whether to enable probability estimates
            #   - random_state: Random seed
            self.model = SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            )


        elif self.method_name == 'Apriori':
            # Apriori Algorithm for Association Rule Mining
            # Note: Apriori is for association rules, not classification
            # We'll adapt it by converting features to binary and using rule-based classification
            '''
            Considering the need to save
            computing resources, feature
            selection was adopted to reduce
            the feature dimension.
            '''
            self.model = 'Apriori'  # Special handling needed


        elif self.method_name == 'EM':
            # Expectation-Maximization (Gaussian Mixture Model)
            # Parameters:
            #   - n_components: Number of mixture components
            #   - covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            #   - max_iter: Maximum iterations
            #   - random_state: Random seed
            self.model = GaussianMixture(
                n_components=NUM_CLASSES,
                covariance_type='full',
                max_iter=100,
                random_state=self.random_state
            )


        elif self.method_name == 'PageRank':
            # PageRank Algorithm (network-based classification)
            # Note: PageRank is for ranking nodes in a graph
            # We'll adapt it using k-NN graph construction
            self.model = 'PageRank'  # Special handling needed


        elif self.method_name == 'AdaBoost':
            # AdaBoost (Adaptive Boosting)
            # Parameters:
            #   - n_estimators: Number of weak learners
            #   - learning_rate: Shrinks contribution of each learner
            #   - algorithm: 'SAMME' or 'SAMME.R'
            #   - random_state: Random seed
            self.model = AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                # algorithm='SAMME.R',
                random_state=self.random_state
            )


        elif self.method_name == 'kNN':
            # k-Nearest Neighbors
            # Parameters:
            #   - n_neighbors: Number of neighbors
            #   - weights: Weight function ('uniform', 'distance')
            #   - algorithm: Algorithm for nearest neighbors ('auto', 'ball_tree', 'kd_tree', 'brute')
            #   - p: Power parameter for Minkowski distance (1=manhattan, 2=euclidean)
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform',
                algorithm='auto',
                p=2
            )


        elif self.method_name == 'NaiveBayes':
            # Gaussian Naive Bayes
            # Parameters:
            #   - var_smoothing: Portion of largest variance added to variances for stability
            self.model = GaussianNB(
                var_smoothing=1e-9
            )


        elif self.method_name == 'CART':
            # CART (Classification and Regression Trees)
            # Use DecisionTreeClassifier with Gini impurity
            # Parameters:
            #   - criterion: 'gini' for Gini impurity (CART style)
            #   - max_depth: Maximum depth of the tree
            #   - min_samples_split: Minimum samples to split
            #   - min_samples_leaf: Minimum samples at leaf
            #   - random_state: Random seed
            self.model = DecisionTreeClassifier(
                criterion='gini',  # Gini impurity (CART)
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=self.random_state
            )

        elif self.method_name == 'IsolationForest':
            # Isolation Forest (anomaly detection, adapted for classification)
            # Parameters:
            #   - n_estimators: Number of trees
            #   - max_samples: Number of samples to draw
            #   - contamination: Expected proportion of outliers
            #   - random_state: Random seed
            self.model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=0.1,
                random_state=self.random_state
            )

        else:
            raise ValueError(f"Unknown method: {self.method_name}")

        # Initialize scaler for methods that benefit from normalization
        if self.method_name in ['SVM', 'kNN', 'k-Means']:
            self.scaler = StandardScaler()

    # Overloading
    def fit(self, X_train, y_train):
        """
        Train the model.

        Args:
            X_train (np.ndarray): Training features [N, D]
            y_train (np.ndarray): Training labels [N]
        """

        # Apply scaling if needed
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)


        # Special handling for clustering-based methods
        if self.method_name == 'k-Means':
            self.model.fit(X_train)
            # Map cluster labels to true labels
            self.cluster_to_class_mapping = self._map_clusters_to_classes(
                self.model.labels_, y_train
            )

        # Special handling for EM (GMM)
        elif self.method_name == 'EM':
            self.model.fit(X_train)
            # Map GMM components to true labels
            self.cluster_to_class_mapping = self._map_clusters_to_classes(
                self.model.predict(X_train), y_train
            )

        # Special handling for Isolation Forest
        elif self.method_name == 'IsolationForest':
            # Isolation Forest returns -1 (outlier) or 1 (inlier)
            # We'll adapt it for multi-class classification
            # Fit separate forests for each class (one-vs-rest)
            self.classifiers = []
            for class_idx in range(NUM_CLASSES):
                y_binary = (y_train == class_idx).astype(int)
                clf = IsolationForest(
                    n_estimators=100,
                    max_samples='auto',
                    contamination=0.1,
                    random_state=self.random_state
                )
                clf.fit(X_train[y_binary == 1])  # Train on positive samples only
                self.classifiers.append(clf)

        # Special handling for Apriori
        elif self.method_name == 'Apriori':
            self._fit_apriori(X_train, y_train, max_features = 10)

        # Special handling for PageRank
        elif self.method_name == 'PageRank':
            self._fit_pagerank(X_train, y_train)

        # Standard sklearn models
        else:
            self.model.fit(X_train, y_train)


    def predict(self, X_test):
        """
        Make predictions.

        Args:
            X_test (np.ndarray): Test features [N, D]

        Returns:
            np.ndarray: Predicted labels [N]
        """
        # Apply scaling if needed
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        # Special handling for k-Means
        if self.method_name == 'k-Means':
            cluster_labels = self.model.predict(X_test)
            predictions = np.array([self.cluster_to_class_mapping[c] for c in cluster_labels])
            return predictions

        # Special handling for EM (GMM)
        elif self.method_name == 'EM':
            component_labels = self.model.predict(X_test)
            predictions = np.array([self.cluster_to_class_mapping[c] for c in component_labels])
            return predictions

        # Special handling for Isolation Forest
        elif self.method_name == 'IsolationForest':
            scores = np.zeros((X_test.shape[0], NUM_CLASSES))
            for class_idx, clf in enumerate(self.classifiers):
                # Higher anomaly score = more likely to belong to this class
                scores[:, class_idx] = clf.score_samples(X_test)
            predictions = np.argmax(scores, axis=1)
            return predictions

        # Special handling for Apriori
        elif self.method_name == 'Apriori':
            return self._predict_apriori(X_test)

        # Special handling for PageRank
        elif self.method_name == 'PageRank':
            return self._predict_pagerank(X_test)

        # Standard sklearn models
        else:
            return self.model.predict(X_test)

    def predict_proba(self, X_test):
        """
        Get prediction probabilities.

        Args:
            X_test (np.ndarray): Test features [N, D]

        Returns:
            np.ndarray: Prediction probabilities [N, num_classes]
        """
        # Apply scaling if needed
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)


        # Special handling for methods without predict_proba
        if self.method_name in ['C4.5', 'CART', 'k-Means', 'EM', 'kNN', 'IsolationForest', 'Apriori', 'PageRank']:
            # Use voting from neighbors or return uniform distribution
            predictions = self.predict(X_test)
            proba = np.zeros((len(X_test), NUM_CLASSES))
            for i, pred in enumerate(predictions):
                proba[i, pred] = 1.0
            return proba

        # Standard sklearn models
        else:
            return self.model.predict_proba(X_test)


    def _map_clusters_to_classes(self, cluster_labels, true_labels):
        """
        Map cluster labels to class labels for unsupervised methods.

        Args:
            cluster_labels (np.ndarray): Cluster labels
            true_labels (np.ndarray): True class labels

        Returns:
            dict: Mapping from cluster to class
        """
        mapping = {}
        for cluster in np.unique(cluster_labels):
            # Find the most common true label for this cluster
            mask = (cluster_labels == cluster)
            if np.sum(mask) > 0:
                true_labels_in_cluster = true_labels[mask]
                most_common = np.bincount(true_labels_in_cluster).argmax()
                mapping[cluster] = most_common
            else:
                mapping[cluster] = 0  # Default to class 0
        return mapping


    def _fit_apriori(self, X_train, y_train, max_features=10):
        """
        Fit Apriori algorithm for association rule mining.
        Convert features to binary and extract rules.
        """

        """
        Use feature selection to reduce the number
        of features
        """
        # Initialize Selector
        selector = SelectKBest(f_classif, k=max_features)
        X_selected = selector.fit_transform(X_train, y_train)

        print(f"Original features count: {X_train.shape[1]}")
        print(f"The number of selected features after SelectKBest selection process: {X_selected.shape[1]}")

        # Discretize continuous features to binary
        X_binary = (X_selected > np.median(X_selected, axis=0)).astype(int)

        # Create DataFrame for mlxtend
        import pandas as pd
        df = pd.DataFrame(X_binary)
        df['label'] = y_train

        # Find frequent itemsets
        frequent_itemsets = apriori(
            df.drop('label', axis=1),
            min_support=0.1,
            use_colnames=True,
            low_memory=True  # Add the low memory option
        )

        # Generate association rules
        rules = association_rules(
            frequent_itemsets,
            metric="confidence",
            min_threshold=0.5
        )

        # Store rules and selector for prediction
        self.apriori_rules = rules
        self.apriori_threshold = np.median(X_train, axis=0)
        self.feature_selector = selector  # Save the selector for prediction
        self.selected_feature_indices = selector.get_support(indices=True)



    def _predict_apriori(self, X_test):
        """
        Predict using Apriori-based rules.
        """
        # For simplicity, use nearest neighbor with binary features
        X_binary = (X_test > self.apriori_threshold).astype(int)

        # Use k-NN as fallback (since Apriori is not directly a classifier)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_binary, np.zeros(len(X_binary)))  # Dummy labels
        # Return random predictions (Apriori is not suitable for this task)
        return np.random.randint(0, NUM_CLASSES, len(X_test))


    def _fit_pagerank(self, X_train, y_train):
        """
        Fit PageRank using k-NN graph construction.
        """
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = min(10, len(X_train))

        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_train)
        distances, indices = nbrs.kneighbors(X_train)

        # Create graph
        G = nx.Graph()
        for i in range(len(X_train)):
            G.add_node(i, label=y_train[i])
            for j, neighbor_idx in enumerate(indices[i]):
                weight = 1.0 / (distances[i][j] + 1e-6)
                G.add_edge(i, neighbor_idx, weight=weight)

        # Compute PageRank scores
        pagerank_scores = nx.pagerank(G, weight='weight')

        # Store for prediction
        self.pagerank_graph = G
        self.pagerank_scores = pagerank_scores
        self.X_train_pagerank = X_train
        self.y_train_pagerank = y_train


    def _predict_pagerank(self, X_test):
        """
        Predict using PageRank-based ranking.
        """
        from sklearn.neighbors import NearestNeighbors
        n_neighbors = min(10, len(self.X_train_pagerank))

        # Find nearest neighbors in training set
        nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(self.X_train_pagerank)
        distances, indices = nbrs.kneighbors(X_test)

        predictions = []
        for i in range(len(X_test)):
            # Weight neighbors by PageRank scores
            neighbor_indices = indices[i]
            neighbor_labels = self.y_train_pagerank[neighbor_indices]

            # Simple majority vote
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            pred = unique[np.argmax(counts)]
            predictions.append(pred)

        return np.array(predictions)


    def get_model_info(self):
        """
        Get model parameter information.

        Returns:
            str: Model parameters and description
        """
        info = f"Method: {self.method_name}\n"
        info += "=" * 80 + "\n"

        if self.method_name == 'C4.5':
            info += "Description: C4.5 Decision Tree (Information Gain)\n"
            info += "Parameters:\n"
            info += f"  - criterion: {self.model.criterion}\n"
            info += f"  - max_depth: {self.model.max_depth}\n"
            info += f"  - min_samples_split: {self.model.min_samples_split}\n"
            info += f"  - min_samples_leaf: {self.model.min_samples_leaf}\n"

        elif self.method_name == 'k-Means':
            info += "Description: k-Means Clustering (adapted for classification)\n"
            info += "Parameters:\n"
            info += f"  - n_clusters: {self.model.n_clusters}\n"
            info += f"  - max_iter: {self.model.max_iter}\n"
            info += f"  - n_init: {self.model.n_init}\n"

        elif self.method_name == 'SVM':
            info += "Description: Support Vector Machine (RBF kernel)\n"
            info += "Parameters:\n"
            info += f"  - C: {self.model.C}\n"
            info += f"  - kernel: {self.model.kernel}\n"
            info += f"  - gamma: {self.model.gamma}\n"
            info += f"  - probability: {self.model.probability}\n"

        elif self.method_name == 'Apriori':
            info += "Description: Apriori Algorithm (Association Rule Mining)\n"
            info += "Note: Apriori is for association rules, adapted via k-NN fallback\n"

        elif self.method_name == 'EM':
            info += "Description: Expectation-Maximization (Gaussian Mixture Model)\n"
            info += "Parameters:\n"
            info += f"  - n_components: {self.model.n_components}\n"
            info += f"  - covariance_type: {self.model.covariance_type}\n"
            info += f"  - max_iter: {self.model.max_iter}\n"

        elif self.method_name == 'PageRank':
            info += "Description: PageRank Algorithm (k-NN graph based)\n"
            info += "Note: PageRank adapted for classification via graph-based ranking\n"

        elif self.method_name == 'AdaBoost':
            info += "Description: AdaBoost (Adaptive Boosting)\n"
            info += "Parameters:\n"
            info += f"  - n_estimators: {self.model.n_estimators}\n"
            info += f"  - learning_rate: {self.model.learning_rate}\n"
            # info += f"  - algorithm: {self.model.algorithm}\n"

        elif self.method_name == 'kNN':
            info += "Description: k-Nearest Neighbors\n"
            info += "Parameters:\n"
            info += f"  - n_neighbors: {self.model.n_neighbors}\n"
            info += f"  - weights: {self.model.weights}\n"
            info += f"  - algorithm: {self.model.algorithm}\n"
            info += f"  - p (Minkowski): {self.model.p}\n"

        elif self.method_name == 'NaiveBayes':
            info += "Description: Gaussian Naive Bayes\n"
            info += "Parameters:\n"
            info += f"  - var_smoothing: {self.model.var_smoothing}\n"

        elif self.method_name == 'CART':
            info += "Description: CART Decision Tree (Gini Impurity)\n"
            info += "Parameters:\n"
            info += f"  - criterion: {self.model.criterion}\n"
            info += f"  - max_depth: {self.model.max_depth}\n"
            info += f"  - min_samples_split: {self.model.min_samples_split}\n"
            info += f"  - min_samples_leaf: {self.model.min_samples_leaf}\n"

        elif self.method_name == 'IsolationForest':
            info += "Description: Isolation Forest (Anomaly Detection)\n"
            info += "Parameters:\n"
            info += f"  - n_estimators: {self.model.n_estimators}\n"
            info += f"  - max_samples: {self.model.max_samples}\n"
            info += f"  - contamination: {self.model.contamination}\n"

        info += "=" * 80 + "\n"
        return info


# ==================== 5. Training Function ====================
def train_model(model, X_train, y_train, use_cv=False, cv_folds=5):
    """
    Train a traditional ML model.

    Args:
        model: TraditionalMLModels instance
        X_train (np.ndarray): Training features [N, D]
        y_train (np.ndarray): Training labels [N]
        use_cv (bool): Whether to use cross-validation
        cv_folds (int): Number of CV folds

    Returns:
        dict: Training results including accuracy and CV scores if applicable
    """
    print("\n" + "=" * 80)
    print(f"Training {model.method_name}...")
    print("=" * 80)
    print(model.get_model_info())

    # Train the model
    model.fit(X_train, y_train)

    # Training predictions
    train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)

    print(f"Training Accuracy: {train_acc*100:.2f}%")


    # Cross-validation (if supported and requested)
    cv_results = None
    if use_cv:
        print(f"\nPerforming {cv_folds}-fold Cross-Validation...")


        # Apply scaling if needed for CV

        if model.scaler is not None:
            # max-min × 2  SVM' | 'kNN' | 'k-Means
            X_train_scaled = model.scaler.transform(X_train)
        else:
            X_train_scaled = X_train


        # Methods that support sklearn's cross_val_score
        if model.method_name not in ['k-Means', 'EM', 'Apriori', 'PageRank', 'IsolationForest']:
            cv_scores = cross_val_score(
                model.model, X_train_scaled, y_train,
                cv=cv_folds, scoring='accuracy'
            )
            print(f"CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
            print(f"Individual fold accuracies: {[f'{s*100:.2f}%' for s in cv_scores]}")


            cv_results = {
                'cv_scores': cv_scores,
                'mean_cv_acc': cv_scores.mean(),
                'std_cv_acc': cv_scores.std()
            }

        else:
            print("Cross-validation not supported for this method")

    print("=" * 80)

    return {
        'train_accuracy': train_acc,
        'cv_results': cv_results
    }



# ==================== 6. Testing Function ====================
def test_model(model, X_test, y_test):
    """
    Test a trained model.

    Args:
        model: Trained TraditionalMLModels instance
        X_test (np.ndarray): Test features [N, D]
        y_test (np.ndarray): Test labels [N]

    Returns:
        dict: Test results including accuracy, confusion matrix, etc.
    """
    print("\n" + "=" * 80)
    print(f"Testing {model.method_name}...")
    print("=" * 80)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate metrics
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)

    # Print results
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    test_results = {
        'accuracy': test_acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': y_pred,
        'labels': y_test,
        'probabilities': y_proba
    }

    print("=" * 80)

    return test_results


# ==================== 7. Visualization Function ====================
def plot_training_results(model_name, train_results, test_results, cv_results=None):
    """
    Plot training and testing results.

    Args:
        model_name (str): Name of the model
        train_results (dict): Training results
        test_results (dict): Test results
        cv_results (dict, optional): Cross-validation results
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Training Accuracy Bar
    axes[0, 0].bar(['Training'], [train_results['train_accuracy']*100],
                   color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylim([0, 100])
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_title(f'{model_name} - Training Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 2. Test Accuracy Bar
    axes[0, 1].bar(['Testing'], [test_results['accuracy']*100],
                   color='coral', alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylim([0, 100])
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title(f'{model_name} - Testing Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')


    # 3. Cross-Validation Results (if available)
    if cv_results is not None and cv_results['cv_scores'] is not None:
        cv_scores = cv_results['cv_scores']
        fold_nums = np.arange(1, len(cv_scores) + 1)
        axes[1, 0].bar(fold_nums, cv_scores * 100,
                      color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1, 0].axhline(y=cv_results['mean_cv_acc'] * 100,
                          color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {cv_results["mean_cv_acc"]*100:.2f}%')
        axes[1, 0].set_ylim([0, 100])
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].set_title(f'{model_name} - Cross-Validation Results', fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Cross-Validation Results',
                        ha='center', va='center', fontsize=14,
                        transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Cross-Validation Results', fontsize=12, fontweight='bold')



    # 4. Confusion Matrix Heatmap
    cm = test_results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[1, 1].set_title(f'{model_name} - Confusion Matrix (Acc: {test_results["accuracy"]*100:.2f}%)',
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Label')
    axes[1, 1].set_ylabel('True Label')

    plt.tight_layout()


    # Save figure
    save_dir = r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\log_boards_model_pred_results'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name.lower()}_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    plt.cla()
    plt.close()
    #
    print(f"Visualization saved to: {save_path}")



# ==================== 8. Main Function ====================
def main():
    """
    Main function to run all 11 traditional ML methods.
    """
    print("\n" + "=" * 80)
    print("=" * 80)
    print("Traditional Machine Learning for Gait Classification")
    print("=" * 80)
    print("=" * 80)

    # ==================== Step 1: Load Data ====================
    print("\n[Step 1/5] Loading Data...")
    print("-" * 80)

    dogen_path = r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\gait64_neodegen_pkls'

    if not os.path.exists(dogen_path):
        print("WARNING: Data path not found. Using simulated data for demonstration...")
        num_samples = 100
        num_features = 300

        # Simulate data
        all_features = np.random.randn(num_samples, num_features).astype(np.float32)
        labels = np.random.randint(0, NUM_CLASSES, num_samples)

        # Split into train/test
        train_X, test_X, train_y, test_y = train_test_split(
            all_features, labels, test_size=0.5, random_state=42, stratify=labels
        )

    else:
        # Load real data
        train_X, train_y, test_X, test_y = load_dogen_data(
            dogen_path, train_ratio=0.5, top_k_entropy_th=200
        )

    print(f"Training set: {train_X.shape[0]} samples, {train_X.shape[1]} features")
    print(f"Test set: {test_X.shape[0]} samples, {test_X.shape[1]} features")

    # ==================== Step 2: Define Methods ====================
    print("\n[Step 2/5] Initializing Traditional ML Methods...")
    print("-" * 80)

    methods = [
        'C4.5',
        'k-Means',
        'SVM',
        'Apriori',
        'EM',
        'PageRank',
        'AdaBoost',
        'kNN',
        'NaiveBayes',
        'CART',
        'IsolationForest'
    ]

    print(f"Methods to evaluate ({len(methods)}):")
    for i, method in enumerate(methods, 1):
        print(f"  {i}. {method}")

    # ==================== Step 3: Train and Test All Methods ====================
    print("\n[Step 3/5] Training and Testing All Methods...")
    print("-" * 80)

    all_results = {}

    for method_name in methods:
        print(f"\n{'='*80}")
        print(f"Evaluating Method: {method_name}")
        print(f"{'='*80}")


        try:
            # Initialize model
            model = TraditionalMLModels(method_name, random_state=42)

            # Train model (with CV for applicable methods)
            use_cv = method_name not in ['k-Means', 'EM', 'Apriori', 'PageRank', 'IsolationForest']
            train_results = train_model(model, train_X, train_y, use_cv=use_cv, cv_folds=5)

            # Test model
            test_results = test_model(model, test_X, test_y)

            # Store results
            all_results[method_name] = {
                'train_results': train_results,
                'test_results': test_results
            }

            # Plot results
            plot_training_results(method_name, train_results, test_results, train_results['cv_results'])

        except Exception as e:
            print(f"\nERROR in {method_name}: {e}")
            import traceback
            traceback.print_exc()
            continue



    # ==================== Step 4: Summary ====================
    print("\n[Step 4/5] Generating Summary...")
    print("-" * 80)

    print("\n" + "=" * 80)
    print("SUMMARY OF ALL METHODS")
    print("=" * 80)
    print(f"{'Method':<20} {'Train Acc (%)':<15} {'Test Acc (%)':<15}")
    print("-" * 80)


    for method_name, results in all_results.items():
        train_acc = results['train_results']['train_accuracy'] * 100
        test_acc = results['test_results']['accuracy'] * 100
        print(f"{method_name:<20} {train_acc:<15.2f} {test_acc:<15.2f}")

    print("=" * 80)

    # Find best method
    best_method = max(all_results.keys(),
                     key=lambda k: all_results[k]['test_results']['accuracy'])
    best_acc = all_results[best_method]['test_results']['accuracy'] * 100

    print(f"\nBest Method: {best_method} (Test Accuracy: {best_acc:.2f}%)")


    # ==================== Step 5: Save Results ====================
    '''
    ... It is necessary to increase the saving
     of the ML model... Currently, only the
    results are saved...
    '''
    print("\n[Step 5/5] Saving Results...")
    print("-" * 80)

    save_dir = r'D:\gait-in-neurodegenerative-disease-database-1.0.0\gait-in-neurodegenerative-disease-database-1.0.0\neodogen_models'
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'traditional_ml_results.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"Results saved to: {save_path}")

    print("\n" + "=" * 80)
    print("All methods evaluated successfully!")
    print("=" * 80)

    return all_results


# # ==================== Additional Dependencies for Apriori ====================
# def check_dependencies():
#     """
#     Check and install additional dependencies if needed.
#     """
#     print("Checking dependencies...")
#
#     missing = []
#
#     try:
#         import mlxtend
#     except ImportError:
#         missing.append('mlxtend')
#
#     try:
#         import networkx
#     except ImportError:
#         missing.append('networkx')
#
#     if missing:
#         print(f"\nWARNING: Missing dependencies: {missing}")
#         print("Please install them using:")
#         print(f"  pip install {' '.join(missing)}")
#         print("\nFor Apriori (mlxtend) and PageRank (networkx) support")
#     else:
#         print("All dependencies are installed!")


if __name__ == "__main__":
    # Check dependencies first
    # check_dependencies()

    # Run main pipeline
    results = main()

    print("\n" + "=" * 80)
    print("Program completed successfully!")
    print("=" * 80)

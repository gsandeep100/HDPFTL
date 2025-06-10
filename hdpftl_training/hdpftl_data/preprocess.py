# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        preprocess.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-24
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import gc
import glob
import os
from glob import glob

import numpy as np
from imblearn.over_sampling import SVMSMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from hdpftl_training.hdpftl_data.sampling import stratified_downsample
from hdpftl_utility.config import OUTPUT_DATASET_ALL_DATA
from hdpftl_utility.utils import named_timer

"""
🎯 What is PCA (Principal Component Analysis)?
PCA is a dimensionality reduction technique that transforms a dataset with many features 
into a smaller set of new variables (called principal components) that still contain most of the original 
information.
Compressing data — like converting a full HD photo into a smaller version while keeping key details intact.

"""

from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import pandas as pd
import warnings


# 🔍 Step 0: Profile
def profile_dataset(X, y):
    print("📐 Feature shape:")
    print(f"  ➤ X shape: {X.shape}")

    print("\n📊 Class distribution:")
    counts = Counter(y)
    for label, count in counts.items():
        print(f"  ➤ Class {label}: {count} samples")

    imbalance_ratio = max(counts.values()) / min(counts.values())
    print(f"\n⚖️  Imbalance Ratio: {imbalance_ratio:.2f}")

    print("\n🔍 Data type inspection:")
    print(pd.DataFrame(X).dtypes.value_counts())


# 🧪 Step 1: PCA
def reduce_dim(X, n_components=30):
    print(f"\n🔧 Reducing dimensions from {X.shape[1]} → {n_components} using PCA")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)


# ⚖️ Step 2: SMOTE
def fast_safe_smote(X, y, k_neighbors=5):
    counts = Counter(y)
    min_class_size = min(counts.values())
    k = min(k_neighbors, min_class_size - 1)
    if k < 1:
        warnings.warn("Too few samples for SMOTE; skipping.")
        return X, y
    print(f"\n⚡ Applying SMOTE with k={k}")
    sm = SMOTE(k_neighbors=k, random_state=42)
    result = sm.fit_resample(X, y)
    # Ensure only X and y are returned, even if more values are present
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    return result


# 🌀 Step 3: Hybrid
def hybrid_balance(X, y):
    print("\n🌀 Applying hybrid balancing (undersample + SMOTE)")
    under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    over = SMOTE(k_neighbors=5, random_state=42)
    pipeline = Pipeline([('under', under), ('over', over)])
    try:
        result = pipeline.fit_resample(X, y)
        if result is None:
            warnings.warn("Pipeline fit_resample returned None.")
            return X, y
        X_res, y_res = result
        return X_res, y_res
    except Exception as e:
        warnings.warn(f"Hybrid balancing failed: {e}")
        return X, y

    # 🎯 Master Function
    # 🎯 Master function with pre-sampling

    """AI is creating summary for 
    Summary Table
'pca_smote'	PCA + SMOTE (faster, low-dimensional)
'hybrid'	Undersample + SMOTE (original features)
'smote_only'	SMOTE only (no PCA)
'none'	No resampling, just profiling    """


def prepare_data(X, y, strategy='pca_smote', n_components=30, pre_sample=False, sample_fraction=0.1):
    print("📊 Running prepare_data with strategy:", strategy)

    # Optional downsampling
    if pre_sample:
        X, y = stratified_downsample(X, y, fraction=sample_fraction)

    profile_dataset(X, y)

    if strategy == 'pca_smote':
        X_reduced = reduce_dim(X, n_components=n_components)
        X_final, y_final = fast_safe_smote(X_reduced, y)
    elif strategy == 'hybrid':
        X_final, y_final = hybrid_balance(X, y)
    elif strategy == 'smote_only':
        X_final, y_final = fast_safe_smote(X, y)
    elif strategy == 'none':
        print("\n🚫 No resampling applied. Returning original X, y.")
        X_final, y_final = X, y
    else:
        raise ValueError("❌ Invalid strategy. Choose from 'pca_smote', 'hybrid', 'smote_only', or 'none'.")

    print("\n✅ Final shape:", X_final.shape)
    print("✅ Final class distribution:", Counter(y_final))
    return X_final, y_final


def safe_smote(X, y):
    counts = Counter(y)
    min_class_size = min(counts.values())
    k = min(5, min_class_size - 1)
    if k < 1:
        print("Too few samples for SMOTE; skipping.")
        return X, y
    smote = SVMSMOTE(k_neighbors=k, random_state=42)
    return smote.fit_resample(X, y)


def load_and_label_all(folder_path, benign_keywords=['benign'], attack_keywords=None):
    all_files = glob(os.path.join(folder_path, "*.csv")) + glob(os.path.join(folder_path, "*.CSV"))

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in: {os.path.abspath(folder_path)}")
    print(f"Found {len(all_files)} CSV files in '{folder_path}'")
    combined_df = []

    for count, file in enumerate(all_files, start=1):
        df = pd.read_csv(file)
        filename = os.path.basename(file).lower()
        print(f"Count: {count}, Processing File: {file}")

        # Determine label from filename
        if any(kw in filename for kw in benign_keywords):
            df['Label'] = 0
        else:
            df['Label'] = 1  # assume attack if not explicitly benign

        """
        df = safe_clean_dataframe(
            df,
            chunk_size=5000,
            invalid_values=[-999, '?'],
            replace_with=np.nan,
            log_progress=True
        )
        """
        pd.set_option('future.no_silent_downcasting', True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # df.infer_objects(copy=False)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1, inplace=True, errors="ignore")
        df.columns = df.columns.str.strip()
        # label_encoder = LabelEncoder()
        # df['Label'] = label_encoder.fit_transform(df['Label'])
        # label = 0 if 'benign' in os.path.basename(path).lower() else 1
        # df['Label'] = label
        df = df.select_dtypes(include=[np.number]).dropna(axis=1)
        combined_df.append(df)

    final_df = pd.concat(combined_df, ignore_index=True)
    return final_df


def safe_clean_dataframe(df: pd.DataFrame,
                         chunk_size: int = 10000,
                         invalid_values=None,
                         replace_with=np.nan,
                         log_progress: bool = True,
                         auto_gc: bool = True) -> pd.DataFrame:
    """
    Safely replaces infinities and other specified invalid values in a large DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to clean.
        chunk_size (int): Rows to process per chunk (default: 10000).
        invalid_values (list): Additional values to replace (e.g., [-999, '?']).
        replace_with: Value to replace with (default: np.nan).
        log_progress (bool): If True, print progress updates.
        auto_gc (bool): If True, run garbage collection after each chunk.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Default values to clean
    if invalid_values is None:
        invalid_values = []
    values_to_replace = [np.inf, -np.inf] + invalid_values

    df = df.copy()
    total_rows = len(df)

    for start in range(0, total_rows, chunk_size):
        end = min(start + chunk_size, total_rows)
        if log_progress:
            print(f"Processing rows {start} to {end - 1}...")

        df.iloc[start:end] = df.iloc[start:end].replace(values_to_replace, replace_with)

        if auto_gc:
            gc.collect()

    if log_progress:
        print("Cleaning complete.")

    return df


def preprocess_data(writer=None):
    # all_files = glob.glob(path + "*.csv")
    # df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    with named_timer("load_and_label_all", writer, tag="load_and_label_all"):
        df = load_and_label_all(OUTPUT_DATASET_ALL_DATA)
    # Replace infinities and -999 or '?' with NaN
    scaler = MinMaxScaler()
    features = df.columns.difference(['Label'])
    df[features] = scaler.fit_transform(df[features])
    X, y = df[features], df['Label']

    # downsampling
    # with named_timer("downsample", writer, tag="downsample"):
    #   X_small, y_small = stratified_downsample(X, y, fraction=0.2)

    # print(df.columns)

    # SVMSMOTE- Create synthetic minority points near SVM boundary (critical zones).
    #           Makes the minority class stronger exactly where it matters — at the decision boundary.
    # SMOTEENN- Clean noisy samples after oversampling using Edited Nearest Neighbors.
    #           Removes confusing and overlapped samples, making the classifier much more accurate.

    # resampling_pipeline = Pipeline([
    #     ('svm_smote', SVMSMOTE(random_state=42)),
    #     ('smote_enn', SMOTEENN(random_state=42))
    # ])
    with named_timer("safe_smote", writer, tag="safe_smote"):
        # Call the function with your dataset
        X_final, y_final = prepare_data(X, y, strategy='hybrid')
        """
        smote_result = safe_smote(X_small, y_small)
        if smote_result is None:
            X_final, y_final = X_small, y_small
        elif isinstance(smote_result, tuple) and len(smote_result) >= 2:
            X_final, y_final = smote_result[:2]
        else:
            X_final, y_final = X_small, y_small
        """
    with named_timer("train_test_split", writer, tag="train_test_split"):
        X_temp, X_test, y_temp, y_test = train_test_split(X_final, y_final, test_size=0.1, random_state=42,
                                                          stratify=y_final)
        X_pretrain, X_finetune, y_pretrain, y_finetune = train_test_split(
            X_temp, y_temp, test_size=0.1, stratify=y_temp, random_state=42
        )
        # Scale features
        scaler = StandardScaler()
        if isinstance(X_pretrain, pd.Series):
            X_pretrain = X_pretrain.to_frame()
        X_pretrain = scaler.fit_transform(X_pretrain)
        X_test = scaler.fit_transform(X_test)

        return X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test


"""
def preprocess_data_small(csv_path, test_size=0.2):
    # Load hdpftl_dataset
    df = pd.read_csv(csv_path + "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)

    # Drop unnamed or constant columns
    df.drop(columns=[col for col in df.columns if 'Unnamed' in col or df[col].nunique() <= 1], inplace=True)

    # Drop rows with NaN or inf
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Separate label
    y = df[' Label']
    df.drop(columns=[' Label'], inplace=True)

    # Drop non-numeric columns (e.g., Flow ID, Source IP, etc.)
    df = df.select_dtypes(include=['float64', 'int64'])

    # Encode label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Optional: show label mapping
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_).tolist())))

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, stratify=y_encoded, random_state=42
    )

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_test, y_train, y_test
    """

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
from sklearn.preprocessing import LabelEncoder

from hdpftl_training.hdpftl_data.sampling import stratified_downsample
from hdpftl_utility.config import OUTPUT_DATASET_ALL_DATA
from hdpftl_utility.log import safe_log
from hdpftl_utility.utils import named_timer, to_float32

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


def calculate_imbalance_ratio(counts):
    """
    Calculate the imbalance ratio of class counts.

    Args:
        counts (dict): Dictionary with class labels as keys and their counts as values.

    Returns:
        float or None: imbalance ratio (max/min) or None if counts is empty.
    """
    if counts and all(v > 0 for v in counts.values()):
        imbalance_ratio = max(counts.values()) / min(counts.values())
        return imbalance_ratio
    else:
        # Handle empty dict or zero counts safely
        print("Warning: counts dictionary is empty or contains zero counts.")
        return None


# 🔍 Step 0: Profile
def profile_dataset(X, y):
    safe_log("📐 Feature shape:")
    safe_log(f"  ➤ X shape: {X.shape}")

    safe_log("\n📊 Class distribution:")
    counts = Counter(y)
    for label, count in counts.items():
        safe_log(f"  ➤ Class {label}: {count} samples")

    imbalance_ratio = calculate_imbalance_ratio(counts)
    if imbalance_ratio is not None:
        safe_log(f"Imbalance ratio: {imbalance_ratio:.2f}")
    else:
        safe_log("Imbalance ratio could not be calculated due to missing or invalid data.")

    safe_log("\n🔍 Data type inspection:")
    safe_log(pd.DataFrame(X).dtypes.value_counts())


# 🧪 Step 1: PCA
def reduce_dim(X, n_components=30):
    safe_log(f"\n🔧 Reducing dimensions from {X.shape[1]} → {n_components} using PCA")
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
    safe_log(f"\n⚡ Applying SMOTE with k={k}")
    sm = SMOTE(k_neighbors=k, random_state=42)
    result = sm.fit_resample(X, y)
    # Ensure only X and y are returned, even if more values are present
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    return result


# 🌀 Step 3: Hybrid
def hybrid_balance(X, y):
    safe_log("\n🌀 Applying hybrid balancing (undersample + SMOTE)")
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
    safe_log("📊 Running prepare_data with strategy:", strategy)

    # Optional downsampling
    if pre_sample:
        X, y = stratified_downsample(X, y, fraction=sample_fraction)

    profile_dataset(X, y)
    # Force float32 to save memory
    X = to_float32(X)

    try:
        if strategy == 'pca_smote':
            X_reduced = reduce_dim(X, n_components=n_components)
            X_final, y_final = fast_safe_smote(X_reduced, y)
        elif strategy == 'hybrid':
            X_final, y_final = hybrid_balance(X, y)
        elif strategy == 'smote_only':
            X_final, y_final = fast_safe_smote(X, y)
        elif strategy == 'none':
            safe_log("\n🚫 No resampling applied. Returning original X, y.")
            X_final, y_final = X, y
        else:
            raise ValueError("❌ Invalid strategy. Choose from 'pca_smote', 'hybrid', 'smote_only', or 'none'.")

        safe_log("\n✅ Final shape:", X_final.shape)
        safe_log("✅ Final class distribution:", Counter(y_final))
        return X_final, y_final

    finally:
        # Cleanup memory
        del X, y
        gc.collect()


def safe_smote(X, y):
    counts = Counter(y)
    min_class_size = min(counts.values())
    k = min(5, min_class_size - 1)
    if k < 1:
        safe_log("Too few samples for SMOTE; skipping.")
        return X, y
    smote = SVMSMOTE(k_neighbors=k, random_state=42)
    return smote.fit_resample(X, y)


def assign_labels(
    df,
    filename,
    benign_keywords=None,
    attack_keywords=None,
    multiclass_keywords=None,
    manual_label_map=None,
    content_label_column_candidates=None
):
    filename_lower = os.path.basename(filename).lower()

    # --- Case 1: Use existing 'Label' column ---
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip()
        label_encoder = LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'])
        safe_log(f"Label mapping from existing column for {filename}")
        return df

    # --- Case 2: Multiclass keywords from filename ---
    if multiclass_keywords:
        for label_value, keywords in multiclass_keywords.items():
            if any(kw.lower() in filename_lower for kw in keywords):
                df['Label'] = label_value
                safe_log(f"Multiclass label {label_value} assigned via filename for {filename}")
                return df

    # --- Case 3: Binary keywords from filename ---
    if benign_keywords and any(kw.lower() in filename_lower for kw in benign_keywords):
        df['Label'] = 0
        safe_log(f"Binary label 0 assigned (benign) via filename for {filename}")
        return df
    elif attack_keywords and any(kw.lower() in filename_lower for kw in attack_keywords):
        df['Label'] = 1
        safe_log(f"Binary label 1 assigned (attack) via filename for {filename}")
        return df

    # --- Case 4: Manual mapping from filename ---
    if manual_label_map:
        label = manual_label_map.get(filename_lower)
        if label is not None:
            df['Label'] = label
            safe_log(f"Manual label {label} assigned via filename for {filename}")
            return df

    # --- ✅ NEW: Case 5 - Content-based keyword search ---
    if content_label_column_candidates:
        for col in content_label_column_candidates:
            if col in df.columns:
                col_values = df[col].astype(str).str.lower()
                if benign_keywords and col_values.str.contains('|'.join(benign_keywords), na=False).any():
                    df['Label'] = 0
                    safe_log(f"Content-based label 0 (benign) assigned using column '{col}' in {filename}")
                    return df
                if attack_keywords and col_values.str.contains('|'.join(attack_keywords), na=False).any():
                    df['Label'] = 1
                    safe_log(f"Content-based label 1 (attack) assigned using column '{col}' in {filename}")
                    return df

    # --- Final fallback ---
    df['Label'] = np.nan
    safe_log(f"⚠️ Could not infer label for: {filename}")
    return df


def load_and_label_all(
        folder_path,
        benign_keywords=None,
        attack_keywords=None,
        multiclass_keywords=None,
        manual_label_map=None
):
    csv_files = glob(os.path.join(folder_path, "*.csv")) + glob(os.path.join(folder_path, "*.CSV"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {os.path.abspath(folder_path)}")

    safe_log(f"Found {len(csv_files)} CSV files in '{folder_path}'")
    all_data = []

    for idx, file in enumerate(csv_files, start=1):
        safe_log(f"[{idx}] Loading: {file}")
        try:
            df = pd.read_csv(file)
        except Exception as e:
            safe_log(f"⚠️ Error reading {file}: {e}")
            continue

        df.columns = df.columns.str.strip()
        df = assign_labels(df, file, benign_keywords, attack_keywords, multiclass_keywords, manual_label_map)

        # Clean
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1, inplace=True, errors='ignore')

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' not in numeric_cols:
            numeric_cols.append('Label')
        df = df[numeric_cols].dropna(axis=1)

        all_data.append(df)
        del df
        gc.collect()

    final_df = pd.concat(all_data, ignore_index=True)
    safe_log(f"Final shape: {final_df.shape}")
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
            safe_log(f"Processing rows {start} to {end - 1}...")

        df.iloc[start:end] = df.iloc[start:end].replace(values_to_replace, replace_with)

        if auto_gc:
            gc.collect()

    if log_progress:
        safe_log("Cleaning complete.")

    return df


from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(selected_folder, writer=None, scaler_type='minmax'):
    with named_timer("load_and_label_all", writer, tag="load_and_label_all"):
        df = load_and_label_all(os.path.join(OUTPUT_DATASET_ALL_DATA, selected_folder))

    features = df.columns.difference(['Label'])
    df[features] = df[features].astype(np.float32)

    X, y = df[features], df['Label']
    X = to_float32(X)

    with named_timer("safe_smote", writer, tag="safe_smote"):
        X_final, y_final = prepare_data(X, y, strategy='hybrid')
        del df, X, y
        gc.collect()

    with named_timer("train_test_split", writer, tag="train_test_split"):
        if len(X_final) > 0 and len(y_final) > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_final, y_final, test_size=0.1, stratify=y_final, random_state=42)
            X_pretrain, X_finetune, y_pretrain, y_finetune = train_test_split(
                X_temp, y_temp, test_size=0.1, stratify=y_temp, random_state=42)
            del X_temp, y_temp
            gc.collect()

            if scaler_type == 'minmax':
                scaler = MinMaxScaler()
                # Fit+transform on all data before splitting is more typical for minmax,
                # but here we do it on pretrain/fine tune/test split to be consistent.
                X_pretrain = scaler.fit_transform(X_pretrain)
                X_finetune = scaler.transform(X_finetune)
                X_test = scaler.transform(X_test)

            elif scaler_type == 'standard':
                scaler = StandardScaler()
                X_pretrain = scaler.fit_transform(X_pretrain)
                X_finetune = scaler.transform(X_finetune)
                X_test = scaler.transform(X_test)

            else:
                raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    return X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test

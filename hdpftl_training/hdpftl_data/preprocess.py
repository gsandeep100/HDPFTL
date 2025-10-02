# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        preprocess.py
   Description:      HPFL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HPFL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-24
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import gc
import glob
import hashlib
import os
import warnings
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from contextlib import nullcontext
from glob import glob
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import hdpftl_training.hdpftl_data.sampling as sampling
import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util


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
    log_util.safe_log("📐 Feature shape:")
    log_util.safe_log(f"  ➤ X shape: {X.shape}")

    log_util.safe_log("\n📊 Class distribution:")
    counts = Counter(y)
    for label, count in counts.items():
        log_util.safe_log(f"  ➤ Class {label}: {count} samples")

    imbalance_ratio = calculate_imbalance_ratio(counts)
    if imbalance_ratio is not None:
        log_util.safe_log(f"Imbalance ratio: {imbalance_ratio:.2f}")
    else:
        log_util.safe_log("Imbalance ratio could not be calculated due to missing or invalid data.")

    log_util.safe_log("\n🔍 Data type inspection:")
    log_util.safe_log(pd.DataFrame(X).dtypes.value_counts())


# 🧪 Step 1: PCA
def reduce_dim(X, n_components=30):
    log_util.safe_log(f"\n🔧 Reducing dimensions from {X.shape[1]} → {n_components} using PCA")
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
    log_util.safe_log(f"\n⚡ Applying SMOTE with k={k}")
    sm = SMOTE(k_neighbors=k, random_state=42)
    result = sm.fit_resample(X, y)
    # Ensure only X and y are returned, even if more values are present
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0], result[1]
    return result

def hybrid_balance(X, y, reduce_dim=True, n_components=50, k_neighbors=5):
    """
    Fast SMOTE-based balancing for large datasets.
    - Optional dimensionality reduction with PCA to speed up nearest neighbor search.
    - Keeps SMOTE-only logic (no undersampling).
    - Falls back gracefully if SMOTE fails.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix
        y (pd.Series or np.ndarray): Target labels
        reduce_dim (bool): Apply PCA to reduce feature dimension
        n_components (int): Number of PCA components if reduce_dim=True
        k_neighbors (int): Number of neighbors for SMOTE

    Returns:
        X_res, y_res: Balanced feature matrix and labels
    """
    log_util.safe_log("\n🌀 Applying modified hybrid balancing (fast SMOTE)")

    try:
        X_orig = X.copy()
        if reduce_dim and X.shape[1] > n_components:
            # Reduce dimensionality for faster SMOTE
            pca = PCA(n_components=n_components, random_state=42)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X.values if hasattr(X, "values") else X

        # Remove n_jobs here; SMOTE doesn't accept it
        sm = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = sm.fit_resample(X_reduced, y)

        # If PCA was used, map back to original space approximately
        if reduce_dim and X.shape[1] > n_components:
            X_res = pca.inverse_transform(X_res)

        return X_res, y_res

    except Exception as e:
        warnings.warn(f"SMOTE failed in hybrid_balance: {e}")
        return X_orig, y


"""AI is creating summary for 
    Summary Table
'pca_smote'	PCA + SMOTE (faster, low-dimensional)
'hybrid'	Undersample + SMOTE (original features)
'smote_only'	SMOTE only (no PCA)
'none'	No resampling, just profiling    """


def prepare_data(X, y, strategy='pca_smote', n_components=30, pre_sample=False, sample_fraction=0.3, verbose=True):
    if verbose:
        log_util.safe_log("📊 Running prepare_data with strategy:", strategy)

    # Optional downsampling
    if pre_sample:
        X, y = sampling.stratified_downsample(X, y, fraction=sample_fraction)

    profile_dataset(X, y)
    # Force float32 to save memory
    X = util.to_float32(X)

    try:
        if strategy == 'pca_smote':
            X_reduced = reduce_dim(X, n_components=n_components)
            X_final, y_final = fast_safe_smote(X_reduced, y)
        elif strategy == 'hybrid':
            X_final, y_final = hybrid_balance(X, y)
        elif strategy == 'smote_only':
            X_final, y_final = fast_safe_smote(X, y)
        elif strategy == 'none':
            if verbose:
                log_util.safe_log("\n🚫 No resampling applied. Returning original X, y.")
            X_final, y_final = X, y
        else:
            raise ValueError("❌ Invalid strategy. Choose from 'pca_smote', 'hybrid', 'smote_only', or 'none'.")

        if verbose:
            log_util.safe_log("\n✅ Final shape:", X_final.shape)
            log_util.safe_log("✅ Final class distribution:", Counter(y_final))
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
        log_util.safe_log("Too few samples for SMOTE; skipping.")
        return X, y
    smote = SVMSMOTE(k_neighbors=k, random_state=42)
    return smote.fit_resample(X, y)


def assign_labels_numeric(df, filename):
    """
    Assign numeric multiclass labels (0–7) to df['Label']
    based on keywords found in the filename.
    """
    multiclass_keywords = {
        0: ["benign"],  # Benign
        1: ["ddos", "botnet"],  # DDoS
        2: ["brute", "force"],  # Brute Force
        3: ["spoof", "fake", "MITM", "DNS"],  # Spoofing
        4: ["dos", "denial"],  # DoS
        5: ["recon", "scan"],  # Recon
        6: ["web", "http", "Uploading_Attack.pcap_Flow", "XSS.pcap_Flow", "SqlInjection.pcap_Flow"],  # Web-based
        7: ["mirai", "malware"]  # Mirai
    }

    filename_lower = os.path.basename(filename).lower()

    # Search for a matching keyword
    for numeric_label, keywords in multiclass_keywords.items():
        if any(kw in filename_lower for kw in keywords):
            df["Label"] = numeric_label
            print(f"[INFO] Assigned numeric label {numeric_label} based on filename: {filename}")
            return df

    # Fallback if no keyword matches
    df["Label"] = -1  # or choose a default like 0
    print(f"[WARN] No keyword match for {filename} → assigned -1")
    return df


def assign_labels(
        df,
        filename,
        benign_keywords=None,
        attack_keywords=None,
        multiclass_keywords=None,
        manual_label_map=None,
        content_label_column_candidates=None
):
    """
    Assign a label column to a dataframe based on:
      1. Existing 'Label' column
      2. 8-class filename keywords (multiclass_keywords)
      3. Binary filename keywords (benign/attack)
      4. Manual filename mapping
      5. Content-based keyword search in columns
      6. Fallback to benign (0)
    """
    filename_lower = os.path.basename(filename).lower()

    # --- Case 1: Existing 'Label' column ---
    if 'Label' in df.columns:
        df['Label'] = df['Label'].astype(str).str.strip()
        label_encoder = LabelEncoder()
        df['Label'] = label_encoder.fit_transform(df['Label'])
        df.attrs['label_mapping'] = dict(
            zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
        )
        print(f"[INFO] Label mapping from existing column for {filename}")
        return df

    # --- Case 2: 8-class filename keywords ---
    if multiclass_keywords:
        for label_value, keywords in multiclass_keywords.items():
            if any(kw.lower() in filename_lower for kw in keywords):
                df['Label'] = label_value
                print(f"[INFO] Multiclass label {label_value} assigned via filename for {filename}")
                return df

    # --- Case 3: Binary filename keywords ---
    if benign_keywords and any(kw.lower() in filename_lower for kw in benign_keywords):
        df['Label'] = 0
        print(f"[INFO] Binary label 0 (benign) assigned via filename for {filename}")
        return df
    elif attack_keywords and any(kw.lower() in filename_lower for kw in attack_keywords):
        df['Label'] = 1
        print(f"[INFO] Binary label 1 (attack) assigned via filename for {filename}")
        return df

    # --- Case 4: Manual filename mapping ---
    if manual_label_map:
        label = manual_label_map.get(filename_lower)
        if label is not None:
            df['Label'] = label
            print(f"[INFO] Manual label {label} assigned via filename for {filename}")
            return df

    # --- Case 5: Content-based keyword search ---
    if content_label_column_candidates:
        for col in content_label_column_candidates:
            if col in df.columns:
                col_values = df[col].astype(str).str.lower()
                if benign_keywords and col_values.str.contains('|'.join(benign_keywords), na=False).any():
                    df['Label'] = 0
                    print(f"[INFO] Content-based label 0 (benign) assigned using column '{col}' in {filename}")
                    return df
                if attack_keywords and col_values.str.contains('|'.join(attack_keywords), na=False).any():
                    df['Label'] = 1
                    print(f"[INFO] Content-based label 1 (attack) assigned using column '{col}' in {filename}")
                    return df

    # --- Fallback ---
    df['Label'] = -1
    print(f"[WARN] ⚠️ Could not infer label for: {filename} — defaulting to 0 (benign)")
    return df


def get_cache_path(folder_path):
    hash_digest = hashlib.md5(folder_path.encode()).hexdigest()
    cache_dir = os.path.join(".cache", "processed_datasets")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{hash_digest}.parquet")


def process_single_file(file,
                        drop_columns=None):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        log_util.safe_log(f"⚠️ Error reading {file}: {e}")
        return None

    df.columns = df.columns.str.strip()
    df = assign_labels_numeric(df, file)

    # 🔐 Convert label to numeric and drop invalids
    df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
    df.dropna(subset=['Label'], inplace=True)
    if df.empty:
        log_util.safe_log(f"❌ Skipping {file}: No valid labels after conversion.")
        return None
    df['Label'] = df['Label'].astype(int)

    # 🧹 Clean
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(drop_columns, axis=1, inplace=True, errors='ignore')

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Label' not in numeric_cols:
        numeric_cols.append('Label')
    df = df[numeric_cols].dropna(axis=1)

    if df.empty:
        log_util.safe_log(f"⚠️ Skipping {file}: Cleaned DataFrame is empty.")
        return None
    df = df.convert_dtypes()
    df = df.infer_objects()
    return df


def plot_skewed_features_log_transform(df, features_per_fig=10):
    """
    Plot original and log1p-transformed distributions of highly skewed features in chunks.

    Args:
        df (pd.DataFrame): Input dataframe containing the features.
        features_per_fig (int, optional): Number of features per figure/page. Default is 10.
    """
    # Your full list of skewed features
    skewed_features = [
        "Total Fwd Packet", "Total Bwd packets", "Total Length of Fwd Packet",
        "Total Length of Bwd Packet", "Fwd Packet Length Max", "Fwd Packet Length Min",
        "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max",
        "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow IAT Std", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max",
        "Bwd IAT Std", "Bwd IAT Max", "Fwd PSH Flags", "Fwd Header Length",
        "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Packet Length Min",
        "Packet Length Max", "Packet Length Mean", "Packet Length Std",
        "Packet Length Variance", "FIN Flag Count", "PSH Flag Count", "ACK Flag Count",
        "CWR Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
        "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Bwd Bytes/Bulk Avg",
        "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg", "Subflow Fwd Packets",
        "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "FWD Init Win Bytes", "Bwd Init Win Bytes", "Fwd Act Data Pkts",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
    ]
    for i in range(0, len(skewed_features), features_per_fig):
        chunk = skewed_features[i:i+features_per_fig]
        n_features = len(chunk)

        plt.figure(figsize=(12, n_features*2))

        for j, feature in enumerate(chunk, 1):
            original = df[feature].fillna(0)
            log_transformed = np.log1p(original)

            # Original feature
            plt.subplot(n_features, 2, 2*j-1)
            sns.histplot(original, bins=50, kde=True, color='skyblue')
            plt.title(f"{feature} (Original) - skew={original.skew():.2f}")
            plt.xlabel("")

            # Log-transformed feature
            plt.subplot(n_features, 2, 2*j)
            sns.histplot(log_transformed, bins=50, kde=True, color='salmon')
            plt.title(f"{feature} (Log1p) - skew={log_transformed.skew():.2f}")
            plt.xlabel("")

        plt.tight_layout()
        plt.show()



def log_transform_skewed_features(
    df,
    target_col='label',
    skew_threshold=2.0,
    extreme_skew_threshold=1000,
    clip_threshold=1e6,
    top_n=None,
    scale_method='standard',
    drop_extreme_skew=True,
    remove_constant=True,
    constant_threshold=1e-6,
    verbose=True,
    report_top_n=10
):
    """
    Fully vectorized log-transform for highly skewed numeric features.
    Automatically excludes target column from transformation to avoid issues in classification.

    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str or None): Target column to exclude from transformation
        skew_threshold (float): |skew| above which log1p is applied
        extreme_skew_threshold (float): |skew| above which columns may be dropped
        clip_threshold (float): Clip numeric values to ±clip_threshold
        top_n (int or None): Only transform top N skewed positive features
        scale_method (str): 'standard', 'minmax', or None
        drop_extreme_skew (bool): Drop extreme-skew columns
        remove_constant (bool): Drop near-constant features
        constant_threshold (float): Variance threshold for near-constant features
        verbose (bool): Print debug info
        report_top_n (int): Number of top skewed features to report

    Returns:
        pd.DataFrame: Transformed dataframe
    """
    df_safe = df.copy()

    # 1️⃣ Identify numeric columns, exclude target
    numeric_cols = df_safe.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    # 2️⃣ Drop extreme-skew features
    skews = df_safe[numeric_cols].skew()
    if drop_extreme_skew:
        extreme_cols = skews[abs(skews) > extreme_skew_threshold].index.tolist()
        if extreme_cols:
            df_safe.drop(columns=extreme_cols, inplace=True)
            numeric_cols = [c for c in numeric_cols if c not in extreme_cols]
            if verbose:
                print(f"❌ Dropped extreme-skew features: {extreme_cols}")

    # Ensure numeric_cols exist after dropping
    numeric_cols = [c for c in numeric_cols if c in df_safe.columns]

    # 3️⃣ Report top skewed features before
    skew_before = df_safe[numeric_cols].skew().abs().sort_values(ascending=False)
    if verbose:
        print(f"\nTop {report_top_n} skewed features BEFORE log-transform:")
        print(skew_before.head(report_top_n))

    # 4️⃣ Select high-skew, positive columns for log1p
    skewed_cols = skew_before[skew_before > skew_threshold].index.tolist()
    positive_cols = [c for c in skewed_cols if (df_safe[c] >= 0).all()]
    high_skew_cols = positive_cols[:top_n] if top_n is not None else positive_cols

    # 5️⃣ Apply log1p transform
    if high_skew_cols:
        df_safe[high_skew_cols] = np.log1p(df_safe[high_skew_cols])
        if verbose:
            print("\n🔄 Log-transformed highly skewed features:")
            for c in high_skew_cols:
                print(f"  • {c} (skew={skew_before[c]:.2f})")
    elif verbose:
        print("✅ No numeric features exceeded skew threshold.")

    # 6️⃣ Clip numeric values safely
    if clip_threshold is not None:
        numeric_cols = [c for c in numeric_cols if c in df_safe.columns]
        df_safe[numeric_cols] = df_safe[numeric_cols].clip(lower=-clip_threshold, upper=clip_threshold)

    # 7️⃣ Scale numeric features safely
    numeric_cols = [c for c in numeric_cols if c in df_safe.columns]
    if scale_method in ('standard', 'minmax') and numeric_cols:
        if scale_method == 'standard':
            df_safe[numeric_cols] = StandardScaler().fit_transform(df_safe[numeric_cols])
        else:
            df_safe[numeric_cols] = MinMaxScaler().fit_transform(df_safe[numeric_cols])
        if verbose:
            print(f"\n✅ Applied {scale_method} scaling to numeric features.")
    elif scale_method is not None and scale_method not in ('standard', 'minmax'):
        raise ValueError("scale_method must be 'standard', 'minmax', or None")

    # 8️⃣ Remove near-constant features safely
    if remove_constant:
        numeric_cols = [c for c in numeric_cols if c in df_safe.columns]
        constant_features = df_safe[numeric_cols].columns[df_safe[numeric_cols].var() < constant_threshold].tolist()
        if constant_features:
            df_safe.drop(columns=constant_features, inplace=True)
            if verbose:
                print(f"🗑 Removed near-constant features: {constant_features}")

    # 9️⃣ Report top skewed features after
    numeric_cols = [c for c in numeric_cols if c in df_safe.columns]
    skew_after = df_safe[numeric_cols].skew().abs().sort_values(ascending=False)
    if verbose:
        print(f"\nTop {report_top_n} skewed features AFTER log-transform:")
        print(skew_after.head(report_top_n))

    return df_safe



"""
def load_and_label_all_parallel(log_path_str, folder_path,
                                benign_keywords=None,
                                attack_keywords=None,
                                multiclass_keywords=None,
                                manual_label_map=None,
                                drop_columns=None,
                                max_workers=4,
                                cache_parquet=True,
                                parquet_file="cached_preprocessed.parquet",
                                skew_threshold=2.0):
    if drop_columns is None:
        drop_columns = [
            'Flow ID',
            'Src IP',
            'Dst IP',
            'Timestamp',
            'Src Port',
            'Dst Port'
        ]

    csv_files = glob(os.path.join(folder_path, "*.csv")) + glob(os.path.join(folder_path, "*.CSV"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {os.path.abspath(folder_path)}")

    parquet_path = os.path.join(log_path_str, parquet_file)
    if cache_parquet and os.path.exists(parquet_path):
        log_util.safe_log(f"📦 Using cached parquet file: {parquet_path}")
        return pd.read_parquet(parquet_path)

    log_util.safe_log(f"🧵 Loading {len(csv_files)} CSVs with ThreadPoolExecutor...")

    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_file, file,
                            benign_keywords, attack_keywords, multiclass_keywords,
                            manual_label_map, drop_columns): file
            for file in csv_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            file = futures[future]
            try:
                df = future.result()
                if df is not None:
                    all_data.append(df)
                    log_util.safe_log(f"[{i}/{len(csv_files)}] ✅ Processed: {file}")
                else:
                    log_util.safe_log(f"[{i}/{len(csv_files)}] ⚠️ Skipped: {file}")
            except Exception as e:
                log_util.safe_log(f"[{i}/{len(csv_files)}] ❌ Exception for {file}: {e}")

    if not all_data:
        raise ValueError("❌ No usable labeled data found in any CSV files.")

    final_df = pd.concat(all_data, ignore_index=True)

    # -----------------------------
    # Preprocessing to reduce overfitting
    # -----------------------------

    # 1️⃣ Log-transform skewed numeric features (Flow Duration included)
    numeric_cols = final_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        skewness = final_df[col].skew()
        if skewness > skew_threshold:
            final_df[col] = np.log1p(final_df[col])
            log_util.safe_log(f"🔄 Log-transformed '{col}' (skew={skewness:.2f})")

    # 2️⃣ Protocol as categorical
    if 'Protocol' in final_df.columns:
        final_df['Protocol'] = final_df['Protocol'].astype('category')

    # 3️⃣ Convert any remaining object columns to string
    object_cols = final_df.select_dtypes(include='object').columns
    for col in object_cols:
        try:
            final_df[col] = final_df[col].astype(str)
        except Exception as e:
            log_util.safe_log(f"❌ Could not convert column '{col}': {e}")

    log_util.safe_log("✅✅ All file processing completed.")
    log_util.safe_log(f"✅ Final shape: ({int(final_df.shape[0])}, {int(final_df.shape[1])})")

    # 4️⃣ Cache to parquet
    if cache_parquet:
        try:
            # final_df.to_parquet(parquet_path, index=False, engine="fastparquet", compression='BROTLI')
            log_util.safe_log(f"📝 Cached to parquet: {parquet_path}")
        except Exception as e:
            log_util.safe_log(f"❌ Failed to cache to parquet: {e}")

    return final_df





"""
def load_and_label_all_parallel(
        log_path_str,
        folder_path,
        drop_columns=None,
        max_workers=4,
        cache_parquet=True,
        parquet_file="cached_preprocessed.parquet",
        skew_threshold=2.0
):
    """
    Load multiple CSVs in parallel, label them, drop unsafe columns,
    normalize skewed numeric features (via log_transform_skewed_features),
    convert Protocol to categorical, and cache to parquet.
    """

    if drop_columns is None:
        drop_columns = [
            'Flow ID',
            'Src IP',
            'Dst IP',
            'Timestamp',
            'Src Port',
            'Dst Port'
        ]

    csv_files = glob(os.path.join(folder_path, "*.csv")) + glob(os.path.join(folder_path, "*.CSV"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {os.path.abspath(folder_path)}")

    parquet_path = os.path.join(log_path_str, parquet_file)
    if cache_parquet and os.path.exists(parquet_path):
        log_util.safe_log(f"📦 Using cached parquet file: {parquet_path}")
        return pd.read_parquet(parquet_path)

    log_util.safe_log(f"🧵 Loading {len(csv_files)} CSVs with ThreadPoolExecutor...")

    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_file,
                file,
                drop_columns
            ): file
            for file in csv_files
        }

        for i, future in enumerate(as_completed(futures), 1):
            file = futures[future]
            try:
                df = future.result()
                if df is not None:
                    all_data.append(df)
                    log_util.safe_log(f"[{i}/{len(csv_files)}] ✅ Processed: {file}")
                else:
                    log_util.safe_log(f"[{i}/{len(csv_files)}] ⚠️ Skipped: {file}")
            except Exception as e:
                log_util.safe_log(f"[{i}/{len(csv_files)}] ❌ Exception for {file}: {e}")

    if not all_data:
        raise ValueError("❌ No usable labeled data found in any CSV files.")

    final_df = pd.concat(all_data, ignore_index=True)

    # -----------------------------
    # 1️⃣ Log-transform highly skewed numeric features using modular function
    # -----------------------------
    target = "Label"

    final_df = log_transform_skewed_features(
        final_df,
        target_col=target,
        skew_threshold=2.0,
        extreme_skew_threshold=1000,
        clip_threshold=1e6,
        top_n=20,
        scale_method='standard',
        drop_extreme_skew=True,
        remove_constant=True,
        report_top_n=10
    )

    y = df[target].values  # keep target untouched
    #plot_skewed_features_log_transform(final_df)
    # -----------------------------
    # 2️⃣ Protocol as categorical
    # -----------------------------
    if 'Protocol' in final_df.columns:
        final_df['Protocol'] = final_df['Protocol'].astype('category')
        log_util.safe_log("🔄 Converted 'Protocol' to categorical")

    # -----------------------------
    # 3️⃣ Convert any remaining object columns to string
    # -----------------------------
    object_cols = final_df.select_dtypes(include='object').columns
    for col in object_cols:
        try:
            final_df[col] = final_df[col].astype(str)
            log_util.safe_log(f"🔄 Converted '{col}' to string")
        except Exception as e:
            log_util.safe_log(f"❌ Could not convert column '{col}': {e}")

    log_util.safe_log("✅ All file processing completed.")
    log_util.safe_log(f"✅ Final shape: ({int(final_df.shape[0])}, {int(final_df.shape[1])})")

    # -----------------------------
    # 4️⃣ Cache to parquet
    # -----------------------------
    if cache_parquet:
        try:
            # final_df.to_parquet(parquet_path, index=False, engine="fastparquet", compression='BROTLI')
            log_util.safe_log(f"📝 Cached to parquet: {parquet_path}")
        except Exception as e:
            log_util.safe_log(f"❌ Failed to cache to parquet: {e}")

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
            log_util.safe_log(f"Processing rows {start} to {end - 1}...")

        df.iloc[start:end] = df.iloc[start:end].replace(values_to_replace, replace_with)

        if auto_gc:
            gc.collect()

    if log_progress:
        log_util.safe_log("Cleaning complete.")

    return df


def safe_preprocess_data(log_path_str, folder_path, scaler_type='minmax'):
    """
    Loads and preprocesses the dataset.
    Returns train/test splits and fine-tune splits safely aligned.
    Only keeps a fraction (`sample_frac`) of the data.
    """
    # --- Load and preprocess ---
    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data(
        log_path_str, folder_path, scaler_type=scaler_type
    )

    # --- Align shapes safely ---
    def align_xy(X, y):
        X_np = X.to_numpy() if isinstance(X, (pd.DataFrame, pd.Series)) else X
        y_np = y.to_numpy() if isinstance(y, (pd.DataFrame, pd.Series)) else y
        min_len = min(X_np.shape[0], y_np.shape[0])
        return X_np[:min_len], y_np[:min_len]

    X_pretrain, y_pretrain = align_xy(X_pretrain, y_pretrain)
    X_finetune, y_finetune = align_xy(X_finetune, y_finetune)
    X_test, y_test = align_xy(X_test, y_test)

    print(f"[INFO] Shapes after preprocessing (10% sampling):")
    print(f"X_pretrain: {X_pretrain.shape}, y_pretrain: {y_pretrain.shape}")
    print(f"X_finetune: {X_finetune.shape}, y_finetune: {y_finetune.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    return X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test


def preprocess_data(log_path_str, selected_folder, writer=None, scaler_type='minmax'):
    with util.named_timer("load_and_label_all_parallel", writer, tag="load_and_label_all_parallel"):
        df = load_and_label_all_parallel(log_path_str, os.path.join(config.OUTPUT_DATASET_ALL_DATA, selected_folder))

    features = df.columns.difference(['Label'])
    df[features] = df[features].astype(np.float32)

    X, y = df[features], df['Label']
    X = util.to_float32(X)

    with util.named_timer("safe_smote", writer, tag="safe_smote"):
        X_final, y_final = prepare_data(X, y, strategy='hybrid', verbose=False)
        del df, X, y
        gc.collect()

    with util.named_timer("train_test_split", writer, tag="train_test_split"):
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


#####################################################################

###To consider all classes should be there even if 1 sample###

#####################################################################


def prepare_data_class_safe(X, y, strategy='hybrid', n_components=30, verbose=True):
    """
    Prepare dataset with optional balancing while keeping all classes.
    """
    X = util.to_float32(X)

    if strategy == 'pca_smote':
        X_reduced = reduce_dim(X, n_components=n_components)
        X_final, y_final = fast_safe_smote_class_safe(X_reduced, y)
    elif strategy == 'hybrid':
        X_final, y_final = hybrid_balance(X, y)
        X_final, y_final = fast_safe_smote_class_safe(X_final, y_final)
    elif strategy == 'smote_only':
        X_final, y_final = fast_safe_smote_class_safe(X, y)
    elif strategy == 'none':
        if verbose:
            log_util.safe_log("🚫 No resampling applied. Returning original X, y.")
        X_final, y_final = X, y
    else:
        raise ValueError("❌ Invalid strategy. Choose from 'pca_smote', 'hybrid', 'smote_only', or 'none'.")

    if verbose:
        log_util.safe_log("✅ Final shape:", X_final.shape)
        log_util.safe_log("✅ Final class distribution:", Counter(y_final))

    return X_final, y_final


def safe_timer_wrapper(name, writer=None, tag=None):
    """
    Returns a valid context manager for timing.
    If util.named_timer returns None, uses nullcontext().
    """
    timer = util.named_timer(name, writer, tag=tag)
    return timer if timer is not None else nullcontext()


def preprocess_data_safe(log_path_str, selected_folder, writer=None, scaler_type='minmax', min_samples_per_class=1):
    """
    Load, balance, and split dataset ensuring all classes are preserved in pretrain, finetune, and test.
    SMOTE is applied only to training splits to avoid leakage.
    Returns:
        X_final, y_final: full balanced dataset after class-safe SMOTE
        X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test
    """
    # --- Load raw dataset ---
    timer1 = util.named_timer("load_and_label_all_parallel", writer, tag="load_and_label_all_parallel") or nullcontext()
    with timer1:
        df = load_and_label_all_parallel(
            log_path_str, os.path.join(config.OUTPUT_DATASET_ALL_DATA, selected_folder)
        )

    features = df.columns.difference(['Label'])
    df[features] = df[features].astype(np.float32)

    X, y = df[features].to_numpy(), df['Label'].to_numpy()
    X = util.to_float32(X)

    # --- Prepare / balance data with class-safe SMOTE ---
    timer2 = util.named_timer("prepare_data_class_safe", writer, tag="prepare_data_class_safe") or nullcontext()
    with timer2:
        X_final, y_final = prepare_data_class_safe(X, y, strategy='hybrid', verbose=False)
        del df, X, y
        gc.collect()

    # --- Check minimum samples per class ---
    counts = Counter(y_final)
    for cls, cnt in counts.items():
        if cnt < min_samples_per_class:
            raise ValueError(f"Class '{cls}' has too few samples ({cnt}).")

    # --- Stratified split: test 10% ---
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, test_idx = next(sss_test.split(X_final, y_final))
    X_temp, X_test = X_final[train_idx], X_final[test_idx]
    y_temp, y_test = y_final[train_idx], y_final[test_idx]

    # --- Stratified split: pretrain / finetune 10% of train ---
    sss_fine = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    pre_idx, fine_idx = next(sss_fine.split(X_temp, y_temp))
    X_pretrain, X_finetune = X_temp[pre_idx], X_temp[fine_idx]
    y_pretrain, y_finetune = y_temp[pre_idx], y_temp[fine_idx]

    del X_temp, y_temp
    gc.collect()

    # --- Apply class-safe SMOTE only to pretrain + finetune ---
    X_pretrain, y_pretrain = fast_safe_smote_class_safe(X_pretrain, y_pretrain)
    X_finetune, y_finetune = fast_safe_smote_class_safe(X_finetune, y_finetune)

    # --- Scaling ---
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unsupported scaler_type: {scaler_type}")

    X_pretrain = scaler.fit_transform(X_pretrain)
    X_finetune = scaler.transform(X_finetune)
    X_test = scaler.transform(X_test)  # test scaling only, no SMOTE

    # --- Verify all classes preserved ---
    expected_classes = set(counts.keys())
    for name, labels in zip(
            ["y_final", "y_pretrain", "y_finetune", "y_test"],
            [y_final, y_pretrain, y_finetune, y_test]
    ):
        present_classes = set(np.unique(labels))
        missing = expected_classes - present_classes
        if missing:
            raise ValueError(f"{name} lost some classes! Missing: {missing}")

    return X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test


def fast_safe_smote_class_safe(X, y, k_neighbors=5):
    """
    Apply SMOTE safely ensuring all classes in `y` remain.
    If any class has only 1 sample, it is kept and not resampled.
    """
    X, y = np.asarray(X), np.asarray(y)
    counts = Counter(y)
    classes = np.unique(y)

    # Identify classes with enough samples for SMOTE
    smote_classes = [cls for cls, cnt in counts.items() if cnt > 1]

    if len(smote_classes) < 2:
        # Too few classes for SMOTE; return original
        warnings.warn("Too few classes with >1 sample. Skipping SMOTE.")
        return X, y

    # Minimum k_neighbors based on smallest class > 1
    min_class_size = min(counts[cls] for cls in smote_classes)
    k = min(k_neighbors, min_class_size - 1)
    if k < 1:
        warnings.warn("Too few samples for SMOTE; skipping.")
        return X, y

    # Apply SMOTE only to eligible classes
    sm = SMOTE(k_neighbors=k, random_state=42)
    try:
        X_res, y_res = sm.fit_resample(X, y)
    except ValueError as e:
        # Fallback if SMOTE fails
        warnings.warn(f"SMOTE failed: {e}. Returning original X, y.")
        return X, y

    # Ensure original single-sample classes are preserved
    present_classes = set(np.unique(y_res))
    missing_classes = set(classes) - present_classes
    for cls in missing_classes:
        idx = np.where(y == cls)[0][0]
        X_res = np.vstack([X_res, X[idx:idx + 1]])
        y_res = np.append(y_res, y[idx])

    return X_res, y_res


def hybrid_balance_class_safe(X, y, k_neighbors=5):
    """
    Hybrid balancing: undersample majority classes + SMOTE oversampling,
    ensuring all original classes remain in output.
    """
    log_util.safe_log("\n🌀 Applying hybrid balancing (undersample + SMOTE)")

    under = RandomUnderSampler(sampling_strategy='auto', random_state=42)
    over = SMOTE(k_neighbors=k_neighbors, random_state=42)
    pipeline = Pipeline([('under', under), ('over', over)])

    try:
        result = pipeline.fit_resample(X, y)
        if result is None:
            warnings.warn("Pipeline fit_resample returned None.")
            return X, y
        X_res, y_res = result

        # --- Ensure all original classes are present ---
        original_classes = np.unique(y)
        missing_classes = set(original_classes) - set(np.unique(y_res))
        for cls in missing_classes:
            idx = np.where(y == cls)[0][0]  # take first sample
            X_res = np.vstack([X_res, X[idx:idx + 1]])
            y_res = np.append(y_res, y[idx])

        return X_res, y_res

    except Exception as e:
        warnings.warn(f"Hybrid balancing failed: {e}")
        return X, y

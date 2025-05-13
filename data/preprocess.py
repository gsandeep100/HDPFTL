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
import glob
from collections import Counter

import numpy as np
import pandas as pd
import torch
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SVMSMOTE, SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

def safe_smote(X, y):
    counts = Counter(y)
    min_class_size = min(counts.values())
    k = min(5, min_class_size - 1)
    if k < 1:
        print("Too few samples for SMOTE; skipping.")
        return X, y
    smote = SVMSMOTE(k_neighbors=k, random_state=42)
    return smote.fit_resample(X, y)

def preprocess_data(path):
    all_files = glob.glob(path + "*.csv")
    df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(['Flow ID', 'Source IP', 'Destination IP', 'Timestamp'], axis=1, inplace=True, errors="ignore")
    label_encoder = LabelEncoder()
    df['Label'] = label_encoder.fit_transform(df['Label'])
    scaler = MinMaxScaler()
    features = df.columns.difference(['Label'])
    df[features] = scaler.fit_transform(df[features])
    X, y = df[features], df['Label']

    # SVMSMOTE- Create synthetic minority points near SVM boundary (critical zones).
    #           Makes the minority class stronger exactly where it matters — at the decision boundary.
    # SMOTEENN- Clean noisy samples after oversampling using Edited Nearest Neighbors.
    #           Removes confusing and overlapped samples, making the classifier much more accurate.

    resampling_pipeline = Pipeline([
        ('svm_smote', SVMSMOTE(random_state=42)),
        ('smote_enn', SMOTEENN(random_state=42))
    ])
    X_final, y_final = safe_smote(X, y)
    return train_test_split(X_final, y_final, test_size=0.2, random_state=42, stratify=y_final)

def preprocess_data_small(csv_path, test_size=0.2):
    # Load dataset
    df = pd.read_csv(csv_path+ "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv", low_memory=False)

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
    print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

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
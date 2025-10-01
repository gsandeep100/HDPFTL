# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        pretrainclass.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-26
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import gc
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util
from hdpftl_training.hdpftl_models.TabularNet import TabularNet

"""
1. Pretraining phase
Use X_pretrain, y_pretrain — large, general dataset (can be synthetic or real).

Pretrain a global model to learn general features.
"""


def pretrain_class(X_train, X_test, y_train, y_test, input_dim, early_stop_patience=5, verbose=True):
    num_classes = len(torch.unique(torch.tensor(y_train.values)))

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long)
    # Wrap in datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.BATCH_SIZE_TRAINING, pin_memory=False)
    val_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.BATCH_SIZE_TRAINING, pin_memory=False)
    log_util.safe_log("\n=== Pretraining Phase (Real Data) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TabularNet(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    os.makedirs(config.EPOCH_DIR, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    epoch_metrics = []

    for epoch in range(config.NUM_EPOCHS_PRE_TRAIN):
        model.train()
        train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_val.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        epoch_metrics.append((train_loss, val_loss, val_acc))

        if verbose:
            log_util.safe_log(
                f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Val Acc = {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            folder_path = config.PRE_MODEL_FOLDER_PATH_TEMPLATE.substitute(n=util.get_today_date())
            util.is_folder_exist(folder_path)
            torch.save(model.state_dict(), config.PRE_MODEL_PATH_TEMPLATE.substitute(n=util.get_today_date()))
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                log_util.safe_log(f"Early stopping at epoch {epoch + 1}", level="warning")
                break

    # Save metrics to file
    np.save(config.EPOCH_FILE_PRE, np.array(epoch_metrics))
    log_util.safe_log("Pretraining complete. Best model saved.")
    # === CLEANUP ===
    del model, optimizer, criterion, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

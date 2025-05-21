# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        hdpftl_main.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:           Sandeep Ghosh
   Created Date:     2025-04-21
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import logging
import warnings

import numpy as np
import torch

from data.preprocess import preprocess_data
from result.final_model import save
from result.predictions import predict
from training.hdpftl_pipeline import hdpftl_pipeline
from transfer.pretrainclass import pretrain_class
from transfer.targetclass import target_class
from utility.config import OUTPUT_DATASET_PATH_2024

warnings.filterwarnings("ignore", category=SyntaxWarning)


# === Setup Logging ===
def setup_logging(log_to_file=True):
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        filename="hdpftl_run.log" if log_to_file else None,
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    setup_logging()
    logging.info("========================Process Started===================================")
    # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
    X_train, X_test, y_train, y_test = preprocess_data(OUTPUT_DATASET_PATH_2024)
    logging.info("Data preprocessing completed.")
    # Step 2: Pretrain global model
    pre_model = pretrain_class()
    logging.info("Pretraining completed.")
    # Step 3: Instantiate target model and train on device
    model = target_class()
    global_model, personalized_models = hdpftl_pipeline(X_train, y_train, X_test, y_test, model)
    logging.info("HDPFTL training and personalization completed.")
    save(global_model, personalized_models)
    logging.info("Models saved.")

    # Sample input: 3 feature vectors
    new_sample = np.random.rand(5, 79).astype(np.float32)  # 3 samples, each with 79 features
    new_sample = torch.tensor(new_sample)

    # Label map
    label_map = {0: "Normal", 1: "DoS", 2: "Probe", 3: "R2L", 4: "U2R"}
    # Predict
    preds, probs = predict(new_sample, global_model, label_map=label_map, return_proba=True)

    logging.info(f"Predictions: {preds}")
    logging.info(f"Probabilities: {probs}")
    logging.info("=========================Process Finished===================================")
    print("Predictions:", preds)
    print("Probabilities:", probs)
    # plot(global_accuracies=preds, personalized_accuracies=probs)

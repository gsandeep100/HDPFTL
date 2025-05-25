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
import warnings

import torch

from hdpftl_evaluation.evaluate_global_model import evaluate_global_model
from hdpftl_evaluation.evaluate_per_client import evaluate_personalized_models_per_client, evaluate_per_client
from hdpftl_plotting.plot import plot_client_accuracies, plot_personalized_vs_global
from hdpftl_training.hdpftl_data.preprocess import preprocess_data
from hdpftl_training.hdpftl_pipeline import hdpftl_pipeline, dirichlet_partition
from hdpftl_training.hdpftl_pre_training.pretrainclass import pretrain_class
from hdpftl_training.hdpftl_pre_training.targetclass import target_class
from hdpftl_utility.config import OUTPUT_DATASET_ALL_DATA
from hdpftl_utility.log import setup_logging, safe_log
from hdpftl_utility.utils import setup_device

warnings.filterwarnings("ignore", category=SyntaxWarning)

if __name__ == "__main__":
    setup_logging()
    safe_log("========================Process Started===================================")
    safe_log("============================================================================")
    safe_log("============================================================================")

    # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
    X_train, X_test, y_train, y_test = preprocess_data(OUTPUT_DATASET_ALL_DATA)
    safe_log("[1]Data preprocessing completed.")
    # Step 2: Pretrain global model
    pre_model = pretrain_class()
    safe_log("[2]Pretraining completed.")
    # Step 3: Instantiate target model and train on device
    model = target_class()
    safe_log("[3]Fine Tuning completed.")

    device = setup_device()
    num_classes = len(torch.unique(y_train))
    client_partitions = dirichlet_partition(X_train, y_train, num_classes, alpha=0.5)
    client_partitions_test = dirichlet_partition(X_test, y_test, num_classes, alpha=0.5)
    safe_log("[4]Partitioning hdpftl_data using Dirichlet...")

    global_model, personalized_models = hdpftl_pipeline(X_train, y_train, X_test, y_test, model,
                                                        client_partitions, client_partitions_test, device)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #######################  EVALUATION  #######################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """During evaluation: Use  global model for generalization tests 
    Use personalized models to report per - client performance"""

    safe_log("\n[10]Evaluating personalized per client...")
    personalised_acc = evaluate_personalized_models_per_client(model, X_test,y_test, client_partitions_test)
    # for cid, model in personalized_models.items():
    #     acc = evaluate_personalized_models_per_client(model, X_test[client_partitions_test[cid]],
    #                                                   y_test[client_partitions_test[cid]], client_partitions_test)
    #     safe_log(f"Client {cid} Accuracy for Personalized Model for clients: {acc[cid]:.4f}")

    safe_log("[11]Evaluating global per client...")
    client_accs = evaluate_per_client(global_model, X_test, y_test, client_partitions_test)

    safe_log("[12] Evaluating global model...")
    global_acc = evaluate_global_model(global_model, X_test, y_test)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #######################  PLOT  #######################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    plot_client_accuracies(client_accs, global_acc=global_acc, title="Per-Client vs Global Model Accuracy")
    plot_personalized_vs_global(personalised_acc, global_acc)
    safe_log("========================Process Completed===================================")
    safe_log("============================================================================")
    safe_log("============================================================================")

"""
print("\n[3] Aggregating fleet models...")
    # global_model = aggregate_models(local_models, base_model_fn)
    global_model = bayesian_aggregate_models(local_models, base_model_fn)
    print("\n[4] Evaluating global model...")
    acc = evaluate_global_model(global_model, X_test, y_test)
    print(f"Global Accuracy Before Personalization: {acc:.4f}")
    logging.info(f"Global Accuracy Before Personalization: {acc:.4f}")

    evaluate_per_client(global_model, X_test, y_test, client_partitions_test)

    print("\n[5] Personalizing each client...")
    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)

    print("\n[6] Evaluating personalized hdpftl_models...")
    for cid, model in personalized_models.items():
        acc = evaluate_global_model(model, X_test[client_partitions_test[cid]], y_test[client_partitions_test[cid]],device)
        print(f"Global Accuracy After Personalization for Client {cid}: {acc:.4f}")
        logging.info(f"Global Accuracy After Personalization for Client {cid}: {acc:.4f}")

        # print(f"Personalized Accuracy for Client {cid}: {acc:.4f}")

    return global_model, personalized_models

"""

""" commented for now...need a way to get the evaluation on a different hdpftl_dataset

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
    """

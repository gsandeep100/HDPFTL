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
import os
import tkinter as tk
import warnings

import numpy as np
import torch

from hdpftl_evaluation.evaluate_global_model import evaluate_global_model, evaluate_global_model_fromfile
from hdpftl_evaluation.evaluate_per_client import evaluate_personalized_models_per_client, evaluate_per_client, \
    load_personalized_models_fromfile
from hdpftl_plotting.plot import plot_client_accuracies, plot_personalized_vs_global, plot_confusion_matrix, \
    plot_accuracy_comparison, plot_training_loss, plot_class_distribution_per_client
from hdpftl_training.hdpftl_data.preprocess import preprocess_data
from hdpftl_training.hdpftl_pipeline import hdpftl_pipeline, dirichlet_partition
from hdpftl_training.hdpftl_pre_training.pretrainclass import pretrain_class
from hdpftl_training.hdpftl_pre_training.targetclass import target_class
from hdpftl_utility.config import OUTPUT_DATASET_ALL_DATA, GLOBAL_MODEL_PATH, EPOCH_FILE_FINE, EPOCH_FILE_PRE
from hdpftl_utility.log import setup_logging, safe_log
from hdpftl_utility.utils import setup_device

warnings.filterwarnings("ignore", category=SyntaxWarning)

if __name__ == "__main__":
    setup_logging()
    safe_log("============================================================================")
    safe_log("======================Process Started=======================================")
    safe_log("============================================================================")

    # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
    X_train, X_test, y_train, y_test = preprocess_data(OUTPUT_DATASET_ALL_DATA)
    safe_log("[1]Data preprocessing completed.")
    device = setup_device()
    num_classes = len(torch.unique(y_train))
    """
    dirichlet_partition is a standard technique to simulate non-IID data — 
    and it's commonly used in federated learning experiments to control the degree of 
    heterogeneity among clients.
    Smaller alpha → more skewed, clients have few classes dominating.
    Larger alpha → more uniform data distribution across clients.
    """
    client_partitions, client_data_dict = dirichlet_partition(X_train, y_train, num_classes, alpha=0.3)
    client_partitions_test, client_data_dict_test = dirichlet_partition(X_test, y_test, num_classes, alpha=0.3)
    safe_log("[4]Partitioning hdpftl_data using Dirichlet...")

    # If fine-tuned model exists, load and return it
    if not os.path.exists(GLOBAL_MODEL_PATH):
        # Step 2: Pretrain global model
        model = pretrain_class()
        safe_log("[2]Pretraining completed.")
        # Step 3: Instantiate target model and train on device
        base_model = target_class()
        safe_log("[3]Fine Tuning completed.")

        global_model, personalized_models = hdpftl_pipeline(X_train, y_train, base_model, client_partitions)

    #######################  EVALUATION LOAD FROM FILE #########
    else:
        # Load global model
        global_model = evaluate_global_model_fromfile()
        # Load personalized models
        personalized_models = load_personalized_models_fromfile()

    # Evaluate
    # global_accs = evaluate_global_model(global_model, X_test, y_test, client_partitions_test)
    # personalized_accs = evaluate_personalized_models_per_client(personalized_models, X_test, y_test, client_partitions_test)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #######################  EVALUATION  #######################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    """During evaluation: Use  global model for generalization tests 
    Use personalized models to report per - client performance"""

    safe_log("\n[10]Evaluating personalized models per client on client partitioned data...")
    personalised_acc = evaluate_personalized_models_per_client(personalized_models, X_test, y_test,
                                                               client_partitions_test)
    # for cid, model in personalized_models.items():
    #     acc = evaluate_personalized_models_per_client(model, X_test[client_partitions_test[cid]],
    #                                                   y_test[client_partitions_test[cid]], client_partitions_test)
    #     safe_log(f"Client {cid} Accuracy for Personalized Model for clients: {acc[cid]:.4f}")

    safe_log("[11]Evaluating global model on client partitioned dataset per client...")
    client_accs = evaluate_per_client(global_model, X_test, y_test, client_partitions_test)

    safe_log("[12] Evaluating global model on global dataset...")
    global_acc = evaluate_global_model(global_model, X_test, y_test)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #######################  PLOT  #############################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    with torch.no_grad():
        outputs = global_model(X_test.to(device))
        _, predictions = torch.max(outputs, 1)
    num_classes = int(max(y_test.max().item(), predictions.cpu().max().item()) + 1)
    print("Number of classes:::", num_classes)
    # plot_confusion_matrix(y_true=y_test, y_pred=predictions, class_names=[str(i) for i in range(num_classes)])

    # plot_training_loss(losses=np.load(EPOCH_FILE_PRE), label='Pre Epoch Losses')
    # plot_training_loss(losses=np.load(EPOCH_FILE_FINE), label='Fine Tuning Epoch Losses')
    # plot_accuracy_comparison(client_accs, personalised_acc)
    # plot_client_accuracies(client_accs, global_acc=global_acc, title="Per-Client vs Global Model Accuracy")
    # plot_personalized_vs_global(personalised_acc, global_acc)

    # GUI setup
    root = tk.Tk()
    root.title("Select a Plot to View")
    root.geometry("750x750")

    tk.Label(root, text="Choose a Plot Type", font=("Arial", 16)).pack(pady=10)

    # Buttons for each plot type
    tk.Button(root, text="Client Labels Distribution", width=20,
              command=lambda: plot_class_distribution_per_client(client_data_dict)).pack(pady=5)

    tk.Button(root, text="Confusion Matrix", width=20,
              command=lambda: plot_confusion_matrix(y_true=y_test, y_pred=predictions,
                                                    class_names=[str(i) for i in range(int(num_classes))],
                                                    normalize=True)).pack(pady=5)
    tk.Button(root, text="Pre Epoch Losses", width=20,
              command=lambda: plot_training_loss(losses=np.load(EPOCH_FILE_PRE), name='epoch_loss_pre.png',
                                                 label='Pre Epoch Losses')).pack(pady=5)
    tk.Button(root, text="Fine Tuning Epoch Losses", width=20,
              command=lambda: plot_training_loss(losses=np.load(EPOCH_FILE_FINE), name='epoch_loss_fine.png',
                                                 label='Fine Tuning Epoch Losses')).pack(pady=5)
    tk.Button(root, text="Global/Personalized Acc/Client", width=20,
              command=lambda: plot_accuracy_comparison(client_accs, personalised_acc)).pack(pady=5)
    tk.Button(root, text="Per-Client Accuracy", width=20,
              command=lambda: plot_client_accuracies(client_accs, global_acc=global_acc,
                                                     title="Per-Client vs Global Model Accuracy")).pack(pady=5)
    tk.Button(root, text="Client Acc: Personalized/Global", width=20,
              command=lambda: plot_personalized_vs_global(personalised_acc, global_acc)).pack(pady=5)

    root.mainloop()
    safe_log("===========================================================================")
    safe_log("===========================Process Completed===============================")
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

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
import threading
import tkinter as tk
import warnings
import time
from tkinter import ttk

import numpy as np
import torch
from torch.distributed.elastic.multiprocessing import start_processes
from torch.utils.tensorboard import SummaryWriter

from hdpftl_cross_validation.cross_validation_model import cross_validate_model_advanced, cross_validate_model
from hdpftl_evaluation.evaluate_global_model import evaluate_global_model, evaluate_global_model_fromfile
from hdpftl_evaluation.evaluate_per_client import evaluate_personalized_models_per_client, evaluate_per_client, \
    load_personalized_models_fromfile
from hdpftl_plotting import predictions
from hdpftl_plotting.plot import plot_client_accuracies, plot_personalized_vs_global, plot_confusion_matrix, \
    plot_training_loss, plot_class_distribution_per_client, cross_validate_model_with_plots
from hdpftl_training.hdpftl_data.preprocess import preprocess_data
from hdpftl_training.hdpftl_pipeline import hdpftl_pipeline, dirichlet_partition_with_devices
from hdpftl_training.hdpftl_pre_training.pretrainclass import pretrain_class
from hdpftl_training.hdpftl_pre_training.finetune_model import finetune_model
from hdpftl_utility.config import EPOCH_FILE_FINE, EPOCH_FILE_PRE, INPUT_DIM, NUM_CLIENTS, \
    NUM_DEVICES_PER_CLIENT, GLOBAL_MODEL_PATH_TEMPLATE, OUTPUT_DATASET_ALL_DATA
from hdpftl_utility.log import setup_logging, safe_log
from hdpftl_utility.utils import named_timer, setup_device, get_output_folders, get_today_date

warnings.filterwarnings("ignore", category=SyntaxWarning)

if __name__ == "__main__":

    global_model = None
    personalized_models = None

    start_time = time.time()
    client_accs = {}
    personalised_acc = {}
    X_test = {}
    y_test = {}
    client_data_dict = {}
    global_acc = {}
    num_classes  = None
    y_pred = None
    predictions = None

    def update_progress(progress):
        root.after(0, lambda: _update_progress_ui(progress))


    def _update_progress_ui(progress):
        progress['value'] = progress
        progress_label.config(text=f"Training Progress: {progress}%")
        if progress == 100:
            progress_label.config(text="Training complete!")

    def start_training():
        progress['value'] = 0
        progress_label.config(text="Training started...")
        # Run training in background thread so UI does not freeze
        threading.Thread(target=start_process, args=(10, update_progress)).start()


    def start_process():
        global global_model
        global personalized_models
        global num_classes
        global predictions
        setup_logging()
        safe_log("============================================================================")
        safe_log("======================Process Started=======================================")
        safe_log("============================================================================")
        writer = SummaryWriter(log_dir="runs/hdpftl_pipeline")

        # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
        with named_timer("Preprocessing", writer, tag="Preprocessing"):
            global X_test, y_test
            X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data()
        # safe_log("[1]Data preprocessing completed.")
        device = setup_device()
        """
        dirichlet_partition is a standard technique to simulate non-IID data — 
        and it's commonly used in federated learning experiments to control the degree of 
        heterogeneity among clients.
        Smaller alpha → more skewed, clients have few classes dominating.
        Larger alpha → more uniform data distribution across clients.
        """
        with named_timer("dirichlet_partition", writer, tag="dirichlet_partition"):
            global client_data_dict
            client_data_dict, hierarchical_data = dirichlet_partition_with_devices(
                X_pretrain, y_pretrain, alpha=0.5, num_clients=NUM_CLIENTS,
                num_devices_per_client=NUM_DEVICES_PER_CLIENT
            )

            client_data_dict_test, hierarchical_data_test = dirichlet_partition_with_devices(
                X_test, y_test, alpha=0.5, num_clients=NUM_CLIENTS, num_devices_per_client=NUM_DEVICES_PER_CLIENT
            )

        # safe_log("[4]Partitioning hdpftl_data using Dirichlet...")

        # If fine-tuned model exists, load and return it
        if not os.path.exists(GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=get_today_date())):
            # Step 2: Pretrain global model
            with named_timer("pretrain_class", writer, tag="pretrain_class"):
                pretrain_class(X_pretrain, X_test, y_pretrain, y_test, input_dim=INPUT_DIM, early_stop_patience=10)
            # safe_log("[2]Pretraining completed.")
            # Step 3: Instantiate target model and train on device
            with named_timer("target_class", writer, tag="target_class"):
                def base_model_fn():
                    return finetune_model(
                        X_finetune,
                        y_finetune,
                        input_dim=X_finetune.shape[1],
                        target_classes=len(np.unique(y_finetune)))

            # safe_log("[3]Fine Tuning completed.")

            with named_timer("hdpftl_pipeline", writer, tag="hdpftl_pipeline"):
                global_model, personalized_models = hdpftl_pipeline(base_model_fn, hierarchical_data, X_test, y_test)

        #######################  LOAD FROM FILE ##################################
        else:
            # Load global model
            with named_timer("Evaluate Global Model From File", writer, tag="EvalFromFile"):
                global_model = evaluate_global_model_fromfile()
            # Load personalized models
            with named_timer("Evaluate Personalized Models From File", writer, tag="PersonalizedEval"):
                personalized_models = load_personalized_models_fromfile()

        # Evaluate
        # global_accs = evaluate_global_model(global_model, X_test, y_test, client_partitions_test)
        # personalized_accs = evaluate_personalized_models_per_client(personalized_models, X_test, y_test, client_partitions_test)

        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #######################  EVALUATION  #######################
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

        """During evaluation: Use  global model for generalization tests 
        Use personalized models to report per - client performance"""

        # safe_log("\n[10]Evaluating personalized models per client on client partitioned data...")
        with named_timer("Evaluate Personalized Models", writer, tag="PersonalizedEval"):
            global personalised_acc
            personalised_acc = evaluate_personalized_models_per_client(personalized_models, client_data_dict_test)

        # safe_log("[11]Evaluating global model on client partitioned dataset per client...")
        with named_timer("Evaluate Personalized Models Per Client", writer, tag="PersonalizedEvalperClient"):
            global client_accs
            client_accs = evaluate_per_client(global_model, client_data_dict_test)

        # safe_log("[12] Evaluating global model on global dataset...")
        with named_timer("Evaluate Global Model", writer, tag="GlobalEval"):
            global global_acc
            global_acc = evaluate_global_model(global_model, X_test, y_test)

        end_time = time.time()
        elapsed_time = end_time - start_time
        mins, secs = divmod(elapsed_time, 60)
        safe_log(f"\n⏱️ Total time taken: {int(mins)} minutes and {int(secs)} seconds")
        """
        safe_log("[12] Cross Validate Model...")
        with named_timer("Cross Validate Model", writer, tag="ValidateModel"):
            accuracies = cross_validate_model(X_test, y_test, k=5, num_epochs=20, lr=0.001)

        safe_log("[12] Cross Validate Model with F1 Score...")
        with named_timer("Cross Validate Model with F1 Score", writer, tag="ValidateModelF1"):
            fold_results = cross_validate_model_advanced(X_test, y_test, k=5, num_epochs=20, early_stopping=True)
        """
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #######################  PLOT  #############################
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(device)
            outputs = global_model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
        num_classes = int(max(y_test.max().item(), predictions.cpu().max().item()) + 1)
        safe_log("Number of classes:::", num_classes)
        # plot_confusion_matrix(y_true=y_test, y_pred=predictions, class_names=[str(i) for i in range(num_classes)])

        # plot_training_loss(losses=np.load(EPOCH_FILE_PRE), label='Pre Epoch Losses')
        # plot_training_loss(losses=np.load(EPOCH_FILE_FINE), label='Fine Tuning Epoch Losses')
        # plot_accuracy_comparison(client_accs, personalised_acc)
        # plot_client_accuracies(client_accs, global_acc=global_acc, title="Per-Client vs Global Model Accuracy")
        # plot_personalized_vs_global(personalised_acc, global_acc)

        safe_log("===========================================================================")
        safe_log("===========================Process Completed===============================")
        safe_log("============================================================================")


    def on_selection(event):
        selection = listbox.curselection()
        if selection:
            index = selection[0]
            selected_folder = listbox.get(index)
            label_selected.config(text=f"📂 Selected Folder: {selected_folder}")

    # GUI setup
    root = tk.Tk()
    root.title("HDPFTL Architecture")
    root.geometry("1200x1200")
    root.minsize(400, 700)


    tk.Label(root, text="Select Database", font=("Arial", 16)).pack(pady=10)

    # Frame for styling and layout
    frame = ttk.Frame(root, padding=10)
    frame.pack(fill="both", expand=True)

    # Scrollbar
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    # Listbox with styles
    listbox = tk.Listbox(
        frame,
        height=8,
        width=50,
        bd=2,
        font=("Helvetica", 12),
        bg="#f5f5f5",
        fg="#333",
        selectbackground="#cce5ff",
        selectforeground="#000",
        activestyle="dotbox",
        borderwidth=2,
        relief="ridge",
        yscrollcommand=scrollbar.set
    )
    listbox.pack(side="left", expand=True,padx=20, pady=20)
    scrollbar.config(command=listbox.yview)

    for folder in get_output_folders(OUTPUT_DATASET_ALL_DATA):
        listbox.insert(tk.END, folder)

    # Label to display selected item
    label_selected = tk.Label(root, text="📂 Selected Folder: None", font=("Helvetica", 12))
    label_selected.pack(pady=10)

    # Bind selection event
    listbox.bind('<<ListboxSelect>>', on_selection)

    # Buttons for each plot type
    tk.Button(root, text="Client Labels Distribution", width=30,
              command=lambda: plot_class_distribution_per_client(client_data_dict)).pack(pady=5)

    tk.Button(root, text="Confusion Matrix", width=30,
              command=lambda: plot_confusion_matrix(y_true=y_test, y_pred=predictions,
                                                    class_names=[str(i) for i in range(int(num_classes))],
                                                    normalize=True)).pack(pady=5)
    tk.Button(root, text="Pre Epoch Losses", width=30,
              command=lambda: plot_training_loss(losses=np.load(EPOCH_FILE_PRE), name='epoch_loss_pre.png',
                                                 label='Pre Epoch Losses')).pack(pady=5)
    tk.Button(root, text="Fine Tuning Epoch Losses", width=30,
              command=lambda: plot_training_loss(losses=np.load(EPOCH_FILE_FINE), name='epoch_loss_fine.png',
                                                 label='Fine Tuning Epoch Losses')).pack(pady=5)
    # tk.Button(root, text="Global/Personalized Acc/Client", width=30,
    #         command=lambda: plot_accuracy_comparison(client_accs, personalised_acc)).pack(pady=5)
    tk.Button(root, text="Personalized vs Global--Dotted", width=30,
              command=lambda: plot_client_accuracies(client_accs, global_acc=global_acc,
                                                     title="Personalized vs Global--Dotted")).pack(pady=5)
    tk.Button(root, text="Personalized vs Global--Bar Chart", width=30,
              command=lambda: plot_personalized_vs_global(personalised_acc, global_acc)).pack(pady=5)

    tk.Button(root, text="Cross Validation Model", width=30,
              command=lambda: cross_validate_model_with_plots(X_test, y_test)).pack(pady=5)

    start_button = tk.Button(root, text="Start Training", command=start_training)
    start_button.pack(side='bottom', pady=10)
    # Create custom style
    style = ttk.Style()
    style.theme_use('default')
    style.configure(
        "Custom.Horizontal.TProgressbar",
        thickness=25,  # Height of the progress bar
        troughcolor='#e0e0e0',  # Background color
        background='#4caf50'  # Progress bar color
    )
    # Create progressbar with full width
    progress = ttk.Progressbar(
        root,
        orient="horizontal",
        mode="determinate",

        length=500,  # Optional; will auto expand with fill
        style="Custom.Horizontal.TProgressbar"
    )
    progress.pack(fill='x', padx=20, pady=20)  # Full width with padding

    # Simulate progress
    #progress['value'] = 40  # Set any value (0–100)
    progress_label = tk.Label(root, text="Progress: 0%")
    progress_label.pack()

    root.mainloop()

"""
safe_log("\n[3] Aggregating fleet models...")
    # global_model = aggregate_models(local_models, base_model_fn)
    global_model = bayesian_aggregate_models(local_models, base_model_fn)
    safe_log("\n[4] Evaluating global model...")
    acc = evaluate_global_model(global_model, X_test, y_test)
    safe_log(f"Global Accuracy Before Personalization: {acc:.4f}")
    logging.info(f"Global Accuracy Before Personalization: {acc:.4f}")

    evaluate_per_client(global_model, X_test, y_test, client_partitions_test)

    safe_log("\n[5] Personalizing each client...")
    personalized_models = personalize_clients(global_model, X_train, y_train, client_partitions)

    safe_log("\n[6] Evaluating personalized hdpftl_models...")
    for cid, model in personalized_models.items():
        acc = evaluate_global_model(model, X_test[client_partitions_test[cid]], y_test[client_partitions_test[cid]],device)
        safe_log(f"Global Accuracy After Personalization for Client {cid}: {acc:.4f}")
        logging.info(f"Global Accuracy After Personalization for Client {cid}: {acc:.4f}")

        # safe_log(f"Personalized Accuracy for Client {cid}: {acc:.4f}")

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
    safe_log("Predictions:", preds)
    print("Probabilities:", probs)
    # plot(global_accuracies=preds, personalized_accuracies=probs)
    """



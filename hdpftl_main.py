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
import time
import tkinter as tk
import warnings
from tkinter import scrolledtext, messagebox
from tkinter import ttk

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from hdpftl_evaluation.evaluate_global_model import evaluate_global_model, evaluate_global_model_fromfile
from hdpftl_evaluation.evaluate_per_client import evaluate_personalized_models_per_client, evaluate_per_client, \
    load_personalized_models_fromfile
from hdpftl_plotting import predictions
from hdpftl_plotting.plot import plot_client_accuracies, plot_personalized_vs_global, plot_confusion_matrix, \
    plot_training_loss, plot_class_distribution_per_client, cross_validate_model_with_plots
from hdpftl_training.hdpftl_data.preprocess import preprocess_data
from hdpftl_training.hdpftl_pipeline import hdpftl_pipeline, dirichlet_partition_with_devices
from hdpftl_training.hdpftl_pre_training.finetune_model import finetune_model
from hdpftl_training.hdpftl_pre_training.pretrainclass import pretrain_class
from hdpftl_utility.config import EPOCH_FILE_FINE, EPOCH_FILE_PRE, INPUT_DIM, NUM_CLIENTS, \
    NUM_DEVICES_PER_CLIENT, GLOBAL_MODEL_PATH_TEMPLATE, OUTPUT_DATASET_ALL_DATA, NUM_FEDERATED_ROUND, \
    NUM_EPOCHS_PRE_TRAIN
from hdpftl_utility.log import setup_logging, safe_log
from hdpftl_utility.utils import named_timer, setup_device, get_output_folders, get_today_date, number_of_data_folders

warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

warnings.filterwarnings("ignore", category=SyntaxWarning)

if __name__ == "__main__":

    num_classes = 0
    client_accs = {}
    X_test = {}
    y_test = {}
    client_data_dict = {}
    global_acc = {}
    personalised_acc = {}
    y_pred = None
    predictions = None
    selected_folder = ""
    start_time_label = None
    end_time_label = None
    hh = None
    mm = None
    ss = None
    after_id = None
    start_time = float(time.time())


    def update_progress(value):
        root.after(0, lambda: _update_progress_ui(value))


    def _update_progress_ui(value):
        progress['value'] = value
        progress_label.config(text=f"Training Progress: {int(value)}%")
        if value == 100:
            progress_label.config(text="✅ Training complete!")


    def start_training():
        # Start infinite progress animation
        global start_time
        progress.config(mode='indeterminate')
        progress.start(10)
        progress_label.config(text="Training in progress...")

        # Start long-running task in a new thread
        threading.Thread(target=start_process).start()


    def complete_progress_bar():
        def finish():
            progress.stop()
            progress.config(mode='determinate', maximum=100)

            end_time = time.time()
            elapsed_time = end_time - start_time
            mins, secs = divmod(elapsed_time, 60)
            hh, mm, ss = convert_to_hms(mins, secs)

            #total_time_taken_label.config(text=f"Total Time: {hh}H:{mm}M:{ss}S")
            time_taken_label.config(text=f"📂 {hh}Hours:{mm}Minutes:{ss}seconds")

            current_time = time.strftime('%H:%M:%S')
            end_time_label.config(text=f"End Time: {current_time}")

            # Animate progress bar to 100%
            for val in range(0, 101, 10):
                root.after(val * 3, lambda v=val: progress.config(value=v))

            root.after(350, lambda: progress_label.config(text="✅ Training complete!"))

        root.after(0, finish)


    def start_process():
        global personalised_acc
        global global_acc
        global client_accs
        global client_data_dict
        global hh, mm, ss

        def update_clock():
            global after_id
            current_time = time.strftime('%H:%M:%S')
            clock_label_start.config(text=f"🕒 {current_time}")
            after_id = root.after(1000, update_clock)  # schedule next update

        def stop_clock():
            global after_id
            if after_id is not None:
                root.after_cancel(after_id)  # cancel the scheduled call
                after_id = None

        def calculate_total_steps():
            total_steps = number_of_data_folders(
                OUTPUT_DATASET_ALL_DATA) * 10  # or total epochs, or total logical steps
            total_steps += NUM_FEDERATED_ROUND * NUM_CLIENTS * NUM_DEVICES_PER_CLIENT * NUM_EPOCHS_PRE_TRAIN * len(
                hierarchical_data)

        update_clock()
        setup_logging()
        safe_log("============================================================================")
        safe_log("======================Process Started=======================================")
        safe_log("============================================================================")
        writer = SummaryWriter(log_dir="runs/hdpftl_pipeline")

        # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
        with named_timer("Preprocessing", writer, tag="Preprocessing"):
            global X_test, y_test
            X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data(
                selected_folder)
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
            global_model, personalized_models = load_from_file(writer)

        personalised_acc, client_accs, global_acc = evaluation(X_test, client_data_dict_test, global_model,
                                                               personalized_models, writer, y_test)
        plot(global_model)
        stop_clock()
        complete_progress_bar()
        # plot_confusion_matrix(y_true=y_test, y_pred=predictions, class_names=[str(i) for i in range(num_classes)])

        # plot_training_loss(losses=np.load(EPOCH_FILE_PRE), label='Pre Epoch Losses')
        # plot_training_loss(losses=np.load(EPOCH_FILE_FINE), label='Fine Tuning Epoch Losses')
        # plot_accuracy_comparison(client_accs, personalised_acc)
        # plot_client_accuracies(client_accs, global_acc=global_acc, title="Per-Client vs Global Model Accuracy")
        # plot_personalized_vs_global(personalised_acc, global_acc)

        safe_log("===========================================================================")
        safe_log("===========================Process Completed===============================")
        safe_log("============================================================================")


    def load_from_file(writer):
        with named_timer("Evaluate Global Model From File", writer, tag="EvalFromFile"):
            global_model = evaluate_global_model_fromfile()
        # Load personalized models
        with named_timer("Evaluate Personalized Models From File", writer, tag="PersonalizedEval"):
            personalized_models = load_personalized_models_fromfile()
        return global_model, personalized_models


    def evaluation(X_test, client_data_dict_test, global_model, personalized_models, writer, y_test):
        # Evaluate
        # global_accs = evaluate_global_model(global_model, X_test, y_test, client_partitions_test)
        # personalized_accs = evaluate_personalized_models_per_client(personalized_models, X_test, y_test, client_partitions_test)
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #######################  EVALUATION  #######################
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        """During evaluation: Use  global model for generalization tests 
                Use personalized models to report per - client performance"""
        with named_timer("Evaluate Personalized Models", writer, tag="PersonalizedEval"):
            personalised_acc = evaluate_personalized_models_per_client(personalized_models, client_data_dict_test)
        with named_timer("Evaluate Personalized Models Per Client", writer, tag="PersonalizedEvalperClient"):
            client_accs = evaluate_per_client(global_model, client_data_dict_test)
        with named_timer("Evaluate Global Model", writer, tag="GlobalEval"):
            global_acc = evaluate_global_model(global_model, X_test, y_test)
        return personalised_acc, client_accs, global_acc


    """
        safe_log("[12] Cross Validate Model...")
        with named_timer("Cross Validate Model", writer, tag="ValidateModel"):
            accuracies = cross_validate_model(X_test, y_test, k=5, num_epochs=20, lr=0.001)

        safe_log("[12] Cross Validate Model with F1 Score...")
        with named_timer("Cross Validate Model with F1 Score", writer, tag="ValidateModelF1"):
            fold_results = cross_validate_model_advanced(X_test, y_test, k=5, num_epochs=20, early_stopping=True)
    """


    def plot(global_model):
        global predictions
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #######################  PLOT  #############################
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(setup_device())
            outputs = global_model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
        num_classes = int(max(y_test.max().item(), predictions.cpu().max().item()) + 1)
        safe_log("Number of classes:::", num_classes)


    def convert_to_hms(mins, secs):
        total_seconds = int(mins * 60 + secs)
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return hh, mm, ss

        # GUI


    def on_selection(event):
        global selected_folder
        selection = listbox.curselection()
        if selection:
            index = selection[0]
            selected_folder = listbox.get(index)
            label_selected.config(text=f"📂 Selected Folder: {selected_folder}")
            start_button.state(["!disabled"])
            client_label_dist_btn.config(state="normal")
            confusion_matrix_btn.config(state="normal")
            pre_epoch_loses_btn.config(state="normal")
            fine_tune_epoch_loses_btn.config(state="normal")
            per_global_dotted_btn.config(state="normal")
            per_global_bar_btn.config(state="normal")
            cross_validation_btn.config(state="normal")


    def open_log_window():
        try:
            with open("hdpftl_run.log", "r") as f:
                log_contents = f.read()
        except FileNotFoundError:
            messagebox.showerror("Error", "hdpftl_run.log not found.")
            return

        log_win = tk.Toplevel(root)
        log_win.title("Log Viewer")
        log_win.geometry("600x400")

        text_area = scrolledtext.ScrolledText(log_win, wrap=tk.WORD)
        text_area.pack(expand=True, fill='both')
        text_area.insert(tk.END, log_contents)
        text_area.config(state='disabled')  # Make read-only


    root = tk.Tk()
    root.title("HDPFTL Architecture")

    # Responsive size
    window_width = 800
    window_height = 600
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.minsize(500, 700)
    # Calculate position to center the window
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)

    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    top_frame = tk.Frame(root)
    top_frame.pack(side='top', pady=20)

    # Label
    tk.Label(top_frame, text="HDPFTL Architecture", font=("Arial", 18, "bold")).grid(row=0, column=1, padx=10)
    tk.Button(top_frame, text="View log.txt", command=open_log_window, width=10).grid(row=0, column=2, padx=10)

    time_taken_label = tk.Label(top_frame, font=('Arial', 12), fg='red', text="Total Time: --:--:--")
    time_taken_label.grid(row=0, column=4, padx=10)

    # Frame for listbox and scrollbar
    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=20, pady=10)

    # Scrollbar
    scrollbar = tk.Scrollbar(frame)
    scrollbar.pack(side="right", fill="y")

    # Listbox
    listbox = tk.Listbox(
        frame,
        height=5,
        width=60,
        font=("Segoe UI", 13),
        selectmode=tk.SINGLE,
        bg="#f9f9f9",
        fg="#333",
        selectbackground="#a0c4ff",
        selectforeground="#000",
        activestyle="dotbox",
        relief="ridge",
        yscrollcommand=scrollbar.set
    )
    listbox.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    scrollbar.config(command=listbox.yview)

    # Insert folders
    for folder in get_output_folders(OUTPUT_DATASET_ALL_DATA):
        listbox.insert(tk.END, folder)

    # Selection label
    label_selected = tk.Label(root, text="📂 Selected Folder: None", font=("Helvetica", 12))
    label_selected.pack(pady=10)

    # Bind selection event
    listbox.bind('<<ListboxSelect>>', on_selection)

    # Buttons for actions
    button_params = {"width": 30, "font": ("Arial", 11), "pady": 5}
    client_label_dist_btn = tk.Button(
        root,
        text="Client Labels Distribution",
        state="disabled",
        command=lambda: (
            start_process(),
            plot_class_distribution_per_client(client_data_dict)
        ),
        **button_params
    )
    client_label_dist_btn.pack()

    confusion_matrix_btn = tk.Button(
        root,
        text="Confusion Matrix",
        state="disabled",
        command=lambda: (
            start_process(),
            plot_confusion_matrix(y_test, predictions, [str(i) for i in range(int(num_classes))], normalize=True)
        ),
        **button_params
    )
    confusion_matrix_btn.pack()

    pre_epoch_loses_btn = tk.Button(
        root,
        text="Pre Epoch Losses",
        state="disabled",
        command=lambda: (
            start_process(),
            plot_training_loss(np.load(EPOCH_FILE_PRE), 'epoch_loss_pre.png', 'Pre Epoch Losses')
        ),
        **button_params
    )
    pre_epoch_loses_btn.pack()

    fine_tune_epoch_loses_btn = tk.Button(
        root,
        text="Fine Tuning Epoch Losses",
        state="disabled",
        command=lambda: (
            start_process(),
            plot_training_loss(np.load(EPOCH_FILE_FINE), 'epoch_loss_fine.png', 'Fine Tuning Epoch Losses')
        ),
        **button_params
    )
    fine_tune_epoch_loses_btn.pack()

    per_global_dotted_btn = tk.Button(
        root,
        text="Personalized vs Global--Dotted",
        state="disabled",
        command=lambda: (
            start_process(),
            plot_client_accuracies(client_accs, global_acc, "Personalized vs Global--Dotted")
        ),
        **button_params
    )
    per_global_dotted_btn.pack()

    per_global_bar_btn = tk.Button(
        root,
        text="Personalized vs Global--Bar Chart",
        state="disabled",
        command=lambda: (
            start_process(),
            plot_personalized_vs_global(personalised_acc, global_acc)
        ),
        **button_params
    )
    per_global_bar_btn.pack()

    cross_validation_btn = tk.Button(
        root,
        text="Cross Validation Model",
        state="disabled",
        command=lambda: (
            start_process(),
            cross_validate_model_with_plots(X_test, y_test)
        ),
        **button_params
    )
    cross_validation_btn.pack()

    # Create a custom style
    style = ttk.Style(root)
    style.theme_use('default')  # Use default theme to enable custom styling

    style.configure(
        "Custom.Horizontal.TProgressbar",
        thickness=25,  # Controls the height
        troughcolor='#e0e0e0',  # Background area color
        background='#4caf50',  # Progress fill color
        bordercolor='#ccc',
        lightcolor='#4caf50',
        darkcolor='#4caf50'
    )

    # Create the progress bar
    progress = ttk.Progressbar(
        root,
        orient="horizontal",
        mode="determinate",
        length=500,
        style="Custom.Horizontal.TProgressbar"
    )
    progress.pack(fill='x', padx=20, pady=(20, 5))

    # Progress label
    progress_label = tk.Label(root, text="Progress: 0%", font=("Segoe UI", 10))
    progress_label.pack(pady=(0, 10))

    # Start training button
    style = ttk.Style()
    style.theme_use('default')
    style.configure("Custom.TButton",
                    foreground="white",
                    background="#d32f2f",

                    font=("Arial", 16, "bold"),
                    padding=10)
    style.map("Custom.TButton",
              background=[('active', '#b71c1c')],
              foreground=[('disabled', 'gray')])

    bottom_frame = tk.Frame(root)
    bottom_frame.pack(side='bottom', pady=20)

    # Start time label
    clock_label_start = tk.Label(bottom_frame, font=('Arial', 12), fg='green', text="Start Time: --:--:--")
    clock_label_start.grid(row=0, column=0, padx=10)

    # Start Training Button
    start_button = ttk.Button(bottom_frame, text="Start Training", command=start_training, style="Custom.TButton",
                              width=25)
    start_button.grid(row=0, column=1, padx=10)
    start_button.state(["disabled"])

    # End time label
    clock_label_end = tk.Label(bottom_frame, font=('Arial', 12), fg='red', text="End Time: --:--:--")
    clock_label_end.grid(row=0, column=2, padx=10)

    start_time_label = clock_label_start
    end_time_label = clock_label_end

    root.mainloop()

# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:        hdpftl_main.py
   Description:      HDPFTL - Preventing Zero-day Attacks on IoT Devices using
                     Hierarchical Decentralized Personalized Federated Transfer Learning (HDPFTL)
                     with ResNet-18 Model for Cross-Silo Collaboration on Heterogeneous Non-IID Data
   Author:          Sandeep Ghosh
   Created Date:     2025-04-21
   Python3 Version:   3.12.8
-------------------------------------------------
"""
import os
import pickle
import threading
import time
import tkinter as tk
import traceback
import warnings

from multiprocessing import Process, Event, Queue
from tkinter import scrolledtext, messagebox
from tkinter import ttk

import numpy as np
import torch
from joblib import dump, load
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
from hdpftl_evaluation.evaluate_global_model import evaluate_global_model, evaluate_global_model_fromfile
from hdpftl_evaluation.evaluate_per_client import evaluate_personalized_models_per_client, evaluate_per_client, \
    load_personalized_models_fromfile
from hdpftl_plotting.plot import plot_client_accuracies, plot_personalized_vs_global, plot_confusion_matrix, \
    plot_training_loss, plot_class_distribution_per_client, cross_validate_model_with_plots
from hdpftl_training.hdpftl_data.preprocess import preprocess_data
from hdpftl_training.hdpftl_pipeline import hdpftl_pipeline, dirichlet_partition_with_devices
from hdpftl_training.hdpftl_pre_training.finetune_model import finetune_model
from hdpftl_training.hdpftl_pre_training.pretrainclass import pretrain_class
from hdpftl_utility.config import EPOCH_FILE_FINE, EPOCH_FILE_PRE, INPUT_DIM, NUM_CLIENTS, \
    NUM_DEVICES_PER_CLIENT, GLOBAL_MODEL_PATH_TEMPLATE, OUTPUT_DATASET_ALL_DATA, LOGS_DIR_TEMPLATE, X_Y_TEST_PATH_TEMPLATE, PARTITIONED_DATA_PATH_TEMPLATE, \
    RESULTS_PATH_TEMPLATE, PREDICTIONS_PATH_TEMPLATE
from hdpftl_utility.log import setup_logging, safe_log
from hdpftl_utility.utils import named_timer, setup_device, get_output_folders, get_today_date, number_of_data_folders, \
    is_folder_exist
from typing import BinaryIO


warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

warnings.filterwarnings("ignore", category=SyntaxWarning)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

if __name__ == "__main__":

    personalized_models = None
    hierarchical_data = {}
    client_data_dict_test = {}
    hierarchical_data_test={}
    client_accs = {}
    predictions = None
    num_classes = ""
    global_model = {}
    global_acc = 0.0
    personalised_acc = {}

    X_test = {}
    y_test = {}
    client_data_dict = {}
    y_pred = None
    selected_folder = ""
    start_time_label = None
    end_time_label = None
    hh = None
    mm = None
    ss = None
    after_id = ""
    start_time = float(time.time())

    def update_progress(value):
        root.after(0, lambda: _update_progress_ui(value))


    def _update_progress_ui(value):
        progress['value'] = value
        progress_label.config(text=f"Training Progress: {int(value)}%")
        if value == 100:
            progress_label.config(text="✅ Training complete!")


    def monitor_process(p, q, done):
        global num_classes,global_acc,client_accs,personalised_acc,predictions
        p.join()  # Wait for process to finish
        if p.exitcode != 0:
            print(f"❌ Process crashed with exit code {p.exitcode}")
            return

        if not q.empty():
            results = q.get()
            if done.is_set():
                print("✅ Process finished (event received).")
                global_acc, client_accs, personalised_acc,predictions, num_classes = results
                print("Global:", global_acc)
                print("Client:", client_accs)
                print("Personalised:", personalised_acc)
                stop_clock()
                complete_progress_bar()
            else:
                print("⚠️ Queue received data but done flag not set.")
        else:
            print("❌ Queue is empty. Process may have crashed before q.put()")


    def start_thread():
        ctx = mp.get_context("spawn")  # Use spawn instead of fork

        q = ctx.Queue()
        done_event = ctx.Event()

        p = Process(target=start_process, args=(q, done_event))
        p.start()
        return p, q, done_event


    def start_training():
        # Start infinite progress animation
        global start_time
        progress.config(mode='indeterminate')
        progress.start(10)
        progress_label.config(text="Training in progress...")

        # Start long-running task in a new thread
        update_clock()
        p, q, done = start_thread()
        # ✅ Run non-blocking monitor in background
        threading.Thread(target=monitor_process, args=(p, q, done), daemon=True).start()


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


    def start_process(q, done_event):
        global hh, mm, ss
        global global_model,personalized_models, X_test, y_test, client_data_dict,hierarchical_data,\
            client_data_dict_test, hierarchical_data_test,personalised_acc, client_accs, global_acc,\
            predictions,num_classes

        try:
            log_path_str = LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder, date=get_today_date())
            is_folder_exist(log_path_str)
            setup_logging(log_path_str)
            safe_log("============================================================================")
            safe_log("======================Process Started=======================================")
            safe_log("============================================================================")
            writer = SummaryWriter(log_dir="runs/hdpftl_pipeline")
            # If fine-tuned model exists, load and return it
            if not os.path.exists(GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=get_today_date())):
                # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
                with named_timer("Preprocessing", writer, tag="Preprocessing"):
                    global X_test, y_test
                    X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data(
                        selected_folder)
                # safe_log("[1]Data preprocessing completed.")
                #device = setup_device()
                """
                dirichlet_partition is a standard technique to simulate non-IID data — 
                and it's commonly used in federated learning experiments to control the degree of 
                heterogeneity among clients.
                Smaller alpha → more skewed, clients have few classes dominating.
                Larger alpha → more uniform data distribution across clients.
                """
                partition_output_path = PARTITIONED_DATA_PATH_TEMPLATE.substitute(n=get_today_date())
                os.makedirs(os.path.dirname(partition_output_path), exist_ok=True)
                xy_output_path = X_Y_TEST_PATH_TEMPLATE.substitute(n=get_today_date())
                os.makedirs(os.path.dirname(xy_output_path), exist_ok=True)
                result_output_path = RESULTS_PATH_TEMPLATE.substitute(n=get_today_date())
                os.makedirs(os.path.dirname(result_output_path), exist_ok=True)
                predictions_output_path = PREDICTIONS_PATH_TEMPLATE.substitute(n=get_today_date())
                os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)

                save_path = os.path.join(xy_output_path + "X_y_test.joblib")
                dump((X_test, y_test), save_path)

                with named_timer("dirichlet_partition", writer, tag="dirichlet_partition"):
                    client_data_dict, hierarchical_data = dirichlet_partition_with_devices(
                        X_pretrain, y_pretrain, alpha=0.5, num_clients=NUM_CLIENTS,
                        num_devices_per_client=NUM_DEVICES_PER_CLIENT
                    )
                with open(partition_output_path+ "partitioned_data.pkl", "wb") as f:
                    pickle.dump((client_data_dict, hierarchical_data), f)

                with named_timer("dirichlet_partition_test", writer, tag="dirichlet_partition_test"):
                    client_data_dict_test, hierarchical_data_test = dirichlet_partition_with_devices(
                        X_test, y_test, alpha=0.5, num_clients=NUM_CLIENTS, num_devices_per_client=NUM_DEVICES_PER_CLIENT
                    )

                with open(partition_output_path+"partitioned_data_test.pkl", "wb") as f:
                    pickle.dump((client_data_dict_test, hierarchical_data_test), f)


                # Step 2: Pretrain global model
                with named_timer("pretrain_class", writer, tag="pretrain_class"):
                    pretrain_class(X_pretrain, X_test, y_pretrain, y_test, input_dim=INPUT_DIM, early_stop_patience=10)
                    # Step 3: Instantiate target model and train on device
                with named_timer("target_class", writer, tag="target_class"):
                    def base_model_fn():
                        return finetune_model(
                            X_finetune,
                            y_finetune,
                            input_dim=X_finetune.shape[1],
                            target_classes=len(np.unique(y_finetune)))

                with named_timer("hdpftl_pipeline", writer, tag="hdpftl_pipeline"):
                    global_model, personalized_models = hdpftl_pipeline(base_model_fn, hierarchical_data, X_test, y_test)

                personalised_acc, client_accs, global_acc = evaluation(X_test, client_data_dict_test, global_model,
                                                                       personalized_models, writer, y_test)
                with open(result_output_path + "results.pkl", "wb") as f:
                    pickle.dump((personalised_acc, client_accs, global_acc), f)

                predictions, num_classes = plot(global_model)
                # Save predictions and num_classes
                with open(predictions_output_path + "predictions.pkl", "wb") as f:
                    pickle.dump((predictions.cpu().numpy(), num_classes), f)

            #######################  LOAD FROM FILES ##################################
            else:
                load_from_files(writer)

            # Ensure client_accs and personalised_acc are CPU-safe (if they are tensors)
            #client_accs = [acc.cpu() if hasattr(acc, 'cpu') else acc for acc in client_accs]
            #if hasattr(personalised_acc, 'cpu'):
                #personalised_acc = personalised_acc.cpu()


            q.put((global_acc, client_accs, personalised_acc,predictions, num_classes))
            # Signal completion
            done_event.set()

            safe_log("===========================================================================")
            safe_log("===========================Process Completed===============================")
            safe_log("============================================================================")

        except Exception as e:
            safe_log("Exception in thread:", e, level="error")
            traceback.print_exc()


    def load_from_files(writer):
        partition_output_path = PARTITIONED_DATA_PATH_TEMPLATE.substitute(n=get_today_date()) + "partitioned_data.pkl"
        partition_output_test_path = PARTITIONED_DATA_PATH_TEMPLATE.substitute(n=get_today_date()) + "partitioned_data_test.pkl"
        xy_output_path = X_Y_TEST_PATH_TEMPLATE.substitute(n=get_today_date())  + "X_y_test.joblib"
        result_output_path = RESULTS_PATH_TEMPLATE.substitute(n=get_today_date()) + "results.pkl"
        predictions_output_path = PREDICTIONS_PATH_TEMPLATE.substitute(n=get_today_date()) + "predictions.pkl"

        global global_model,personalized_models, X_test, y_test, client_data_dict,hierarchical_data,\
            client_data_dict_test, hierarchical_data_test,personalised_acc, client_accs, global_acc,\
            predictions,num_classes
        with named_timer("Evaluate Global Model From File", writer, tag="EvalFromFile"):
            global_model = evaluate_global_model_fromfile()
        with named_timer("Evaluate Personalized Models From File", writer, tag="PersonalizedEval"):
            personalized_models = load_personalized_models_fromfile()
        X_test, y_test = load(xy_output_path)
        with open(partition_output_path, "rb") as f:
            client_data_dict, hierarchical_data = pickle.load(f)
        with open(partition_output_test_path, "rb") as f:
            client_data_dict_test, hierarchical_data_test = pickle.load(f)
        with open(result_output_path, "rb") as f:
            personalised_acc, client_accs, global_acc = pickle.load(f)
        with open(predictions_output_path, "rb") as f:
            predictions, num_classes = pickle.load(f)

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
        global predictions,num_classes
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        #######################  PLOT  #############################
        """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(setup_device())
            outputs = global_model(X_test_tensor)
            _, predictions = torch.max(outputs, 1)
        num_classes = int(max(y_test.max().item(), predictions.cpu().max().item()) + 1)
        safe_log("Number of classes:::", num_classes)
        return predictions, num_classes


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
            view_log_btn.config(state="normal")


    def open_log_window():
        try:
            log_path_str = LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder, date=get_today_date())
            with open(log_path_str+"hdpftl_run.log", "r") as f:
                log_contents = f.read()
        except FileNotFoundError:
            messagebox.showerror("Error", "hdpftl_run.log not found.")
            return

        log_win = tk.Toplevel(root)
        log_win.title("Log Viewer for dataset:"+ selected_folder+ " and dated:" + get_today_date())
        log_win.geometry("600x400")

        text_area = scrolledtext.ScrolledText(log_win, wrap=tk.WORD)
        text_area.pack(expand=True, fill='both')
        text_area.insert(tk.END, log_contents)
        text_area.config(state='disabled')  # Make read-only


    root = tk.Tk()
    root.title("HDPFTL Architecture")

    # Responsive size
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = int(screen_width * 0.8)
    window_height = int(screen_height * 0.8)

    # Calculate position to center the window
    # Calculate position to center the window
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Apply minimum size if you want
    root.minsize(500, 700)  # optional, keep or remove

    root.geometry(f"{window_width}x{window_height}+{x}+{y}")

    top_frame = tk.Frame(root)
    top_frame.pack(side='top', pady=20)

    # Label
    tk.Label(top_frame, text="HDPFTL Architecture", font=("Arial", 18, "bold")).grid(row=0, column=1, padx=10)
    view_log_btn  = tk.Button(top_frame, text="View Log", command=open_log_window, width=10,state="disabled")
    view_log_btn.grid(row=0, column=2, padx=10)

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


    def handle_client_label_distribution():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot_class_distribution_per_client(client_data_dict)
        else:
            print("❌ Subprocess failed or did not complete properly.")


    client_label_dist_btn = tk.Button(
        root,
        text="Client Labels Distribution",
        state="disabled",
        command=handle_client_label_distribution,
        **button_params
    )
    client_label_dist_btn.pack()


    def handle_confusion_matrix():
        global num_classes,global_acc,client_accs,personalised_acc,predictions
        p, q, done_event = start_thread()
        p.join()  # Wait for the process to complete

        if p.exitcode == 0 and done_event.is_set():
            try:
                # Try to get predictions and num_classes from queue if available
                if not q.empty():
                    global_acc, client_accs, personalised_acc,predictions, num_classes = q.get()

                # Validate num_classes
                if isinstance(num_classes, int) and num_classes > 0:
                    class_labels = [str(i) for i in range(num_classes)]
                    plot_confusion_matrix(y_test, predictions, class_labels, normalize=True)
                else:
                    print("❌ num_classes is invalid or not properly set.")
            except Exception as e:
                print(f"❌ Error during confusion matrix plotting: {e}")
        else:
            print("❌ Subprocess failed or did not complete properly.")


    confusion_matrix_btn = tk.Button(
        root,
        text="Confusion Matrix",
        state="disabled",
        command=handle_confusion_matrix,
        **button_params
    )

    confusion_matrix_btn.pack()


    def handle_pre_epoch_losses():
        p, q, done_event = start_thread()
        p.join()  # Wait for process to finish

        if p.exitcode == 0 and done_event.is_set():
            plot_training_loss(np.load(EPOCH_FILE_PRE), 'epoch_loss_pre.png', 'Pre Epoch Losses')
        else:
            print("❌ Failed to complete pre-epoch process.")


    pre_epoch_loses_btn = tk.Button(
        root,
        text="Pre Epoch Losses",
        state="disabled",
        command=handle_pre_epoch_losses,
        **button_params
    )

    pre_epoch_loses_btn.pack()


    def handle_fine_tune_losses():
        p, q, done_event = start_thread()
        p.join()  # Wait for process to finish

        if p.exitcode == 0 and done_event.is_set():
            plot_training_loss(np.load(EPOCH_FILE_FINE), 'epoch_loss_fine.png', 'Fine Tuning Epoch Losses')
        else:
            print("❌ Fine-tuning process failed or did not signal completion.")


    fine_tune_epoch_loses_btn = tk.Button(
        root,
        text="Fine Tuning Epoch Losses",
        state="disabled",
        command=handle_fine_tune_losses,
        **button_params
    )

    fine_tune_epoch_loses_btn.pack()


    def handle_plot_personalised_vs_global():
        p, q, done_event = start_thread()
        p.join()  # Wait for process to finish

        if p.exitcode == 0 and done_event.is_set():
            plot_client_accuracies(client_accs, global_acc, "Personalized vs Global--Dotted")
        else:
            print("❌ Failed to generate plot. Process exited with error or did not complete.")


    per_global_dotted_btn = tk.Button(
        root,
        text="Personalized vs Global--Dotted",
        state="disabled",
        command=handle_plot_personalised_vs_global,
        **button_params
    )

    per_global_dotted_btn.pack()


    def handle_personalized_vs_global_bar():
        p, q, done_event = start_thread()
        p.join()  # Wait for process to finish

        if p.exitcode == 0 and done_event.is_set():
            plot_personalized_vs_global(personalised_acc, global_acc)
        else:
            print("❌ Process failed or did not finish properly.")


    per_global_bar_btn = tk.Button(
        root,
        text="Personalized vs Global--Bar Chart",
        state="disabled",
        command=handle_personalized_vs_global_bar,
        **button_params
    )

    per_global_bar_btn.pack()


    def handle_cross_validation():
        p, q, done_event = start_thread()
        p.join()  # Wait for the subprocess to finish

        if p.exitcode == 0 and done_event.is_set():
            cross_validate_model_with_plots(X_test, y_test)
        else:
            print("❌ Cross-validation process failed or didn’t signal completion.")


    cross_validation_btn = tk.Button(
        root,
        text="Cross Validation Model",
        state="disabled",
        command=handle_cross_validation,
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

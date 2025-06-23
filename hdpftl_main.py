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
import multiprocessing as mp
import os
import pickle
import platform
import re
import shutil
import threading
import time
import tkinter as tk
import traceback
import warnings
from multiprocessing import Process
from tkinter import scrolledtext, messagebox, font
from tkinter import ttk

import numpy as np
import torch
from joblib import dump, load
from torch.utils.tensorboard import SummaryWriter

from hdpftl_evaluation.evaluate_global_model import evaluate_global_model, evaluate_global_model_fromfile
from hdpftl_evaluation.evaluate_per_client import evaluate_personalized_models_per_client, evaluate_per_client, \
    load_personalized_models_fromfile
from hdpftl_plotting.plot import plot_client_accuracies, plot_personalized_vs_global, plot_confusion_matrix, \
    plot_training_loss, plot_class_distribution_per_client, cross_validate_model_with_plots
from hdpftl_training.hdpftl_data.preprocess import preprocess_data
from hdpftl_training.hdpftl_pipeline import hdpftl_pipeline, dirichlet_partition_with_devices
from hdpftl_training.hdpftl_pre_training.finetune_model import finetune_model
from hdpftl_training.hdpftl_pre_training.pretrainclass import pretrain_class
from hdpftl_utility import config
from hdpftl_utility.config import EPOCH_FILE_FINE, EPOCH_FILE_PRE, NUM_CLIENTS, \
    NUM_DEVICES_PER_CLIENT, GLOBAL_MODEL_PATH_TEMPLATE, OUTPUT_DATASET_ALL_DATA, LOGS_DIR_TEMPLATE, \
    TRAINED_MODEL_FOLDER_PATH, EPOCH_DIR, PLOT_PATH
from hdpftl_utility.log import setup_logging, safe_log
from hdpftl_utility.utils import named_timer, setup_device, get_output_folders, get_today_date, is_folder_exist

warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

warnings.filterwarnings("ignore", category=SyntaxWarning)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

personalized_models = None
hierarchical_data = {}
client_data_dict_test = {}
hierarchical_data_test = {}
client_accs = {}
predictions = None
num_classes = ""
global_model = {}
global_acc = 0.0
personalised_acc = {}
writer = SummaryWriter(log_dir="runs/hdpftl_pipeline")
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
is_training = None
training_process: Process | None = None
done_flag = None
result_buttons = {}
# Global config dictionary to hold parameters
config_params = {
    "BATCH_SIZE": 5,
    "BATCH_SIZE_TRAINING": 16,
    "NUM_CLIENTS": 10,
    "NUM_DEVICES_PER_CLIENT": 5,
    "NUM_EPOCHS_PRE_TRAIN": 5,
    "NUM_FEDERATED_ROUND": 5
}


def open_settings_window():
    settings_win = tk.Toplevel(root)
    settings_win.title("Training Settings")
    # Remove fixed geometry for auto-sizing
    # settings_win.geometry("350x350")
    # settings_win.resizable(False, False)
    settings_win.minsize(300, 200)
    settings_win.resizable(True, False)

    settings_win.columnconfigure(1, weight=1)  # Make entry column expand

    entries = {}

    for idx, (key, val) in enumerate(config_params.items()):
        ttk.Label(settings_win, text=f"{key}:").grid(row=idx, column=0, padx=10, pady=8, sticky="w")
        entry = ttk.Entry(settings_win)
        entry.grid(row=idx, column=1, padx=10, pady=8, sticky="ew")
        entry.insert(0, str(val))
        entries[key] = entry

    def save_settings():
        # Update config module in memory first
        for key, entry in entries.items():
            val = entry.get()
            if val.isdigit():
                setattr(config, key, int(val))
            else:
                try:
                    setattr(config, key, float(val))
                except ValueError:
                    setattr(config, key, val)

        config_path = os.path.join(os.path.dirname(__file__), "hdpftl_utility/config.py")
        if not os.path.exists(config_path):
            print("config.py not found! Cannot update.")
            return

        # Read all lines of config.py
        with open(config_path, "r") as f:
            lines = f.readlines()

        # Prepare regex patterns to match each key assignment
        patterns = {key: re.compile(rf"^{key}\s*=\s*.*$") for key in entries.keys()}

        # Prepare replacement lines for updated keys
        replacements = {}
        for key in entries.keys():
            val = getattr(config, key)
            if isinstance(val, str):
                replacements[key] = f'{key} = "{val}"\n'
            else:
                replacements[key] = f"{key} = {val}\n"

        # Update lines if key found, else append new assignments at end
        updated_keys = set()
        for i, line in enumerate(lines):
            for key, pattern in patterns.items():
                if pattern.match(line):
                    lines[i] = replacements[key]
                    updated_keys.add(key)
                    break

        # Append keys not found in file (new keys)
        for key in entries.keys():
            if key not in updated_keys:
                lines.append(replacements[key])

        # Write back updated lines
        with open(config_path, "w") as f:
            f.writelines(lines)

        print("Config updated with changed values.")
        settings_win.destroy()

    # Save and Cancel buttons
    btn_frame = ttk.Frame(settings_win)
    btn_frame.grid(row=len(config_params), column=0, columnspan=2, pady=20)

    ttk.Button(btn_frame, text="Save", command=save_settings).pack(side="left", padx=10)
    ttk.Button(btn_frame, text="Cancel", command=settings_win.destroy).pack(side="left", padx=10)


def evaluation(X_test_param, client_data_dict_test_param, global_model_param, personalized_models_param, writer_param,
               y_test_param):
    global personalised_acc, client_accs, global_acc
    result_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date())
    os.makedirs(os.path.dirname(result_output_path), exist_ok=True)
    predictions_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date())
    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)

    # Evaluate
    # global_accs = evaluate_global_model(global_model, X_test, y_test, client_partitions_test)
    # personalized_accs = evaluate_personalized_models_per_client(personalized_models, X_test, y_test, client_partitions_test)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #######################  EVALUATION  #######################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """During evaluation: Use  global model for generalization tests 
            Use personalized models to report per - client performance"""
    with named_timer("Evaluate Personalized Models", writer_param, tag="PersonalizedEval"):
        personalised_acc = evaluate_personalized_models_per_client(personalized_models_param,
                                                                   client_data_dict_test_param)
    with named_timer("Evaluate Personalized Models Per Client", writer, tag="PersonalizedEvalperClient"):
        client_accs = evaluate_per_client(global_model_param, client_data_dict_test_param)
    with named_timer("Evaluate Global Model", writer, tag="GlobalEval"):
        global_acc = evaluate_global_model(global_model_param, X_test_param, y_test_param)

        with open(result_output_path + "results.pkl", "wb") as f:
            pickle.dump((personalised_acc, client_accs, global_acc), f)

        prediction, num_of_classes = plot(global_model)
        # Save predictions and num_classes
        with open(predictions_output_path + "predictions.pkl", "wb") as f:
            pickle.dump((prediction.cpu().numpy(), num_of_classes), f)

    return personalised_acc, client_accs, global_acc


def plot(global_model_param):
    global predictions, num_classes
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #######################  PLOT  #############################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(setup_device())
        outputs = global_model_param(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
    num_classes = int(max(y_test.max().item(), predictions.cpu().max().item()) + 1)
    safe_log("Number of classes:::", num_classes)
    return predictions, num_classes


def load_from_files(writer_param):
    partition_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "partitioned_data.pkl"
    partition_output_test_path = TRAINED_MODEL_FOLDER_PATH.substitute(
        n=get_today_date()) + "partitioned_data_test.pkl"
    xy_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "X_y_test.joblib"
    result_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "results.pkl"
    predictions_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "predictions.pkl"

    global global_model, personalized_models, X_test, y_test, client_data_dict, hierarchical_data, \
        client_data_dict_test, hierarchical_data_test, personalised_acc, client_accs, global_acc, \
        predictions, num_classes
    with named_timer("Evaluate Global Model From File", writer_param, tag="EvalFromFile"):
        global_model = evaluate_global_model_fromfile()
    with named_timer("Evaluate Personalized Models From File", writer_param, tag="PersonalizedEval"):
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
    return global_model, personalized_models, X_test, y_test, client_data_dict, hierarchical_data, \
        client_data_dict_test, hierarchical_data_test, personalised_acc, client_accs, global_acc, \
        predictions, num_classes


def start_process(selected_folder_param, done_event):
    global hh, mm, ss
    global global_model, personalized_models, X_test, y_test, client_data_dict, hierarchical_data, \
        client_data_dict_test, hierarchical_data_test, personalised_acc, client_accs, global_acc, \
        predictions, num_classes

    try:
        log_path_str = LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder_param, date=get_today_date())
        is_folder_exist(log_path_str)
        setup_logging(log_path_str)
        safe_log("============================================================================")
        safe_log("======================Process Started=======================================")
        safe_log("============================================================================")
        # If fine-tuned model exists, load and return it
        if not os.path.exists(GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=get_today_date())):
            # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
            with named_timer("Preprocessing", writer, tag="Preprocessing"):
                global X_test, y_test
                # For deep learning:
                X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data(
                    selected_folder_param, scaler_type='minmax')
                # For classical ML:----Dont Delete the below comment...its for the different parameter different situations models
                # X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess_data(selected_folder,scaler_type='standard')
            # safe_log("[1]Data preprocessing completed.")
            # device = setup_device()
            """
            dirichlet_partition is a standard technique to simulate non-IID data — 
            and it's commonly used in federated learning experiments to control the degree of 
            heterogeneity among clients.
            Smaller alpha → more skewed, clients have few classes dominating.
            Larger alpha → more uniform data distribution across clients.
            """
            partition_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date())
            os.makedirs(os.path.dirname(partition_output_path), exist_ok=True)
            xy_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date())
            os.makedirs(os.path.dirname(xy_output_path), exist_ok=True)

            save_path = os.path.join(xy_output_path + "X_y_test.joblib")
            dump((X_test, y_test), save_path)

            with named_timer("dirichlet_partition", writer, tag="dirichlet_partition"):
                client_data_dict, hierarchical_data = dirichlet_partition_with_devices(
                    X_pretrain, y_pretrain, alpha=0.5, num_clients=NUM_CLIENTS,
                    num_devices_per_client=NUM_DEVICES_PER_CLIENT
                )
            with open(partition_output_path + "partitioned_data.pkl", "wb") as f:
                pickle.dump((client_data_dict, hierarchical_data), f)

            with named_timer("dirichlet_partition_test", writer, tag="dirichlet_partition_test"):
                client_data_dict_test, hierarchical_data_test = dirichlet_partition_with_devices(
                    X_test, y_test, alpha=0.5, num_clients=NUM_CLIENTS,
                    num_devices_per_client=NUM_DEVICES_PER_CLIENT
                )

            with open(partition_output_path + "partitioned_data_test.pkl", "wb") as f:
                pickle.dump((client_data_dict_test, hierarchical_data_test), f)

            # Step 2: Pretrain global model
            with named_timer("pretrain_class", writer, tag="pretrain_class"):
                pretrain_class(X_pretrain, X_test, y_pretrain, y_test, input_dim=X_pretrain.shape[1],
                               early_stop_patience=10)
                # Step 3: Instantiate target model and train on device
            with named_timer("target_class", writer, tag="target_class"):
                def base_model_fn():
                    return finetune_model(
                        X_finetune,
                        y_finetune,
                        input_dim=X_finetune.shape[1],
                        target_classes=len(np.unique(y_finetune)))

            with named_timer("hdpftl_pipeline", writer, tag="hdpftl_pipeline"):
                global_model, personalized_models = hdpftl_pipeline(base_model_fn, hierarchical_data, X_test,
                                                                    y_test)

            personalised_acc, client_accs, global_acc = evaluation(X_test, client_data_dict_test, global_model,
                                                                   personalized_models, writer, y_test)
        #######################  LOAD FROM FILES ##################################
        else:
            load_from_files(writer)

        # Ensure client_accs and personalised_acc are CPU-safe (if they are tensors)
        # client_accs = [acc.cpu() if hasattr(acc, 'cpu') else acc for acc in client_accs]
        # if hasattr(personalised_acc, 'cpu'):
        # personalised_acc = personalised_acc.cpu()
        """
        def to_cpu_deep(obj):
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: to_cpu_deep(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_cpu_deep(v) for v in obj]
            elif isinstance(obj, tuple):
                return tuple(to_cpu_deep(v) for v in obj)
            else:
                return obj
        """

        # results = (global_acc, client_accs, personalised_acc, predictions, num_classes)
        # safe_results = to_cpu_deep(results)  # ✅ ONLY send CPU-safe objects
        # q.put(safe_results)

        # Signal completion
        done_event.set()
        time.sleep(1)  # Give monitor thread time to get the data

        # safe_log("Sending to queue:")
        # for i, item in enumerate(safe_results):
        # safe_log(f" - Item {i}: {type(item)}, CUDA: {getattr(item, 'is_cuda', False)}")

        safe_log("===========================================================================")
        safe_log("===========================Process Completed===============================")
        safe_log("============================================================================")

    except Exception as e:
        safe_log("Exception in thread:", e, level="error")
        traceback.print_exc()


def disable_result_buttons():
    for label, btn in result_buttons.items():
        btn.config(state="disabled")


if __name__ == "__main__":
    def update_progress(value):
        root.after(0, lambda: _update_progress_ui(value))


    def _update_progress_ui(value):
        progress['value'] = value
        progress_label.config(text=f"Training Progress: {int(value)}%")
        if value == 100:
            progress_label.config(text="Progress: 100% - Training Complete!")


    def monitor_process(p, q, done_event):
        global num_classes, global_acc, client_accs, personalised_acc, predictions, writer, is_training
        global global_model, personalized_models, X_test, y_test, client_data_dict, hierarchical_data, \
            client_data_dict_test, hierarchical_data_test, personalised_acc, client_accs, global_acc, \
            predictions, num_classes

        p.join()  # Wait for process to finish
        if p.exitcode != 0:
            print(f"❌ Process crashed with exit code {p.exitcode}")
            is_training = False
            start_button.config(text="Start Training")
            stop_clock()
            complete_progress_bar()
            return

        # if not q.empty():
        # try :
        # results = q.get(timeout=5)
        # If tensors somehow still leak through, convert to CPU
        # results = tuple(r.cpu() if isinstance(r, torch.Tensor) and r.is_cuda else r for r in results)
        # except Exception as e:
        #    print(f"❌ Failed to get results from queue: {e}")
        #     return
        if done_event.is_set():
            print("✅ Process finished (event received).")
            writer.add_scalar("Accuracy/Global", global_acc)
            global_model, personalized_models, X_test, y_test, client_data_dict, hierarchical_data, \
                client_data_dict_test, hierarchical_data_test, personalised_acc, client_accs, global_acc, \
                predictions, num_classes = load_from_files(writer)
            # global_acc, client_accs, personalised_acc,predictions, num_classes = results
            print("Global:", global_acc)
            print("Client:", client_accs)
            print("Personalised:", personalised_acc)
            writer.close()
            is_training = False
            start_button.config(text="Start Training")
            partition_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "partitioned_data.pkl"
            xy_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "X_y_test.joblib"
            result_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "results.pkl"
            predictions_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "predictions.pkl"

            # Define a dictionary mapping button labels to their required file paths
            file_paths = {
                "📊 Client Labels Distribution": partition_output_path,
                "📉 Confusion Matrix": [xy_output_path, predictions_output_path],
                "📈 Pre Epoch Losses": os.path.join(PLOT_PATH + get_today_date() + "/", 'epoch_loss_pre.png'),
                "🛠️ Fine Tuning Epoch Losses": os.path.join(PLOT_PATH + get_today_date() + "/", 'epoch_loss_fine.png'),
                "🔁 Personalized vs Global--Bar Chart": result_output_path,
                "🔄 Personalized vs Global--Dotted": result_output_path,
                " Cross Validation Model": TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "X_y_test.joblib",
                "📄 View Log": LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder,
                                                           date=get_today_date()) + "hdpftl_run.log"
            }

            for label, btn in result_buttons.items():
                paths = file_paths.get(label, [])
                # Normalize to list if single path string
                if isinstance(paths, str):
                    paths = [paths]
                # Check if all files exist (change to any() if OR needed)
                if all(os.path.exists(p) for p in paths):
                    btn.config(state="normal")
                else:
                    btn.config(state="disabled")
            stop_clock()
            complete_progress_bar()
        else:
            print("⚠️ Queue received data but done flag not set.")
        # else:
        #    print("❌ Queue is empty. Process may have crashed before q.put()")


    def start_thread():
        ctx = mp.get_context("spawn")  # Use spawn instead of fork

        q = ctx.Queue()
        done_event = ctx.Event()

        p = Process(target=start_process, args=(selected_folder, done_event,))
        p.start()
        return p, q, done_event


    def clear_trainings():
        # Remove directories
        dirs_to_remove = [
            LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder, date=get_today_date()),
            EPOCH_DIR,
            TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()),
            LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder, date=get_today_date()) + "hdpftl_run.log"
        ]
        for dir_path in dirs_to_remove:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    print(f"Removed directory: {dir_path}")
                except Exception as e:
                    print(f"Failed to remove {dir_path}: {e}")
            else:
                print(f"Directory does not exist: {dir_path}")
        disable_result_buttons()


    def start_training():
        global is_training, start_time
        global training_process, done_flag
        # Start infinite progress animation
        if is_training:
            if training_process and training_process.is_alive():
                print("⚠️ Terminating training process...")
                training_process.terminate()
                training_process.join()
                start_time_label.config(text="Start Time: --:--:--")
                end_time_label.config(text="End Time: --:--:--")
                time_taken_label.config(text="Total Time: --:--:--")

            is_training = False
            start_button.config(text="Start Training")
            stop_clock()
            complete_progress_bar()
            # Disable buttons immediately
            disable_result_buttons()
            # Close writer if open
            if writer and not writer.close:
                writer.close()
            clear_trainings()

            return

        is_training = True
        progress.config(mode='indeterminate')
        progress.start(10)
        progress_label.config(text="Training in progress...")
        start_button.config(text="Stop Training")
        # Start long-running task in a new thread
        update_clock()
        training_process, q, done_event = start_thread()
        # ✅ Run non-blocking monitor in background
        threading.Thread(target=monitor_process, args=(training_process, q, done_event), daemon=True).start()


    def complete_progress_bar():
        def finish():
            progress.stop()
            progress.config(mode='determinate', maximum=100)

            end_time = time.time()
            elapsed_time = end_time - start_time
            mins, secs = divmod(elapsed_time, 60)
            hh, mm, ss = convert_to_hms(mins, secs)

            # total_time_taken_label.config(text=f"Total Time: {hh}H:{mm}M:{ss}S")
            time_taken_label.config(text=f"Total Time: {hh}Hours:{mm}Minutes:{ss}seconds")

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


    """
        safe_log("[12] Cross Validate Model...")
        with named_timer("Cross Validate Model", writer, tag="ValidateModel"):
            accuracies = cross_validate_model(X_test, y_test, k=5, num_epochs=20, lr=0.001)

        safe_log("[12] Cross Validate Model with F1 Score...")
        with named_timer("Cross Validate Model with F1 Score", writer, tag="ValidateModelF1"):
            fold_results = cross_validate_model_advanced(X_test, y_test, k=5, num_epochs=20, early_stopping=True)
    """


    def convert_to_hms(mins, secs):
        total_seconds = int(mins * 60 + secs)
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return hh, mm, ss

        # GUI


    def on_selection(event):
        global selected_folder, result_buttons
        selection = listbox.curselection()
        if selection:
            index = selection[0]
            selected_folder = listbox.get(index)
            label_selected.config(text=f"📂 Selected Folder: {selected_folder}")
            start_button.state(["!disabled"])

            partition_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "partitioned_data.pkl"
            partition_output_test_path = TRAINED_MODEL_FOLDER_PATH.substitute(
                n=get_today_date()) + "partitioned_data_test.pkl"
            xy_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "X_y_test.joblib"
            result_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "results.pkl"
            predictions_output_path = TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "predictions.pkl"

            # Define a dictionary mapping button labels to their required file paths
            file_paths = {
                "📊 Client Labels Distribution": partition_output_path,
                "📉 Confusion Matrix": [xy_output_path, predictions_output_path],
                "📈 Pre Epoch Losses": os.path.join(PLOT_PATH + get_today_date() + "/", 'epoch_loss_pre.png'),
                "🛠️ Fine Tuning Epoch Losses": os.path.join(PLOT_PATH + get_today_date() + "/", 'epoch_loss_fine.png'),
                "🔁 Personalized vs Global--Bar Chart": result_output_path,
                "🔄 Personalized vs Global--Dotted": result_output_path,
                " Cross Validation Model": TRAINED_MODEL_FOLDER_PATH.substitute(n=get_today_date()) + "X_y_test.joblib",
                "📄 View Log": LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder,
                                                           date=get_today_date()) + "hdpftl_run.log"
            }

            for label, btn in result_buttons.items():
                paths = file_paths.get(label, [])
                # Normalize to list if single path string
                if isinstance(paths, str):
                    paths = [paths]
                # Check if all files exist (change to any() if OR needed)
                if all(os.path.exists(p) for p in paths):
                    btn.config(state="normal")
                else:
                    btn.config(state="disabled")
        else:
            label_selected.config(text="📂 Selected Folder: None")
            start_button.state(["disabled"])  # Disable start button
            for label, btn in result_buttons.items():
                btn.config(state="disabled")


    def open_log_window():
        try:
            log_path_str = LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder, date=get_today_date())
            with open(log_path_str + "hdpftl_run.log", "r") as f:
                log_contents = f.read()
        except FileNotFoundError:
            messagebox.showerror("Error", "hdpftl_run.log not found.")
            return

        log_win = tk.Toplevel(root)
        log_win.title("Log Viewer for dataset:" + selected_folder + " and dated:" + get_today_date())
        log_win.geometry("600x400")

        text_area = scrolledtext.ScrolledText(log_win, wrap=tk.WORD)
        text_area.pack(expand=True, fill='both')
        text_area.insert(tk.END, log_contents)
        text_area.config(state='disabled')  # Make read-only


    # ---------- Set Window Size ----------
    mp.set_start_method("spawn")
    root = tk.Tk()
    root.title("HDPFTL Architecture")
    # 1. Set default font for all widgets in this root window
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family="Helvetica", size=9)  # Smaller font
    root.option_add("*Font", default_font)

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

    # Header
    header_label = tk.Label(root, text="HDPFTL Architecture", font=("Arial", 18, "bold"))
    header_label.pack(pady=(15, 5))

    # Main frame (use grid or pack consistently)
    main_frame = tk.Frame(root)
    main_frame.pack(fill='both', expand=True)

    # ---------- Frame 1: Selection Area ----------
    style = ttk.Style(root)
    style.theme_use('default')
    # Style for selection_frame LabelFrame
    style.configure(
        "Selection.TLabelframe",
        background="#fff7e6",  # very light warm/yellow background
        bordercolor="#f5a623",  # warm orange border
        relief="solid",
        borderwidth=3,
        padding=10
    )

    style.configure(
        "Selection.TLabelframe.Label",
        font=("Arial", 16, "bold"),
        foreground="#b35e00"  # deep orange for title text
    )
    # Define reusable "Distinct" style if not already defined
    style.configure("Distinct.TLabelframe",
                    background="#f0f8ff",  # very light blue
                    bordercolor="#0288d1",  # bright blue border
                    relief="solid",
                    borderwidth=3)

    style.configure("Distinct.TLabelframe.Label",
                    font=("Arial", 14, "bold"),
                    foreground="#01579b")  # deep blue title text

    # Create selection_frame using same style class
    selection_frame = ttk.LabelFrame(main_frame, text="🗂 Select Dataset", style="Distinct.TLabelframe", padding=10)
    selection_frame.pack(fill="x", padx=10, pady=(10, 5))

    # Selection label
    label_selected = tk.Label(
        selection_frame,
        text="📂 Selected Folder: None",
        font=("Helvetica", 24, "bold"),
        anchor='center',
        justify='center',
    )
    label_selected.pack(pady=10, fill='x')

    # Scrollbar
    scrollbar = tk.Scrollbar(selection_frame)
    scrollbar.pack(side="right", fill="y")

    # Listbox
    listbox = tk.Listbox(
        selection_frame,
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

    # Bind selection event
    listbox.bind('<<ListboxSelect>>', on_selection)


    # ---------- Frame 2: Control Area ----------
    # ------------------ Tooltip Class ------------------ #
    class ToolTip:
        def __init__(self, widget, text):
            self.widget = widget
            self.text = text
            self.tip_window = None
            widget.bind("<Enter>", self.show_tip)
            widget.bind("<Leave>", self.hide_tip)

        def show_tip(self, event=None):
            if self.tip_window or not self.text:
                return
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + 20
            self.tip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            label = tk.Label(tw, text=self.text, justify='left',
                             background="#ffffe0", relief='solid', borderwidth=1,
                             font=("Segoe UI", 9))
            label.pack(ipadx=5, ipady=3)

        def hide_tip(self, event=None):
            if self.tip_window:
                self.tip_window.destroy()
                self.tip_window = None


    # ------------------ Control Area ------------------ #
    # Create style for LabelFrame
    style = ttk.Style(root)
    style.theme_use('default')

    style.configure(
        "Distinct.TLabelframe",
        background="#e6f0ff",  # very light blue background
        bordercolor="#1a73e8",  # bright blue border
        relief="solid",
        borderwidth=3,
        padding=10
    )

    style.configure(
        "Distinct.TLabelframe.Label",
        font=("Arial", 16, "bold"),
        foreground="#0b5394"  # deep blue for title text
    )

    # Create the custom label frame with style
    control_frame = ttk.LabelFrame(main_frame, text="🛠️ Process", style="Distinct.TLabelframe", padding=10)
    control_frame.pack(fill="both", padx=10, pady=(10, 5))

    for i in range(4):
        control_frame.columnconfigure(i, weight=1)

    # Style setup
    style = ttk.Style(root)
    style.theme_use('default')
    style.configure(
        "Custom.Horizontal.TProgressbar",
        thickness=30,  # taller bar
        troughcolor="#cccccc",  # light gray background
        background="#0b84a5",  # vibrant blue (or try "#4caf50", "#ff9800", etc.)
        bordercolor="#000000",  # optional
        lightcolor="#0b84a5",
        darkcolor="#0b84a5"
    )
    style.configure("Custom.TButton",
                    foreground="white",
                    background="#d32f2f",
                    font=("Arial", 14, "bold"),
                    padding=10)
    style.map("Custom.TButton",
              background=[('active', '#b71c1c')],
              foreground=[('disabled', 'gray')])

    # Start time label
    clock_label_start = tk.Label(control_frame, font=('Arial', 12), fg='green', text="🕒 Start Time: --:--:--")
    clock_label_start.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
    ToolTip(clock_label_start, "When the training process starts")
    start_time_label = clock_label_start
    # Start button
    start_button = ttk.Button(control_frame, text="🚀 Start Training", command=start_training, style="Custom.TButton")
    start_button.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
    start_button.state(["disabled"])
    ToolTip(start_button, "Start the federated training process")

    # End time label
    clock_label_end = tk.Label(control_frame, font=('Arial', 12), fg='red', text="🛑 End Time: --:--:--")
    clock_label_end.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
    ToolTip(clock_label_end, "When training finishes")
    end_time_label = clock_label_end
    # Total time label
    time_taken_label = tk.Label(control_frame, font=('Arial', 12), fg='blue', text="⏱️ Total Time: --:--:--")
    time_taken_label.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
    ToolTip(time_taken_label, "Total duration of training")

    # Separator
    ttk.Separator(control_frame, orient="horizontal").grid(row=1, column=0, columnspan=4, sticky="ew", pady=(10, 5))

    # Progress bar
    progress = ttk.Progressbar(
        control_frame,
        orient="horizontal",
        mode="determinate",
        style="Custom.Horizontal.TProgressbar"
    )
    progress.grid(row=2, column=0, columnspan=4, sticky="ew", padx=10, pady=(15, 5))

    # Progress label (clearly visible)
    progress_label = tk.Label(control_frame, text="Progress: 0%", font=("Segoe UI", 11, "bold"), fg="#0b84a5")
    progress_label.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(0, 10))
    ToolTip(progress_label, "Shows real-time training progress")

    # ------------------ Dark/Light Theme Toggle ------------------ #
    is_dark_mode = False


    def toggle_theme():
        global is_dark_mode
        if is_dark_mode:
            root.tk_setPalette(background='white', foreground='black')
            style.configure(".", background='white', foreground='black')
            style.configure("Custom.Horizontal.TProgressbar", troughcolor='#f0f0f0', background='#4caf50')
        else:
            root.tk_setPalette(background='#2e2e2e', foreground='white')
            style.configure(".", background='#2e2e2e', foreground='white')
            style.configure("Custom.Horizontal.TProgressbar", troughcolor='#3c3c3c', background='#81c784')
        is_dark_mode = not is_dark_mode


    # Button to open settings window
    settings_button = ttk.Button(control_frame, text="⚙️ Settings", command=open_settings_window)
    settings_button.grid(row=4, column=1, sticky="e", padx=5, pady=5)
    ToolTip(settings_button, "Modify Settings")

    theme_button = ttk.Button(control_frame, text="🌓 Toggle Theme", command=toggle_theme)
    theme_button.grid(row=4, column=2, sticky="e", padx=5, pady=5)
    ToolTip(theme_button, "Switch between dark and light mode")

    # Add Clear Training button in column 4
    clear_training_button = ttk.Button(
        control_frame,
        text="🧹 Clear Training",
        command=clear_trainings
    )
    clear_training_button.grid(row=4, column=3, sticky="ew", padx=5, pady=5)
    ToolTip(clear_training_button, "Clear all training logs and outputs")

    # ------------------ Optional: Animated Progress ------------------ #
    is_training = False  # Controlled externally


    def animate_progress_label():
        current = progress_label.cget("text")
        if current.endswith("..."):
            progress_label.config(text="Progress: Training")
        else:
            progress_label.config(text=current + ".")
        if is_training:
            root.after(500, animate_progress_label)


    # ---------- Frame 3: Result Area ----------

    # --- Button styling ---
    button_params = {
        "font": ("Arial", 7),
        "height": 2,
        "width": 25
    }

    # Style for result_frame LabelFrame
    style.configure(
        "Results.TLabelframe",
        background="#e6fff0",  # very light mint/green background
        bordercolor="#34a853",  # fresh green border
        relief="solid",
        borderwidth=3,
        padding=10
    )
    style.configure(
        "Results.TLabelframe.Label",
        font=("Arial", 16, "bold"),
        foreground="#20733e"  # deep green for title text
    )

    # Result Frame
    result_frame = ttk.LabelFrame(main_frame, text="📊 Results", style="Results.TLabelframe", padding=10)
    result_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

    # Inner frame for grid layout
    button_grid_frame = ttk.Frame(result_frame, padding=5)
    button_grid_frame.pack(fill="both", expand=True)


    # --- Button action handlers ---
    def handle_client_label_distribution():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot_class_distribution_per_client(client_data_dict)
        else:
            print("❌ Subprocess failed or did not complete properly.")


    def handle_confusion_matrix():
        global num_classes, global_acc, client_accs, personalised_acc, predictions
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            try:
                if isinstance(num_classes, int) and num_classes > 0:
                    class_labels = [str(i) for i in range(num_classes)]
                    plot_confusion_matrix(y_test, predictions, class_labels, normalize=True)
                else:
                    print("❌ num_classes is invalid or not properly set.")
            except Exception as e:
                print(f"❌ Error during confusion matrix plotting: {e}")
        else:
            print("❌ Subprocess failed or did not complete properly.")


    def handle_pre_epoch_losses():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot_training_loss(np.load(EPOCH_FILE_PRE), 'epoch_loss_pre.png', 'Pre Epoch Losses')
        else:
            print("❌ Failed to complete pre-epoch process.")


    def handle_fine_tune_losses():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot_training_loss(np.load(EPOCH_FILE_FINE), 'epoch_loss_fine.png', 'Fine Tuning Epoch Losses')
        else:
            print("❌ Fine-tuning process failed or did not signal completion.")


    def handle_plot_personalised_vs_global():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot_client_accuracies(client_accs, global_acc, "Personalized vs Global--Dotted")
        else:
            print("❌ Failed to generate plot. Process exited with error or did not complete.")


    def handle_personalized_vs_global_bar():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot_personalized_vs_global(personalised_acc, global_acc)
        else:
            print("❌ Process failed or did not finish properly.")


    def handle_cross_validation():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            cross_validate_model_with_plots(X_test, y_test)
        else:
            print("❌ Cross-validation process failed or didn’t signal completion.")


    # --- Button list ---
    buttons = [
        ("📊 Client Labels Distribution", handle_client_label_distribution),
        ("📉 Confusion Matrix", handle_confusion_matrix),
        ("📈 Pre Epoch Losses", handle_pre_epoch_losses),
        ("🛠️ Fine Tuning Epoch Losses", handle_fine_tune_losses),
        ("🔁 Personalized vs Global--Bar Chart", handle_personalized_vs_global_bar),
        ("🔄 Personalized vs Global--Dotted", handle_plot_personalised_vs_global),
        ("🔬 Cross Validation Model", handle_cross_validation),
        ("📄 View Log", open_log_window)
    ]


    # --- Hover effect ---
    def on_enter(e):
        e.widget.config(bg="#005f73", fg="purple", cursor="hand2")


    def on_leave(e):
        system = platform.system()
        if system == "Windows":
            default_bg = "SystemButtonFace"
        elif system == "Darwin":  # macOS
            default_bg = "#ececec"
        else:  # Linux (Pop!_OS etc)
            default_bg = "#f0f0f0"

        e.widget.config(bg=default_bg, fg="black", cursor="arrow")


    # --- Add buttons in a 3-column grid ---
    for idx, (label, command) in enumerate(buttons):
        row = idx // 3
        col = idx % 3
        state = "normal" if label == "View Log" else "disabled"
        btn = tk.Button(
            button_grid_frame,
            text=f"🔹 {label}",
            command=command,
            state="disabled",
            **button_params
        )
        btn.grid(row=row, column=col, padx=10, pady=10, sticky="ew")
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)
        button_grid_frame.grid_columnconfigure(col, weight=1)
        result_buttons[label] = btn

    root.mainloop()

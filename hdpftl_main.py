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
import shutil
import tkinter as tk
import config
import threading
import time
import traceback
import warnings
from multiprocessing import Process
from tkinter import scrolledtext, messagebox, font
from tkinter import ttk, filedialog

global use_all_files_var, loaded_files
import numpy as np
import torch
from joblib import dump, load
from torch.utils.tensorboard import SummaryWriter

import hdpftl_evaluation.evaluate_global_model as evaluate_global_model
import hdpftl_evaluation.evaluate_per_client as evaluate_per_client
import hdpftl_plotting.plot as plot
import hdpftl_training.hdpftl_data.preprocess as preprocess
import hdpftl_training.hdpftl_pipeline as pipeline
import hdpftl_training.hdpftl_pre_training.finetune_model as finetune_model
import hdpftl_training.hdpftl_pre_training.pretrainclass as pretrainclass
import hdpftl_utility.config as config
import hdpftl_utility.log as log_util
import hdpftl_utility.utils as util

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
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
loaded_files = []
use_all_files_var = None  # default unchecked
# Global config dictionary to hold parameters

log_stop_event = None
log_thread = None
config_params = util.sync_config_params(config.saved_config_params) #initialization


def enable_disable_button():
    partition_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(
        n=util.get_today_date()) + "partitioned_data.pkl"
    xy_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date()) + "X_y_test.joblib"
    result_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date()) + "results.pkl"
    predictions_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date()) + "predictions.pkl"

    # Define a dictionary mapping button labels to their required file paths
    file_paths = {
        "📊 Client Labels Distribution": partition_output_path,
        "📉 Confusion Matrix": [xy_output_path, predictions_output_path],
        "📈 Pre Epoch Losses": os.path.join(config.PLOT_PATH + util.get_today_date() + "/", 'epoch_loss_pre.png'),
        "🛠️ Fine Tuning Epoch Losses": os.path.join(config.PLOT_PATH + util.get_today_date() + "/",
                                                    'epoch_loss_fine.png'),
        "🔁 Personalized vs Global--Bar Chart": result_output_path,
        "🔄 Personalized vs Global--Dotted": result_output_path,
        "🔬 Cross Validation Model": config.TRAINED_MODEL_FOLDER_PATH.substitute(
            n=util.get_today_date()) + "X_y_test.joblib",
        "📄 View Log": config.LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder,
                                                          date=util.get_today_date()) + "hdpftl_run.log"
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

def disable_result_buttons():
    for label, btn in result_buttons.items():
        btn.config(state="disabled")

def open_settings_window():
    import os
    import shutil
    import tkinter as tk
    from tkinter import ttk, filedialog
    import re

    try:
        from tkdnd2 import TkDND  # pip install TkinterDnD2
        DND_AVAILABLE = True
    except ImportError:
        DND_AVAILABLE = False

    global use_all_files_var, loaded_files
    util.sync_config_params(config_params)  # Ensure config_params synced
    loaded_files = list(getattr(config, "TEST_CSV_PATHS", []))

    settings_win = tk.Toplevel(root)
    settings_win.title("Training Settings")
    settings_win.minsize(450, 400)
    settings_win.resizable(False, False)
    settings_win.columnconfigure(1, weight=1)

    entries = {}

    # ---------- Checkbox ----------
    use_all_files_var = tk.BooleanVar(value=getattr(config, "USE_UPLOADED_TEST_FILES", False))
    checkbox = ttk.Checkbutton(
        settings_win,
        text="Use all uploaded CSV files for testing",
        variable=use_all_files_var
    )
    checkbox.grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=5)

    # ---------- Config Entry Fields ----------
    for idx, key in enumerate(config_params.keys()):
        if key == "USE_UPLOADED_TEST_FILES":
            continue
        ttk.Label(settings_win, text=f"{key}:", anchor="w").grid(
            row=idx + 1, column=0, padx=15, pady=5, sticky="w"
        )
        entry = ttk.Entry(settings_win)
        entry.grid(row=idx + 1, column=1, padx=15, pady=5, sticky="ew")
        entry.insert(0, str(getattr(config, key, config_params[key])))
        entries[key] = entry

    # ---------- CSV Selection ----------
    csv_label = ttk.Label(settings_win, text="No CSVs selected", foreground="gray", anchor="w")
    csv_label.grid(row=len(config_params) + 1, column=0, columnspan=2, sticky="w", padx=15, pady=(15, 5))

    listbox_frame = ttk.Frame(settings_win, borderwidth=1, relief="solid")
    listbox_frame.grid(row=len(config_params) + 3, column=0, columnspan=2, padx=15, pady=(0, 15), sticky="nsew")
    settings_win.rowconfigure(len(config_params) + 3, weight=1)

    scrollbar = ttk.Scrollbar(listbox_frame, orient="vertical")
    csv_listbox = tk.Listbox(
        listbox_frame,
        height=6,
        yscrollcommand=scrollbar.set,
        selectmode="browse",
        exportselection=False,
        borderwidth=0,
        highlightthickness=0,
    )
    scrollbar.config(command=csv_listbox.yview)
    csv_listbox.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    def on_select(event):
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            value = event.widget.get(index)
            print(f"Selected: {value}")

    csv_listbox.bind("<<ListboxSelect>>", on_select)

    # ---------- Helper: Add CSV Files ----------
    def add_csv_files(file_paths):
        global loaded_files
        new_files = []
        for path in file_paths:
            if os.path.isfile(path) and path.lower().endswith(".csv"):
                filename = os.path.basename(path)
                dest_path = os.path.join(config.OUTPUT_DATASET_SELECTED_TEST_DATA, filename)
                if not os.path.exists(config.OUTPUT_DATASET_SELECTED_TEST_DATA):
                    os.makedirs(config.OUTPUT_DATASET_SELECTED_TEST_DATA)
                try:
                    shutil.copy2(path, dest_path)
                    if path not in loaded_files:
                        loaded_files.append(path)
                        new_files.append(path)
                except Exception as e:
                    print(f"Error copying {path}: {e}")

        csv_listbox.delete(0, tk.END)
        for f in loaded_files:
            csv_listbox.insert(tk.END, os.path.basename(f))
        csv_label.config(text=f"Loaded {len(loaded_files)} file(s)", foreground="green")

    def select_test_csv():
        file_paths = filedialog.askopenfilenames(
            title="Select One or More Test CSV Files",
            filetypes=[("CSV Files", "*.csv")],
            defaultextension=".csv"
        )
        if file_paths:
            add_csv_files(file_paths)

    csv_button = ttk.Button(settings_win, text="Select Test CSV(s)", command=select_test_csv)
    csv_button.grid(row=len(config_params) + 2, column=0, columnspan=2, pady=(0, 10), padx=15, sticky="ew")

    # ---------- Drag-and-Drop Support ----------
    if DND_AVAILABLE:
        dnd = TkDND(settings_win)
        def drop_callback(event):
            files = settings_win.tk.splitlist(event.data)
            add_csv_files(files)
        dnd.bindtarget(csv_listbox, drop_callback, 'text/uri-list')

    # ---------- Load previous CSVs ----------
    def load_previous_test_csvs():
        csv_listbox.delete(0, tk.END)
        loaded_files.clear()
        if hasattr(config, "TEST_CSV_PATHS") and config.TEST_CSV_PATHS:
            for path in config.TEST_CSV_PATHS:
                csv_listbox.insert(tk.END, os.path.basename(path))
            loaded_files.extend(config.TEST_CSV_PATHS)
            csv_label.config(text=f"Loaded {len(config.TEST_CSV_PATHS)} file(s)", foreground="green")
        else:
            csv_label.config(text="No CSVs selected", foreground="gray")

        use_all_files_var.set(bool(getattr(config, "USE_UPLOADED_TEST_FILES", False)))

    load_previous_test_csvs()

    # ---------- Save Settings ----------
    def save_settings():
        # 1️⃣ Save all entry values in-memory
        for key, entry in entries.items():
            val = entry.get()
            if val.isdigit():
                setattr(config, key, int(val))
            else:
                try:
                    setattr(config, key, float(val))
                except ValueError:
                    setattr(config, key, val)

        # 2️⃣ Save CSV paths and checkbox in-memory
        setattr(config, "TEST_CSV_PATHS", tuple(loaded_files))
        use_uploaded = bool(use_all_files_var.get())
        setattr(config, "USE_UPLOADED_TEST_FILES", use_uploaded)
        config_params["USE_UPLOADED_TEST_FILES"] = use_uploaded

        # 3️⃣ Update config.py safely without deleting unrelated lines
        config_path = os.path.join(os.path.dirname(__file__), "hdpftl_utility/config.py")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                lines = f.readlines()
        else:
            lines = []

        patterns = {key: re.compile(rf"^{key}\s*=\s*.*$") for key in entries.keys()}
        patterns["TEST_CSV_PATHS"] = re.compile(r"^TEST_CSV_PATHS\s*=\s*.*$")
        patterns["USE_UPLOADED_TEST_FILES"] = re.compile(r"^USE_UPLOADED_TEST_FILES\s*=\s*.*$")

        replacements = {key: f"{key} = {repr(getattr(config, key))}\n" for key in entries.keys()}
        replacements["TEST_CSV_PATHS"] = f"TEST_CSV_PATHS = {repr(tuple(loaded_files))}\n"
        replacements["USE_UPLOADED_TEST_FILES"] = f"USE_UPLOADED_TEST_FILES = {use_uploaded}\n"

        updated_keys = set()
        for i, line in enumerate(lines):
            for key, pattern in patterns.items():
                if pattern.match(line):
                    lines[i] = replacements[key]
                    updated_keys.add(key)
                    break

        for key, new_line in replacements.items():
            if key not in updated_keys:
                lines.append(new_line)

        with open(config_path, "w") as f:
            f.writelines(lines)

        util.reload_config()
        print("✅ Config updated. Window will close now.")
        settings_win.destroy()

    # ---------- Buttons ----------
    btn_frame = ttk.Frame(settings_win)
    btn_frame.grid(row=len(config_params) + 4, column=0, columnspan=2, pady=10, padx=15, sticky="ew")

    save_btn = ttk.Button(btn_frame, text="Save", command=save_settings)
    save_btn.pack(side="right", padx=5)

    cancel_btn = ttk.Button(btn_frame, text="Cancel", command=settings_win.destroy)
    cancel_btn.pack(side="right", padx=5)

    # ---------- Close Handlers ----------
    def on_close():
        settings_win.grab_release()
        settings_win.destroy()

    settings_win.protocol("WM_DELETE_WINDOW", on_close)
    settings_win.bind("<Escape>", lambda e: on_close())
    settings_win.transient(root)
    settings_win.grab_set()
    settings_win.focus_set()

def load_from_files(writer_param):
    partition_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(
        n=util.get_today_date()) + "partitioned_data.pkl"
    partition_output_test_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(
        n=util.get_today_date()) + "partitioned_data_test.pkl"
    xy_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date()) + "X_y_test.joblib"
    result_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date()) + "results.pkl"
    predictions_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date()) + "predictions.pkl"

    global global_model, personalized_models, X_test, y_test, client_data_dict, hierarchical_data, \
        client_data_dict_test, hierarchical_data_test, personalised_acc, client_accs, global_acc, \
        predictions, num_classes
    with util.named_timer("Evaluate Global Model From File", writer_param, tag="EvalFromFile"):
        global_model = evaluate_global_model.evaluate_global_model_fromfile()
    with util.named_timer("Evaluate Personalized Models From File", writer_param, tag="PersonalizedEval"):
        personalized_models = evaluate_per_client.load_personalized_models_fromfile()
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

def start_log_watcher(log_path, log_text):
    def on_new_line(line):
        def append():
            log_text.config(state="normal")
            # Optional: add color tag detection
            if "ERROR" in line:
                log_text.insert(tk.END, line, "ERROR")
            elif "WARNING" in line:
                log_text.insert(tk.END, line, "WARNING")
            elif "INFO" in line:
                log_text.insert(tk.END, line, "INFO")
            else:
                log_text.insert(tk.END, line)

            if getattr(log_text, "auto_scroll", True):
                log_text.see(tk.END)

            log_text.config(state="disabled")

        log_text.after(0, append)

    # Optional: pass a stop_event to allow future clean shutdown
    stop_event = threading.Event()
    watcher_thread = threading.Thread(
        target=tail_log_file,
        args=(log_path, on_new_line, stop_event),
        daemon=True
    )
    watcher_thread.start()
    return stop_event, watcher_thread  # Return stop_event if you want to stop watching later

def stop_log_watcher():
    if log_stop_event:
        log_stop_event.set()  # ✅ This will stop the thread loop
        print("Log watcher stopping...")

    # Optional: wait for thread to exit gracefully
    if log_thread:
        log_thread.join(timeout=2)
        print("Log thread stopped.")

def tail_log_file(filepath, on_line_callback, stop_event):
    print(f"Monitoring log file: {filepath}")

    file = None
    file_inode = None

    while not stop_event.is_set():
        if os.path.isfile(filepath):
            try:
                current_inode = os.stat(filepath).st_ino
                if file is None or file_inode != current_inode:
                    if file:
                        file.close()
                    file = open(filepath, 'r')
                    file_inode = current_inode
                    file.seek(0, os.SEEK_END)  # Go to the end for new lines only

                while not stop_event.is_set():
                    line = file.readline()
                    if line:
                        on_line_callback(line)
                    else:
                        # Check if file was truncated or replaced
                        try:
                            new_inode = os.stat(filepath).st_ino
                            if new_inode != file_inode:
                                break  # file replaced, reopen next loop
                        except FileNotFoundError:
                            break  # file deleted
                        time.sleep(0.1)

            except Exception as e:
                print(f"Error reading log file: {e}")
                time.sleep(1)
        else:
            if file:
                file.close()
                file = None
                file_inode = None
            time.sleep(0.5)

def evaluation(X_test_param, client_data_dict_test_param, global_model_param, personalized_models_param, writer_param,
               y_test_param):
    global personalised_acc, client_accs, global_acc
    result_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date())
    os.makedirs(os.path.dirname(result_output_path), exist_ok=True)
    predictions_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date())
    os.makedirs(os.path.dirname(predictions_output_path), exist_ok=True)

    # Evaluate
    # global_accs = evaluate_global_model(global_model, X_test, y_test, client_partitions_test)
    # personalized_accs = evaluate_personalized_models_per_client(personalized_models, X_test, y_test, client_partitions_test)
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #######################  EVALUATION  #######################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """During evaluation: Use  global model for generalization tests 
            Use personalized models to report per - client performance"""
    with util.named_timer("Evaluate Personalized Models", writer_param, tag="PersonalizedEval"):
        personalised_acc = evaluate_per_client.evaluate_personalized_models_per_client(personalized_models_param,
                                                                                       client_data_dict_test_param)
    with util.named_timer("Evaluate Personalized Models Per Client", writer, tag="PersonalizedEvalperClient"):
        client_accs = evaluate_per_client.evaluate_per_client(global_model_param, client_data_dict_test_param)
    with util.named_timer("Evaluate Global Model", writer, tag="GlobalEval"):
        global_acc = evaluate_global_model.evaluate_global_model(global_model_param, X_test_param, y_test_param)

        with open(result_output_path + "results.pkl", "wb") as f:
            pickle.dump((personalised_acc, client_accs, global_acc), f)

        prediction, num_of_classes = plot(global_model)
        # Save predictions and num_classes
        with open(predictions_output_path + "predictions.pkl", "wb") as f:
            pickle.dump((prediction.cpu().numpy(), num_of_classes), f)

    return personalised_acc, client_accs, global_acc

def start_process(selected_folder_param, done_event):
    global hh, mm, ss
    global global_model, personalized_models, X_test, y_test, client_data_dict, hierarchical_data, \
        client_data_dict_test, hierarchical_data_test, personalised_acc, client_accs, global_acc, \
        predictions, num_classes

    try:
        log_path_str = config.LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder_param, date=util.get_today_date())
        util.is_folder_exist(log_path_str)
        log_util.setup_logging(log_path_str)

        log_util.safe_log("============================================================================")
        log_util.safe_log("======================Process Started=======================================")
        log_util.safe_log("============================================================================")
        # If fine-tuned model exists, load and return it
        if not os.path.exists(config.GLOBAL_MODEL_PATH_TEMPLATE.substitute(n=util.get_today_date())):
            # download_dataset(INPUT_DATASET_PATH_2024, OUTPUT_DATASET_PATH_2024)
            with util.named_timer("Preprocessing", writer, tag="Preprocessing"):
                global X_test, y_test
                # For deep learning:
                X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess.preprocess_data(
                    log_path_str,
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
            partition_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date())
            os.makedirs(os.path.dirname(partition_output_path), exist_ok=True)
            xy_output_path = config.TRAINED_MODEL_FOLDER_PATH.substitute(n=util.get_today_date())
            os.makedirs(os.path.dirname(xy_output_path), exist_ok=True)

            save_path = os.path.join(xy_output_path + "X_y_test.joblib")
            dump((X_test, y_test), save_path)

            with util.named_timer("dirichlet_partition", writer, tag="dirichlet_partition"):
                client_data_dict, hierarchical_data = pipeline.dirichlet_partition_with_devices(
                    X_pretrain, y_pretrain, alpha=0.5, num_clients=config.NUM_CLIENTS,
                    num_devices_per_client=config.NUM_DEVICES_PER_CLIENT
                )
            with open(partition_output_path + "partitioned_data.pkl", "wb") as f:
                pickle.dump((client_data_dict, hierarchical_data), f)

            with util.named_timer("dirichlet_partition_test", writer, tag="dirichlet_partition_test"):
                client_data_dict_test, hierarchical_data_test = pipeline.dirichlet_partition_with_devices(
                    X_test, y_test, alpha=0.5, num_clients=config.NUM_CLIENTS,
                    num_devices_per_client=config.NUM_DEVICES_PER_CLIENT
                )

            with open(partition_output_path + "partitioned_data_test.pkl", "wb") as f:
                pickle.dump((client_data_dict_test, hierarchical_data_test), f)

            # Step 2: Pretrain global model
            with util.named_timer("pretrain_class", writer, tag="pretrain_class"):
                pretrainclass.pretrain_class(X_pretrain, X_test, y_pretrain, y_test, input_dim=X_pretrain.shape[1],
                                             early_stop_patience=10)
                # Step 3: Instantiate Finetune model and train on device
            with util.named_timer("target_class", writer, tag="target_class"):
                def base_model_fn():
                    return finetune_model.init_model(input_dim=X_finetune.shape[1],
                                                     target_classes=len(np.unique(y_finetune)))

            with open(partition_output_path + "general_data_test.pkl", "wb") as f:
                pickle.dump((X_pretrain.shape[1], X_finetune.shape[1], y_finetune, len(np.unique(y_finetune))), f)

                with util.named_timer("pfl_pipeline", writer, tag="pfl_pipeline"):
                    global_model, personalized_models = pipeline.run_pfl(
                        base_model_fn,
                        client_data_dict,  # use training client partitions
                        X_test, y_test,
                        num_rounds=config.NUM_FEDERATED_ROUND,
                        local_epochs=2
                    )

            #with util.named_timer("hdpftl_pipeline", writer, tag="hdpftl_pipeline"):
             #   global_model, personalized_models = pipeline.hdpftl_pipeline(base_model_fn, hierarchical_data, X_test,
             #                                                                y_test)

        #######################  LOAD FROM FILES ##################################
        else:
            with util.named_timer("Preprocessing", writer, tag="Preprocessing"):
                X_final, y_final, X_pretrain, y_pretrain, X_finetune, y_finetune, X_test, y_test = preprocess.preprocess_data(
                    "selected_test", scaler_type='minmax')

            load_from_files(writer)

        # if getattr(config, "USE_UPLOADED_TEST_FILES", False) and hasattr(config, "test_dfs") and config.test_dfs:
        # if use_all_files_var:

        personalised_acc, client_accs, global_acc = evaluation(X_test, client_data_dict_test, global_model,
                                                               personalized_models, writer, y_test)

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

        log_util.safe_log("===========================================================================")
        log_util.safe_log("===========================Process Completed===============================")
        log_util.safe_log("============================================================================")

    except Exception as e:
        log_util.safe_log("Exception in thread:", e, level="error")
        traceback.print_exc()

def plot(global_model_param):
    global predictions, num_classes
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    #######################  PLOT  #############################
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).float().to(util.setup_device())
        outputs = global_model_param(X_test_tensor)
        _, predictions = torch.max(outputs, 1)
    num_classes = int(max(y_test.max().item(), predictions.cpu().max().item()) + 1)
    log_util.safe_log("Number of classes:::", num_classes)
    return predictions, num_classes




if __name__ == "__main__":
    mp.set_start_method("spawn")
    root = tk.Tk()


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
            stop_log_watcher()
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

            enable_disable_button()
            stop_clock()
            complete_progress_bar()
            stop_log_watcher()
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
        for dir_path in config.dirs_to_remove:
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
            stop_log_watcher()
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


    def convert_to_hms(mins, secs):
        total_seconds = int(mins * 60 + secs)
        hh = total_seconds // 3600
        mm = (total_seconds % 3600) // 60
        ss = total_seconds % 60
        return hh, mm, ss

        # GUI


    def on_selection(event):
        global selected_folder, result_buttons,log_stop_event, log_thread
        selection = listbox.curselection()
        if selection:
            index = selection[0]
            selected_folder = listbox.get(index)
            log_path_str = config.LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder, date=util.get_today_date())
            log_stop_event, log_thread = start_log_watcher(log_path_str + "hdpftl_run.log", log_text)
            label_selected.config(text=f"📂 Selected Folder: {selected_folder}")
            start_button.state(["!disabled"])
            enable_disable_button()

        else:
            label_selected.config(text="📂 Selected Folder: None")
            start_button.state(["disabled"])  # Disable start button
            for label, btn in result_buttons.items():
                btn.config(state="disabled")


    # --- Button action handlers ---
    def handle_client_label_distribution():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot.plot_class_distribution_per_client(client_data_dict)
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
                    plot.plot_confusion_matrix(y_test, predictions, class_labels, normalize=True)
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
            plot.plot_training_loss(np.load(config.EPOCH_FILE_PRE), 'epoch_loss_pre.png', 'Pre Epoch Losses')
        else:
            print("❌ Failed to complete pre-epoch process.")


    def handle_fine_tune_losses():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot.plot_training_loss(np.load(config.EPOCH_FILE_FINE), 'epoch_loss_fine.png', 'Fine Tuning Epoch Losses')
        else:
            print("❌ Fine-tuning process failed or did not signal completion.")


    def handle_plot_personalised_vs_global():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot.plot_client_accuracies(client_accs, global_acc, "Personalized vs Global--Dotted")
        else:
            print("❌ Failed to generate plot. Process exited with error or did not complete.")


    def handle_personalized_vs_global_bar():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot.plot_personalized_vs_global(personalised_acc, global_acc)
        else:
            print("❌ Process failed or did not finish properly.")


    def handle_cross_validation():
        p, q, done_event = start_thread()
        p.join()
        if p.exitcode == 0 and done_event.is_set():
            plot.cross_validate_model_with_plots(X_test, y_test)
        else:
            print("❌ Cross-validation process failed or didn’t signal completion.")


    def toggle_theme():
        global is_dark_mode
        if is_dark_mode:
            # Light Theme
            root.tk_setPalette(background='white', foreground='black')
            style.configure(".", background='white', foreground='black')
            style.configure("TLabel", background='white', foreground='black')
            style.configure("TFrame", background='white')
            style.configure("Custom.Horizontal.TProgressbar", troughcolor='#f0f0f0', background='#4caf50')
        else:
            # Dark Theme
            root.tk_setPalette(background='#2e2e2e', foreground='white')
            style.configure(".", background='#2e2e2e', foreground='white')
            style.configure("TLabel", background='#2e2e2e', foreground='white')
            style.configure("TFrame", background='#2e2e2e')
            style.configure("Custom.Horizontal.TProgressbar", troughcolor='#3c3c3c', background='#81c784')

        is_dark_mode = not is_dark_mode


    def open_log_window():
        try:
            log_path_str = config.LOGS_DIR_TEMPLATE.substitute(dataset=selected_folder, date=util.get_today_date())
            with open(log_path_str + "hdpftl_run.log", "r") as f:
                log_contents = f.read()
        except FileNotFoundError:
            messagebox.showerror("Error", "hdpftl_run.log not found.")
            return

        log_win = tk.Toplevel(root)
        log_win.title("Log Viewer for dataset:" + selected_folder + " and dated:" + util.get_today_date())
        log_win.geometry("600x400")

        text_area = scrolledtext.ScrolledText(log_win, wrap=tk.WORD)
        text_area.pack(expand=True, fill='both')
        text_area.insert(tk.END, log_contents)
        text_area.config(state='disabled')  # Make read-only


    """
        safe_log("[12] Cross Validate Model...")
        with named_timer("Cross Validate Model", writer, tag="ValidateModel"):
            accuracies = cross_validate_model(X_test, y_test, k=5, num_epochs=20, lr=0.001)

        safe_log("[12] Cross Validate Model with F1 Score...")
        with named_timer("Cross Validate Model with F1 Score", writer, tag="ValidateModelF1"):
            fold_results = cross_validate_model_advanced(X_test, y_test, k=5, num_epochs=20, early_stopping=True)
    """


    def create_ui():
        global use_all_files_var, is_dark_mode, x, style, label_selected, listbox, clock_label_start, start_time_label, start_button, end_time_label, time_taken_label, progress, progress_label, animate_progress_label
        # ---------- Set Window Size ----------
        use_all_files_var = tk.BooleanVar(value=config.USE_UPLOADED_TEST_FILES)  # Now safe to create
        root.title("HDPFTL Architecture")
        # Main menu bar
        menu_bar = tk.Menu(root)
        # Submenu: Reports
        reports_files_menu = tk.Menu(menu_bar, tearoff=0)
        reports_files_menu.add_command(label="📊 Client Labels Distribution", command=handle_client_label_distribution)
        reports_files_menu.add_command(label="📉 Confusion Matrix", command=handle_confusion_matrix)
        reports_files_menu.add_command(label="📈 Pre Epoch Losses", command=handle_pre_epoch_losses)
        reports_files_menu.add_command(label="🛠️ Fine Tuning Epoch Losses", command=handle_fine_tune_losses)
        reports_files_menu.add_command(label="🔁 Personalized vs Global--Bar Chart",
                                       command=handle_personalized_vs_global_bar)
        reports_files_menu.add_command(label="🔄 Personalized vs Global--Dotted",
                                       command=handle_plot_personalised_vs_global)
        reports_files_menu.add_command(label="🔬 Cross Validation Model", command=handle_cross_validation)
        menu_bar.add_cascade(label="📊 Reports", menu=reports_files_menu)
        # Submenu: Settings
        is_dark_mode = False
        view_menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Clear All", command=clear_trainings)
        view_menu.add_command(label="Logs", command=open_log_window)
        view_menu.add_checkbutton(label="Dark Mode", command=toggle_theme)
        menu_bar.add_cascade(label="🧰 Utility", menu=view_menu)
        # Submenu: Settings
        settings_files_menu = tk.Menu(menu_bar, tearoff=0)
        settings_files_menu.add_command(label="🛠️ Settings", command=open_settings_window)
        menu_bar.add_cascade(label="⚙️ Settings", menu=settings_files_menu)
        # Submenu: Settings
        exit_menu = tk.Menu(menu_bar, tearoff=0)
        exit_menu.add_command(label="Exit", command=on_close)
        menu_bar.add_cascade(label="Exit", menu=exit_menu)
        # Display the menu bar
        root.config(menu=menu_bar)
        # 1. Set default font for all widgets in this root window
        default_font = font.nametofont("TkDefaultFont")
        default_font.configure(family="Helvetica", size=12)  # Smaller font
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
        # header_label = tk.Label(root, text="HDPFTL Architecture", font=("Arial", 18, "bold"))
        # header_label.pack(pady=(15, 5))
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
        for folder in util.get_output_folders(config.OUTPUT_DATASET_ALL_DATA):
            if folder == "selected_test":
                continue
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
        start_button = ttk.Button(control_frame, text="🚀 Start Training", command=start_training,
                                  style="Custom.TButton")
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

        # ------------------ Optional: Animated Progress ------------------ #
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
        if util.get_os() == "macOS":
            button_params = {
                "font": ("Arial", 13),
                "height": 2,
                "width": 25
            }
        else:
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
        result_frame = ttk.LabelFrame(main_frame, text="📝 Logs", style="Results.TLabelframe", padding=10)
        result_frame.pack(fill="both", expand=True, padx=10, pady=(5, 10))

        log_text = scrolledtext.ScrolledText(
            result_frame,
            wrap="word",
            height=10,
            font=("Courier", 16, "bold"),
            background="black",  # Black background
            foreground="white",  # Default text color
            insertbackground="white"  # White cursor
        )
        log_text.pack(fill="both", expand=True)
        log_text.config(state="disabled")
        # === Color tags ===
        log_text.tag_config("INFO", foreground="lightgreen")
        log_text.tag_config("WARNING", foreground="orange")
        log_text.tag_config("ERROR", foreground="red")
        log_text.tag_config("CRITICAL", foreground="darkred")
        log_text.tag_config("DEBUG", foreground="gray")


        # === Optional Controls ===
        log_text.auto_scroll = True
        return log_text


    def on_close():
        root.destroy()


    log_text = create_ui()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

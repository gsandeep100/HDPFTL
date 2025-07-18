# === Setup Logging ===
import logging
import os
import queue
import sys
import threading
import time

import numpy as np
import psutil

# Global variables
_log_queue = queue.Queue()
_thread = None
_stop_event = threading.Event()
_main_callback = None


def set_main_callback(callback):
    """Register the callback method from main.py to receive log records asynchronously."""
    global _main_callback
    _main_callback = callback


def _thread_worker():
    global _log_queue, _main_callback, _stop_event
    print(f"[_thread_worker] Queue ID: {id(_log_queue)}")  # Debug
    while not _stop_event.is_set():
        try:
            log_record = _log_queue.get()
            print("Got log_record from queue", log_record)  # <- Should appear
            if _main_callback:
                print("Calling main callback")
                _main_callback(log_record)
            else:
                print("Callback is None")
            _log_queue.task_done()
        except queue.Empty:
            continue
    print("Log thread stopped.")


def enqueue_log(log_record):
    global _log_queue
    print(f"[enqueue_log] Queue ID: {id(_log_queue)}")  # Debug
    _log_queue.put(log_record)


def start_worker_thread():
    """Start the background thread if not already running."""
    global _stop_event, _thread
    if _thread and _thread.is_alive():
        print("Thread already running.")
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_thread_worker, daemon=True)
    _thread.start()
    print("Thread started.")


def stop_worker_thread():
    global _stop_event, _thread
    """Stop the background thread gracefully."""
    _stop_event.set()
    if _thread:
        _thread.join()
    print("Thread stopped.")


def setup_logging(log_path, log_to_file=True):
    log_format = "%(asctime)s [%(levelname)s] %(message)s"

    handlers = []
    if log_to_file:
        handlers.append(logging.FileHandler(log_path + "hdpftl_run.log"))

    # Always also log to console
    handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,  # Set the minimum level you want to capture (e.g. INFO or DEBUG)
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )


def safe_log(message, extra="", level="info"):
    # Convert non-string inputs to string
    if not isinstance(message, str):
        message = str(message)
    if not isinstance(extra, str):
        extra = str(extra)

    full_msg = message + extra

    # Log and print based on specified level
    if level.lower() == "debug":
        logging.debug(full_msg)
    elif level.lower() == "info":
        logging.info(full_msg)
    elif level.lower() == "warning":
        logging.warning(full_msg)
    elif level.lower() == "error":
        logging.error(full_msg)
    elif level.lower() == "critical":
        logging.critical(full_msg)
    else:
        logging.info(full_msg)  # default

    print(full_msg)
    log_memory(message)

    # Define conversion to JSON-safe format
    def make_json_safe(obj):
        """Convert numpy types to built-in types."""
        if isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj  # fallback

    # Build structured log record with safe values
    log_record = {
        "message": make_json_safe(full_msg),
        "level": make_json_safe(level.lower()),
        "custom": "",  # handle this later if needed
        "timestamp": make_json_safe(time.time()),
    }

    enqueue_log(log_record)


def log_memory(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"💾 [MEM] {tag} — {mem_mb:.2f} MB")

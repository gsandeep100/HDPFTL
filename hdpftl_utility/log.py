# === Setup Logging ===
import json
import logging
import os
import sys
import time
from datetime import datetime
import numpy as np
import psutil


# Global variables for structured JSON logs
_structured_json_path = None
_current_log_dir = None
_run_counter = None

def setup_logging(log_path, log_to_file=True):
    """
    Set up logging to both console and a file.
    """
    # Ensure log directory exists
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, "hdpftl_run.log")

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    if log_to_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # Initial log entry
    logger.info(f"Logging initialized. Log file: {log_file if log_to_file else 'Console only'}")

def safe_log(message, extra="", level="info"):
    """
    Safe logging function that logs to console, file, memory,
    and writes structured logs to a timestamped JSON file with pretty-print.
    If writing fails, the log file is renamed with a .error extension.
    """

    global _structured_json_path
    global _current_log_dir
    global _run_counter

    # Initialize run counter if not already
    if "_run_counter" not in globals():
        _run_counter = 0

    # Convert non-string inputs to strings
    if not isinstance(message, str):
        message = str(message)
    if not isinstance(extra, str):
        extra = str(extra)

    full_msg = message + extra

    # Log based on specified level (file + console)
    level_upper = level.upper()
    if level_upper == "DEBUG":
        logging.debug(full_msg)
    elif level_upper == "INFO":
        logging.info(full_msg)
    elif level_upper == "WARNING":
        logging.warning(full_msg)
    elif level_upper == "ERROR":
        logging.error(full_msg)
    elif level_upper == "CRITICAL":
        logging.critical(full_msg)
    else:
        logging.info(full_msg)  # default

    # Optional: memory tracking
    if "log_memory" in globals():
        try:
            log_memory(message)
        except Exception as e:
            logging.debug(f"log_memory() failed: {e}")

    # Helper: convert numpy objects to JSON-safe format
    def make_json_safe(obj):
        if isinstance(obj, (np.int64, np.int32, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.floating)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    # Structured log record
    log_record = {
        "message": make_json_safe(full_msg),
        "level": make_json_safe(level.lower()),
        "custom": "",
        "timestamp": make_json_safe(time.time()),
        "human_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    # Write structured log to JSON if _current_log_dir is set
    if _current_log_dir is not None:
        os.makedirs(_current_log_dir, exist_ok=True)
        if _structured_json_path is None:
            # Increment run counter for separate file if rerun on same date
            _run_counter += 1
            run_timestamp = datetime.now().strftime("%Y-%m-%d")
            _structured_json_path = os.path.join(
                _current_log_dir, f"structured_logs_{run_timestamp}_run{_run_counter}.json"
            )

        # Append structured log as one pretty-printed line
        try:
            with open(_structured_json_path, "a", encoding="utf-8") as jf:
                json_line = json.dumps(
                    log_record, ensure_ascii=False, indent=None, separators=(',', ': ')
                )
                jf.write(json_line + "\n")
        except Exception as e:
            logging.error(f"Failed to write structured log to JSON: {e}")
            try:
                # Rename log file to indicate error
                error_path = _structured_json_path.rsplit('.', 1)[0] + ".error"
                os.rename(_structured_json_path, error_path)
                logging.error(f"Log file renamed to indicate error: {error_path}")
                _structured_json_path = error_path  # update global path
            except Exception as rename_e:
                logging.error(f"Failed to rename log file to .error: {rename_e}")

def log_memory(tag=""):
    """
    Logs current process memory usage to both console and file.
    """
    try:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 ** 2)
        mem_msg = f"💾 [MEM] {tag} — {mem_mb:.2f} MB"

        # Log to both file and console (via configured logger)
        logging.info(mem_msg)

        # Also print explicitly for immediate visibility
        #print(mem_msg)

    except Exception as e:
        logging.error(f"Memory logging failed: {e}")
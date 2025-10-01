# === Setup Logging ===
import logging
import os
import sys
import time

import numpy as np
import psutil


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
    if level.upper() == "debug":
        logging.debug(full_msg)
    elif level.upper() == "info":
        logging.info(full_msg)
    elif level.upper() == "warning":
        logging.warning(full_msg)
    elif level.upper() == "error":
        logging.error(full_msg)
    elif level.upper() == "critical":
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


def log_memory(tag=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"💾 [MEM] {tag} — {mem_mb:.2f} MB")

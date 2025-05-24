# === Setup Logging ===
import logging


def setup_logging(log_to_file=True):
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        filename="hdpftl_run.log" if log_to_file else None,
        level=logging.INFO,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

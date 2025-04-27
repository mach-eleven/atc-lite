import argparse
from datetime import datetime
from pathlib import Path

import logging 
logger = logging.getLogger("train.type_checkers")

def checkpoint_type(value):
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Checkpoint path does not exist: {value}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Checkpoint path is not a file: {value}")
    return path
    
def outdir_type(value):
    outdir_path = Path(value)
    if outdir_path.exists():
        # move current contents to a directory called "outdir_path_<timestamp>"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = outdir_path.parent / f"backup_{outdir_path.name}_{timestamp}"
        outdir_path.rename(backup_dir)
        logger.info(f"Existing directory moved to: '{backup_dir}'")

    # create a new directory
    outdir_path.mkdir(parents=True, exist_ok=False)
    return outdir_path

def log_list_type(value):
    # Split the input string by commas and strip whitespace
    log_types = [x.strip() for x in value.split(',')]
    # Check if all log types are valid
    valid_log_types = ['csv', 'tensorboard', 'stdout', 'file']
    for log_type in log_types:
        if log_type not in valid_log_types:
            raise argparse.ArgumentTypeError(f"Invalid log type: {log_type}. Valid options are: {valid_log_types}")
    return log_types

def gt_0(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Value must be greater than 0: {value}")
    return ivalue

def gt_0_float(value):
    fvalue = float(value)
    if fvalue <= 0:
        raise argparse.ArgumentTypeError(f"Value must be greater than 0: {value}")
    return fvalue

def parse_log_config(log_config):
    """
    Parse the log configuration string into a list of log types.
    log-where is a comma-separated string of log types (e.g., 'csv,tensorboard,stdout,file').
    This function converts it into 3 separate boolean flags.
    Args:
        log_config (str or list): The log configuration string or list.
    Returns:
        tuple: A tuple of boolean flags indicating whether to log to each type.
    """
    if isinstance(log_config, str):
        log_config = log_config.split(',')
    log_csv = 'csv' in log_config
    log_tensorboard = 'tensorboard' in log_config
    log_file = 'file' in log_config
    return log_csv, log_tensorboard, log_file
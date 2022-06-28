import json
import os
from .ErrorLogEntry import ErrorLogEntry

__all__ = ['log_last_error', 'finalize_error_log']

err_log_file_path = 'error_log.json'

def log_last_error(critical=False, user_message=''):
    all_errors = []
    if os.path.exists(err_log_file_path):
        with open(err_log_file_path, 'r') as infile:
            all_errors = json.load(infile)
    all_errors.append(ErrorLogEntry(critical=critical, user_message=user_message).to_serializable())
    with open(err_log_file_path, 'w') as outfile:
        json.dump(all_errors, outfile, indent=4)


def finalize_error_log():
    if not os.path.exists(err_log_file_path):
        all_errors = []
        with open(err_log_file_path, 'w') as outfile:
            json.dump(all_errors, outfile, indent=4)

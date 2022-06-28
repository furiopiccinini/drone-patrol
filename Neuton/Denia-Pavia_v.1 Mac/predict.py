import sys
import os
import common.import_paths
import run
import warnings
from neuton.local_prediction import *
from common.logging import *

warnings.filterwarnings("ignore")

USAGE = f"Usage: python {sys.argv[0]} (--help | <path-to-test-csv-file>)\n" \
        f"Please see the readme file for requirements"


def main():
    args = sys.argv[1:]
    if not args:
        raise SystemExit(USAGE)

    if args[0] == "--help":
        print(USAGE)
        return
    else:
        file_path = args[0]

    # create output folder
    if not os.path.exists('output'):
        os.mkdir('output')

    # Launching Preprocessing and Feature Engineering
    update_config_test_path(file_path)
    run.freeze_support()
    run.do_everything()
    predict(file_path)
    finalize_error_log()
    return 0


if __name__ == '__main__':
    sys.exit(main())

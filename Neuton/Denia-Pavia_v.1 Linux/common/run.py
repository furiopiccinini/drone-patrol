from multiprocessing import freeze_support
import import_paths
import os
import gc
import numpy as np
import random as rn


def do_everything():
    randomseed = 42
    os.environ['PYTHONHASHSEED'] = str(randomseed)
    np.random.seed(randomseed)
    rn.seed(randomseed)
    not_windows = os.name != 'nt'
    if not_windows:
        import resource

    from workflow import Workflow
    workflow = Workflow()
    workflow.execute()
    del workflow
    gc.collect()

    if not_windows:
        print(
            f'Memory load after preprocessing: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024} mb')


if __name__ == "__main__":
    freeze_support()
    do_everything()

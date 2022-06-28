import json
import os
import timeit
from .logging import log_last_error

__all__ = ['enclose_in_lines', 'stopwatch', 'log_and_skip_on_error']


def log_and_skip_on_error(func):
    def log_and_skip_on_error_wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except:
            log_last_error()

    return log_and_skip_on_error_wrapper


def enclose_in_lines(func):
    def enclose_in_lines_wrapper(*args, **kwargs):
        print('=' * 60)
        func(*args, **kwargs)
        print('=' * 60)

    return enclose_in_lines_wrapper


def stopwatch(name):
    def decorator(func):
        def stopwatch_wrapper(*args, **kwargs):
            start = timeit.default_timer()
            func(*args, **kwargs)
            stop = timeit.default_timer()
            print('\n   - {} finished, time: {:.2f} minutes'.format(name, (stop - start) / 60))

        return stopwatch_wrapper

    return decorator

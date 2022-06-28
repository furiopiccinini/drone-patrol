from datetime import datetime
import json
import os
import traceback

class ErrorLogEntry:
    _timestamp = None
    _critical = None
    _stacktrace = None
    _error = None
    _user_message = None

    def __init__(self, critical=False, user_message=''):
        self._timestamp = datetime.now()
        self._critical = critical
        self._user_message = user_message
        trace_list = map(lambda x: x.splitlines(), traceback.format_stack()[:-3])
        self._stacktrace = [item for sublist in trace_list for item in sublist]
        self._error = traceback.format_exc().splitlines()[1:]
        for line in self._error:
            print(line)


    def to_serializable(self):
        return {
            'timestamp': self._timestamp.isoformat(),
            'critical': self._critical,
            'stacktrace': self._stacktrace,
            'error': self._error,
            'user_message': self._user_message
        }


import pickle
import os
from steps.processing_step import ProcessingStep
from ErrorFixer import ErrorFixer
from data.processing_data import ProcessingData


class ErrorFixerTestStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        error_fixer_file_path = 'dict/error_fixer.p'
        if os.path.isfile(error_fixer_file_path):
            with open(error_fixer_file_path, 'rb') as pickle_in:
                error_fixer = pickle.load(pickle_in)
                info.data = error_fixer.transform(info.data)

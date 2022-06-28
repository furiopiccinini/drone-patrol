import pickle
from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData


class DropColumnsTestStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        with open('dict/droped_columns.p', 'rb') as pickle_in:
            dropped_columns = pickle.load(pickle_in)
        for col in dropped_columns:
            if col in info.data.columns:
                info.data.drop(col, axis=1, inplace=True)
        del dropped_columns


import os
from steps.processing_step import ProcessingStep
from methods_temp import csv_to_bin
from data.processing_data import ProcessingData


class CsvToBinTestStep(ProcessingStep):
    def __init__(self, csv_path: str):
        self.__csv_path = csv_path

    def execute(self, info: ProcessingData):
        try:
            csv_to_bin(inputs=info.data, data_path=self.__csv_path, taskTypeStr='', firstTargetIdx=-1, outputsCount=0,
                       normalization=info.config.normalization)
        except (ModuleNotFoundError, ImportError):
            local = not os.path.exists('data_preprocessing')
            if not local:
                print('Can not convert csv to bin')

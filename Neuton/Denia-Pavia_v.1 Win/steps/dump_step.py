import gc
from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_text_preprocessing import dump


class DumpStep(ProcessingStep):
    def __init__(self, csv_path: str):
        self.__csv_path = csv_path

    def execute(self, info: ProcessingData):
        dump(info.data, self.__csv_path)

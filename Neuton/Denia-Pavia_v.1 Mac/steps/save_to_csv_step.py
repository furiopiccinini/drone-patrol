from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_import_csv import get_delimiter


class SaveToCsvStep(ProcessingStep):
    def __init__(self, path: str, save_index: bool, get_delimiter: bool = False):
        self.__path = path
        self.__save_index = save_index
        self.__get_delimiter = get_delimiter

    def execute(self, info: ProcessingData):
        if self.__get_delimiter:
            sep = get_delimiter(info.config.path)
            info.data.to_csv(self.__path, index=self.__save_index, sep=sep)
        else:
            info.data.to_csv(self.__path, index=self.__save_index)

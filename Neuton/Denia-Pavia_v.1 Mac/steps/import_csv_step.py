from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_import_csv import import_csv


class ImportCSVStep(ProcessingStep):
    def __init__(self, dump: bool, target: bool = False, metric: bool = False, datetime_col: bool = False,
                 path: str = None, data_type: bool = False):
        self.__dump = dump
        self.__target = target
        self.__metric = metric
        self.__datetime_col = datetime_col
        self.__path = path
        self.__data_type = data_type

    def execute(self, info: ProcessingData):
        self.__target = info.config.target if self.__target else None
        self.__metric = info.config.metric if self.__metric else None
        self.__datetime_col = info.config.datetime_col if self.__datetime_col else None
        if self.__path is None:
            self.__path = info.config.path
        self.__data_type = info.config.data_type if self.__data_type else None

        result = import_csv(self.__path, self.__metric, self.__target, self.__datetime_col, self.__dump, self.__data_type)
        if self.__target:
            if self.__datetime_col:
                info.data, info.config.target, info.config.datetime_col = result
            else:
                info.data, info.config.target = result
        else:
            if self.__datetime_col:
                info.data, info.config.datetime_col = result
            else:
                info.data = result


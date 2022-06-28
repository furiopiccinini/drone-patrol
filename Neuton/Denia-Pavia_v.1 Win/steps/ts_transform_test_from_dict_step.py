from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_ts_transform_test_from_dict import ts_transform_test_from_dict


class TsTransformTestFromDictStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.data = ts_transform_test_from_dict(info.data)

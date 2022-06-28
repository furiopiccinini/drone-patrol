from steps.processing_step import ProcessingStep
from MOD_ts_scale_test import ts_scale_test
from data.processing_data import ProcessingData


class TsScaleTestStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.data = ts_scale_test(info.data)

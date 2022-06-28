from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_ts_test_stack import ts_test_stacking


class TsStackingTestStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.data = ts_test_stacking(info.data, info.config.metric)

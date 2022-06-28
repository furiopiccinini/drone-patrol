from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_ts_test_mean_target import ts_test_mean_target


class TsMeanTargetTestStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.data = ts_test_mean_target(info.data)

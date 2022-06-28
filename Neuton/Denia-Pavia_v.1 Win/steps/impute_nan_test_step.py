from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_impute_nan_downloadable import impute_nan_test


class ImputeNanTestStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.data = impute_nan_test(info.data)

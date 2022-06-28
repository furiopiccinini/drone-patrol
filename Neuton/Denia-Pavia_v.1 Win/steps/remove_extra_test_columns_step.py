from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_remove_extra_test_columns import remove_extra_test_columns


class RemoveExtraTestColumnsStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.data = remove_extra_test_columns(info.data)

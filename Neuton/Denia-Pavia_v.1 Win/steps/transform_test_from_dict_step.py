from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from MOD_transform_test_from_dict import transform_test_from_dict


class TransformTestFromDictStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.data = transform_test_from_dict(info.data)

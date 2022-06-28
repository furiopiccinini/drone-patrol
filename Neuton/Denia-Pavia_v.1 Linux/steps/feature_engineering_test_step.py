from steps.processing_step import ProcessingStep
from MOD_add_feats_test import add_feats_test
from data.processing_data import ProcessingData


class FeatureEngineeringTestStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.data = add_feats_test(info.data)

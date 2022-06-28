from steps.processing_step import ProcessingStep
from MOD_text_preprocessing import perform_vectorization
from data.processing_data import ProcessingData


class TextVectorizationStep(ProcessingStep):
    def __init__(self, use_existing: bool = False):
        self.__use_existing = use_existing

    def execute(self, info: ProcessingData):
        info.data = perform_vectorization(info.data, self.__use_existing)


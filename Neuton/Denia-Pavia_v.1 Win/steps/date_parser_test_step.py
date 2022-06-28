import pickle
import os
from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData
from DateParser import DateParser


class DateParserTestStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        date_parser_path = 'dict/date_parser.p'
        if os.path.exists(date_parser_path):
            with open('dict/date_parser.p', 'rb') as f:
                date_parser = pickle.load(f)
                info.data = date_parser.transform(info.data)

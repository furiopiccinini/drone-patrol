from typing import List
from steps.processing_step import ProcessingStep
from data.processing_data import ProcessingData


class Scenario(object):
    def __init__(self):
        raise NotImplementedError()

    def __init__(self, steps: List[ProcessingStep], data: ProcessingData):
        self.__steps = steps
        self.__data = data

    def execute(self):
        for step in self.__steps:
            step.execute(self.__data)

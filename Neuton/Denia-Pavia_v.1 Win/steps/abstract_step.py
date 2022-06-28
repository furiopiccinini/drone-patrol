import sys
from abc import *

sys.path.append('..')
from data.abstract_data import AbstractData


class AbstractStep(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def execute(self, info: AbstractData):
        pass

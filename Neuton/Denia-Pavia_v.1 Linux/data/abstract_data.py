from .config import Config
from pandas import DataFrame
from abc import *


class AbstractData(metaclass=ABCMeta):
    data: DataFrame
    config = Config()

    @abstractmethod
    def __init__(self):
        pass

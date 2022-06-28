import os
import sys
from typing import List
import inspect
sys.path.append('../..')
from data.processing_data import ProcessingData
from steps.processing_step import ProcessingStep


class BaseStepOrder(object):
    __mode: str
    __preprocessing: bool
    __feature_engineering: bool
    __time_series: bool
    __tiny: bool
    __mech_vibration: bool
    __training_was_restarted: bool

    def get_steps(self) -> List[ProcessingStep]:
        raise NotImplementedError()

    def get_file_name(self):
        return os.path.basename(inspect.getfile(self.__class__))

    def select(self, info: ProcessingData):
        self.__set_attributes()
        mode = self.__mode == info.config.mode
        pp = self.__preprocessing == info.config.preprocessing
        fe = self.__feature_engineering == info.config.feature_engineering
        ts = self.__time_series == (info.config.datetime_col is not None)
        tiny = self.__tiny == info.config.tiny
        mv = self.__mech_vibration == info.config.mech_vibration
        re = self.__training_was_restarted == info.config.training_was_restarted

        # print(self.__training_was_restarted, info.config.training_was_restarted)
        # print(mode, pp, fe, ts, tiny, re)
        return mode and pp and fe and ts and tiny and mv and re

    def __set_attributes(self):
        self.__mode = 'Abstract'
        name = self.__class__.__name__
        for mode in ('Train', 'Test', 'Valid'):
            if mode in name:
                self.__mode = mode.lower()

        self.__preprocessing = 'Pp' in name
        self.__feature_engineering = 'Fe' in name
        self.__time_series = 'Ts' in name
        self.__tiny = 'Tiny' in name
        self.__mech_vibration = 'Mv' in name
        self.__training_was_restarted = 'Re' in name

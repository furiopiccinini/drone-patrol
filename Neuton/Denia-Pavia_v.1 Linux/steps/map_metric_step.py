from steps.processing_step import ProcessingStep
from temp_metric_map import metric_map_func
from data.processing_data import ProcessingData


class MapMetricStep(ProcessingStep):
    def __init__(self):
        pass

    @staticmethod
    def execute(info: ProcessingData):
        info.config.metric = metric_map_func(info.config.metric)

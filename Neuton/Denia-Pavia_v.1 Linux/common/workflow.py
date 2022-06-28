from common.logging import *
from data.processing_data import ProcessingData
from common.decorators import *
from composer import Composer
from scenario import Scenario
from common.tracing_decorator import tracing_decorator


class Workflow(object):
    @enclose_in_lines
    @stopwatch('Preprocessing and feature engineering pipeline')
    def execute(self):
        info = ProcessingData()
        composer = Composer()
        steps = composer.get_steps(info)
        composer.save_step_order_file_name()
        scenario = Scenario(steps, info)
        scenario.execute()
        # self.foobar(info)
        finalize_error_log()

    @tracing_decorator(__file__)
    def foobar(self, info):
        pass

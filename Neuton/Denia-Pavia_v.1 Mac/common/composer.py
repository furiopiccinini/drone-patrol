from typing import List
import inspect
import json
import os
from data.processing_data import ProcessingData
from steps.processing_step import ProcessingStep
from steps.orders.base_step_order import BaseStepOrder


class Composer(object):
    def __init__(self):
        self.step_order_file_name = None
        self.save_file_name = 'step_order_file_name.json'

    @staticmethod
    def __get_step_orders():
        step_order_classes = list()
        orders_path = 'steps/orders/'
        for file in os.listdir(orders_path):
            child_namespace = os.path.splitext(file)[0]
            module = __import__(orders_path.replace('/', '.') + child_namespace)
            module = module.orders.__dict__[child_namespace]
            classes = [x[1] for x in inspect.getmembers(module, inspect.isclass)]
            step_order_classes.extend([x for x in classes if issubclass(x, BaseStepOrder)])

        return set(step_order_classes)

    def get_steps(self, info: ProcessingData) -> List[ProcessingStep]:
        step_order_classes = self.__get_step_orders()
        for step_order_class in step_order_classes:
            step_order = step_order_class()
            if step_order.select(info):
                self.step_order_file_name = step_order.get_file_name()
                return step_order.get_steps()

    def save_step_order_file_name(self):
        with open(self.save_file_name, 'w') as f:
            json.dump(self.step_order_file_name, f)

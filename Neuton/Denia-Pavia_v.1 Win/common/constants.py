import os
import json
import os


class Constants:
    __instance = None

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Constants, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.load()

    def load(self, path='data/constants.json'):
        if os.path.exists(path):
            with open(path, 'r') as file:
                temp = json.load(file)
                for key in temp:
                    self.__dict__[key] = temp[key]
        else:
            print(f'file "{path}" not found')

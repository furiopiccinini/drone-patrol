import json
from pathlib import Path
from typing import Optional


class Config:
    __instance = None
    path: Optional[Path]
    target: str
    metric: str
    preprocessing: bool
    feature_engineering: bool
    datetime_col: Optional[bool]
    mode: str
    normalization: str
    training_was_restarted: bool
    tiny: bool
    mech_vibration: bool
    data_type: str

    def __new__(cls, path="data/config.json"):
        if not hasattr(cls, "instance"):
            cls.instance = super(Config, cls).__new__(cls)
            cls.instance.load(path)
        return cls.instance

    # def __init__(self):
    #     self.load()

    def load(self, path="data/config.json"):
        with open(path, "r") as file:
            temp = json.load(file)
            self.absorb(temp)

    def absorb(self, data: dict):
        self.absorb_mode(data)
        self.absorb_path(data)
        self.absorb_target_column(data)
        self.absorb_target_metric(data)
        self.absorb_data_preparation(data)
        self.absorb_feature_engineering(data)
        self.absorb_datetime_column(data)
        self.absorb_training_was_restarted(data)
        self.absorb_normalization_type(data)
        self.absorb_tinyml(data)
        self.absorb_df_type(data)

    def absorb_mode(self, data: dict):
        ok = ("train", "valid", "test")
        if "mode" in data:
            data["mode"] = data["mode"].lower()
            if data["mode"] in ok:
                self.mode = data["mode"]
            else:
                raise Exception(f'mode is not in {ok}')
        else:
            raise Exception(f'mode is not in data/config.json')

    def absorb_path(self, data: dict):
        found = False
        if "dataSources" in data:
                sources = data["dataSources"]
                if type(sources) is list:
                    for source in sources:
                        if "datasetPurpose" in source:
                            if source["datasetPurpose"].lower() == 'predict':
                                source["datasetPurpose"] = 'test'

                            if str(source["datasetPurpose"]).lower() == self.mode:
                                self.path = source["dataSourcePath"]
                                found = True
                                break

                    if not found:
                        raise Exception(f'no datasource of datasetPurpose equal to mode={self.mode}')
                else:
                    raise Exception(f'dataSources is not a list')
        else:
            raise Exception(f'dataSources is not in data/config.json')

    def absorb_target_column(self, data: dict):
        if "targetColumn" in data:
            self.target = data["targetColumn"]
        else:
            raise Exception(f'targetColumn is not in data/config.json')

    def absorb_target_metric(self, data: dict):
        if "targetMetric" in data:
            self.metric = data["targetMetric"]
        else:
            raise Exception(f'targetMetric is not in data/config.json')

    def absorb_data_preparation(self, data: dict):
        if "dataPreparation" in data:
            if str(data["dataPreparation"]).lower() == "true":
                self.preprocessing = True
            elif str(data["dataPreparation"]).lower() == "false":
                self.preprocessing = False
            else:
                raise Exception(f'Cannot convert {data["dataPreparation"]} to boolean')
        else:
            raise Exception(f'dataPreparation is not in data/config.json')

    def absorb_feature_engineering(self, data: dict):
        if "featureEngineering" in data:
            if str(data["featureEngineering"]).lower() == "true":
                self.feature_engineering = True
            elif str(data["featureEngineering"]).lower() == "false":
                self.feature_engineering = False
            else:
                raise Exception(f'Cannot convert {data["featureEngineering"]} to boolean')
        else:
            raise Exception(f'featureEngineering is not in data/config.json')

    def absorb_datetime_column(self, data: dict):
        self.datetime_col = None
        if "timeSeriesSettings" in data:
            if data["timeSeriesSettings"] is not None:
                if "columnsTimeSeriesSettings" in data["timeSeriesSettings"]:
                    columns = data["timeSeriesSettings"]["columnsTimeSeriesSettings"]
                    if type(columns) is list:
                        found = False
                        for column in columns:
                            if "isTimeSeriesColumn" in column:
                                if str(column["isTimeSeriesColumn"]).lower() == "true":
                                    if "columnName" in column:
                                        self.datetime_col = column["columnName"]
                                        found = True
                                        break
                        if not found:
                            raise Exception("no column had isTimeSeriesColumn=true in data/config.json")
                    else:
                        print("columnsTimeSeriesSettings in timeSeriesSettings is not a list in data/config.json")
                else:
                    print("columnsTimeSeriesSettings is not in timeSeriesSettings in data/config.json")
            else:
                print("timeSeriesSettings is None in data/config.json")
        else:
            print("timeSeriesSettings is not in data/config.json")

    def absorb_training_was_restarted(self, data: dict):
        if "trainingWasRestarted" in data:
            if str(data["trainingWasRestarted"]).lower() == "true":
                self.training_was_restarted = True
            elif str(data["trainingWasRestarted"]).lower() == "false":
                self.training_was_restarted = False
            else:
                raise Exception(f'Cannot convert {data["trainingWasRestarted"]} to boolean')
        else:
            raise Exception(f'trainingWasRestarted is not in data/config.json')

    def absorb_normalization_type(self, data: dict):
        ok = ("NONE", "SINGLE", "UNIQUE")
        if "normalizationType" in data:
            if data["normalizationType"] in ok:
                self.normalization = data["normalizationType"]
            else:
                raise Exception(f'normalizationType is not in {ok}')
        else:
            raise Exception(f'normalizationType is not in data/config.json')

    def absorb_tinyml(self, data: dict):
        if "tinyMlSettings" in data:
            if data["tinyMlSettings"] is not None:
                if "isTinyMlOn" in data["tinyMlSettings"]:
                    if str(data["tinyMlSettings"]["isTinyMlOn"]).lower() == "true":
                        self.tiny = True
                        print('Pipeline for tinyml:')
                    elif str(data["tinyMlSettings"]["isTinyMlOn"]).lower() == "false":
                        self.tiny = False
                    else:
                        raise Exception(f'Cannot convert {data["tinyMlSettings"]["isTinyMlOn"]} to boolean')
                else:
                    raise Exception(f'isTinyMlOn is not in tinyMlSettings in data/config.json')
                if "performMechVibrationPP" in data["tinyMlSettings"]:
                    if str(data["tinyMlSettings"]["performMechVibrationPP"]).lower() == "true":
                        self.mech_vibration = True
                        print('Mechanical vibration pipeline:')
                    elif str(data["tinyMlSettings"]["performMechVibrationPP"]).lower() == "false":
                        self.mech_vibration = False
                    else:
                        raise Exception(f'Cannot convert {data["tinyMlSettings"]["performMechVibrationPP"]} to boolean')
                else:
                    raise Exception(f'performMechVibrationPP is not in tinyMlSettings in data/config.json')
            else:
                raise Exception(f'tinyMlSettings is None in data/config.json')
        else:
            raise Exception(f'tinyMlSettings is not in data/config.json')

    def absorb_df_type(self, data: dict):
        self.data_type = None
        if "interpretDataTypeAs" in data:
            if data["interpretDataTypeAs"] is not None:
                self.data_type = str(data["interpretDataTypeAs"]).lower()
                print(f'DataType is {self.data_type}')
        else:
            raise Exception(f'interpretDataTypeAs is not in data/config.json')

import os
import json
from pandas import DataFrame

tracing_config = None


def get_tracing_config():
    default_config = {
        'save': False,
        'tracing_dir_post': 'dummydir/post',
        'tracing_dir_pre': 'dummydir/pre'
    }
    tracing_config_path = 'common/tracing_decorator.json'
    if os.path.exists(tracing_config_path):
        with open(tracing_config_path) as f:
            config_json = json.load(f)
            if 'save' in config_json:
                if config_json['save'].lower() == 'true':
                    default_config['save'] = True
                    default_config['tracing_dir_post'] = f'tests/tracing/{config_json["version"]}/{config_json["name"]}/post/'
                    default_config['tracing_dir_pre'] = f'tests/tracing/{config_json["version"]}/{config_json["name"]}/pre/'

    return default_config


def init_tracing_config():
    global tracing_config
    tracing_config = get_tracing_config()


init_tracing_config()


def tracing_decorator(name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print('in tracing wrapper')
            global tracing_config
            if tracing_config is None:
                tracing_config = get_tracing_config()

            if tracing_config['save']:
                path_pre = f'{tracing_config["tracing_dir_pre"]}{os.path.basename(name)}.{func.__name__}.csv'
                path_post = f'{tracing_config["tracing_dir_post"]}{os.path.basename(name)}.{func.__name__}.csv'
                if not os.path.exists(tracing_config['tracing_dir_post']):
                    os.makedirs(tracing_config['tracing_dir_post'])

                if not os.path.exists(tracing_config['tracing_dir_pre']):
                    os.makedirs(tracing_config['tracing_dir_pre'])

            for data in args:
                if type(data) is DataFrame:
                    if tracing_config['save']:
                        data.to_csv(path_pre, index=None)

                    result = func(*args, **kwargs)
                    df = result
                    if isinstance(result, list):
                        for df in result:
                            if type(data) is DataFrame:
                                break

                    if tracing_config['save'] and df is not None:
                        df.to_csv(path_post, index=None)

                    return result

                if hasattr(data, 'data'):
                    if tracing_config['save']:
                        data.data.to_csv(path_pre, index=None)

                    func(*args, **kwargs)
                    if tracing_config['save']:
                        data.data.to_csv(path_post, index=None)

                    break

        return wrapper

    return decorator

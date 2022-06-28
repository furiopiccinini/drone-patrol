import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.append('feature_engineering')
sys.path.append('feature_engineering/mods')
sys.path.append('mods')
#from methods_temp import select_method
import numpy as np
import pandas as pd
from MOD_text_preprocessing import dump


def test_feats(data):
    from MOD_add_feats_test import add_feats_test
    
    test = add_feats_test(data)
    if test.shape == data.shape:
        pass
        print('Data preprocessing baseline metric was not improved\nUse test_processed.csv for model prediction')
    else:
        dump(test, 'output/test_features.csv')
        print('\nFeature engineering yielded better results\ntest_features.csv is saved to output folder')
    return test

if __name__ == "__main__":
    args = sys.argv[1:]
    test_feats(args[0])


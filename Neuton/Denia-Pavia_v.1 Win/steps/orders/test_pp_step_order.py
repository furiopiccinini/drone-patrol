from typing import List
import sys
import os
sys.path.append('../..')
from data.processing_data import ProcessingData
from steps.processing_step import ProcessingStep
from steps.orders.base_step_order import BaseStepOrder
from steps.import_csv_step import ImportCSVStep
from steps.remove_extra_test_columns_step import RemoveExtraTestColumnsStep
from steps.text_vectorization_step import TextVectorizationStep
from steps.date_parser_test_step import DateParserTestStep
from steps.drop_columns_test_step import DropColumnsTestStep
from steps.error_fixer_test_step import ErrorFixerTestStep
from steps.impute_nan_test_step import ImputeNanTestStep
from steps.transform_test_from_dict_step import TransformTestFromDictStep
from steps.save_to_csv_step import SaveToCsvStep
from steps.dump_step import DumpStep
from steps.csv_to_bin_test_step import CsvToBinTestStep


class TestPpStepOrder(BaseStepOrder):
    @staticmethod
    def get_steps() -> List[ProcessingStep]:
        local = not os.path.exists('data_preprocessing')
        final_dump_path = 'output/' if local else 'data_preprocessing/output/'
        return [ImportCSVStep(dump=False),
                RemoveExtraTestColumnsStep(),
                TextVectorizationStep(use_existing=True),
                DateParserTestStep(),
                DropColumnsTestStep(),
                ErrorFixerTestStep(),
                ImputeNanTestStep(),
                TransformTestFromDictStep(),
                SaveToCsvStep(path=final_dump_path + 'pp_only_test.csv', save_index=False),
                DumpStep(csv_path=final_dump_path + 'processed_test.csv'),
                CsvToBinTestStep(final_dump_path + 'processed_test.csv')
                ]
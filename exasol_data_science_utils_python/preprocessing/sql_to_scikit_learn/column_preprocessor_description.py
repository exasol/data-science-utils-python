from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_factory import \
    ColumnPreprocessorFactory
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_selector import ColumnSelector


class ColumnPreprocessorDescription:
    def __init__(self,
                 column_selector: ColumnSelector,
                 column_preprocessor_factory: ColumnPreprocessorFactory):
        self.column_preprocessor_factory = column_preprocessor_factory
        self.column_selector = column_selector


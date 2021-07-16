import pandas as pd

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_one_hot_transformer import \
    SKLearnPrefittedOneHotTransformer
from exasol_data_science_utils_python.preprocessing.sql.encoding.sql_ordinal_encoder import SQLOrdinalEncoder
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor import ColumnPreprocessor, \
    SQLBasedColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_factory import \
    ColumnPreprocessorFactory
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class OneHotEncoderFactory(ColumnPreprocessorFactory):
    def create(self,
               sql_executor: SQLExecutor,
               source_column: Column,
               target_schema: SchemaName,
               experiment_name: ExperimentName) -> ColumnPreprocessor:
        parameter_tables = \
            SQLOrdinalEncoder().fit(sql_executor, source_column.name, target_schema, experiment_name)
        dictionary_parameter_tables = \
            [parameter_table for parameter_table in parameter_tables
             if parameter_table.purpose == SQLOrdinalEncoder.PURPOSE_DICTIONARY_TABLE]
        dictionary_parameter_table = dictionary_parameter_tables[0]
        result_set = sql_executor.execute(
            f"""SELECT "VALUE" FROM {dictionary_parameter_table.table.name.fully_qualified()}""")
        dictionary = pd.DataFrame(data=result_set.fetchall(), columns=result_set.column_names())
        dictionary = dictionary.set_index("VALUE", drop=False)
        transformer = SKLearnPrefittedOneHotTransformer(dictionary)
        column_preprocessor = SQLBasedColumnPreprocessor(source_column,
                                                         target_schema,
                                                         experiment_name,
                                                         transformer,
                                                         parameter_tables)
        return column_preprocessor

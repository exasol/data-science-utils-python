from typing import List, Tuple, Dict

from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_prefitted_column_transformer import \
    SKLearnPrefittedColumnTransformer
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_description import \
    ColumnPreprocessorDescription
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_factory import \
    ColumnPreprocessorFactory
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_set_preprocessor import \
    ColumnSetPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor_factory import \
    TablePreprocessorFactory
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.schema.column_type import ColumnType
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class ColumnDescriptionBasedTablePreprocessorFactory(TablePreprocessorFactory):
    def __init__(self,
                 input_column_preprocessor_descriptions: List[ColumnPreprocessorDescription],
                 target_column_preprocessor_descriptions: List[ColumnPreprocessorDescription]):
        self._target_column_preprocessor_descriptions = target_column_preprocessor_descriptions
        self._input_column_preprocessor_descriptions = input_column_preprocessor_descriptions

    def create_table_processor(self,
                               sql_executor: SQLExecutor,
                               source_table: TableName,
                               target_schema: SchemaName,
                               experiment_name: ExperimentName) -> TablePreprocessor:
        query = f"""
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM SYS.EXA_ALL_COLUMNS 
            WHERE COLUMN_SCHEMA='{source_table.schema_name.name}'
            AND COLUMN_TABLE='{source_table.name}'
            """
        result_set = sql_executor.execute(query)
        rows = result_set.fetchall()
        columns = [Column(ColumnNameBuilder.create(row[0], source_table), ColumnType(row[1])) for row in rows]
        input_column_preprocessor_factory_mapping = \
            self._create_column_preprocessor_factory_mapping(
                columns, self._input_column_preprocessor_descriptions)
        target_column_preprocessor_factory_mapping = \
            self._create_column_preprocessor_factory_mapping(
                columns, self._target_column_preprocessor_descriptions)
        self._check_if_input_and_target_columns_are_disjunct(
            input_column_preprocessor_factory_mapping, target_column_preprocessor_factory_mapping)
        input_column_set_preprocessors = \
            self._create_column_set_preprocessors(input_column_preprocessor_factory_mapping,
                                                  sql_executor, target_schema, source_table, experiment_name)
        target_column_set_preprocessors = \
            self._create_column_set_preprocessors(target_column_preprocessor_factory_mapping,
                                                  sql_executor, target_schema, source_table, experiment_name)
        table_preproccesor = \
            TablePreprocessor(input_column_set_preprocessors,
                              target_column_set_preprocessors,
                              source_table, target_schema, experiment_name)
        return table_preproccesor

    def _create_column_set_preprocessors(
            self,
            column_preprocessor_factory_mapping: Dict[str, Tuple[Column, ColumnPreprocessorFactory]],
            sql_executor: SQLExecutor,
            target_schema: SchemaName,
            source_table: TableName,
            experiment_name: ExperimentName) \
            -> ColumnSetPreprocessor:
        column_preprocessors = \
            self._create_column_preprocessors(column_preprocessor_factory_mapping,
                                              sql_executor, target_schema, experiment_name)
        column_transformer = self._create_column_transformers(column_preprocessors)
        column_set_preprocessor = ColumnSetPreprocessor(
            column_preprocessors=column_preprocessors,
            column_transformer=column_transformer,
            source_table=source_table,
            target_schema=target_schema
        )
        return column_set_preprocessor

    def _create_column_transformers(self, column_preprocessors: List[ColumnPreprocessor]):
        column_transformer = SKLearnPrefittedColumnTransformer(
            [(column_preprocessor.source_column.name.name, column_preprocessor.transformer)
             for column_preprocessor in column_preprocessors])
        return column_transformer

    def _create_column_preprocessors(
            self,
            column_preprocessor_factory_mapping: Dict[str, Tuple[Column, ColumnPreprocessorFactory]],
            sql_executor: SQLExecutor,
            target_schema: SchemaName,
            experiment_name: ExperimentName):
        column_preprocessors = \
            [column_preprocessor_factory.create(sql_executor, column, target_schema, experiment_name)
             for column, column_preprocessor_factory
             in column_preprocessor_factory_mapping.values()]
        return column_preprocessors

    def _check_if_input_and_target_columns_are_disjunct(self, input_column_preprocessor_factory_mapping,
                                                        target_column_preprocessor_factory_mapping):
        input_column_names = input_column_preprocessor_factory_mapping.keys()
        target_column_names = target_column_preprocessor_factory_mapping.keys()
        if not input_column_names.isdisjoint(target_column_names):
            raise ValueError(
                f"The selected input columns {list(input_column_names)} and "
                " target columns {list(target_column_names)} are not disjoint")

    def _create_column_preprocessor_factory_mapping(
            self,
            columns: List[Column],
            column_preprocessor_descriptions: List[ColumnPreprocessorDescription]) \
            -> Dict[str, Tuple[Column, ColumnPreprocessorFactory]]:
        input_column_preprocessor_factory_mapping = {}
        for column in columns:
            for column_preprocessor_description in column_preprocessor_descriptions:
                if column_preprocessor_description.column_selector.column_accepted(column):
                    input_column_preprocessor_factory_mapping[column.name.name] = \
                        (column, column_preprocessor_description.column_preprocessor_factory)
                    break
        return input_column_preprocessor_factory_mapping

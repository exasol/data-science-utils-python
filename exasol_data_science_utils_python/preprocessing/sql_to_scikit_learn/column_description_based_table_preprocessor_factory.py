from typing import List, Tuple, Dict

from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_type import ColumnType
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_description import \
    ColumnPreprocessorDescription
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_factory import \
    ColumnPreprocessorFactory
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor_factory import \
    TablePreprocessorFactory
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
                               target_schema: SchemaName) -> TablePreprocessor:
        query = f"""
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM SYS.EXA_ALL_COLUMNS 
            WHERE COLUMN_SCHEMA='{source_table.schema_name.name}'
            AND COLUMN_TABLE='{source_table.name}'
            """
        result_set = sql_executor.execute(query)
        rows = result_set.fetchall()
        columns = [Column(ColumnName(row[0], source_table), ColumnType(row[1])) for row in rows]
        input_column_preprocessor_factory_mapping = \
            self._create_column_preprocessor_factory_mapping(
                columns, self._input_column_preprocessor_descriptions)
        target_column_preprocessor_factory_mapping = \
            self._create_column_preprocessor_factory_mapping(
                columns, self._target_column_preprocessor_descriptions)
        self._check_if_input_and_target_columns_are_disjunct(
            input_column_preprocessor_factory_mapping, target_column_preprocessor_factory_mapping)
        input_column_preprocessors = \
            self._create_column_preprocessors(input_column_preprocessor_factory_mapping,
                                              sql_executor, target_schema)
        target_column_preprocessors = \
            self._create_column_preprocessors(target_column_preprocessor_factory_mapping,
                                              sql_executor, target_schema)
        table_preproccesor = \
            TablePreprocessor(input_column_preprocessors,
                              target_column_preprocessors,
                              source_table, target_schema)
        return table_preproccesor

    def _create_column_preprocessors(
            self,
            column_preprocessor_factory_mapping: Dict[str, Tuple[Column, ColumnPreprocessorFactory]],
            sql_executor: SQLExecutor,
            target_schema: SchemaName):
        column_preprocessors = \
            [column_preprocessor_factory.create(sql_executor, column, target_schema)
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

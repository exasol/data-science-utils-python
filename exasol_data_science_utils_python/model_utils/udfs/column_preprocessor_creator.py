from typing import List, Dict

import pandas as pd
from sklearn.compose import ColumnTransformer

from exasol_data_science_utils_python.model_utils.sklearn_contrast_matrix_transformer import \
    SKLearnContrastMatrixTransformer
from exasol_data_science_utils_python.model_utils.sklearn_min_max_scalar import SKLearnMinMaxScalar
from exasol_data_science_utils_python.model_utils.udfs.abstract_column_preprocessor_creator import \
    ColumnPreprocssorCreatorResult, AbstractColumnPreprocessorCreator
from exasol_data_science_utils_python.preprocessing.encoding.ordinal_encoder import OrdinalEncoder
from exasol_data_science_utils_python.preprocessing.normalization.min_max_scaler import MinMaxScaler
from exasol_data_science_utils_python.preprocessing.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.table_preprocessor import TablePreprocessor
from exasol_data_science_utils_python.preprocessing.column_preprocessor_definition import ColumnPreprocessorDefinition
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class ColumnPreprocessorCreator(AbstractColumnPreprocessorCreator):

    def generate_column_transformers(self,
                                     sql_executor: SQLExecutor,
                                     input_columns: List[ColumnName],
                                     target_columns: List[ColumnName],
                                     source_table: TableName,
                                     target_schema: SchemaName) -> ColumnPreprocssorCreatorResult:
        columns = input_columns + target_columns
        column_types = self._get_column_types(columns, source_table, sql_executor)
        fit_tables = self._fit_table_preprocessor(column_types, columns, source_table, sql_executor, target_schema)
        input_columns_transformer = self._create_column_transformer(column_types, fit_tables, input_columns,
                                                                    source_table, sql_executor)
        target_column_transformer = self._create_column_transformer(column_types, fit_tables, target_columns,
                                                                    source_table, sql_executor)
        return ColumnPreprocssorCreatorResult(input_columns_transformer,
                                              target_column_transformer)

    def _get_column_types(self,
                          columns: List[ColumnName],
                          source_table: TableName,
                          sql_executor: SQLExecutor):
        column_name_list = ",".join(f"'{column.name}'" for column in columns)
        query = f"""
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM SYS.EXA_ALL_COLUMNS 
            WHERE COLUMN_SCHEMA='{source_table.schema_name.name}'
            AND COLUMN_TABLE='{source_table.name}'
            AND COLUMN_NAME in ({column_name_list})
            """
        result_set = sql_executor.execute(query)
        column_types = dict(result_set.fetchall())
        return column_types

    def _fit_table_preprocessor(
            self,
            column_types: Dict[str, str],
            columns: List[ColumnName],
            source_table: TableName,
            sql_executor: SQLExecutor,
            target_schema: SchemaName):
        column_preprocessor_defintions = []
        for column in columns:
            column_type = column_types[column.name]
            if column_type == 'DECIMAL(18,0)':
                column_preprocessor_defintions.append(ColumnPreprocessorDefinition(column.name, OrdinalEncoder()))
            elif column_type.startswith('VARCHAR'):
                column_preprocessor_defintions.append(ColumnPreprocessorDefinition(column.name, OrdinalEncoder()))
            elif column_type == 'DOUBLE':
                column_preprocessor_defintions.append(ColumnPreprocessorDefinition(column.name, MinMaxScaler()))
            else:
                raise Exception(f"Type '{column_type}' of column {column.fully_qualified()} not supported")
        table_preprocessor = TablePreprocessor(target_schema, source_table, column_preprocessor_defintions)
        fit_tables = table_preprocessor.fit(sql_executor)
        return fit_tables

    def _create_column_transformer(
            self,
            column_types: Dict[str, str],
            fit_tables: List[TableName],
            columns: List[ColumnName],
            source_table: TableName,
            sql_executor: SQLExecutor):
        sklearn_transformer = []
        for column in columns:
            column_type = column_types[column.name]
            if column_type == 'DECIMAL(18,0)':
                transformer = self._create_onehot_transformer(column, fit_tables, sql_executor)
                sklearn_transformer.append((column.name, transformer, [column.name]))
            elif column_type.startswith('VARCHAR'):
                transformer = self._create_onehot_transformer(column, fit_tables, sql_executor)
                sklearn_transformer.append((column.name, transformer, [column.name]))
            elif column_type == 'DOUBLE':
                transformer = self._create_minmax_transformer(column, fit_tables, sql_executor)
                sklearn_transformer.append((column.name, transformer, [column.name]))
            else:
                raise Exception(f"Type '{column_type}' of column {column.fully_qualified()} not supported")
        column_transformer = ColumnTransformer(transformers=sklearn_transformer)
        select_list = self._get_select_list_for_columns(columns)
        dummy_data_for_fit = sql_executor.execute(
            f"""SELECT {select_list} FROM {source_table.fully_qualified()} LIMIT 2""").fetchall()
        dummy_df_for_fit = pd.DataFrame(data=dummy_data_for_fit, columns=[column.name for column in columns])
        return column_transformer.fit(dummy_df_for_fit)

    def _create_onehot_transformer(
            self,
            column: ColumnName,
            fit_tables: List[TableName],
            sql_executor: SQLExecutor):
        fit_table_for_column = [table for table in fit_tables if f"_{column.name}_" in table.name]
        if len(fit_table_for_column) != 1:
            raise Exception(f"Couldn't find fit_table for column '{column.name}'")
        result_set = sql_executor.execute(
            f"""SELECT "VALUE" FROM {fit_table_for_column[0].fully_qualified()}""")
        dictionary = pd.DataFrame(data=result_set.fetchall(), columns=result_set.column_names())
        dictionary = dictionary.set_index("VALUE", drop=False)
        contrast_matrix = pd.get_dummies(dictionary, columns=["VALUE"], drop_first=True)
        transformer = SKLearnContrastMatrixTransformer(contrast_matrix)
        return transformer

    def _create_minmax_transformer(
            self,
            column: ColumnName,
            fit_tables: List[TableName],
            sql_executor: SQLExecutor):
        fit_table_for_column = [table for table in fit_tables if f"_{column.name}_" in table.name]
        if len(fit_table_for_column) != 1:
            raise Exception(f"Couldn't find fit_table for column '{column.name}'")
        result_set = sql_executor.execute(
            f"""SELECT "MIN", "RANGE"  FROM {fit_table_for_column[0].fully_qualified()}""")
        rows = result_set.fetchall()
        min_value = rows[0][0]
        range_value = rows[0][1]
        transformer = SKLearnMinMaxScalar(min_value=min_value, range_value=range_value)
        return transformer

    def _get_select_list_for_columns(self, columns: List[ColumnName]):
        column_name_list = self._get_column_name_list(columns)
        return ",".join(f'"{column_name}"' for column_name in column_name_list)

    def _get_column_name_list(self, columns: List[ColumnName]):
        column_name_list = [column.name for column in columns]
        return column_name_list

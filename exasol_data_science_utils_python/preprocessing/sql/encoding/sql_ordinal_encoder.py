import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.sql.sql_column_preprocessor import SQLColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.preprocessing.sql.schema.column_type import ColumnType
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table import Table
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql.transform_select_clause_part import TransformSelectClausePart
from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor

ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX = "ORDINAL_ENCODER_DICTIONARY"


class SQLOrdinalEncoder(SQLColumnPreprocessor):
    """
    This ColumnPreprocessor implements a OrdinalEncoder.
        It was inspired by the
    `OrdinalEncoder of scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html>`_

    """

    def _get_dictionary_table_alias(self, target_schema: SchemaName, source_column: ColumnName):
        return self._get_table_alias(
            target_schema, source_column,
            ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX)

    def _get_dictionary_table_name(self, target_schema: SchemaName, source_column: ColumnName):
        return self._get_target_table(
            target_schema, source_column,
            ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX)

    def _get_id_column(self, table: TableName = None):
        min_column = ColumnName("ID", table)
        return min_column

    def _get_value_column(self, table: TableName = None):
        range_column = ColumnName("VALUE", table)
        return range_column

    def requires_global_transformation_for_training_data(self) -> bool:
        return False

    def fit(self, sqlexecutor: SQLExecutor, source_column: ColumnName, target_schema: SchemaName) -> List[
        ParameterTable]:
        """
        This method creates a dictionary table from the source column where every distinct value of the source column
        is mapped to an id between 0 and number of distinct values - 1

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param target_schema: Schema where the result tables of the fit-queries should be stored
        :return: List of fit-queries as strings
        """
        dictionary_table_name = self._get_dictionary_table_name(target_schema, source_column)
        id_column_name = self._get_id_column()
        value_column_name = self._get_value_column()
        query = textwrap.dedent(f"""
                CREATE OR REPLACE TABLE {dictionary_table_name.fully_qualified()} AS
                SELECT
                    CAST(rownum - 1 AS INTEGER) as {id_column_name.fully_qualified()},
                    {value_column_name.quoted_name()}
                FROM (
                    SELECT DISTINCT {source_column.fully_qualified()} as {value_column_name.quoted_name()}
                    FROM {source_column.table_name.fully_qualified()}
                    ORDER BY {source_column.fully_qualified()}
                );
                """)
        sqlexecutor.execute(query)
        parameter_table = ParameterTable(
            source_column=source_column,
            table=Table(
                name=dictionary_table_name,
                columns=[
                    Column(name=id_column_name, type=ColumnType("INTEGER")),
                    Column(name=value_column_name, type=ColumnType("ANY"))
                ]),
            purpose="DictionaryTable",
        )
        return [parameter_table]

    def create_transform_from_clause_part(self,
                                          sql_executor: SQLExecutor,
                                          source_column: ColumnName,
                                          input_table: TableName,
                                          target_schema: SchemaName) -> List[str]:
        """
        This method generates a LEFT OUTER JOIN with the dictionary table and the input table.
        The LEFT OUTER JOIN is important to keep all rows, also those which contain a NULL.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of from-clause parts which can be concatenated with "\n"
        """
        dictionary_table = self._get_dictionary_table_name(target_schema, source_column)
        alias = self._get_dictionary_table_alias(target_schema, source_column)
        input_column = ColumnName(source_column.name, input_table)
        value_column = self._get_value_column(alias)
        from_clause_part = textwrap.dedent(f"""
            LEFT OUTER JOIN {dictionary_table.fully_qualified()}
            AS {alias.fully_qualified()}
            ON
                {value_column.fully_qualified()} = 
                {input_column.fully_qualified()}
            """)
        return [from_clause_part]

    def create_transform_select_clause_part(self,
                                            sql_executor: SQLExecutor,
                                            source_column: ColumnName,
                                            input_table: TableName,
                                            target_schema: SchemaName) -> List[TransformSelectClausePart]:
        """
        This method replaces the value in the input_table with the id in the dictionary.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of select-clause parts which can be concatenated with ","
        """
        alias = self._get_dictionary_table_alias(target_schema, source_column)
        id_column = self._get_id_column(alias)
        transformation_column_name = ColumnName(f"{source_column.name}_ID")
        select_clause_part_expression = \
            textwrap.dedent(f'{id_column.fully_qualified()} AS {transformation_column_name.quoted_name()}')
        select_clause_part = TransformSelectClausePart(
            select_clause_part_expression=select_clause_part_expression,
            tranformation_column=TransformationColumn(
                source_column=source_column,
                input_column=ColumnNameBuilder(source_column).with_table_name(input_table).build(),
                column=Column(name=transformation_column_name, type=ColumnType("INTEGER")),
                purpose="ReplaceValueByID"
            )
        )
        return [select_clause_part]

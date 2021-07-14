import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.schema.column_name import Column
from exasol_data_science_utils_python.preprocessing.schema.schema_name import Schema
from exasol_data_science_utils_python.preprocessing.schema.table_name import Table
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor

ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX = "ORDINAL_ENCODER_DICTIONARY"


class OrdinalEncoder(ColumnPreprocessor):
    """
    This ColumnPreprocessor implements a OrdinalEncoder.
        It was inspired by the
    `OrdinalEncoder of scikit-learn <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html>`_

    """

    def _get_dictionary_table_alias(self, target_schema: Schema, source_column: Column):
        return self._get_table_alias(
            target_schema, source_column,
            ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX)

    def _get_dictionary_table_name(self, target_schema: Schema, source_column: Column):
        return self._get_target_table(
            target_schema, source_column,
            ORDINAL_ENCODER_DICTIONARY_TABLE_PREFIX)

    def _get_id_column(self, table: Table = None):
        min_column = Column("ID", table)
        return min_column

    def _get_value_column(self, table: Table = None):
        range_column = Column("VALUE", table)
        return range_column

    def fit(self, sqlexecutor: SQLExecutor, source_column: Column, target_schema: Schema) -> List[Table]:
        """
        This method creates a dictionary table from the source column where every distinct value of the source column
        is mapped to an id between 0 and number of distinct values - 1

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param target_schema: Schema where the result tables of the fit-queries should be stored
        :return: List of fit-queries as strings
        """
        dictionary_table = self._get_dictionary_table_name(target_schema, source_column)
        id_column = self._get_id_column()
        value_column = self._get_value_column()
        value_column_alias = Column("VALUE")
        query = textwrap.dedent(f"""
                CREATE OR REPLACE TABLE {dictionary_table.fully_qualified()} AS
                SELECT
                    CAST(rownum - 1 AS INTEGER) as {id_column.fully_qualified()},
                    {value_column_alias.fully_qualified()}
                FROM (
                    SELECT DISTINCT {source_column.fully_qualified()} as {value_column_alias.fully_qualified()}
                    FROM {source_column.table.fully_qualified()}
                    ORDER BY {source_column.fully_qualified()}
                );
                """)
        sqlexecutor.execute(query)
        return [dictionary_table]

    def create_transform_from_clause_part(self,
                                          sql_executor: SQLExecutor,
                                          source_column: Column,
                                          input_table: Table,
                                          target_schema: Schema) -> List[str]:
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
        input_column = Column(source_column.name, input_table)
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
                                            source_column: Column,
                                            input_table: Table,
                                            target_schema: Schema) -> List[str]:
        """
        This method replaces the value in the input_table with the id in the dictionary.

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of select-clause parts which can be concatenated with ","
        """
        alias = self._get_dictionary_table_alias(target_schema, source_column)
        id_column = self._get_id_column(alias)
        select_clause_part = textwrap.dedent(f'{id_column.fully_qualified()} AS "{source_column.name}_ID"')
        return [select_clause_part]

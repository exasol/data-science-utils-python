import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class ColumnPreprocesserDefinition:
    def __init__(self, column_name: str, column_preprocessor: ColumnPreprocessor):
        self.column_preprocessor = column_preprocessor
        self.column_name = column_name


class TablePreprocessor():
    def __init__(self, target_schema: Schema, source_table: Table,
                 column_preprocessor_defintions: List[ColumnPreprocesserDefinition]):
        self.source_table = source_table
        self.target_schema = target_schema
        self.column_preprocessor_defintions = column_preprocessor_defintions

    def fit(self, sql_executor: SQLExecutor) -> List[Table]:
        """
        This method calls the create_fit_queries for all column preprocessor definitions
        and returns the collected queries.
        Fit-queries are used to collect global statistics about the Source Table
        which the transformation query later uses for the transformation.
        This method is inspired by the
        `interface of scikit-learn <https://scikit-learn.org/stable/developers/develop.html>`_

        :return: List of created tables or views
        """
        result = []
        for column_preprocessor_defintion in self.column_preprocessor_defintions:
            source_column = Column(column_preprocessor_defintion.column_name, self.source_table)
            preprocessor = column_preprocessor_defintion.column_preprocessor
            created_tables = preprocessor.fit(sql_executor, source_column, self.target_schema)
            result.extend(created_tables)
        return result

    def transform(self, sql_executor: SQLExecutor, input_table: Table) -> Table:
        """
        This method creates the transform_query by calling create_transform_from_clause_part and
        create_transform_select_clause_part for all column preprocessor definitions.
        Transform queries apply the transformation and might use the collected global state from the fit-queries.
        This method is inspired by the
        `interface of scikit-learn <https://scikit-learn.org/stable/developers/develop.html>`_

        :return: created table or view
        """
        select_clause_parts_str = self._create_transform_select_clause_parts(sql_executor, input_table)
        from_clause_parts_str = self._create_transform_from_clause_parts(sql_executor, input_table)
        transformation_table = Table(
            f"{input_table.schema.name}_{input_table.name}_TRANSFORMED",
            self.target_schema)
        query = textwrap.dedent(
f"""CREATE OR REPLACE VIEW {transformation_table.fully_qualified()} AS
SELECT
{select_clause_parts_str}
FROM {input_table.fully_qualified()}
{from_clause_parts_str}""")
        sql_executor.execute(query)
        return transformation_table

    def _create_transform_from_clause_parts(self, sql_executor: SQLExecutor, input_table: Table) -> str:
        from_clause_parts = []
        for column_preprocessor_defintion in self.column_preprocessor_defintions:
            source_column = Column(column_preprocessor_defintion.column_name, self.source_table)
            column_preprocessor = column_preprocessor_defintion.column_preprocessor
            parts = column_preprocessor.create_transform_from_clause_part(
                sql_executor,
                source_column,
                input_table,
                self.target_schema)
            from_clause_parts.extend(parts)
        from_clause_parts_str = "\n".join(from_clause_parts)
        return from_clause_parts_str

    def _create_transform_select_clause_parts(self, sql_executor: SQLExecutor, input_table: Table) -> str:
        select_clause_parts = []
        for column_preprocessor_defintion in self.column_preprocessor_defintions:
            source_column = Column(column_preprocessor_defintion.column_name, self.source_table)
            preprocessor = column_preprocessor_defintion.column_preprocessor
            parts = preprocessor.create_transform_select_clause_part(
                sql_executor,
                source_column,
                input_table,
                self.target_schema)
            select_clause_parts.extend(parts)
        select_clause_parts_str = ",\n".join(select_clause_parts)
        return select_clause_parts_str

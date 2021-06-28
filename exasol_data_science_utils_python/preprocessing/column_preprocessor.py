from abc import ABC, abstractmethod
from typing import List

from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table


class ColumnPreprocessor(ABC):
    """
    A ColumnProcessor generates queries or parts of queries for a specific transformation of this column.
    """

    def _get_table_alias(self, target_schema: Schema, source_column: Column, prefix: str):
        target_schema_name = target_schema.name
        source_schema_name = source_column.table.schema.name
        source_table_name = source_column.table.name
        alias = Table(f"{target_schema_name}_{source_schema_name}_{source_table_name}_{source_column.name}_{prefix}")
        return alias

    def _get_target_table(self, target_schema: Schema, source_column: Column, prefix: str):
        source_schema_name = source_column.table.schema.name
        source_table_name = source_column.table.name
        target_table = Table(f"{source_schema_name}_{source_table_name}_{source_column.name}_{prefix}", target_schema)
        return target_table

    @abstractmethod
    def create_fit_queries(self,
                           source_column: Column,
                           target_schema: Schema) -> List[str]:
        """
        Subclasses need to implement this method to generate the fit-queries.
        Fit-queries are used to collect global statistics about the Source Table
        which the transformation query later uses for the transformation.
        This method is inspired by the
        `interface of scikit-learn <https://scikit-learn.org/stable/developers/develop.html>`_

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param target_schema: Schema where the result tables of the fit-queries should be stored
        :return: List of fit-queries as strings
        """
        pass

    @abstractmethod
    def create_transform_from_clause_part(self,
                                          source_column: Column,
                                          input_table: Table,
                                          target_schema: Schema) -> List[str]:
        """
        Subclasses need to implement this method to generate the from-clause parts of the transformation query-
        Transform queries apply the transformation and might use the collected global state from the fit-queries.
        This method is inspired by the
        `interface of scikit-learn <https://scikit-learn.org/stable/developers/develop.html>`_

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of from-clause parts which can be concatenated with "\n"
        """
        pass

    @abstractmethod
    def create_transform_select_clause_part(self,
                                            source_column: Column,
                                            input_table: Table,
                                            target_schema: Schema) -> List[str]:
        """
        Subclasses need to implement this method to generate the select-clause parts of the transformation query
        Transform queries apply the transformation and might use the collected global state from the fit-queries.
        This method is inspired by the
        `interface of scikit-learn <https://scikit-learn.org/stable/developers/develop.html>`_

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param input_table: Table to apply the transformation too
        :param target_schema: Schema where result table of the transformation should be stored
        :return: List of select-clause parts which can be concatenated with ","
        """
        pass

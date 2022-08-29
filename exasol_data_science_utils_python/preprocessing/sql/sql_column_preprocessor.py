from abc import ABC, abstractmethod
from typing import List

from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.schema import ColumnName
from exasol_data_science_utils_python.schema import ExperimentName
from exasol_data_science_utils_python.schema import SchemaName
from exasol_data_science_utils_python.schema import TableName
from exasol_data_science_utils_python.preprocessing.sql.transform_select_clause_part import TransformSelectClausePart
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class SQLColumnPreprocessor(ABC):
    """
    A ColumnProcessor generates queries or parts of queries for a specific transformation of this column.
    """

    def _get_table_alias(self, target_schema: SchemaName, source_column: ColumnName, prefix: str):
        target_schema_name = target_schema.name
        source_schema_name = source_column.table_name.schema_name.name
        source_table_name = source_column.table_name.name
        alias = TableName(
            f"{target_schema_name}_{source_schema_name}_{source_table_name}_{source_column.name}_{prefix}")
        return alias

    def _get_target_table(self,
                          target_schema: SchemaName,
                          source_column: ColumnName,
                          experiment_name: ExperimentName, prefix: str):
        source_schema_name = source_column.table_name.schema_name.name
        source_table_name = source_column.table_name.name
        target_table = TableName(
            f"{experiment_name.name}_{source_schema_name}_{source_table_name}_{source_column.name}_{prefix}",
            target_schema)
        return target_table

    @abstractmethod
    def requires_global_transformation_for_training_data(self) -> bool:
        pass

    @abstractmethod
    def fit(self,
            sql_processor: SQLExecutor,
            source_column: ColumnName,
            target_schema: SchemaName,
            experiment_name: ExperimentName) -> List[ParameterTable]:
        """
        Subclasses need to implement this method to generate the parameter tables.
        Parameter tables are used to collect global statistics about the Source Table
        which the transformation query later uses for the transformation.
        This method is inspired by the
        `interface of scikit-learn <https://scikit-learn.org/stable/developers/develop.html>`_

        :param source_column: Column in the source table which was used to fit this ColumnPreprocessor
        :param target_schema: Schema where the result tables of the fit-queries should be stored
        :return: List of parameter tables or views
        """
        pass

    @abstractmethod
    def create_transform_from_clause_part(self,
                                          sql_executor: SQLExecutor,
                                          source_column: ColumnName,
                                          input_table: TableName,
                                          target_schema: SchemaName,
                                          experiment_name: ExperimentName) -> List[str]:
        """
        Subclasses need to implement this method to generate the from-clause parts of the transformation query-
        Transform queries apply the transformation and might use the collected global state from the parameter tables.
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
                                            sql_executor: SQLExecutor,
                                            source_column: ColumnName,
                                            input_table: TableName,
                                            target_schema: SchemaName,
                                            experiment_name: ExperimentName) -> List[TransformSelectClausePart]:
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

    def __repr__(self):
        return generate_repr_for_object(self)

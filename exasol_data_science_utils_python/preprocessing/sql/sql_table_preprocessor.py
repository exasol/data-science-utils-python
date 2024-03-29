import textwrap
from typing import List, Tuple

from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.preprocessing.sql.sql_column_preprocessor_definition import \
    SQLColumnPreprocessorDefinition
from exasol_data_science_utils_python.preprocessing.sql.tranformation_table import TransformationTable
from exasol_data_science_utils_python.preprocessing.sql.transform_select_clause_part import TransformSelectClausePart
from exasol_data_science_utils_python.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.schema.table_name_builder import TableNameBuilder
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor
from exasol_data_science_utils_python.utils.repr_generation_for_object import generate_repr_for_object


class SQLTablePreprocessor:
    def __init__(self,
                 target_schema: SchemaName,
                 source_table: TableName,
                 experiment_name: ExperimentName,
                 column_preprocessor_definitions: List[SQLColumnPreprocessorDefinition]):
        self.experiment_name = experiment_name
        self.source_table = source_table
        self.target_schema = target_schema
        self.column_preprocessor_definitions = column_preprocessor_definitions

    def fit(self, sql_executor: SQLExecutor) -> List[ParameterTable]:
        """
        This method calls the create_fit_queries for all column preprocessor definitions
        and returns the collected queries.
        Fit-queries are used to collect global statistics about the Source Table
        which the transformation query later uses for the transformation.
        This method is inspired by the
        `interface of scikit-learn <https://scikit-learn.org/stable/developers/develop.html>`_

        :return: List of parameter tables or views
        """
        result = []
        for column_preprocessor_definition in self.column_preprocessor_definitions:
            source_column = ColumnNameBuilder.create(column_preprocessor_definition.column_name, self.source_table)
            preprocessor = column_preprocessor_definition.column_preprocessor
            parameter_tables = preprocessor.fit(sql_executor, source_column, self.target_schema, self.experiment_name)
            result.extend(parameter_tables)
        return result

    def transform(self, sql_executor: SQLExecutor, input_table: TableName) -> TransformationTable:
        """
        This method creates the transform_query by calling create_transform_from_clause_part and
        create_transform_select_clause_part for all column preprocessor definitions.
        Transform queries apply the transformation and might use the collected global state from the fit-queries.
        This method is inspired by the
        `interface of scikit-learn <https://scikit-learn.org/stable/developers/develop.html>`_

        :return: created transformation table or view
        """
        select_clause_parts_str, select_clause_parts = \
            self._create_transform_select_clause_parts(sql_executor, input_table)
        from_clause_parts_str = self._create_transform_from_clause_parts(sql_executor, input_table)
        transformation_table_name = \
            TableNameBuilder.create(
                f"{self.experiment_name.name}_{input_table.schema_name.name}_{input_table.name}_TRANSFORMED",
                self.target_schema)
        transformation_columns = [select_clause_part.tranformation_column for select_clause_part in select_clause_parts]
        transformation_table = TransformationTable(transformation_table_name, transformation_columns)
        query = textwrap.dedent(
            f"""CREATE OR REPLACE VIEW {transformation_table_name.fully_qualified} AS
SELECT
{select_clause_parts_str}
FROM {input_table.fully_qualified}
{from_clause_parts_str}""")
        sql_executor.execute(query)
        return transformation_table

    def _create_transform_from_clause_parts(self, sql_executor: SQLExecutor, input_table: TableName) -> str:
        from_clause_parts = []
        for column_preprocessor_definition in self.column_preprocessor_definitions:
            source_column = ColumnNameBuilder.create(column_preprocessor_definition.column_name, self.source_table)
            column_preprocessor = column_preprocessor_definition.column_preprocessor
            parts = column_preprocessor.create_transform_from_clause_part(
                sql_executor,
                source_column,
                input_table,
                self.target_schema,
                self.experiment_name)
            from_clause_parts.extend(parts)
        from_clause_parts_str = "\n".join(from_clause_parts)
        return from_clause_parts_str

    def _create_transform_select_clause_parts(self, sql_executor: SQLExecutor, input_table: TableName) \
            -> Tuple[str, List[TransformSelectClausePart]]:
        select_clause_parts = []
        for column_preprocessor_definition in self.column_preprocessor_definitions:
            source_column = ColumnNameBuilder.create(column_preprocessor_definition.column_name, self.source_table)
            preprocessor = column_preprocessor_definition.column_preprocessor
            parts = preprocessor.create_transform_select_clause_part(
                sql_executor,
                source_column,
                input_table,
                self.target_schema,
                self.experiment_name)
            select_clause_parts.extend(parts)
        select_clause_parts_strs = [select_clause_part.select_clause_part_expression for select_clause_part in
                                    select_clause_parts]
        combined_select_clause_parts_str = ",\n".join(select_clause_parts_strs)
        return combined_select_clause_parts_str, select_clause_parts

    def __repr__(self):
        return generate_repr_for_object(self)

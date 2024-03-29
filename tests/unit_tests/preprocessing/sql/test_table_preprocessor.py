import textwrap
from typing import List

from exasol_data_science_utils_python.preprocessing.sql.parameter_table import ParameterTable
from exasol_data_science_utils_python.preprocessing.sql.sql_column_preprocessor import SQLColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.sql.sql_column_preprocessor_definition import \
    SQLColumnPreprocessorDefinition
from exasol_data_science_utils_python.preprocessing.sql.sql_table_preprocessor import SQLTablePreprocessor
from exasol_data_science_utils_python.preprocessing.sql.tranformation_table import TransformationTable
from exasol_data_science_utils_python.preprocessing.sql.transform_select_clause_part import TransformSelectClausePart
from exasol_data_science_utils_python.preprocessing.sql.transformation_column import TransformationColumn
from exasol_data_science_utils_python.schema.column import Column
from exasol_data_science_utils_python.schema.column import ColumnName
from exasol_data_science_utils_python.schema.column import ColumnType
from exasol_data_science_utils_python.schema.column_name_builder import ColumnNameBuilder
from exasol_data_science_utils_python.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table import Table
from exasol_data_science_utils_python.schema.table_name import TableName
from exasol_data_science_utils_python.schema.table_name_builder import TableNameBuilder
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor
from exasol_data_science_utils_python.udf_utils.testing.mock_sql_executor import MockSQLExecutor


class MyColumnPreprocessor(SQLColumnPreprocessor):

    def requires_global_transformation_for_training_data(self):
        return False

    def fit(self,
            sqlexecutor: SQLExecutor,
            source_column: ColumnName,
            target_schema: SchemaName,
            experiment_name: ExperimentName) \
            -> List[ParameterTable]:
        target_table_name = self._get_target_table(target_schema, source_column, experiment_name, "PREFIX")
        query = textwrap.dedent(f'''
                   CREATE OR REPLACE TABLE {target_table_name.fully_qualified} AS 
                   SELECT 1 AS "VALUE"
                   ''')
        sqlexecutor.execute(query)
        column = Column(ColumnNameBuilder.create("VALUE", target_table_name), ColumnType("INTEGER"))
        target_table = Table(target_table_name, [column])
        parameter_table = ParameterTable(source_column, target_table, "purpose")
        return [parameter_table]

    def create_transform_from_clause_part(self,
                                          sql_executor: SQLExecutor,
                                          source_column: ColumnName,
                                          input_table: TableName,
                                          target_schema: SchemaName,
                                          experiment_name: ExperimentName) -> List[str]:
        target_table = self._get_target_table(target_schema, source_column, experiment_name, "PREFIX")
        return [f"CROSS JOIN {target_table.fully_qualified}"]

    def create_transform_select_clause_part(self,
                                            sql_executor: SQLExecutor,
                                            source_column: ColumnName,
                                            input_table: TableName,
                                            target_schema: SchemaName,
                                            experiment_name: ExperimentName) \
            -> List[TransformSelectClausePart]:
        transformation_column_name = ColumnNameBuilder.create(f"{source_column.name}_VALUE")
        transformation_column = Column(transformation_column_name, ColumnType("INTEGER"))
        input_column_name = ColumnNameBuilder(column_name=source_column).with_table_like_name(input_table).build()
        transformation_column = TransformationColumn(source_column=source_column,
                                                     input_column=input_column_name,
                                                     column=transformation_column,
                                                     purpose="purpose")
        transform_select_clause_part = TransformSelectClausePart(transformation_column,
                                                                 f'1 AS {transformation_column_name.quoted_name}')
        return [transform_select_clause_part]


def test_table_preprocessor_create_fit_queries():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableNameBuilder.create("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column1 = ColumnNameBuilder.create("SRC_COLUMN1", source_table)
    source_column2 = ColumnNameBuilder.create("SRC_COLUMN2", source_table)
    column_preprocessor_definitions = [
        SQLColumnPreprocessorDefinition(source_column1.name, MyColumnPreprocessor()),
        SQLColumnPreprocessorDefinition(source_column2.name, MyColumnPreprocessor()),
    ]
    experiment = ExperimentName("EXPERIMENT")
    table_preprocessor = SQLTablePreprocessor(target_schema,
                                              source_table,
                                              experiment,
                                              column_preprocessor_definitions)
    mock_sql_executor = MockSQLExecutor()
    parameter_tables = table_preprocessor.fit(mock_sql_executor)
    source_column1_create_table = textwrap.dedent('''
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_PREFIX" AS 
        SELECT 1 AS "VALUE"
        ''')
    source_column2_create_table = textwrap.dedent('''
        CREATE OR REPLACE TABLE "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN2_PREFIX" AS 
        SELECT 1 AS "VALUE"
        ''')
    expected_parameter_tables = get_expected_parameter_table()
    assert source_column1_create_table == mock_sql_executor.queries[0]
    assert source_column2_create_table == mock_sql_executor.queries[1]
    assert expected_parameter_tables == parameter_tables


def get_expected_parameter_table():
    target_table1_name = TableNameBuilder.create("EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_PREFIX",
                                                 SchemaName("TGT_SCHEMA"))
    target_table2_name = TableNameBuilder.create("EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN2_PREFIX",
                                                 SchemaName("TGT_SCHEMA"))
    expected_parameter_tables = [
        ParameterTable(
            source_column=ColumnNameBuilder.create("SRC_COLUMN1",
                                                   TableNameBuilder.create("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
            table=Table(target_table1_name,
                        columns=[
                            Column(ColumnNameBuilder.create("VALUE", target_table1_name), ColumnType("INTEGER"))], ),
            purpose="purpose"
        ),
        ParameterTable(
            source_column=ColumnNameBuilder.create("SRC_COLUMN2",
                                                   TableNameBuilder.create("SRC_TABLE", SchemaName("SRC_SCHEMA"))),
            table=Table(target_table2_name,
                        columns=[
                            Column(ColumnNameBuilder.create("VALUE", target_table2_name), ColumnType("INTEGER"))], ),
            purpose="purpose"
        )
    ]
    return expected_parameter_tables


def test_table_preprocessor_create_transform_query():
    source_schema = SchemaName("SRC_SCHEMA")
    source_table = TableNameBuilder.create("SRC_TABLE", source_schema)
    target_schema = SchemaName("TGT_SCHEMA")
    source_column1 = ColumnNameBuilder.create("SRC_COLUMN1", source_table)
    source_column2 = ColumnNameBuilder.create("SRC_COLUMN2", source_table)
    input_schema = SchemaName("IN_SCHEMA")
    input_table = TableNameBuilder.create("IN_TABLE", input_schema)
    experiment = ExperimentName("EXPERIMENT")

    column_preprocessor_definitions = [
        SQLColumnPreprocessorDefinition(source_column1.name, MyColumnPreprocessor()),
        SQLColumnPreprocessorDefinition(source_column2.name, MyColumnPreprocessor()),
    ]

    table_preprocessor = SQLTablePreprocessor(target_schema,
                                              source_table,
                                              experiment,
                                              column_preprocessor_definitions)
    mock_sql_executor = MockSQLExecutor()
    transformation_table = table_preprocessor.transform(mock_sql_executor, input_table)
    expected_transformation_table = get_expected_transformation_table()
    expected = textwrap.dedent(
        '''CREATE OR REPLACE VIEW "TGT_SCHEMA"."EXPERIMENT_IN_SCHEMA_IN_TABLE_TRANSFORMED" AS
SELECT
1 AS "SRC_COLUMN1_VALUE",
1 AS "SRC_COLUMN2_VALUE"
FROM "IN_SCHEMA"."IN_TABLE"
CROSS JOIN "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN1_PREFIX"
CROSS JOIN "TGT_SCHEMA"."EXPERIMENT_SRC_SCHEMA_SRC_TABLE_SRC_COLUMN2_PREFIX"''')
    # TODO assert table
    assert expected == mock_sql_executor.queries[0]
    assert expected_transformation_table == transformation_table


def get_expected_transformation_table():
    expected_transformation_table_name = \
        TableNameBuilder.create("EXPERIMENT_IN_SCHEMA_IN_TABLE_TRANSFORMED",
                                SchemaName("TGT_SCHEMA"))
    expected_input_table_name = TableNameBuilder.create("IN_TABLE", SchemaName("IN_SCHEMA"))
    expected_source_table_name = TableNameBuilder.create("SRC_TABLE", SchemaName("SRC_SCHEMA"))
    expected_transformation_table = \
        TransformationTable(
            table_name=expected_transformation_table_name,
            transformation_columns=[
                TransformationColumn(
                    source_column=ColumnName("SRC_COLUMN1", expected_source_table_name),
                    input_column=ColumnName("SRC_COLUMN1", expected_input_table_name),
                    column=Column(
                        ColumnName("SRC_COLUMN1_VALUE"),
                        ColumnType("INTEGER")
                    ),
                    purpose="purpose"
                ),
                TransformationColumn(
                    source_column=ColumnName("SRC_COLUMN2", expected_source_table_name),
                    input_column=ColumnName("SRC_COLUMN2", expected_input_table_name),
                    column=Column(
                        ColumnName("SRC_COLUMN2_VALUE"),
                        ColumnType("INTEGER")
                    ),
                    purpose="purpose"
                )

            ]
        )
    return expected_transformation_table

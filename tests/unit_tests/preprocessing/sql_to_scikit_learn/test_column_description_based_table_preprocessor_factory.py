from exasol_data_science_utils_python.preprocessing.scikit_learn.sklearn_identity_transformer import \
    SKLearnIdentityTransformer
from exasol_data_science_utils_python.preprocessing.sql.schema.column import Column
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.column_type import ColumnType
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_description_based_table_preprocessor_factory import \
    ColumnDescriptionBasedTablePreprocessorFactory
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor import ColumnPreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_description import \
    ColumnPreprocessorDescription
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.column_preprocessor_factory import \
    ColumnPreprocessorFactory
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.exact_column_name_selector import \
    ExactColumnNameSelector
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor
from tests.unit_tests.mock_result_set import MockResultSet
from tests.unit_tests.mock_sql_executor import MockSQLExecutor


class MockColumnPreprocessorFactory(ColumnPreprocessorFactory):

    def create(self,
               sql_executor: SQLExecutor,
               source_column: Column,
               target_schema: SchemaName) -> ColumnPreprocessor:
        return ColumnPreprocessor(source_column,
                                  target_schema,
                                  SKLearnIdentityTransformer())


def test_happy_path():
    factory = ColumnDescriptionBasedTablePreprocessorFactory(
        input_column_preprocessor_descriptions=[
            ColumnPreprocessorDescription(
                ExactColumnNameSelector("a"),
                MockColumnPreprocessorFactory()
            ),
            ColumnPreprocessorDescription(
                ExactColumnNameSelector("b"),
                MockColumnPreprocessorFactory()
            )
        ],
        target_column_preprocessor_descriptions=[
            ColumnPreprocessorDescription(
                ExactColumnNameSelector("c"),
                MockColumnPreprocessorFactory()
            ),
            ColumnPreprocessorDescription(
                ExactColumnNameSelector("d"),
                MockColumnPreprocessorFactory()
            )
        ]
    )
    sqlexecutor = MockSQLExecutor(
        result_sets=[
            MockResultSet(
                rows=[
                    ("a", "INTEGER"),
                    ("b", "VARCHAR(20000)"),
                    ("c", "DOUBLE"),
                    ("d", "DOUBLE"),
                    ("e", "INTEGER")
                ]
            )
        ]
    )
    source_table = TableName("SRC_TABLE", SchemaName("SRC_SCHEMA"))
    target_schema = SchemaName("TGT_SCHEMA")
    table_preproccesor = factory.create_table_processor(sqlexecutor,
                                                        source_table,
                                                        target_schema)
    get_columns_query = """
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM SYS.EXA_ALL_COLUMNS 
            WHERE COLUMN_SCHEMA='SRC_SCHEMA'
            AND COLUMN_TABLE='SRC_TABLE'
            """
    assert len(sqlexecutor.queries) == 1
    assert sqlexecutor.queries[0] == get_columns_query
    assert table_preproccesor.source_table == source_table
    assert table_preproccesor.target_schema == target_schema
    assert len(table_preproccesor.input_column_preprocessors) == 2
    assert len(table_preproccesor.target_column_preprocessors) == 2
    assert table_preproccesor.input_column_preprocessors[0].source_column == \
           Column(ColumnName("a", source_table), ColumnType("INTEGER"))
    assert table_preproccesor.input_column_preprocessors[1].source_column == \
           Column(ColumnName("b", source_table), ColumnType("VARCHAR(20000)"))
    assert table_preproccesor.target_column_preprocessors[0].source_column == \
           Column(ColumnName("c", source_table), ColumnType("DOUBLE"))
    assert table_preproccesor.target_column_preprocessors[1].source_column == \
           Column(ColumnName("d", source_table), ColumnType("DOUBLE"))

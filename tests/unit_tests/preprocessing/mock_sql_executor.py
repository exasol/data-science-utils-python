from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor, ResultSet
from tests.unit_tests.preprocessing.mock_result_set import MockResultSet


class MockSQLExecutor(SQLExecutor):
    def __init__(self):
        self.queries = []

    def execute(self, sql: str) -> ResultSet:
        self.queries.append(sql)
        return MockResultSet()
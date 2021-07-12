from typing import List, Dict, Any, Tuple

import pyexasol
from pyexasol import ExaStatement

from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor, ResultSet

DEFAULT_FETCHMANY_SIZE = 10000


class PyExasolResultSet(ResultSet):
    def __init__(self, statement: ExaStatement):
        self.statement = statement

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[Any]:
        return self.statement.__next__()

    def fetchone(self) -> Tuple[Any]:
        return self.statement.fetchone()

    def fetchmany(self, size=DEFAULT_FETCHMANY_SIZE) -> List[Tuple[Any]]:
        return self.statement.fetchmany(size)

    def fetchall(self) -> List[Tuple[Any]]:
        return self.statement.fetchall()

    def rowcount(self):
        return self.statement.rowcount()

    def columns(self) -> Dict[str, Dict[str, Any]]:
        """
        dataType (object) => column metadata
            type (string) => column data type
            precision (number, optional) => column precision
            scale (number, optional) => column scale
            size (number, optional) => maximum size in bytes of a column value
            characterSet (string, optional) => character encoding of a text column
            withLocalTimeZone (true | false, optional) => specifies if a timestamp has a local time zone
            fraction (number, optional) => fractional part of number
            srid (number, optional) => spatial reference system identifier
        :return:
        """
        return self.statement.columns()

    def column_names(self) -> List[str]:
        return self.statement.column_names()

    def close(self):
        return self.statement.close()


class PyexasolSQLExecutor(SQLExecutor):

    def __init__(self, connection: pyexasol.ExaConnection):
        self.connection = connection

    def execute(self, sql: str) -> ResultSet:
        return self.connection.execute(sql)

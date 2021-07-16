import pytest

from exasol_data_science_utils_python.preprocessing.sql.sql_column_preprocessor import SQLColumnPreprocessor


def test_column_preprocessor():
    with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
        SQLColumnPreprocessor()
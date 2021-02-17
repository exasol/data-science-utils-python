import pytest

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor


def test_column_preprocessor():
    with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
        ColumnPreprocessor()

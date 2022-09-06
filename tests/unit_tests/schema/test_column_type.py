import pytest

from exasol_data_science_utils_python.schema.column import ColumnName
from exasol_data_science_utils_python.schema.column_type import ColumnType
from exasol_data_science_utils_python.schema.schema_name import SchemaName
from exasol_data_science_utils_python.schema.table_name_builder import TableNameBuilder
from exasol_data_science_utils_python.schema.table_name_impl import TableNameImpl


def test_correct_types():
    ColumnType(name="COLUMN",
               precision=0,
               scale=0,
               size=0,
               characterSet="UTF-8",
               withLocalTimeZone=True,
               fraction=0,
               srid=0)


def test_optionals():
    ColumnType(name="COLUMN")


def test_name_missing():
    with pytest.raises(TypeError):
        ColumnType()


def test_name_wrong_type():
    with pytest.raises(TypeError):
        ColumnType(name=1)


def test_precision_wrong_type():
    with pytest.raises(TypeError):
        ColumnType(name="Test", precision="")


def test_scale_wrong_type():
    with pytest.raises(TypeError):
        ColumnType(name="Test", scale="")


def test_size_wrong_type():
    with pytest.raises(TypeError):
        ColumnType(name="Test", size="")


def test_characterSet_wrong_type():
    with pytest.raises(TypeError):
        ColumnType(name="Test", characterSet=1)


def test_withLocalTimeZone_wrong_type():
    with pytest.raises(TypeError):
        ColumnType(name="Test", withLocalTimeZone=1)


def test_fraction_wrong_type():
    with pytest.raises(TypeError):
        ColumnType(name="Test", fraction="")


def test_srid_wrong_type():
    with pytest.raises(TypeError):
        ColumnType(name="Test", fraction="")


def test_equality():
    column1 = ColumnType(name="COLUMN",
                         precision=0,
                         scale=0,
                         size=0,
                         characterSet="UTF-8",
                         withLocalTimeZone=True,
                         fraction=0,
                         srid=0)
    column2 = ColumnType(name="COLUMN",
                         precision=0,
                         scale=0,
                         size=0,
                         characterSet="UTF-8",
                         withLocalTimeZone=True,
                         fraction=0,
                         srid=0)
    assert column1 == column2


def test_inequality_name():
    column1 = ColumnType(name="COLUMN1")
    column2 = ColumnType(name="COLUMN2")
    assert column1 != column2


def test_inequality_precision():
    column1 = ColumnType(name="COLUMN", precision=0)
    column2 = ColumnType(name="COLUMN", precision=1)
    assert column1 != column2


def test_inequality_scale():
    column1 = ColumnType(name="COLUMN", scale=0)
    column2 = ColumnType(name="COLUMN", scale=1)
    assert column1 != column2


def test_inequality_size():
    column1 = ColumnType(name="COLUMN", size=0)
    column2 = ColumnType(name="COLUMN", size=1)
    assert column1 != column2


def test_inequality_characterSet():
    column1 = ColumnType(name="COLUMN", characterSet="UTF-8")
    column2 = ColumnType(name="COLUMN", characterSet="ASCII", )
    assert column1 != column2


def test_inequality_withLocalTimeZone():
    column1 = ColumnType(name="COLUMN", withLocalTimeZone=True)
    column2 = ColumnType(name="COLUMN", withLocalTimeZone=False)
    assert column1 != column2


def test_inequality_fraction():
    column1 = ColumnType(name="COLUMN", fraction=0)
    column2 = ColumnType(name="COLUMN", fraction=1)
    assert column1 != column2


def test_inequality_srid():
    column1 = ColumnType(name="COLUMN", srid=0)
    column2 = ColumnType(name="COLUMN", srid=1)
    assert column1 != column2


def test_inequality_table():
    column1 = ColumnName("column", TableNameBuilder.create("table1"))
    column2 = ColumnName("column", TableNameBuilder.create("table2"))
    assert column1 != column2


def test_equality():
    column1 = ColumnType(name="COLUMN",
                         precision=0,
                         scale=0,
                         size=0,
                         characterSet="UTF-8",
                         withLocalTimeZone=True,
                         fraction=0,
                         srid=0)
    column2 = ColumnType(name="COLUMN",
                         precision=0,
                         scale=0,
                         size=0,
                         characterSet="UTF-8",
                         withLocalTimeZone=True,
                         fraction=0,
                         srid=0)
    assert column1 == column2


def test_hash_inequality_name():
    column1 = ColumnType(name="COLUMN1")
    column2 = ColumnType(name="COLUMN2")
    assert hash(column1) != hash(column2)


def test_hash_inequality_precision():
    column1 = ColumnType(name="COLUMN", precision=0)
    column2 = ColumnType(name="COLUMN", precision=1)
    assert hash(column1) != hash(column2)


def test_hash_inequality_scale():
    column1 = ColumnType(name="COLUMN", scale=0)
    column2 = ColumnType(name="COLUMN", scale=1)
    assert hash(column1) != hash(column2)


def test_hash_inequality_size():
    column1 = ColumnType(name="COLUMN", size=0)
    column2 = ColumnType(name="COLUMN", size=1)
    assert hash(column1) != hash(column2)


def test_hash_inequality_characterSet():
    column1 = ColumnType(name="COLUMN", characterSet="UTF-8")
    column2 = ColumnType(name="COLUMN", characterSet="ASCII", )
    assert hash(column1) != hash(column2)


def test_hash_inequality_withLocalTimeZone():
    column1 = ColumnType(name="COLUMN", withLocalTimeZone=True)
    column2 = ColumnType(name="COLUMN", withLocalTimeZone=False)
    assert hash(column1) != hash(column2)


def test_hash_inequality_fraction():
    column1 = ColumnType(name="COLUMN", fraction=0)
    column2 = ColumnType(name="COLUMN", fraction=1)
    assert hash(column1) != hash(column2)


def test_hash_inequality_srid():
    column1 = ColumnType(name="COLUMN", srid=0)
    column2 = ColumnType(name="COLUMN", srid=1)
    assert hash(column1) != hash(column2)


def test_hash_inequality_table():
    column1 = ColumnName("column", TableNameBuilder.create("table1"))
    column2 = ColumnName("column", TableNameBuilder.create("table2"))
    assert hash(column1) != hash(column2)

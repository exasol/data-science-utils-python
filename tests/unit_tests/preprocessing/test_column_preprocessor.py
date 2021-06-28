import pytest

from exasol_data_science_utils_python.preprocessing.column_preprocessor import ColumnPreprocessor


def test_column_preprocessor():
    with pytest.raises(TypeError, match=r"Can't instantiate abstract class"):
        ColumnPreprocessor()


def test_sqlalchemy():
    import sqlalchemy as sa
    from sqlalchemy.sql.elements import quoted_name

    table1 = sa.table("table1", sa.column("id", sa.Integer), sa.column("payload", sa.String), schema="TEST")
    t1_alias = table1.alias("t1")
    t2_alias = table1.alias("t2")
    q = sa.select(t1_alias.c.id.label(quoted_name("id1", quote=True)), t2_alias.c.id.label("id1")).select_from(t1_alias,
                                                                                                               t2_alias)

    print()
    print(str(q))
    print()
    meta = sa.MetaData()
    table2 = sa.Table("table2", meta,
                      sa.Column("id", sa.Integer),
                      sa.Column("payload", sa.String),
                      schema="TEST")
    print(sa.schema.CreateTable(table2))
    print(sa.schema.DropTable(table2, if_exists=True))

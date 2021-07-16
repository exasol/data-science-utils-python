import pytest

from tests.unit_tests.mock_result_set import MockResultSet


def test_fetchall_rows_none_():
    result_set = MockResultSet()
    with pytest.raises(NotImplementedError):
        result_set.fetchall()

def test_fetchone_rows_none_():
    result_set = MockResultSet()
    with pytest.raises(NotImplementedError):
        result_set.fetchone()

def test_fetchmany_rows_none_():
    result_set = MockResultSet()
    with pytest.raises(NotImplementedError):
        result_set.fetchmany()

def test_iter_rows_none_():
    result_set = MockResultSet()
    with pytest.raises(NotImplementedError):
        result_set.__iter__()

def test_next_rows_none_():
    result_set = MockResultSet()
    with pytest.raises(NotImplementedError):
        result_set.__next__()

def test_rowcount_rows_none_():
    result_set = MockResultSet()
    with pytest.raises(NotImplementedError):
        result_set.__next__()


def test_columns_columns_none_():
    result_set = MockResultSet()
    with pytest.raises(NotImplementedError):
        result_set.columns()

def test_column_names_columns_none_():
    result_set = MockResultSet()
    with pytest.raises(NotImplementedError):
        result_set.column_names()
        
def test_for_loop():
    input = [("a", 1), ("b", 2), ("c", 4)]
    result_set = MockResultSet(rows=input)
    result = []
    for row in result_set:
        result.append(row)
    assert input == result

def test_fetchall():
    input = [("a", 1), ("b", 2), ("c", 4)]
    result_set = MockResultSet(rows=input)
    result = result_set.fetchall()
    assert input == result

def test_fetchmany():
    input = [("a", 1), ("b", 2), ("c", 4)]
    result_set = MockResultSet(rows=input)
    result = result_set.fetchmany(2)
    assert input[0:2] == result
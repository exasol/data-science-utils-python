import pytest

from exasol_data_science_utils_python.model_utils.udfs.training_parameter import TrainingParameter
from exasol_data_science_utils_python.preprocessing.schema.column import Column


@pytest.mark.parametrize(
    "epochs,batch_size,shuffle_buffer_size,split_per_node,number_of_random_partitions,split_by_columns",
    [
        (1, 2, 3, True, 4, [Column("abc")]),
        (1, 2, 3, True, None, [Column("abc")]),
        (1, 2, 3, True, None, []),
    ]
)
def test_valid_input_types(epochs, batch_size, shuffle_buffer_size, split_per_node, number_of_random_partitions,
                           split_by_columns):
    TrainingParameter(epochs, batch_size, shuffle_buffer_size, split_per_node, number_of_random_partitions,
                      split_by_columns)


@pytest.mark.parametrize(
    "epochs,batch_size,shuffle_buffer_size,split_per_node,number_of_random_partitions,split_by_columns",
    [
        (1, 2, 3, True, 4, None),
        (1, 2, 3, True, 4, "None"),
        (1, 2, 3, True, 4, [None]),
        (1, 2, 3, None, 4, [Column("abc")]),
        (1, 2, None, True, 4, [Column("abc")]),
        (1, None, 3, True, 4, [Column("abc")]),
        (None, 2, 3, True, 4, [Column("abc")]),
        ("1", 2, 3, True, 4, [Column("abc")]),
        (1, "2", 3, True, 4, [Column("abc")]),
        (1, 2, "3", True, 4, [Column("abc")]),
        (1, 2, 3, "True", 4, [Column("abc")]),
        (1, 2, 3, True, "4", [Column("abc")]),
    ]
)
def test_invalid_input_types(epochs, batch_size, shuffle_buffer_size, split_per_node, number_of_random_partitions,
                             split_by_columns):
    with pytest.raises(TypeError):
        TrainingParameter(epochs, batch_size, shuffle_buffer_size, split_per_node, number_of_random_partitions,
                          split_by_columns)


@pytest.mark.parametrize(
    "epochs,batch_size,shuffle_buffer_size,split_per_node,number_of_random_partitions,split_by_columns",
    [
        (-1, 2, 3, True, 4, []),
        (1, -2, 3, True, 4, []),
        (1, 2, -3, True, 4, []),
        (1, 2, -3, True, -4, []),
    ]
)
def test_invalid_input_values(epochs, batch_size, shuffle_buffer_size, split_per_node, number_of_random_partitions,
                              split_by_columns):
    with pytest.raises(ValueError):
        TrainingParameter(epochs, batch_size, shuffle_buffer_size, split_per_node, number_of_random_partitions,
                          split_by_columns)

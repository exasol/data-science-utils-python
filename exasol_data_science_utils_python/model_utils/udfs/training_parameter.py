from typing import List, Union

from typeguard import typechecked

from exasol_data_science_utils_python.preprocessing.schema.column import Column


class TrainingParameter:
    @typechecked
    def __init__(self,
                 epochs:int,
                 batch_size:int,
                 shuffle_buffer_size:int,
                 split_per_node:bool,
                 number_of_random_partitions:Union[None,int],
                 split_by_columns:List[Column]):
        self.split_by_columns = split_by_columns
        self.number_of_random_partitions = number_of_random_partitions
        self.split_per_node = split_per_node
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
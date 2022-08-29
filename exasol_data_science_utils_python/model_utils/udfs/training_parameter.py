from typing import List, Union

from typeguard import typechecked

from exasol_data_science_utils_python.schema import ColumnName


class TrainingParameter:
    @typechecked
    def __init__(self,
                 epochs:int,
                 batch_size:int,
                 shuffle_buffer_size:int,
                 split_per_node:bool,
                 number_of_random_partitions:Union[None,int],
                 split_by_columns:List[ColumnName]):
        self.split_by_columns = split_by_columns
        self.number_of_random_partitions = number_of_random_partitions
        self.split_per_node = split_per_node
        self.shuffle_buffer_size = shuffle_buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        if not epochs>0:
            raise ValueError("epochs needs to be >0")
        if not batch_size>0:
            raise ValueError("batch_size needs to be >0")
        if not shuffle_buffer_size>0:
            raise ValueError("shuffle_buffer_size needs to be >0")
        if isinstance(number_of_random_partitions,int) and not number_of_random_partitions>0:
            raise ValueError("number_of_random_partitions needs to be >0")

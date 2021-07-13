from pathlib import PurePosixPath

from exasol_data_science_utils_python.model_utils.udfs.abstract_column_preprocessor_creator import \
    AbstractColumnPreprocessorCreator
from exasol_data_science_utils_python.model_utils.udfs.connection_object import ConnectionObject
from exasol_data_science_utils_python.model_utils.udfs.training_parameter import TrainingParameter
from exasol_data_science_utils_python.model_utils.udfs.training_runner import TrainingRunner
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table


class TrainUDF:

    def __init__(self):
        self.counter = 0

    def run(self, exa, ctx, model, column_preprocessor_creator: AbstractColumnPreprocessorCreator):
        model_connection_name = ctx.model_connection
        path_under_model_connection = PurePosixPath(ctx.path_under_model_connection)
        db_connection_name = ctx.db_connection
        source_schema = Schema(ctx.source_schema_name)
        target_schema = Schema(ctx.target_schema_name)
        source_table = Table(ctx.source_table_name, schema=source_schema)
        input_columns = [Column(column_name, table=source_table)
                         for column_name in ctx.input_columns.split(",")]
        target_column = [Column(ctx.target_column, table=source_table)]
        split_by_columns = []
        if ctx.split_by_columns is not None and ctx.split_by_columns != "":
            split_by_columns = [Column(column_name, source_table)
                                for column_name in ctx.split_by_columns.split(",")]
        training_parameter = TrainingParameter(batch_size=ctx.batch_size,
                                               epochs=ctx.epochs,
                                               shuffle_buffer_size=ctx.shuffle_buffer_size,
                                               split_per_node=ctx.split_per_node,
                                               number_of_random_partitions=ctx.number_of_random_partitions,
                                               split_by_columns=split_by_columns)
        model_connection = exa.get_connection(model_connection_name)
        db_connection = exa.get_connection(db_connection_name)
        model_connection_object = ConnectionObject(name=model_connection_name,
                                                   address=model_connection.address,
                                                   user=model_connection.user,
                                                   password=model_connection.password)
        db_connection_object = ConnectionObject(name=db_connection_name,
                                                address=db_connection.address,
                                                user=db_connection.user,
                                                password=db_connection.password)
        job_id = str(exa.meta.statement_id)
        model_id = f"{job_id}_{exa.meta.node_id}_{exa.meta.vm_id}_{self.counter}"
        training_runner = TrainingRunner(
            job_id,
            model_id,
            model_connection_object,
            path_under_model_connection,
            db_connection_object,
            training_parameter,
            input_columns,
            target_column,
            source_table,
            target_schema,
            model,
            column_preprocessor_creator)
        model_info = training_runner.run()
        ctx.emit(model_info["job_id"],
                 model_info["model_id"],
                 model_info["model_connection_name"],
                 model_info["path_under_model_connection"],
                 model_info["model_path"])
        self.counter += 1

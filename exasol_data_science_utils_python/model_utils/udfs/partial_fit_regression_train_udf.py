from pathlib import PurePosixPath

from exasol_data_science_utils_python.model_utils.udfs.connection_object import ConnectionObject
from exasol_data_science_utils_python.model_utils.udfs.training_parameter import TrainingParameter
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regression_training_runner import PartialFitRegressionTrainingRunner
from exasol_data_science_utils_python.schema import ColumnName
from exasol_data_science_utils_python.schema import ExperimentName
from exasol_data_science_utils_python.schema import SchemaName
from exasol_data_science_utils_python.schema import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor_factory import \
    TablePreprocessorFactory


class PartialFitRegressionTrainUDF:

    def __init__(self):
        self.counter = 0

    def run(self, exa, ctx, model, table_preprocessor_factory: TablePreprocessorFactory):
        model_connection_name = ctx.model_connection
        path_under_model_connection = PurePosixPath(ctx.path_under_model_connection)
        download_retry_seconds = ctx.download_retry_seconds
        db_connection_name = ctx.db_connection
        source_schema = SchemaName(ctx.source_schema_name)
        target_schema = SchemaName(ctx.target_schema_name)
        experiment_name = ExperimentName(ctx.experiment_name)
        source_table = TableName(ctx.source_table_name, schema=source_schema)
        columns = [ColumnName(column_name, table_name=source_table)
                         for column_name in ctx.columns.split(",")]
        split_by_columns = []
        if ctx.split_by_columns is not None and ctx.split_by_columns != "":
            split_by_columns = [ColumnName(column_name, source_table)
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
        training_runner = PartialFitRegressionTrainingRunner(
            job_id,
            model_id,
            model_connection_object,
            path_under_model_connection,
            download_retry_seconds,
            db_connection_object,
            training_parameter,
            columns,
            source_table,
            target_schema,
            experiment_name,
            model,
            table_preprocessor_factory)
        model_info = training_runner.run()
        ctx.emit(model_info["job_id"],
                 model_info["model_id"],
                 model_info["model_connection_name"],
                 model_info["path_under_model_connection"],
                 model_info["model_path"])
        self.counter += 1

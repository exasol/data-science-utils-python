from exasol_data_science_utils_python.model_utils.udfs.column_preprocessor_creator import ColumnPreprocessorCreator
from exasol_data_science_utils_python.model_utils.udfs.connection_object import ConnectionObject
from exasol_data_science_utils_python.model_utils.udfs.training_parameter import TrainingParameter
from exasol_data_science_utils_python.model_utils.udfs.training_runner import TrainingRunner
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table


class TrainUDF:

    def run(self, exa, ctx, model, column_preprocessor_creator:ColumnPreprocessorCreator):
        model_connection_name = ctx.model_connection
        db_connection_name = ctx.db_connection
        source_schema = Schema(ctx.source_schema_name)
        target_schema = Schema(ctx.target_schema_name)
        source_table = Table(ctx.source_table_name, schema=source_schema)
        input_columns = [Column(column_name, table=source_table) for column_name in
                         ctx.input_columns.split(",")]
        target_column = [Column(ctx.target_column, table=source_table)]

        training_parameter = TrainingParameter(batch_size=ctx.batch_size,
                                               epochs=ctx.epochs,
                                               shuffle_buffer_size=ctx.shuffle_buffer_size)
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
        training_runner = TrainingRunner(model_connection_object,
                             db_connection_object,
                             training_parameter,
                             input_columns,
                             target_column,
                             source_table,
                             target_schema,
                             model,
                             column_preprocessor_creator)
        training_runner.run()

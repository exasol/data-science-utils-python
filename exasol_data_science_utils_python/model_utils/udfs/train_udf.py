from typing import List

import pyexasol

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.udfs import sql_udf_stub_creator
from exasol_data_science_utils_python.model_utils.udfs.abstract_column_preprocessor_creator import \
    ColumnPreprocssorCreatorResult, AbstractColumnPreprocessorCreator
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.bucketfs_location import BucketFSLocation
from exasol_data_science_utils_python.udf_utils.pyexasol_sql_executor import PyexasolSQLExecutor
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class TrainUDF:
    def __init__(self, exa, model, column_preprocessor_creator: AbstractColumnPreprocessorCreator):
        self.exa = exa
        self.model = model
        self.column_preprocessor_creator = column_preprocessor_creator

    def run(self, ctx):
        model_connection_name = ctx.model_connection
        db_connection_name = ctx.db_connection
        source_schema = Schema(ctx.source_schema_name)
        target_schema = Schema(ctx.target_schema_name)
        source_table = Table(ctx.source_table_name, schema=source_schema)
        input_columns = [Column(column_name, table=source_table) for column_name in ctx.input_columns.split(",")]
        target_column = [Column(ctx.target_column, table=source_table)]
        epochs = ctx.epochs
        batch_size = ctx.batch_size
        shuffle_buffer_size = ctx.shuffle_buffer_size

        columns = input_columns + target_column
        model_connection = self.exa.get_connection(model_connection_name)
        db_connection = self.exa.get_connection(db_connection_name)
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=model_connection.address,
                user=model_connection.user,
                pwd=model_connection.password,
                base_path=None
            )
        c = pyexasol.connect(dsn=db_connection.address, user=db_connection.user, password=db_connection.password)
        sql_executor = PyexasolSQLExecutor(c)

        column_transformer_creator_result = \
            self.column_preprocessor_creator.generate_column_transformers(
                sql_executor,
                input_columns,
                target_column,
                source_table,
                target_schema)
        self.upload_model_prototype(
            model_bucketfs_location,
            column_transformer_creator_result,
            self.model)
        sql_udf_stub_creator.create_partial_fit_regressor_udf(sql_executor, target_schema)
        sql_udf_stub_creator.create_combine_to_voting_regressor_udf(sql_executor, target_schema)

        self.run_train_queries(batch_size, columns, epochs, model_connection_name, shuffle_buffer_size, source_table,
                               sql_executor, target_schema)

    def run_train_queries(self,
                          batch_size: int,
                          columns: List[Column],
                          epochs: int,
                          model_connection_name: str,
                          shuffle_buffer_size: int,
                          source_table: Table,
                          sql_executor: SQLExecutor,
                          target_schema: Schema):
        column_name_list = ",".join(self.get_column_name_list(columns))
        query = f"""
            CREATE OR REPLACE TABLE {target_schema.fully_qualified()}."FITTED_BASE_ESTIMATORS" AS 
            SELECT {target_schema.fully_qualified()}."PARTIAL_FIT_REGRESSOR_UDF"(
                '{model_connection_name}',
                '{column_name_list}',
                {epochs},
                {batch_size},
                {shuffle_buffer_size},
                {self.get_select_list_for_columns(columns)}) 
            FROM {source_table.fully_qualified()}
            GROUP BY IPROC(), floor(rand(1,4)) -- parallelize and distribute
            """
        sql_executor.execute(query)
        query = f"""
            CREATE OR REPLACE TABLE {target_schema.fully_qualified()}."FITTED_ENSEMBLE" AS 
            SELECT {target_schema.fully_qualified()}."COMBINE_TO_VOTING_REGRESSOR_UDF"(
                '{model_connection_name}',
                output_model_path) 
            FROM {target_schema.fully_qualified()}."FITTED_BASE_ESTIMATORS"
            """
        sql_executor.execute(query)

    def upload_model_prototype(self,
                               model_bucketfs_location: BucketFSLocation,
                               column_transformer_creator_result: ColumnPreprocssorCreatorResult,
                               model):
        regressor_partial_fit_iterator = RegressorPartialFitIterator(
            input_preprocessor=column_transformer_creator_result.input_columns_transformer,
            output_preprocessor=column_transformer_creator_result.target_column_transformer,
            model=model
        )
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(regressor_partial_fit_iterator,
                                                                     PartialFitRegressorUDF.INIT_MODEL_FILE_NAME)

    def get_select_list_for_columns(self, columns: List[Column]):
        column_name_list = self.get_column_name_list(columns)
        return ",".join(f'"{column_name}"' for column_name in column_name_list)

    def get_column_name_list(self, columns: List[Column]):
        column_name_list = [column.name for column in columns]
        return column_name_list

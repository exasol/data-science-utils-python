from typing import List

import pyexasol
from numpy.random import RandomState
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.udfs import preprocessor_udf, sql_udf_stub_creator
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.bucketfs_location import BucketFSLocation
from exasol_data_science_utils_python.udf_utils.pyexasol_sql_executor import PyexasolSQLExecutor
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class TrainUDF:
    def __init__(self, exa):
        self.exa = exa

    def run(self, ctx):
        model_connection_name = ctx.model_connection
        model_connection = self.exa.get_connection(model_connection_name)
        db_connection_name = ctx.db_connection
        db_connection = self.exa.get_connection(db_connection_name)
        source_schema = Schema(ctx.source_schema_name)
        target_schema = Schema(ctx.target_schema_name)
        source_table = Table(ctx.source_table_name, schema=source_schema)
        input_columns = [Column(column_name, table=source_table) for column_name in ctx.input_columns.split(",")]
        target_column = Column(ctx.target_column, table=source_table)
        epochs = ctx.epochs
        batch_size = ctx.batch_size
        shuffle_buffer_size = ctx.shuffle_buffer_size

        columns = input_columns + [target_column]
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

        column_types = self.get_column_types(columns, source_schema, source_table, sql_executor)
        input_columns_transformer, target_column_transformer = \
            preprocessor_udf.generate_column_transformers(
                column_types,
                columns,
                input_columns,
                source_table,
                sql_executor,
                target_column,
                target_schema)
        self.upload_model_prototype(input_columns_transformer, model_bucketfs_location, target_column_transformer)
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
        column_name_list = ",".join(preprocessor_udf.get_column_name_list(columns))
        query = f"""
            CREATE OR REPLACE TABLE {target_schema.fully_qualified()}."FITTED_BASE_ESTIMATORS" AS 
            SELECT {target_schema.fully_qualified()}."PARTIAL_FIT_REGRESSOR_UDF"(
                '{model_connection_name}',
                '{column_name_list}',
                {epochs},
                {batch_size},
                {shuffle_buffer_size},
                {preprocessor_udf.get_select_list_for_columns(columns)}) 
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
                               input_columns_transformer: ColumnTransformer,
                               model_bucketfs_location: BucketFSLocation,
                               target_column_transformer: ColumnTransformer):
        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False, fit_intercept=True,
                             eta0=0.9, power_t=0.1, learning_rate='invscaling')
        regressor_partial_fit_iterator = RegressorPartialFitIterator(
            input_preprocessor=input_columns_transformer,
            output_preprocessor=target_column_transformer,
            model=model
        )
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(regressor_partial_fit_iterator,
                                                                     PartialFitRegressorUDF.INIT_MODEL_FILE_NAME)

    def get_column_types(self,
                         columns: List[Column],
                         source_schema: Schema,
                         source_table: Table,
                         sql_executor: SQLExecutor):
        column_name_list = ",".join(f"'{column.name}'" for column in columns)
        query = f"""
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM SYS.EXA_ALL_COLUMNS 
            WHERE COLUMN_SCHEMA='{source_schema.name}'
            AND COLUMN_TABLE='{source_table.name}'
            AND COLUMN_NAME in ({column_name_list})
            """
        result_set = sql_executor.execute(query)
        column_types = dict(result_set.fetchall())
        return column_types

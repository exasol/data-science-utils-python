from typing import List

import pyexasol

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.udfs import sql_udf_stub_creator
from exasol_data_science_utils_python.model_utils.udfs.abstract_column_preprocessor_creator import \
    ColumnPreprocssorCreatorResult, AbstractColumnPreprocessorCreator
from exasol_data_science_utils_python.model_utils.udfs.connection_object import ConnectionObject
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.model_utils.udfs.training_parameter import TrainingParameter
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.bucketfs_location import BucketFSLocation
from exasol_data_science_utils_python.udf_utils.pyexasol_sql_executor import PyexasolSQLExecutor
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class TrainingRunner:
    def __init__(self,
                 model_connection_object: ConnectionObject,
                 path_under_model_connection: str,
                 db_connection_object: ConnectionObject,
                 training_parameter: TrainingParameter,
                 input_columns: List[Column],
                 target_columns: List[Column],
                 source_table: Table,
                 target_schema: Schema,
                 model,
                 column_preprocessor_creator: AbstractColumnPreprocessorCreator):
        self.path_under_model_connection = path_under_model_connection
        self.source_table = source_table
        self.target_schema = target_schema
        self.target_columns = target_columns
        self.input_columns = input_columns
        self.training_parameter = training_parameter
        self.db_connection_object = db_connection_object
        self.model_connection_object = model_connection_object
        self.model = model
        self.column_preprocessor_creator = column_preprocessor_creator
        self.columns = self.input_columns + self.target_columns
        if any(column.table != self.columns[0].table for column in self.columns):
            raise ValueError(
                f"Not all columns in {[column.fully_qualified() for column in self.columns]} are from the same table.")

    def run(self):
        bucket_fs_factory = BucketFSFactory()
        model_bucketfs_location = \
            bucket_fs_factory.create_bucketfs_location(
                url=self.model_connection_object.address,
                user=self.model_connection_object.user,
                pwd=self.model_connection_object.password,
                base_path=self.path_under_model_connection
            )
        c = pyexasol.connect(dsn=self.db_connection_object.address, user=self.db_connection_object.user,
                             password=self.db_connection_object.password)
        sql_executor = PyexasolSQLExecutor(c)

        column_transformer_creator_result = \
            self.column_preprocessor_creator.generate_column_transformers(
                sql_executor,
                self.input_columns,
                self.target_columns,
                self.source_table,
                self.target_schema)
        self.upload_model_prototype(
            model_bucketfs_location,
            column_transformer_creator_result,
            self.model)
        sql_udf_stub_creator.create_partial_fit_regressor_udf(sql_executor, self.target_schema)
        sql_udf_stub_creator.create_combine_to_voting_regressor_udf(sql_executor, self.target_schema)

        self.run_train_queries(sql_executor)

    def run_train_queries(self,
                          sql_executor: SQLExecutor):
        self.run_base_estimator_training(sql_executor)
        self.run_combine_to_voting_regressor_udf(sql_executor)

    def run_combine_to_voting_regressor_udf(self, sql_executor):
        create_table = f"""
        CREATE TABLE IF NOT EXISTS {self.target_schema.fully_qualified()}."FITTED_COMBINED_MODEL" (
            session_id VARCHAR(2000000),
            model_connection_name VARCHAR(2000000),
            path_under_model_connection VARCHAR(2000000),
            model_path VARCHAR(2000000)
        )
        """
        sql_executor.execute(create_table)
        query = f"""
            INSERT INTO {self.target_schema.fully_qualified()}."FITTED_COMBINED_MODEL"
            SELECT  
                CURRENT_SESSION, 
                model_connection_name, 
                path_under_model_connection, 
                combined_model_path
            FROM (
                SELECT {self.target_schema.fully_qualified()}."COMBINE_TO_VOTING_REGRESSOR_UDF"(
                    model_connection_name,
                    path_under_model_connection,
                    model_path) 
                FROM {self.target_schema.fully_qualified()}."FITTED_BASE_MODELS"
                WHERE session_id = CURRENT_SESSION
                AND model_connection_name = '{self.model_connection_object.name}'
                AND (
                    path_under_model_connection = {self._get_path_under_model_connection_as_sql_value()}
                    OR (
                        {self._get_path_under_model_connection_as_sql_value()} is null AND
                        path_under_model_connection is null
                    )
                )
            )
            """
        print(query)
        sql_executor.execute(query)

    def _get_path_under_model_connection_as_sql_value(self):
        if self.path_under_model_connection is None:
            return "null"
        else:
            return f"'{self.path_under_model_connection}'"

    def run_base_estimator_training(self, sql_executor):
        column_name_list = ",".join(self.get_column_name_list(self.columns))
        select_list_for_columns = self.get_select_list_for_columns(self.columns)
        create_table = f"""
        CREATE TABLE IF NOT EXISTS {self.target_schema.fully_qualified()}."FITTED_BASE_MODELS" (
            session_id VARCHAR(2000000),
            model_connection_name VARCHAR(2000000),
            path_under_model_connection VARCHAR(2000000),
            model_path VARCHAR(2000000),
            training_score_sum DOUBLE, 
            training_score_count INTEGER
        )
        """
        sql_executor.execute(create_table)
        query = f"""
            INSERT INTO {self.target_schema.fully_qualified()}."FITTED_BASE_MODELS" 
            SELECT 
                CURRENT_SESSION, 
                model_connection_name, 
                path_under_model_connection, 
                output_model_path,
                training_score_sum,
                training_score_count
            FROM (
                SELECT {self.target_schema.fully_qualified()}."PARTIAL_FIT_REGRESSOR_UDF"(
                    '{self.model_connection_object.name}',
                    {self._get_path_under_model_connection_as_sql_value()},
                    '{column_name_list}',
                    {self.training_parameter.epochs},
                    {self.training_parameter.batch_size},
                    {self.training_parameter.shuffle_buffer_size},
                    {select_list_for_columns}) 
                FROM {self.source_table.fully_qualified()}
                GROUP BY IPROC(), floor(rand(1,4)) -- parallelize and distribute
            )
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
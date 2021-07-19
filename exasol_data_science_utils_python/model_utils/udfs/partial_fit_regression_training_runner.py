from pathlib import PurePosixPath
from typing import List, Any, Dict

import pyexasol

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.udfs import sql_udf_stub_creator
from exasol_data_science_utils_python.model_utils.udfs.connection_object import ConnectionObject
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.model_utils.udfs.training_parameter import TrainingParameter
from exasol_data_science_utils_python.preprocessing.sql.schema.column_name import ColumnName
from exasol_data_science_utils_python.preprocessing.sql.schema.experiment_name import ExperimentName
from exasol_data_science_utils_python.preprocessing.sql.schema.schema_name import SchemaName
from exasol_data_science_utils_python.preprocessing.sql.schema.table_name import TableName
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor import TablePreprocessor
from exasol_data_science_utils_python.preprocessing.sql_to_scikit_learn.table_preprocessor_factory import \
    TablePreprocessorFactory
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.bucketfs_location import BucketFSLocation
from exasol_data_science_utils_python.udf_utils.pyexasol_sql_executor import PyexasolSQLExecutor
from exasol_data_science_utils_python.udf_utils.sql_executor import SQLExecutor


class PartialFitRegressionTrainingRunner:
    def __init__(self,
                 job_id: str,
                 model_id: str,
                 model_connection_object: ConnectionObject,
                 path_under_model_connection: PurePosixPath,
                 download_retry_seconds: int,
                 db_connection_object: ConnectionObject,
                 training_parameter: TrainingParameter,
                 columns: List[ColumnName],
                 source_table: TableName,
                 target_schema: SchemaName,
                 experiment_name: ExperimentName,
                 model,
                 table_preprocessor_factory: TablePreprocessorFactory):
        self.job_id = job_id
        self.model_id = model_id
        self.path_under_model_connection = path_under_model_connection
        self.source_table = source_table
        self.target_schema = target_schema
        self.experiment_name = experiment_name
        self.training_parameter = training_parameter
        self.db_connection_object = db_connection_object
        self.model_connection_object = model_connection_object
        self.model = model
        self.table_preprocessor_factory = table_preprocessor_factory
        self.columns = columns
        self.download_retry_seconds = download_retry_seconds
        if any(column.table_name != self.columns[0].table_name for column in self.columns):
            raise ValueError(
                f"Not all columns in {[column.fully_qualified() for column in self.columns]} are from the same table.")

    def run(self) -> Dict[str, Any]:
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

        table_preprocessor = \
            self.table_preprocessor_factory.create_table_processor(
                sql_executor,
                self.source_table,
                self.target_schema,
                self.experiment_name
            )
        self.upload_model_prototype(
            model_bucketfs_location,
            table_preprocessor,
            self.model)
        sql_udf_stub_creator.create_partial_fit_regressor_udf(sql_executor, self.target_schema)
        sql_udf_stub_creator.create_combine_to_voting_regressor_udf(sql_executor, self.target_schema)

        self.run_train_queries(sql_executor)
        model_info = self.get_combined_model_info(sql_executor)
        return model_info

    def run_train_queries(self,
                          sql_executor: SQLExecutor):
        self.run_base_estimator_training(sql_executor)
        self.run_combine_to_voting_regressor_udf(sql_executor)

    def run_combine_to_voting_regressor_udf(self, sql_executor):
        self.create_fitted_combined_model_table(sql_executor)
        self.insert_combined_model(sql_executor)

    def get_combined_model_info(self, sql_executor) -> Dict[str, Any]:
        result_query = f"""
            SELECT 
                job_id,
                model_id,
                model_connection_name,
                path_under_model_connection,
                model_path
            FROM {self.target_schema.fully_qualified()}."FITTED_COMBINED_MODEL"
            WHERE job_id = '{self.job_id}' 
            AND model_id = '{self.model_id}'
            AND model_connection_name = '{self.model_connection_object.name}'
            AND (
                path_under_model_connection = {self._get_path_under_model_connection_as_sql_value()}
                OR (
                    {self._get_path_under_model_connection_as_sql_value()} is null AND
                    path_under_model_connection is null
                )
            )
        """
        rs = sql_executor.execute(result_query)
        rows = rs.fetchall()
        if rs.rowcount() != 1:
            raise RuntimeError(f"Internal Error: Got not exactly one model from query {result_query}")
        row = rows[0]
        model_info = {
            "job_id": row[0],
            "model_id": row[1],
            "model_connection_name": row[2],
            "path_under_model_connection": row[3],
            "model_path": row[4]
        }
        return model_info

    def insert_combined_model(self, sql_executor):
        query = f"""
            INSERT INTO {self.target_schema.fully_qualified()}."FITTED_COMBINED_MODEL"
            SELECT  
                '{self.job_id}',
                '{self.model_id}',
                model_connection_name, 
                path_under_model_connection, 
                combined_model_path
            FROM (
                SELECT {self.target_schema.fully_qualified()}."COMBINE_TO_VOTING_REGRESSOR_UDF"(
                    model_connection_name,
                    path_under_model_connection,
                    model_path,
                    {self.download_retry_seconds}) 
                FROM {self.target_schema.fully_qualified()}."FITTED_BASE_MODELS"
                WHERE job_id = '{self.job_id}'
                AND model_id = '{self.model_id}'
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
        sql_executor.execute(query)

    def create_fitted_combined_model_table(self, sql_executor):
        create_table = f"""
        CREATE TABLE IF NOT EXISTS {self.target_schema.fully_qualified()}."FITTED_COMBINED_MODEL" (
            job_id VARCHAR(2000000),
            model_id VARCHAR(2000000),
            model_connection_name VARCHAR(2000000),
            path_under_model_connection VARCHAR(2000000),
            model_path VARCHAR(2000000)
        )
        """
        sql_executor.execute(create_table)

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
            job_id VARCHAR(2000000),
            model_id VARCHAR(2000000),
            model_connection_name VARCHAR(2000000),
            path_under_model_connection VARCHAR(2000000),
            model_path VARCHAR(2000000),
            training_score_sum DOUBLE, 
            training_score_count INTEGER
        )
        """
        group_by_clause = self.generate_group_by_clause()
        sql_executor.execute(create_table)
        query = f"""
            INSERT INTO {self.target_schema.fully_qualified()}."FITTED_BASE_MODELS" 
            SELECT 
                '{self.job_id}',
                '{self.model_id}',
                model_connection_name, 
                path_under_model_connection, 
                output_model_path,
                training_score_sum,
                training_score_count
            FROM (
                SELECT {self.target_schema.fully_qualified()}."PARTIAL_FIT_REGRESSOR_UDF"(
                    '{self.model_connection_object.name}',
                    {self._get_path_under_model_connection_as_sql_value()},
                    {self.download_retry_seconds},
                    '{column_name_list}',
                    {self.training_parameter.epochs},
                    {self.training_parameter.batch_size},
                    {self.training_parameter.shuffle_buffer_size},
                    {select_list_for_columns}) 
                FROM {self.source_table.fully_qualified()}
                {group_by_clause}
            )
            """
        sql_executor.execute(query)

    def generate_group_by_clause(self):
        group_by_clause_parts = []
        if self.training_parameter.split_per_node:
            group_by_clause_parts.append("IPROC()")
        partitions = self.training_parameter.number_of_random_partitions
        if partitions is not None:
            if self.training_parameter.split_per_node:
                # We use the floot(partitions/NPROC()), because in case of training using too many partitions too small
                # partitions can starve the estimators. With the floor function we use the lower bound of partitions
                # which fit into the requested number of partitions
                partitions_expression = f"floor({partitions} / NPROC())"
            else:
                partitions_expression = f"{partitions}"
            group_by_clause_parts.append(f"least(floor(rand(0,{partitions_expression})),{partitions_expression}-1)")
        if len(self.training_parameter.split_by_columns) != 0:
            group_by_clause_parts += [column.quoted_name() for column in self.training_parameter.split_by_columns]
        if len(group_by_clause_parts) > 0:
            group_by_clause = "GROUP BY " + ",".join(group_by_clause_parts)
        else:
            group_by_clause = ""
        return group_by_clause

    def upload_model_prototype(self,
                               model_bucketfs_location: BucketFSLocation,
                               table_preprocessor: TablePreprocessor,
                               model):
        regressor_partial_fit_iterator = RegressorPartialFitIterator(
            table_preprocessor=table_preprocessor,
            model=model
        )
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(regressor_partial_fit_iterator,
                                                                     PartialFitRegressorUDF.INIT_MODEL_FILE_NAME)

    def get_select_list_for_columns(self, columns: List[ColumnName]):
        column_name_list = self.get_column_name_list(columns)
        return ",".join(f'"{column_name}"' for column_name in column_name_list)

    def get_column_name_list(self, columns: List[ColumnName]):
        column_name_list = [column.name for column in columns]
        return column_name_list

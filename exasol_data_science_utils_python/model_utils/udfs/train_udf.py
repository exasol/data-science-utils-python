import textwrap

import pandas as pd
import pyexasol
from numpy.random import RandomState
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDRegressor

from exasol_data_science_utils_python.model_utils.partial_fit_iterator import RegressorPartialFitIterator
from exasol_data_science_utils_python.model_utils.sklearn_contrast_matrix_transformer import \
    SKLearnContrastMatrixTransformer
from exasol_data_science_utils_python.model_utils.sklearn_min_max_scalar import SKLearnMinMaxScalar
from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
from exasol_data_science_utils_python.preprocessing.encoding.ordinal_encoder import OrdinalEncoder
from exasol_data_science_utils_python.preprocessing.normalization.min_max_scaler import MinMaxScaler
from exasol_data_science_utils_python.preprocessing.schema.column import Column
from exasol_data_science_utils_python.preprocessing.schema.schema import Schema
from exasol_data_science_utils_python.preprocessing.schema.table import Table
from exasol_data_science_utils_python.preprocessing.table_preprocessor import ColumnPreprocesserDefinition, \
    TablePreprocessor
from exasol_data_science_utils_python.udf_utils.bucketfs_factory import BucketFSFactory
from exasol_data_science_utils_python.udf_utils.pyexasol_sql_executor import PyexasolSQLExecutor


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
        sql_executor.execute(
            "ALTER SESSION SET SCRIPT_LANGUAGES='PYTHON3=localzmq+protobuf:///bfsdefault/default/test/python-3.6-minimal-EXASOL-6.2.0-release-ml?lang=python#buckets/bfsdefault/default/test/python-3.6-minimal-EXASOL-6.2.0-release-ml/exaudf/exaudfclient_py3';")
        sql_executor.execute(f"DROP CONNECTION {model_connection_name}")
        sql_executor.execute(
            f"CREATE CONNECTION {model_connection_name} TO 'http://localhost:6583/default/model;bfsdefault' USER '{model_connection.user}' IDENTIFIED BY '{model_connection.password}';")

        column_types = self.get_column_types(columns, source_schema, source_table, sql_executor)
        fit_tables = self.fit_table_preprocessor(column_types, columns, source_table, sql_executor, target_schema)
        input_columns_transformer = self.create_column_transformer(column_types, fit_tables, input_columns,
                                                                   source_table, sql_executor)
        target_column_transformer = self.create_column_transformer(column_types, fit_tables, [target_column],
                                                                   source_table, sql_executor)
        self.upload_model_prototype(input_columns_transformer, model_bucketfs_location, target_column_transformer)
        self.create_partial_fit_regressor_udf(sql_executor, target_schema)
        self.create_combine_to_voting_regressor_udf(sql_executor, target_schema)
        column_name_list = ",".join(self.get_column_name_list(columns))
        epochs = 1000
        batch_size = 100
        shuffle_buffer_size = 10000
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

    def upload_model_prototype(self, input_columns_transformer, model_bucketfs_location, target_column_transformer):
        model = SGDRegressor(random_state=RandomState(0), loss="squared_loss", verbose=False)
        regressor_partial_fit_iterator = RegressorPartialFitIterator(
            input_preprocessor=input_columns_transformer,
            output_preprocessor=target_column_transformer,
            model=model
        )
        model_bucketfs_location.upload_object_to_bucketfs_via_joblib(regressor_partial_fit_iterator,
                                                                     PartialFitRegressorUDF.INIT_MODEL_FILE_NAME)

    def create_combine_to_voting_regressor_udf(self, sql_executor, target_schema):
        udf = textwrap.dedent(f"""
        CREATE OR REPLACE PYTHON3 SET SCRIPT {target_schema.fully_qualified()}."COMBINE_TO_VOTING_REGRESSOR_UDF"(
        model_connection varchar(10000),
        input_model_path varchar(10000)
        ) 
        EMITS (output_model_path varchar(10000)) AS
        from exasol_data_science_utils_python.model_utils.udfs.combine_to_voting_regressor_udf import \
            CombineToVotingRegressorUDF
    
        udf = CombineToVotingRegressorUDF(exa)
    
        def run(ctx):
            udf.run(ctx)
        """)
        sql_executor.execute(udf)

    def create_partial_fit_regressor_udf(self, sql_executor, target_schema):
        udf = textwrap.dedent(f"""
        CREATE OR REPLACE PYTHON3 SET SCRIPT {target_schema.fully_qualified()}."PARTIAL_FIT_REGRESSOR_UDF"(...) 
        EMITS (output_model_path varchar(10000)) AS
        from exasol_data_science_utils_python.model_utils.udfs.partial_fit_regressor_udf import PartialFitRegressorUDF
    
        udf = PartialFitRegressorUDF(exa)
    
        def run(ctx):
            udf.run(ctx)
        """)
        sql_executor.execute(udf)

    # ET SCRIPT merge_sgd_classifier(bucketfs_path varchar(100000))
    # EMITS (bucketfs_path varchar(10000)) AS
    def get_column_types(self, columns, source_schema, source_table, sql_executor):
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

    def fit_table_preprocessor(self, column_types, columns, source_table, sql_executor, target_schema):
        column_preprocessor_defintions = []
        for column in columns:
            column_type = column_types[column.name]
            if column_type == 'DECIMAL(18,0)':
                column_preprocessor_defintions.append(ColumnPreprocesserDefinition(column.name, OrdinalEncoder()))
            elif column_type.startswith('VARCHAR'):
                column_preprocessor_defintions.append(ColumnPreprocesserDefinition(column.name, OrdinalEncoder()))
            elif column_type == 'DOUBLE':
                column_preprocessor_defintions.append(ColumnPreprocesserDefinition(column.name, MinMaxScaler()))
            else:
                raise Exception(f"Type '{column_type}' of column {column.fully_qualified()} not supported")
        table_preprocessor = TablePreprocessor(target_schema, source_table, column_preprocessor_defintions)
        fit_tables = table_preprocessor.fit(sql_executor)
        return fit_tables

    def create_column_transformer(self, column_types, fit_tables, columns, source_table, sql_executor):
        sklearn_transformer = []
        for column in columns:
            column_type = column_types[column.name]
            if column_type == 'DECIMAL(18,0)':
                transformer = self.create_onehot_transformer(column, fit_tables, sql_executor)
                sklearn_transformer.append((column.name, transformer, [column.name]))
            elif column_type.startswith('VARCHAR'):
                transformer = self.create_onehot_transformer(column, fit_tables, sql_executor)
                sklearn_transformer.append((column.name, transformer, [column.name]))
            elif column_type == 'DOUBLE':
                transformer = self.create_minmax_transformer(column, fit_tables, sql_executor)
                sklearn_transformer.append((column.name, transformer, [column.name]))
            else:
                raise Exception(f"Type '{column_type}' of column {column.fully_qualified()} not supported")
        column_transformer = ColumnTransformer(transformers=sklearn_transformer)
        select_list = self.get_select_list_for_columns(columns)
        dummy_data_for_fit = sql_executor.execute(
            f"""SELECT {select_list} FROM {source_table.fully_qualified()} LIMIT 2""").fetchall()
        dummy_df_for_fit = pd.DataFrame(data=dummy_data_for_fit, columns=[column.name for column in columns])
        return column_transformer.fit(dummy_df_for_fit)

    def get_column_name_list(self, columns):
        column_name_list = [column.name for column in columns]
        return column_name_list

    def create_onehot_transformer(self, column, fit_tables, sql_executor):
        fit_table_for_column = [table for table in fit_tables if f"_{column.name}_" in table.name]
        if len(fit_table_for_column) != 1:
            raise Exception(f"Couldn't find fit_table for column '{column.name}'")
        result_set = sql_executor.execute(
            f"""SELECT "VALUE" FROM {fit_table_for_column[0].fully_qualified()}""")
        dictionary = pd.DataFrame(data=result_set.fetchall(), columns=result_set.column_names())
        dictionary = dictionary.set_index("VALUE", drop=False)
        contrast_matrix = pd.get_dummies(dictionary, columns=["VALUE"], drop_first=True)
        transformer = SKLearnContrastMatrixTransformer(contrast_matrix)
        return transformer

    def create_minmax_transformer(self, column, fit_tables, sql_executor):
        fit_table_for_column = [table for table in fit_tables if f"_{column.name}_" in table.name]
        if len(fit_table_for_column) != 1:
            raise Exception(f"Couldn't find fit_table for column '{column.name}'")
        result_set = sql_executor.execute(
            f"""SELECT "MIN", "RANGE"  FROM {fit_table_for_column[0].fully_qualified()}""")
        rows = result_set.fetchall()
        min_value = rows[0][0]
        range_value = rows[0][1]
        transformer = SKLearnMinMaxScalar(min_value=min_value, range_value=range_value)
        return transformer

    def get_select_list_for_columns(self, columns):
        column_name_list = self.get_column_name_list(columns)
        return ",".join(f'"{column_name}"' for column_name in column_name_list)

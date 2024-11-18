### This step is going to apply a preprocesing to my 2 dataframes (surveys_data_df and lod_factor_df)
### and then is going to merge them into a single df.
### After this is done it 


from subprocess import check_call
from sys import executable

STEP = "CLUSTERIZE"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

# General
import pandas as pd
from pandas import DataFrame
from pandas.tseries.offsets import MonthEnd
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import os
import numpy as np
import boto3
import s3fs
from itertools import combinations
import pickle
import json
import re
import gc
import argparse
import logging
from os import environ
import utils
from boto3 import resource
from pandas import read_csv
import yaml 
# Sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    log_loss,
    classification_report,
    confusion_matrix,
    make_scorer,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta

SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())


# We define the Arguments class that inherits from the AbstractArguments abstract class.
class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        # We call the constructor of the parent class.
        super().__init__()

        # We create an ArgumentParser object that will contain all the necessary arguments for the script.
        parser = argparse.ArgumentParser(description=f"Inputs for the {STEP} step.")

        # We define the arguments that will be passed to the script.
        # "--s3_bucket": is the name of the S3 bucket where the data will be stored or from where it will be read.
        parser.add_argument("--s3_bucket", type=str)

        # "--s3_path_read": is the path in the S3 bucket from where the data will be read.
        # parser.add_argument("--s3_path_read", type=str)

        # "--s3_path_write": is the path in the S3 bucket where the data will be written.
        parser.add_argument("--s3_path_write", type=str)

        # "--str_execution_date": is the execution date of the script.
        parser.add_argument("--str_execution_date", type=str)

        # Finally, we parse the arguments and store them in the self.args property of the class.
        self.args = parser.parse_args()


def classify_into_clusters(df, touchpoints, scaler, kmeans):
    """
    Aplica el modelo K-means entrenado a un dataframe dado y asigna las etiquetas de cluster.

    Args:
    df (pd.DataFrame): DataFrame con los datos a segmentar.
    touchpoints (list): Lista de columnas de touchpoints.
    scaler (StandardScaler): Objeto StandardScaler ya entrenado.
    kmeans (KMeans): Objeto KMeans ya entrenado.

    Returns:
    pd.DataFrame: DataFrame original con las etiquetas de cluster asignadas.
    """
    # Copiar el dataframe original para evitar modificaciones accidentales
    df_copy = df.copy()

    # Filtrar el dataframe
    filtered_df = df_copy[touchpoints]

    # Reemplazar NaNs con -1 (u otro valor que no interfiera en tu análisis)
    filtered_df.fillna(-1, inplace=True)

    # Normalización de las variables utilizando el mismo scaler entrenado
    X_new_scaled = scaler.transform(filtered_df)

    # Aplicar el modelo K-means entrenado al nuevo dataframe
    new_clusters = kmeans.predict(X_new_scaled)

    # Asignar las etiquetas de cluster al dataframe original
    df_copy['cluster'] = new_clusters

    return df_copy

def plot_cluster_proportions(df):
    """
    Genera un histograma con las proporciones de cada cluster.
    
    Args:
    df (pd.DataFrame): DataFrame con las etiquetas de cluster asignadas.
    """
    # Calcular las proporciones de cada cluster
    cluster_counts = df['cluster'].value_counts().sort_index()
    cluster_proportions = cluster_counts / cluster_counts.sum()

    # Generar el histograma con diferentes colores para cada barra utilizando la paleta 'viridis'
    palette = sns.color_palette("viridis", len(cluster_proportions))  # Obtener una paleta de colores viridis
    colors = palette.as_hex()  # Convertir la paleta a formato hexadecimal

    plt.figure(figsize=(10, 6))
    cluster_proportions.plot(kind='bar', color=colors)
    plt.title('Proporción de clientes en cada cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Proporción de clientes')
    plt.xticks(rotation=0)
    plt.show()


def get_names_from_pipeline(preprocessor):
    """
    This function returns the names of the columns that are outputted by the preprocessor.

    Parameters:
    preprocessor (ColumnTransformer): The preprocessor to get output column names from.

    Returns:
    output_columns (list): List of the output column names.
    """
    output_columns = []

    # For each transformer in the preprocessor
    for name, transformer, cols in preprocessor.transformers_:
        # If the transformer is 'drop' or columns are 'drop', continue to the next transformer
        if transformer == 'drop' or cols == 'drop':
            continue

        # If the transformer is a Pipeline, get the last step of the pipeline
        if isinstance(transformer, Pipeline):
            transformer = transformer.steps[-1][1]  # get the last step of the pipeline

        # Depending on the type of the transformer, get the transformed column names
        if isinstance(transformer, ce.TargetEncoder):
            names = [f'{col}_target_enc' for col in cols]
            output_columns += names
        elif isinstance(transformer, ce.WOEEncoder):
            names = [f'{col}_woe_enc' for col in cols]
            output_columns += names
        elif isinstance(transformer, prep.OneHotEncoder):
            names = [f'{col}_enc' for col in transformer.get_feature_names_out(cols)]
            output_columns += names
        else:
            output_columns += cols

    # Return the list of output column names
    return output_columns


def read_data(prefix) -> DataFrame:
    """This function automatically reads a dataframe processed
    with all features in S3 and return this dataframe with
    cid as index

    Parameters
    ----------

    Returns
    -------
    Pandas dataframe containing all features
    """

    s3_keys = [item.key for item in s3_resource.Bucket(S3_BUCKET).objects.filter(Prefix=prefix) if item.key.endswith(".csv")]
    preprocess_paths = [f"s3://{S3_BUCKET}/{key}" for key in s3_keys]
    SAGEMAKER_LOGGER.info(f"preprocess_paths: {preprocess_paths}")
    df_features = pd.DataFrame()
    for file in preprocess_paths:
        df = pd.read_csv(file, error_bad_lines=False)
        df_features = pd.concat([df_features, df], axis=0)
    SAGEMAKER_LOGGER.info(f"Data size: {str(len(df_features))}")
    SAGEMAKER_LOGGER.info(f"Columns: {df_features.columns}")
    df_features.index = df_features[config['VARIABLES_ETL']['ID']]
    df_features.index.name = config['VARIABLES_ETL']['ID']

    return df_features

def read_csv_from_s3(bucket_name, object_key):
    # Create a boto3 S3 client
    s3_client = boto3.client('s3')
    
    # Get the object from S3
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    
    # Read the CSV content
    csv_string = response['Body'].read().decode('utf-8')
    
    # Convert to a Pandas DataFrame
    df = pd.read_csv(StringIO(csv_string))
    
    return df


if __name__ == "__main__":

    """Main functionality of the script."""
    # DEFINE ARGUMENTS
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()
    config = utils.read_config_data()
    S3_BUCKET = args.s3_bucket
    S3_PATH_WRITE = args.s3_path_write
    STR_EXECUTION_DATE = args.str_execution_date
    IS_LAST_DATE = 1
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]
    
    # Convert to datetime object
    execution_date = datetime.strptime(STR_EXECUTION_DATE, "%Y-%m-%d")

    # Format dates as strings for S3 prefixes
    today_date_str = execution_date.strftime("%Y-%m-%d")

    # s3 object
    s3_resource = boto3.resource("s3")

    # Execute clusterize
    columns_to_save = config.get("VARIABLES_CLUSTERIZE").get('COLUMNS_TO_SAVE')
    touchpoints = config.get("VARIABLES_CLUSTERIZE").get('FEATURES')
    model_name = config.get("VARIABLES_CLUSTERIZE").get('MODEL_NAME')[0]
    scaler_name = config.get("VARIABLES_CLUSTERIZE").get('SCALER_NAME')[0]
    
    # path
    src_path_filtered = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{year}{month}{day}/filtered.csv"
    src_path_sampled = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{year}{month}{day}/sampled.csv"
    out_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/01_clusterize_step/{year}{month}{day}"
    model_path = f"customer/simulations/sbx/01_clusterize_step/model/{model_name}.pkl"
    scaler_path = f"customer/simulations/sbx/01_clusterize_step/model/{scaler_name}.pkl"
    
    SAGEMAKER_LOGGER.info("userlog: Model_path:", model_path)
    SAGEMAKER_LOGGER.info("userlog: Scaler_path:", scaler_path)
        
    # Load the trained model from S3
    s3_resource = resource("s3")
    model = (
        s3_resource.Bucket(S3_BUCKET)
        .Object(f"{model_path}")
        .get()
    )
    kmeans_model = pickle.loads(model["Body"].read())      
    
    s3_resource = resource("s3")
    scaler = (
        s3_resource.Bucket(S3_BUCKET)
        .Object(f"{scaler_path}")
        .get()
    )
    kmeans_scaler = pickle.loads(scaler["Body"].read())  

    df_filtered = pd.read_csv(src_path_filtered)
    df_sampled = pd.read_csv(src_path_sampled)
    
    df_f = classify_into_clusters(df_filtered, touchpoints, kmeans_scaler, kmeans_model)
    df_s = classify_into_clusters(df_sampled, touchpoints, kmeans_scaler, kmeans_model)

    df_f[columns_to_save].to_csv(f"{out_path}/clusterized.csv", index=False)
    df_s[columns_to_save].to_csv(f"{out_path}/sampled_clusterized.csv", index=False)
    

from subprocess import check_call
from sys import executable

STEP = "AGG_PREDICT_EXPLAIN"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

# General
import pandas as pd
from pandas.tseries.offsets import MonthEnd
from datetime import datetime, timedelta
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import os
import numpy as np
import datetime
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

import darts
from darts import TimeSeries
from darts.utils.timeseries_generation import (
    gaussian_timeseries,
    linear_timeseries,
    sine_timeseries,
)

from darts.metrics import mape, smape, mae
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries

from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor


import lightgbm
import shap

from darts.models import LightGBMModel

from darts.models import LightGBMModel, RandomForest, LinearRegressionModel
from darts.utils.statistics import check_seasonality, plot_acf, plot_residuals_analysis

from darts.explainability.shap_explainer import ShapExplainer
import pickle
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from darts.models import LinearRegressionModel, LightGBMModel, RandomForest
from calendar import month_name as mn
import os

# Random
import random

#Warnings
import warnings
warnings.filterwarnings("ignore")


SAGEMAKER_LOGGER = logging.getLogger("sagemaker")
SAGEMAKER_LOGGER.setLevel(logging.INFO)
SAGEMAKER_LOGGER.addHandler(logging.StreamHandler())


# Inherits from the AbstractArguments class
class Arguments(utils.AbstractArguments):
    """Class to define the arguments used in the main functionality."""

    def __init__(self):
        """Class constructor."""
        # Call to the constructor of the parent class
        super().__init__()

        # Create an ArgumentParser object
        parser = argparse.ArgumentParser(description=f"Inputs for {STEP} step.")

        # Add the command line arguments that will be used
        parser.add_argument("--s3_bucket", type=str)  # S3 bucket name
        parser.add_argument("--s3_path_write", type=str)  # S3 path to write data
        parser.add_argument("--str_execution_date", type=str)  # Execution date
        parser.add_argument("--is_last_date", type=str, default="1")  # Indicator for the last date

        # Parse the arguments and store them in the 'args' attribute
        self.args = parser.parse_args()
        
def compute_shap_and_prediction(row, key, features_cols, future_scaler, model):
    """
    Computes SHAP values and the predicted NPS for a given row.
    
    Parameters:
    - row_df: The DataFrame row for which to compute SHAP values and prediction.
    - key: The key identifying the specific model and scaler to use.
    - features_cols: List of column names representing features used by the model.
    
    Returns:
    - A tuple containing SHAP values as a dictionary and the predicted NPS.
    """
    # Logic to prepare the row for SHAP value computation and prediction
    aux_nps_ts = TimeSeries.from_series(pd.Series([0]))
    aux_row = pd.DataFrame(0, index=[0], columns=row.columns)
    row_df = pd.concat([aux_row, row]).reset_index(drop=True)
    
    future_covariates_ts = TimeSeries.from_dataframe(row_df[features_cols])[-1:]
    future_covariates_ts_scaled = future_scaler.transform(future_covariates_ts)
    
    # Compute SHAP values and prediction
    shap_explain = ShapExplainer(model=model)
    shap_explained = shap_explain.explain(aux_nps_ts, foreground_future_covariates=future_covariates_ts_scaled)
    shap_explanation = shap_explained.get_shap_explanation_object(horizon=1)

    shap_values = shap_explanation[0].values
    base_value = shap_explanation[0].base_values
    pred_value = base_value + shap_values.sum()
    feature_names=[]
    for feat in shap_explanation.feature_names:
        name = [f for f in features_cols if f in feat]
        feature_names.append(name[0])
    
    
    # Convert SHAP values to a dictionary and adjust the logic based on your ShapExplainer
    shap_values_dict = {f"{feature}_nps": value for feature, value in zip(feature_names, shap_values)}
    shap_values_dict["out_prob_base"] = base_value,
    shap_values_dict["out_prob_nps"] = pred_value,
    
    return shap_values_dict

def compute_shap_and_prediction(row, model, features_cols):
    """
    Computes SHAP values and the predicted NPS for a given row using a LightGBM model.
    
    Parameters:
    - row: The DataFrame row for which to compute SHAP values and prediction.
    - model: The trained LightGBM model used for prediction.
    - features_cols: List of column names representing features used by the model.
    
    Returns:
    - A dictionary containing SHAP values and predicted output in the desired format.
    """
    
    # Prepare the row for prediction
    row_df = row[features_cols].values.reshape(1, -1)  # Prepare the row for prediction
    
    # Make the prediction using the model
    pred_value = model.predict(row_df)[0]  # Predict using LightGBM
    
    # Compute SHAP values
    explainer = shap.Explainer(model)  # SHAP explainer for LightGBM
    shap_values = explainer(row_df)  # Compute SHAP values for the given row
    
    # Extract SHAP values and base value
    shap_values_array = shap_values.values[0]  # SHAP values for each feature
    base_value = shap_values.base_values[0]  # Base value from the SHAP explanation
    
    # Adjust the prediction based on SHAP values
    feature_contributions = shap_values_array.sum()
    adjusted_prediction = base_value + feature_contributions
    
    # Build the SHAP values dictionary with the specified names
    shap_values_dict = {f"{feature}_nps": value for feature, value in zip(features_cols, shap_values_array)}
    shap_values_dict["out_prob_base"] = base_value
    shap_values_dict["pred_nps"] = pred_value
    shap_values_dict["out_prob_nps"] = adjusted_prediction
    
    return shap_values_dict


if __name__ == "__main__":
    """Main functionality of the script."""

    # Log the start of the step
    SAGEMAKER_LOGGER.info("userlog: Starting %s step...", STEP)

    # Initialize the Arguments class and get the arguments
    arguments = Arguments()
    arguments.info(logger=SAGEMAKER_LOGGER)
    args = arguments.get_arguments()

    # Extract the argument values
    S3_BUCKET = args.s3_bucket
    S3_PATH_WRITE = args.s3_path_write
    STR_EXECUTION_DATE = args.str_execution_date
    IS_LAST_DATE = args.is_last_date

    # Parse date from STR_EXECUTION_DATE
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]

    # Load the configuration data
    config = utils.read_config_data()
    
    keys = list(config['VARIABLES_AGG_PREDICT']['CABIN_HAULS'])
    model_names = list(config['VARIABLES_AGG_PREDICT']['MODEL_NAME'])
    scaler_names = list(config['VARIABLES_AGG_PREDICT']['SCALER_NAME'])
    features = list(config['VARIABLES_AGG_PREDICT']['FEATURES'])
    cols_to_save = list(config['VARIABLES_AGG_PREDICT']['COLUMNS_SAVE'])

    # Initialize boto3 S3 client
    s3 = boto3.client('s3')

    # Define the paths for reading data and the trained model
    read_path = f"{S3_PATH_WRITE}/11_agg_preprocess_step/{year}{month}{day}"

    # Load the data to predict
    df_predict = pd.read_csv(f"s3://{S3_BUCKET}/{read_path}/data_for_prediction.csv")

    day_predict_df, day_predict_df_grouped_dfs = utils.process_dataframe(df_predict)

    # Initialize a dictionary to store the augmented DataFrames, models, and scalers
    augmented_dfs = {}
    lgbm_model = {}
    future_scalers = {}

    for key in day_predict_df_grouped_dfs.keys():
        # Initialize a list to collect augmented rows
        augmented_rows = []

        # Load the pre-trained model and scaler from S3
        path = f"customer/nps_aggregated_explainability/prod/targets_pretrained_model/2024-09-23"
        future_scaler_key = f"{path}/future_scaler_{key} (1).pkl"
        model_key = f"{path}/best_tuned_mae_model_{key}_LightGBMModel (1).pkl"
        model_key = f"{path}/LightGBM_{key}.pkl"
        

        # Load scaler
        scaler_response = s3.get_object(Bucket=S3_BUCKET, Key=future_scaler_key)
        future_scalers[key] = pickle.loads(scaler_response['Body'].read())

        # Load model
        model_response = s3.get_object(Bucket=S3_BUCKET, Key=model_key)
        lgbm_model[key] = pickle.loads(model_response['Body'].read())

        for index in range(len(day_predict_df_grouped_dfs[key])):
            # Access the row by its index using .iloc
            row_df = day_predict_df_grouped_dfs[key].iloc[[index]]

            # Compute SHAP values and predicted NPS here...
            # Assuming `compute_shap_and_prediction` is a function you'd implement
            # This function should return SHAP values as a dict and the predicted NPS
            # shap_values = compute_shap_and_prediction(row_df, key, features, future_scalers[key], lgbm_model[key])
            shap_values = compute_shap_and_prediction(row_df, lgbm_model[key], features)

            # For each feature, add its SHAP value to the row
            for feature_name, shap_value in shap_values.items():
                row_df[f'{feature_name}'] = shap_value

            # Add base value and predicted NPS columns
            # row_df['Base Value'] = shap_values['base_value']  # Adjust based on how you obtain the base value
            # row_df['Predicted NPS'] = predicted_nps
            # Append the augmented row to the list
            augmented_rows.append(row_df)


        # Concatenate all augmented rows to form the complete augmented DataFrame
        augmented_dfs[key] = pd.concat(augmented_rows).reset_index(drop=True)
        
        # Reconstruir el DataFrame original
    df = pd.concat(augmented_dfs.values())
    df.reset_index(drop=True, inplace=True)
    
    df['insert_date_ci'] = f'{year}-{month}-{day}'
    # Rename columns, add insert date and select columns to save
    df = df[cols_to_save]
    SAGEMAKER_LOGGER.info(f"userlog: {df.info()}")
    

    # Save the prediction results to S3
    save_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/12_agg_predict_explain_step/{year}{month}{day}/predictions.csv"
    SAGEMAKER_LOGGER.info("userlog: Saving information for predict step in %s.", save_path)
    df.to_csv(save_path, index=False)




### This step is going to apply a preprocesing to my 2 dataframes (surveys_data_df and lod_factor_df)
### and then is going to merge them into a single df.
### After this is done it 


from subprocess import check_call
from sys import executable

STEP = "SIMULATE"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

# General
from io import StringIO
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
import random
from functools import partial

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

from scipy.spatial.distance import euclidean

import random

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
        
        # "--str_targets": is the execution date of the script.
        parser.add_argument("--df_targets", type=str)

        # Finally, we parse the arguments and store them in the self.args property of the class.
        self.args = parser.parse_args()
        
def calculate_nps(promoters, detractors, total_responses):
    """Calcula el Net Promoter Score (NPS)."""
    if total_responses == 0:
        return np.nan
    return ((promoters - detractors) / total_responses) * 100

def calculate_weighted_nps(group_df):
    """Calcula el NPS ponderado para un grupo de datos."""
    promoters_weight = group_df.loc[group_df['nps_100'] > 8, 'monthly_weight'].sum()
    detractors_weight = group_df.loc[group_df['nps_100'] <= 6, 'monthly_weight'].sum()
    total_weight = group_df['monthly_weight'].sum()
    
    if total_weight == 0:
        return np.nan
    return (promoters_weight - detractors_weight) / total_weight * 100

# def calculate_satisfaction(df, variable):
#     """Calcula la tasa de satisfacción para una variable dada, utilizando pesos mensuales si están disponibles."""
#     # Comprobar si la columna 'monthly_weight' existe y no está completamente vacía para los datos relevantes
#     if 'monthly_weight' in df.columns and not df[df[variable].notnull()]['monthly_weight'].isnull().all():
#         # Suma de los pesos donde la variable es >= 8 y satisface la condición de estar satisfecho
#         satisfied_weight = df[df[variable] >= 8]['monthly_weight'].sum()
#         # Suma de todos los pesos donde la variable no es NaN
#         total_weight = df[df[variable].notnull()]['monthly_weight'].sum()
#         # Calcula el porcentaje de satisfacción usando los pesos
#         if total_weight == 0:
#             return np.nan
#         return (satisfied_weight / total_weight) * 100
#     else:
#         # Contar respuestas satisfechas
#         satisfied_count = df[df[variable] >= 8].shape[0]
#         # Contar total de respuestas válidas
#         total_count = df[variable].notnull().sum()
#         # Calcula el porcentaje de satisfacción usando conteo
#         if total_count == 0:
#             return np.nan
#         return (satisfied_count / total_count) * 100

def calculate_satisfaction(df, variable):
    """Calcula la tasa de satisfacción para una variable dada, utilizando pesos mensuales si están disponibles."""
    ## filter only the entries where the variable is not null
    df_aux = df[df[variable].notnull()]
    # Comprobar si la columna 'monthly_weight' existe y no está completamente vacía
    if 'monthly_weight' in df_aux.columns and not df_aux['monthly_weight'].isnull().all():
        
        satisfied_weight = np.sum(np.where(df_aux[variable] >= 8, df_aux['monthly_weight'].fillna(0), 0))
        total_weight = np.sum(df_aux['monthly_weight'])
        
        return (satisfied_weight / total_weight) * 100 if total_weight != 0 else np.nan
    
    else:

        satisfied_count = np.sum(df_aux[variable] >= 8)
        total_count = np.sum(df_aux[variable].notnull())
        
        return (satisfied_count / total_count) * 100 if total_count != 0 else np.nan


def calculate_otp(df, n):
    """Calcula el On-Time Performance (OTP) como el porcentaje de valores igual a 1."""
    on_time_count = (df[f'otp{n}_takeoff'] == 0).sum()
    total_count = df[f'otp{n}_takeoff'].notnull().sum()
    return (on_time_count / total_count) * 100 if total_count > 0 else 0


def calculate_load_factor(df, pax_column, capacity_column):
    """Calcula el factor de carga para una cabina específica."""
    total_pax = df[pax_column].sum()
    total_capacity = df[capacity_column].sum()
    # Evitar la división por cero
    if total_capacity > 0:
        return (total_pax / total_capacity) * 100
    else:
        return 0

def calculate_mean(df, variable):
    """Calcula la media de una variable dada."""
    return df[variable].mean()
    
def calculate_metrics_summary(df, start_date, end_date, touchpoints):
    # Filtrar por rango de fechas
    df_filtered = df[(df['date_flight_local'] >= pd.to_datetime(start_date)) & (df['date_flight_local'] <= pd.to_datetime(end_date))]
    
    # Mapeo de cabinas a columnas de pax y capacidad
    cabin_mapping = {
        'Economy': ('pax_economy', 'capacity_economy'),
        'Business': ('pax_business', 'capacity_business'),
        'Premium Economy': ('pax_premium_ec', 'capacity_premium_ec')
    }
    
    results_list = []
    
    for (cabin, haul), group_df in df_filtered.groupby(['cabin_in_surveyed_flight', 'haul']):
        
        print(f'CABIN/HAUL: {cabin}/{haul}')
        result = {
            'start_date': start_date,
            'end_date': end_date,
            'cabin_in_surveyed_flight': cabin,
            'haul': haul,
            'ticket_price': group_df['ticket_price'].mean(),
            # 'otp15_takeoff': calculate_otp(group_df, 15),
            # 'mean_delay': group_df[group_df['delay_departure']>0]['delay_departure'].mean()
        }
        
        # Calcula el NPS para el grupo
        promoters = (group_df['nps_100'] >= 9).sum()
        detractors = (group_df['nps_100'] <= 6).sum()
        total_responses = group_df['nps_100'].notnull().sum()
        result['NPS'] = calculate_nps(promoters, detractors, total_responses) if total_responses else None
        
        # Calcula el NPS ponderado para el grupo
        result['NPS_weighted'] = calculate_weighted_nps(group_df)
        
        # Satisfacción para cada touchpoint
        for tp in touchpoints:
            result[f'{tp}_satisfaction'] = calculate_satisfaction(group_df, tp)
            
        
        # Calcula el factor de carga para la cabina
        pax_column, capacity_column = cabin_mapping.get(cabin, (None, None))
        if pax_column and capacity_column:
            result['load_factor'] = calculate_load_factor(group_df, pax_column, capacity_column)
        
        results_list.append(result)
    
    return pd.DataFrame(results_list)

import time

def soft_manual_sim_causal(df, targets, touchpoints, df_original, threshold=0.05):
    """
    Adjusts customer scores in the DataFrame to meet satisfaction targets.
    Ensures final satisfactions are within desired intervals based on the necessary change.
    Args:
        df (pd.DataFrame): Adjusted DataFrame after previous modifications.
        targets (pd.DataFrame): DataFrame with satisfaction targets.
        touchpoints (list): List of touchpoints to adjust.
        df_original (pd.DataFrame): Original DataFrame before any adjustments.
        threshold (float): Threshold to consider when adjustments have reached the target.
    Returns:
        pd.DataFrame: Adjusted DataFrame.
    """
    df_adjusted = df.copy()
    adjusted_clients_count = 0
    sum_differences = 0  # To store the sum of differences

    # Track the start time
    start_time = time.time()

    # Calculate original satisfactions from the original DataFrame
    original_satisfactions = {}
    for variable in targets.columns:
        target = targets[variable].values[0]
        if variable.endswith('_satisfaction'):
            touchpoint = variable.replace('_satisfaction', '')
            if touchpoint in touchpoints:
                original_satisfaction = calculate_satisfaction(df_original, touchpoint)
                original_satisfactions[variable] = original_satisfaction
                sum_differences += original_satisfaction - target

    for variable in targets.columns:
        # Break if the execution time exceeds 10 minutes
        if time.time() - start_time > 300:
            print("Execution stopped: exceeded time limit of 10 minutes.")
            break

        target = targets[variable].values[0]
        print(variable, target)
        if variable.endswith('_satisfaction'):
            touchpoint = variable.replace('_satisfaction', '')
            if touchpoint in touchpoints:
                current_satisfaction = calculate_satisfaction(df_adjusted, touchpoint)
                original_satisfaction = original_satisfactions[variable]

                # Determine desired interval based on whether satisfaction needed to increase or decrease
                if original_satisfaction < target:
                    # Satisfaction needed to increase
                    lower_bound = target
                    upper_bound = target + threshold
                elif original_satisfaction > target:
                    # Satisfaction needed to decrease
                    lower_bound = target - threshold
                    upper_bound = target
                else:
                    # Satisfaction was equal to target
                    if sum_differences > 0:
                        # Adjust upwards
                        lower_bound = original_satisfaction
                        upper_bound = original_satisfaction + threshold
                    else:
                        # Adjust downwards
                        lower_bound = original_satisfaction - threshold
                        upper_bound = original_satisfaction

                # Adjust satisfaction upwards
                if current_satisfaction < lower_bound:
                    for value in [7, 6, 5, 4, 3, 2, 1]:
                        while current_satisfaction < lower_bound:
                            # Break if the execution time exceeds 10 minutes
                            if time.time() - start_time > 600:
                                print("Execution stopped: exceeded time limit of 10 minutes.")
                                break

                            to_adjust = df_adjusted[df_adjusted[touchpoint] == value]
                            if to_adjust.empty:
                                break
                            to_adjust_sample = to_adjust.sample(n=1)
                            df_adjusted.loc[to_adjust_sample.index, touchpoint] = 8
                            adjusted_clients_count += 1
                            current_satisfaction = calculate_satisfaction(df_adjusted, touchpoint)
                            if current_satisfaction >= lower_bound:
                                break
                        if current_satisfaction >= lower_bound:
                            break

                # Adjust satisfaction downwards
                elif current_satisfaction > upper_bound:
                    for value in [8, 9, 10]:
                        while current_satisfaction > upper_bound:
                            # Break if the execution time exceeds 10 minutes
                            if time.time() - start_time > 600:
                                print("Execution stopped: exceeded time limit of 10 minutes.")
                                break

                            to_adjust = df_adjusted[df_adjusted[touchpoint] == value]
                            if to_adjust.empty:
                                break
                            to_adjust_sample = to_adjust.sample(n=1)
                            df_adjusted.loc[to_adjust_sample.index, touchpoint] = 7
                            adjusted_clients_count += 1
                            current_satisfaction = calculate_satisfaction(df_adjusted, touchpoint)
                            if current_satisfaction <= upper_bound:
                                break
                        if current_satisfaction <= upper_bound:
                            break

                else:
                    print(f"Current satisfaction for {touchpoint} is within the desired interval.")

            else:
                print(f"Touchpoint {touchpoint} not in touchpoints list.")

        else:
            # For non-satisfaction variables (e.g., mean scores)
            current_mean = calculate_mean(df_adjusted, variable)
            original_mean = calculate_mean(df_original, variable)
            if original_mean < target:
                desired_range = (target, target + threshold)
            elif original_mean > target:
                desired_range = (target - threshold, target)
            else:
                if sum_differences > 0:
                    desired_range = (original_mean, original_mean + threshold)
                else:
                    desired_range = (original_mean - threshold, original_mean)

            if current_mean < desired_range[0]:
                while current_mean < desired_range[0]:
                    # Break if the execution time exceeds 10 minutes
                    if time.time() - start_time > 600:
                        print("Execution stopped: exceeded time limit of 10 minutes.")
                        break

                    n = desired_range[0] - current_mean
                    adjustment_needed = min(n, threshold)
                    df_adjusted[variable] += adjustment_needed
                    adjusted_clients_count += 1
                    current_mean = calculate_mean(df_adjusted, variable)
            elif current_mean > desired_range[1]:
                while current_mean > desired_range[1]:
                    # Break if the execution time exceeds 10 minutes
                    if time.time() - start_time > 600:
                        print("Execution stopped: exceeded time limit of 10 minutes.")
                        break

                    n = current_mean - desired_range[1]
                    adjustment_needed = min(n, threshold)
                    df_adjusted[variable] -= adjustment_needed
                    adjusted_clients_count += 1
                    current_mean = calculate_mean(df_adjusted, variable)
            else:
                print(f"Current mean for {variable} is within the desired interval.")

        print(f"Total clients adjusted: {adjusted_clients_count}")

    return df_adjusted



def hard_manual_sim_rand_cluster_causal(df, targets, touchpoints, df_original, threshold=0.05, num_clients_per_iter=1):
    """
    Adjusts customer scores in the DataFrame to meet satisfaction targets
    by changing multiple touchpoints for a group of clients in each iteration.
    Ensures final satisfactions are within desired intervals based on the necessary change.
    Args:
        df (pd.DataFrame): Adjusted DataFrame after previous modifications.
        targets (pd.DataFrame): DataFrame with satisfaction targets.
        touchpoints (list): List of touchpoints to adjust.
        df_original (pd.DataFrame): Original DataFrame before any adjustments.
        threshold (float): Threshold to consider when adjustments have reached the target.
        num_clients_per_iter (int): Number of clients to select in each iteration.
    Returns:
        pd.DataFrame: Adjusted DataFrame.
    """
    df_adjusted = df.copy()
    df_adjusted['flag_reached'] = 0
    selected_clients = set()

    variables_to_increase = []
    variables_to_decrease = []

    adjusted_clients_count = 0
    sum_differences = 0  # To store the sum of differences
    original_satisfactions = {}

    # Calculate original satisfactions from the original DataFrame
    for variable in targets.columns:
        target = targets[variable].values[0]

        if variable.endswith('_satisfaction'):
            touchpoint = variable.replace('_satisfaction', '')
            if touchpoint in touchpoints:
                original_satisfaction = calculate_satisfaction(df_original, touchpoint)
                original_satisfactions[variable] = original_satisfaction
                sum_differences += original_satisfaction - target

                if original_satisfaction < target:
                    variables_to_increase.append((touchpoint, target))
                elif original_satisfaction > target:
                    variables_to_decrease.append((touchpoint, target))
                else:
                    # Satisfaction was equal to target
                    if sum_differences > 0:
                        # Need to increase within [original, original + threshold]
                        variables_to_increase.append((touchpoint, original_satisfaction + threshold))
                    else:
                        # Need to decrease within [original - threshold, original]
                        variables_to_decrease.append((touchpoint, original_satisfaction - threshold))
                        
    # Ordenar las variables según el orden en touchpoints
    variables_to_increase.sort(key=lambda x: touchpoints.index(x[0]))
    variables_to_decrease.sort(key=lambda x: touchpoints.index(x[0]))
    
    # Increase scores
    while variables_to_increase:
        # if not df_adjusted[df_adjusted['cluster'] == 2].empty:
        #     available_indices = df_adjusted[df_adjusted['cluster'] == 2].index.difference(selected_clients)
        #     if not available_indices.empty:
        #         to_adjust_samples = df_adjusted.loc[available_indices].sample(n=min(num_clients_per_iter, len(available_indices)))
        #     else:
        #         to_adjust_samples = df_adjusted.sample(n=num_clients_per_iter)
        # else:
        #     to_adjust_samples = df_adjusted.sample(n=num_clients_per_iter)
            
        available_indices = df_adjusted.index.difference(selected_clients)

        # Seleccionar un grupo de clientes que no hayan sido seleccionados previamente
        to_adjust_samples = df_adjusted.loc[available_indices].sample(n=min(num_clients_per_iter, len(available_indices)))
            
        

        selected_clients.update(to_adjust_samples.index)

        for value in [7, 6, 5, 4, 3, 2, 1, 0]:
            adjusted = False
            for touchpoint, target in variables_to_increase.copy():
                current_satisfaction = calculate_satisfaction(df_adjusted, touchpoint)
                if current_satisfaction >= target:
                    print(f"Variable {touchpoint} has reached the target satisfaction.")
                    variables_to_increase.remove((touchpoint, target))
                    continue

                mask = (to_adjust_samples[touchpoint] == value) | (to_adjust_samples[touchpoint] == value - 1)
                if mask.any():
                    new_value = random.randint(8, 10)
                    df_adjusted.loc[to_adjust_samples[mask].index, touchpoint] = new_value
                    adjusted = True
            if adjusted:
                adjusted_clients_count += len(to_adjust_samples[mask])
                break

    # Decrease scores
    while variables_to_decrease:
        # if not df_adjusted[df_adjusted['cluster'] == 2].empty:
        #     available_indices = df_adjusted[df_adjusted['cluster'] == 2].index.difference(selected_clients)
        #     if not available_indices.empty:
        #         to_adjust_samples = df_adjusted.loc[available_indices].sample(n=min(num_clients_per_iter, len(available_indices)))
        #     else:
        #         to_adjust_samples = df_adjusted.sample(n=num_clients_per_iter)
        # else:
        #     to_adjust_samples = df_adjusted.sample(n=num_clients_per_iter)
        
        available_indices = df_adjusted.index.difference(selected_clients)
        # Seleccionar un grupo de clientes que no hayan sido seleccionados previamente
        to_adjust_samples = df_adjusted.loc[available_indices].sample(n=min(num_clients_per_iter, len(available_indices)))

        selected_clients.update(to_adjust_samples.index)
        

        for value in [8, 9, 10]:
            adjusted = False
            for touchpoint, target in variables_to_decrease.copy():
                current_satisfaction = calculate_satisfaction(df_adjusted, touchpoint)
                if current_satisfaction <= target:
                    print(f"Variable {touchpoint} has reached the target satisfaction.")
                    variables_to_decrease.remove((touchpoint, target))
                    continue

                mask = (to_adjust_samples[touchpoint] == value) | (to_adjust_samples[touchpoint] == value + 1)
                if mask.any():
                    new_value = random.randint(0, 7)
                    df_adjusted.loc[to_adjust_samples[mask].index, touchpoint] = new_value
                    adjusted = True
            if adjusted:
                adjusted_clients_count += len(to_adjust_samples[mask])
                break

    print(f"Total clients adjusted: {adjusted_clients_count}")

    # Final verification
    verification_results = {}
    for variable in targets.columns:
        if variable.endswith('_satisfaction'):
            touchpoint = variable.replace('_satisfaction', '')
            if touchpoint in touchpoints:
                final_satisfaction = calculate_satisfaction(df_adjusted, touchpoint)
                verification_results[variable] = (final_satisfaction, targets[variable].values[0])
        else:
            final_mean = calculate_mean(df_adjusted, variable)
            verification_results[variable] = (final_mean, targets[variable].values[0])

    for var, (final, target) in verification_results.items():
        print(f"Variable: {var}, Final: {final:.2f}, Target: {target:.2f}, Met: {abs(final - target) <= threshold}")

    return df_adjusted
    
def calculate_total_distance(df, targets, touchpoints):
    """
    Calcula la distancia total entre las satisfacciones actuales y los targets utilizando pesos suaves.
    
    Args:
    df (pd.DataFrame): DataFrame con las puntuaciones de los clientes.
    targets (pd.DataFrame): DataFrame con una fila que contiene los targets de satisfacción.
    touchpoints (list): Lista de touchpoints a considerar.
    
    Returns:
    float: Distancia total ponderada.
    """
    current_satisfaction = np.array([calculate_satisfaction(df, tp) for tp in touchpoints])
    target_values = targets[[f'{tp}_satisfaction' for tp in touchpoints]].to_numpy().flatten()
    
    ## get the euclidean distance between the two arrays
    total_distance = np.linalg.norm(current_satisfaction - target_values)
    
    return total_distance

def causal_swapping(
    df, 
    targets, 
    touchpoints, 
    max_iterations=None, 
    threshold=1, 
    patience=None, 
    distance_threshold=1,
    focus_feat=None, 
    auto_focus_feat=True, 
    cycle_limit=5, 
    target_len=None, 
    n=1,
    df_replacement=None,
    graph_convergence=False,
):
    """
    Adjusts the customer population by swapping clients between clusters until targets are met or no further improvement.
    Ensures final satisfactions are within desired intervals based on the necessary change.
    Args:
        df (pd.DataFrame): DataFrame with customer scores.
        targets (pd.DataFrame): DataFrame with satisfaction targets.
        touchpoints (list): List of touchpoints to adjust.
        df_original (pd.DataFrame): Original DataFrame before any adjustments.
        max_iterations (int): Maximum number of iterations.
        threshold (float): Threshold to consider when adjustments have reached the target.
        patience (int): Number of additional iterations after the distance stops improving.
    Returns:
        pd.DataFrame: Adjusted DataFrame.
    """

    ##################################################################################################################################################################################
    ## HEADER
    ##################################################################################################################################################################################

    iteration = 0
    patience_counter = 0
    cycle_counter = 0
    convergence_flag = False
    cycle_results = {}
    distance_history = []
    
    if target_len:
        print("Reshaping...")
        calculate_total_distance_partial = partial(calculate_total_distance, targets=targets, touchpoints=touchpoints)
        df_original, _ = reshape_dataframe(df, target_len, calculate_total_distance_partial)
        print(f"The original {len(df)}-rows dataframe has been reshaped to {len(df_original)} rows.")
        print(f"Within the reshape, distance has change from {calculate_total_distance(df, targets, touchpoints)} to { calculate_total_distance(df_original, targets, touchpoints)}")
        
    else:
        df_original = df.copy()
    
    df_current = df_original.copy()
    df_original['swapped'] = False
    df_current['swapped'] = False

    original_satisfactions = {tp: calculate_satisfaction(df_original, tp) for tp in touchpoints}
    initial_distance = calculate_total_distance(df_original, targets, touchpoints)
    print("Initial distance between dataframe satisfactions and targets:", initial_distance)

    ## display initial satisfactions in a dataframe
    initial_conditions_df = pd.DataFrame({
        'Original Satisfaction': [original_satisfactions[tp] for tp in touchpoints],
        'Lower Bound': [targets[f'{tp}_satisfaction'].values[0] - threshold for tp in touchpoints],
        'Target': [targets[f'{tp}_satisfaction'].values[0] for tp in touchpoints],
        'Upper Bound': [targets[f'{tp}_satisfaction'].values[0] + threshold for tp in touchpoints],
    }, index=touchpoints).T
    
    display(round(initial_conditions_df, 3))
    
    target_intervals = initial_conditions_df.loc[['Lower Bound', 'Upper Bound']].to_dict('list')
    target_values = initial_conditions_df.loc['Target'].values.flatten()
    current_satisfactions = np.array([calculate_satisfaction(df_current, tp) for tp in touchpoints])
    
    ##################################################################################################################################################################################
    ## AUTO-SET PARAMETERS
    ##################################################################################################################################################################################
    
    if auto_focus_feat and not focus_feat: 
        ## check if any touchpoint has an initial difference of more than 50
        large_diff_touchpoints = [
            tp for tp, satisfaction in original_satisfactions.items()
            if abs(satisfaction - targets[f'{tp}_satisfaction'].values[0]) > 30
        ]
        ## if there is only one touchpoint in the list, it is seat as the focus_feat
        if len(large_diff_touchpoints) == 1:
            focus_feat = large_diff_touchpoints[0]
            print('Auto-focus feat detected:', focus_feat)
    current_focus_feat = focus_feat

    ## auto set parameters patience and max_iterations when not defined
    population_size = len(df_original)
    if population_size > 0:
        if not patience:
            patience = 1000 if population_size < 1000 else 500 + 0.5 * population_size
            print(f"Parameter patience has been set automatically to {patience}. Population size: {population_size}")
        if not max_iterations:
            max_iterations = 1000 if population_size < 1000 else population_size
            print(f"Parameter max_iterations has been set automatically to {max_iterations}. Population size: {population_size}")        
    
    print("\nHere we start the loop...\n")

    ##################################################################################################################################################################################
    ## LOOP
    ##################################################################################################################################################################################

    current_distance = initial_distance  
    while cycle_counter < cycle_limit:
                
        adjustments_made = False
        mean_difference = np.mean(current_satisfactions - target_values)
        sum_difference = np.sum(np.abs(current_satisfactions - target_values))
        
        if iteration % 250 == 0: ##  
            print("Iteration number: ", iteration)
            print("Mean difference: ", mean_difference)
            print("Sum difference: ", sum_difference)
        
        df_test = df_current.copy()

        ##################################################################################################################################################################################
        ## FOCUS ITERATION
        ##################################################################################################################################################################################
        if current_focus_feat:
            print(f"Performing a focused iteration ({iteration})...")
            
            focus_difference = (calculate_satisfaction(df_current, current_focus_feat) - targets[f'{current_focus_feat}_satisfaction']).values[0]
            focusly_adjusted_clients = df_current['swapped'].sum()
            non_null_focus_feat = df_current[current_focus_feat].notnull().sum()
            
            print("Focus distance is: ", focus_difference)            
            print('Focusly adjusted clients:', focusly_adjusted_clients)
            print('Non-null entries on the focus feat:', non_null_focus_feat)

            focus_condition_1 = abs(focus_difference) > 0.2 * abs(sum_difference)
            focus_condition_2 = focusly_adjusted_clients < non_null_focus_feat
            focus_condition_3 = patience_counter < non_null_focus_feat - focusly_adjusted_clients
            
            if focus_condition_1 and focus_condition_2 and focus_condition_3:            
                adjustments_made, df_test = focus_swap_client(df_test, touchpoints=touchpoints, focus_feat=current_focus_feat, increase=(focus_difference < 0))                    
            else:
                ## there is no longer necessary to focus on that feature
                current_focus_feat = False
        
        ##################################################################################################################################################################################
        ## NORMAL ITERATION
        ##################################################################################################################################################################################                
        else:
            if mean_difference < 0:
                ## if the mean difference is negative, increase satisfaction (swap a client from level 0 to level 2)
                adjustments_made, df_test = simple_swap_client(df_test, remove_from_cluster=0, duplicate_from_cluster=2, touchpoints=touchpoints, n=n, df_replacement=df_replacement)
                
            elif mean_difference > 0:
                ## if the mean difference is positive, decrease satisfaction (swap a client from level 2 to level 0)
                adjustments_made, df_test = simple_swap_client(df_test, remove_from_cluster=2, duplicate_from_cluster=0, touchpoints=touchpoints, n=n, df_replacement=df_replacement)
        
        ## if an adjustment was made, recalculate the distance
        if adjustments_made:

            new_distance = calculate_total_distance(df_test, targets, touchpoints)

            if new_distance < current_distance:
                ## if the distance is smaller, we copy the adjustment to 'df_current' and recalculate satisfactions 

                df_current = df_test.copy()

                current_satisfactions = np.array([calculate_satisfaction(df_current, tp) for tp in touchpoints])                
                current_distance = new_distance
                patience_counter = 0
                if iteration % 250 == 0: # 
                    print(f"Distance after adjustment: {new_distance:.4f}. Mininum distance found so far in this cycle: {current_distance:.4f}")
                    print(f"Efficient iteration!")
                
            else:
                patience_counter += 1
                if iteration % 250 == 0: #
                    print(f"Distance after adjustment: {new_distance:.4f}. Mininum distance found so far in this cycle: {current_distance:.4f}")
                    print(f"Unnefficient iteration ({iteration}). Patience counter increased to: {patience_counter}")
                    
            distance_history.append(current_distance)
            
        else:
            print(f"No adjustment was made at iteration {iteration}. Probably the algorithm is switching from focus to normal mode.")

        ##################################################################################################################################################################################
        ## STOPPERS
        ##################################################################################################################################################################################
        if new_distance < distance_threshold:
            print(f"New_distance < {distance_threshold}. Stopping.")
            convergence_flag = True
            break        
        elif all_within_limits(current_satisfactions, target_intervals, touchpoints):
            print(f"All satisfactions are within the target intervals at iteration {iteration}. Stopping.")
            convergence_flag = True
            break
            
        elif patience_counter >= patience or iteration > max_iterations:

            if patience_counter >= patience:
                print(f"\n The algorithm has run out of patience at iteration {iteration} with distance {current_distance:.4f}. End of cycle {cycle_counter}. \n\n\n")
            else:
                print(f"\n The algorithm has reached maximun iteration ({iteration}) with distance {current_distance:.4f}. End of cycle {cycle_counter}. \n\n\n") 

            ## save this cycle results
            cycle_results[cycle_counter] = [df_current.copy(), current_distance, distance_history]
            ## reset all variables
            cycle_counter += 1
            patience_counter = 0
            iteration = 0
            df_current = df_original.copy()
            current_distance = initial_distance
            current_focus_feat = focus_feat
            distance_history = []
        else:    
            iteration += 1
        
    ##################################################################################################################################################################################
    ## OUTPUT
    ##################################################################################################################################################################################

    if convergence_flag:
        best_cycle, best_df, best_distance, best_distance_history = cycle_counter, df_current, current_distance, distance_history
        print(f"Converge reached at cycle {best_cycle} with a distance of {best_distance:.4f}.")
    else:
        ## get best cycle results
        best_cycle, (best_df, best_distance, best_distance_history) = min(cycle_results.items(), key=lambda x: x[1][1])
        print(f"Converge not reached. Best cycle was {best_cycle} with a distance of {best_distance:.4f}.")

    ## display final satisfactions in a dataframe
    final_satisfactions = {tp: calculate_satisfaction(best_df, tp) for tp in touchpoints}
    final_conditions_df = pd.DataFrame({
        'Final Satisfactions': [final_satisfactions[tp] for tp in touchpoints],
        'Lower Bound': [targets[f'{tp}_satisfaction'].values[0] - threshold for tp in touchpoints],
        'Target': [targets[f'{tp}_satisfaction'].values[0] for tp in touchpoints],
        'Upper Bound': [targets[f'{tp}_satisfaction'].values[0] + threshold for tp in touchpoints],
        'Original number of nulls': df_original.isnull().sum()[touchpoints],
        'Final number of nulls': best_df.isnull().sum()[touchpoints],
    }, index=touchpoints).T  
    display(round(final_conditions_df, 3))
    
    ## plot the convergence curve in the best cycle
    if graph_convergence:
        distance_graph(best_distance_history)
    
    return best_df

def simple_swap_client(df_adjusted, remove_from_cluster, duplicate_from_cluster, touchpoints, n=1, df_replacement=None):
    
    clients_to_remove = df_adjusted[df_adjusted['cluster'] == remove_from_cluster]
    
    if df_replacement is not None:
        clients_to_duplicate = df_replacement[df_replacement['cluster'] == duplicate_from_cluster]
    else:
        clients_to_duplicate = df_adjusted[df_adjusted['cluster'] == duplicate_from_cluster]
    
    if not clients_to_remove.empty and not clients_to_duplicate.empty:
            
        ## select a client from the 'clients_to_remove' dataframe to be removed
        client_to_remove = clients_to_remove.sample(n=n)
        df_adjusted.drop(client_to_remove.index, inplace=True)

        ## select a client from the 'clients_to_duplicate' dataframe to be duplicated
        client_to_duplicate = clients_to_duplicate.sample(n=n)
        df_adjusted = pd.concat([df_adjusted, client_to_duplicate], ignore_index=True)
        
        return True, df_adjusted
    return False, df_adjusted

def focus_swap_client(df_adjusted, touchpoints, focus_feat, increase=True):
    if not df_adjusted.empty:
        min_focus_feat = df_adjusted[focus_feat].min()
        max_focus_feat = df_adjusted[focus_feat].max()
        
        # print("Min:", min_focus_feat, "Max: ", max_focus_feat)
        # display(df_adjusted.nlargest(6, focus_feat)[['respondent_id', focus_feat, 'swapped']])
        # display(df_adjusted.nsmallest(6, focus_feat)[['respondent_id', focus_feat, 'swapped']])
        
        if increase:
            ## drop the client with the smallest value in focus_feat
            client_to_remove = df_adjusted[df_adjusted[focus_feat] == min_focus_feat].sample(n=1)
            df_adjusted.drop(client_to_remove.index, inplace=True)

            ## duplicate the client withe the bigger value in focus_feat
            client_to_add = df_adjusted[df_adjusted[focus_feat] == max_focus_feat].sample(n=1)
            df_adjusted = pd.concat([df_adjusted, client_to_add], ignore_index=True)
            df_adjusted.loc[df_adjusted['respondent_id'] == client_to_add['respondent_id'].values[0], 'swapped'] = True
        else:
            ## do the opposite
            client_to_remove = df_adjusted[df_adjusted[focus_feat] == max_focus_feat].sample(n=1)
            df_adjusted.drop(client_to_remove.index, inplace=True)
            
            client_to_add = df_adjusted[df_adjusted[focus_feat] == min_focus_feat].sample(n=1)
            df_adjusted = pd.concat([df_adjusted, client_to_add], ignore_index=True)
            df_adjusted.loc[df_adjusted['respondent_id'] == client_to_add['respondent_id'].values[0], 'swapped'] = True

        return True, df_adjusted
    return False, df_adjusted


def all_within_limits(satisfactions, intervals, touchpoints):
    return all(
        intervals[tp][0] <= satisfactions[i] <= intervals[tp][1]
        for i, tp in enumerate(touchpoints)
    )

def distance_graph(distance_history):
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12)) 

    ## first graph
    ax1.plot(distance_history, label='Distancia Total a Objetivo', color='b')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Distancia a los Objetivos de Satisfacción')
    ax1.set_title('Convergencia del Algoritmo de Ajuste de Satisfacción (Escala Completa)')
    ax1.legend()
    ax1.grid(True)
    
    ## second graph
    ax2.plot(distance_history, label='Distancia Total a Objetivo (Escala Limitada)', color='g')
    ax2.set_ylim(0, 5)
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Distancia a los Objetivos de Satisfacción')
    ax2.set_title('Convergencia del Algoritmo de Ajuste de Satisfacción (Escala Limitada)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
def reshape_dataframe(df, target_len, eval_function, iterations=1000, random_state=42):
    """
    Optimiza el número de filas de un DataFrame para minimizar una función de evaluación.
    
    Parámetros:
    - df: DataFrame original a modificar.
    - target_len: Número deseado de filas en el DataFrame final.
    - eval_function: Función a minimizar que toma un DataFrame y devuelve un valor.
    - iterations: Número de intentos para encontrar el mejor DataFrame. (Default: 100)
    - random_state: Semilla para reproducibilidad de resultados.
    
    Retorna:
    - best_df: El DataFrame que obtuvo el mejor puntaje en la evaluación.
    - best_score: El puntaje obtenido con el mejor DataFrame.
    """

    np.random.seed(random_state)
        
    best_df = None
    best_score = float('inf')
    
    actual_len = len(df)    
    if actual_len == target_len:
        return df, best_score

    increase = actual_len < target_len
    
    for _ in range(iterations):
    
        extra_rows = target_len - len(df)
        if increase:
            rows_to_duplicate = np.random.choice(df.index, extra_rows, replace=True)
            modified_df = pd.concat([df, df.iloc[rows_to_duplicate]], ignore_index=True)
        else:
            rows_to_drop = np.random.choice(df.index, abs(extra_rows), replace=False)
            modified_df = df.drop(rows_to_drop).reset_index(drop=True)
        
        score = eval_function(modified_df)

        if score < best_score:
            best_score, best_df = score, modified_df
    
    return best_df, best_score

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
    DF_TARGETS = args.df_targets
    
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
    columns_to_save = config.get("VARIABLES_SIMULATE").get('COLUMNS_TO_SAVE')
    touchpoints = config.get("VARIABLES_SIMULATE").get('TOUCHPOINTS')
    headers = config.get("VARIABLES_SIMULATE").get('HEADERS_TARGETS')
    
    # If the DataFrame was passed as a CSV string
    targets = pd.read_csv(StringIO(DF_TARGETS), header=None)
    
    # Asignar los encabezados al DataFrame
    targets.columns = headers
    
    SAGEMAKER_LOGGER.info("userlog: Targets are...", targets)
    
    # path
    src_path_characterized = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/015_predict_explain_ori_step/{year}{month}{day}/predictions.csv"
    src_path_replacement = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/015_predict_explain_ori_step/{year}{month}{day}/sampled_predictions.csv"
    out_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/02_simulate_step/{year}{month}{day}"
    

    df = pd.read_csv(src_path_characterized)
    
    # Identificar las columnas en el DataFrame targets que tienen valores faltantes
    missing_targets = targets.columns[targets.isna().any()].tolist()

    # Llenar los valores faltantes utilizando la función Calculate_satisfaction
    for variable in missing_targets:
            SAGEMAKER_LOGGER.info("userlog: missing targets activated")
            # Calcular la satisfacción usando la función
            satisfaction_value = calculate_satisfaction(df, variable)            
            # Si el valor calculado es NaN, rellenar con -1 en ambos DataFrames
            if pd.isna(satisfaction_value):
                targets[variable] = targets[variable].fillna(-1)
                df[variable] = df[variable].fillna(-1)
            else:
                # Si no es NaN, llenar targets solamente con el valor calculado
                targets[variable] = targets[variable].fillna(satisfaction_value)
    
        
    # Assume 'df' is your original DataFrame
    # df_original = df.copy()
    ## reshape
    if len(df) < 100:
        target_len = 100
    else: 
        target_len = None
    ## replacement
    if len(df) < 2000:
        df_replacement = pd.read_csv(src_path_replacement)
    else:
        df_replacement = None

    swapped_df = causal_swapping(
        df, 
        targets=targets, 
        touchpoints=touchpoints, 
        threshold=0.25, ## fijalo tu
        distance_threshold=0.5, ## fijalo tu
        target_len=target_len,
        df_replacement=df_replacement, 
        cycle_limit=5, 
    )
    
    # Añadir la columna 'simulation_client_type' a cada DataFrame
    swaped_df['simulation_client_type'] = 'swaped_simulated'
    # Rename columns, add insert date and select columns to save
    swaped_df['insert_date_ci'] = STR_EXECUTION_DATE
    # swaped_df['model_version']=f'{model_year}-{model_month}-{model_day}'
    swaped_df = swaped_df[columns_to_save]
    swaped_df.to_csv(f"{out_path}/swaped_simulated.csv", index=False)

    # Apply the causal soft and hard simulations
    # soft_sim_df = soft_manual_sim_causal(df, targets, touchpoints, df_original)
    # hard_sim_df = hard_manual_sim_rand_cluster_causal(df, targets, touchpoints, df_original)
    
    # swaped_df[columns_to_save].to_csv(f"{out_path}/swaped_simulated.csv", index=False)
    # soft_sim_df[columns_to_save].to_csv(f"{out_path}/soft_simulated.csv", index=False)
    # hard_sim_df[columns_to_save].to_csv(f"{out_path}/hard_simulated.csv", index=False)



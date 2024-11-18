from subprocess import check_call
from sys import executable

STEP = "AGG_PREPROCESS"

check_call([executable, "-m", "pip", "install", "-r", f"./{STEP.lower()}.txt"])

import pandas as pd
import logging
from datetime import datetime
import argparse
import boto3
from io import StringIO
import utils



# Configuring the logger
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
        parser.add_argument("--s3_bucket", type=str)
        parser.add_argument("--s3_path_read", type=str)
        parser.add_argument("--s3_path_write", type=str)
        parser.add_argument("--str_execution_date", type=str)
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

def calculate_satisfaction(df, variable):
    """Calcula la tasa de satisfacción para una variable dada, utilizando pesos mensuales si están disponibles."""
    # Comprobar si la columna 'monthly_weight' existe y no está completamente vacía para los datos relevantes
    if 'monthly_weight' in df.columns and not df[df[variable].notnull()]['monthly_weight'].isnull().all():
        # Suma de los pesos donde la variable es >= 8 y satisface la condición de estar satisfecho
        satisfied_weight = df[df[variable] >= 8]['monthly_weight'].sum()
        # Suma de todos los pesos donde la variable no es NaN
        total_weight = df[df[variable].notnull()]['monthly_weight'].sum()
        # Calcula el porcentaje de satisfacción usando los pesos
        if total_weight == 0:
            return np.nan
        return (satisfied_weight / total_weight) * 100
    else:
        # Contar respuestas satisfechas
        satisfied_count = df[df[variable] >= 8].shape[0]
        # Contar total de respuestas válidas
        total_count = df[variable].notnull().sum()
        # Calcula el porcentaje de satisfacción usando conteo
        if total_count == 0:
            return np.nan
        return (satisfied_count / total_count) * 100




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

    
def calculate_aggregated_metrics(df, touchpoints):
    df_filtered = df.copy()
    # Mapeo de cabinas a columnas de pax y capacidad
    cabin_mapping = {
        'Economy': ('pax_economy', 'capacity_economy'),
        'Business': ('pax_business', 'capacity_business'),
        'Premium Economy': ('pax_premium_ec', 'capacity_premium_ec')
    }
    
    results_list = []
    
    for (cabin, haul), group_df in df_filtered.groupby(['cabin_in_surveyed_flight', 'haul']):
        
        # print(f'CABIN/HAUL: {cabin}/{haul}')
        result = {
            'cabin_in_surveyed_flight': cabin,
            'haul': haul,
            'ticket_price': group_df['ticket_price'].mean(),
        }
        
        # Calcula el factor de carga para la cabina
        pax_column, capacity_column = cabin_mapping.get(cabin, (None, None))
        if pax_column and capacity_column:
            result['load_factor'] = calculate_load_factor(group_df, pax_column, capacity_column)
        
        # Satisfacción para cada touchpoint
        for tp in touchpoints:
            result[f'{tp}_satisfaction'] = calculate_satisfaction(group_df, tp)           
        

        
        results_list.append(result)
    
    return pd.DataFrame(results_list)



def read_csv_from_s3(bucket_name, object_key):
    # Create a boto3 S3 client
    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    csv_string = response['Body'].read().decode('utf-8')
    return pd.read_csv(StringIO(csv_string))


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
    
    date = STR_EXECUTION_DATE.split("/")[-1].split("=")[-1].replace("-", "")
    year, month, day = date[:4], date[4:6], date[6:]

    
    src_path_filtered = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/00_etl_step/{year}{month}{day}/agg_filtered.csv"
    out_path = f"s3://{S3_BUCKET}/{S3_PATH_WRITE}/11_agg_preprocess_step/{year}{month}{day}/data_for_prediction.csv"
    
    headers = config.get("VARIABLES_SIMULATE").get('HEADERS_TARGETS')
    
    # If the DataFrame was passed as a CSV string
    targets = pd.read_csv(StringIO(DF_TARGETS), header=None)
    
    # Asignar los encabezados al DataFrame
    targets.columns = headers

    # Read data
    df = pd.read_csv(src_path_filtered)
    df['date_flight_local'] = pd.to_datetime(df['date_flight_local'])

    # Calculate aggregated metrics
    touchpoints = config.get("VARIABLES_AGG_PREPROCESS").get('TOUCHPOINTS')
    aggregated_metrics = calculate_aggregated_metrics(df, touchpoints)
    SAGEMAKER_LOGGER.info(f"userlog: Aggregated after reading: {str(aggregated_metrics.shape)}")
    
        
    targets['cabin_in_surveyed_flight'] = aggregated_metrics['cabin_in_surveyed_flight'].unique()[0]
    targets['haul'] = aggregated_metrics['haul'].unique()[0]
    
    targets['ticket_price'] = aggregated_metrics['ticket_price']
    
    # Concatenate targets with aggregated metrics
    concatenated_df = pd.concat([aggregated_metrics, targets], axis=0, ignore_index=True)
    
    # Write results to S3
    concatenated_df.to_csv(out_path, index=False)
    SAGEMAKER_LOGGER.info(f"userlog: Aggregated metrics shape: {str(concatenated_df.shape)}")
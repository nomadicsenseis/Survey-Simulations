import logging
import sagemaker
from os import chdir, pardir, sep
from os.path import dirname
from os.path import join as path_join
from os.path import realpath
from typing import Optional

from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep
from production.pipelines_code import utils
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingOutput
from sagemaker.session import get_execution_role
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.estimator import Estimator
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# Set the base directory for the project
BASE_DIR = dirname(realpath(f"{__file__}{sep}{pardir}"))
chdir(BASE_DIR)

def get_pipeline(
    region: str,  # AWS region
    pipeline_name: str,  # Name of the pipeline
    base_job_prefix: str,  # Prefix for the job names
    role: Optional[str] = None,  # IAM role
    default_bucket: Optional[str] = None,  # Default S3 bucket
    default_bucket_prefix: Optional[str] = None,  # Default S3 key
) -> Pipeline:

    # Get a Sagemaker session
    sagemaker_session = utils.get_session(region=region, default_bucket=default_bucket,
                                          default_bucket_prefix=default_bucket_prefix)

    # If the role is not provided, get the execution role
    if role is None:
        role = get_execution_role(sagemaker_session)

    # Define pipeline parameters
    processing_instance_count = ParameterInteger(name="processing_instance_count", default_value=3)
    param_str_execution_date = ParameterString(name="str_execution_date", default_value="2023-03-01")
    param_s3_bucket_nps = ParameterString(name="s3_bucket_nps", default_value="iberia-data-lake")
    param_s3_bucket_lf = ParameterString(name="s3_bucket_lf", default_value="ibdata-prod-ew1-s3-customer")
    param_s3_path_read_nps = ParameterString(name="s3_path_read_nps")
    param_s3_path_read_lf = ParameterString(name="s3_path_read_lf")
    param_s3_path_write = ParameterString(name="s3_path_write")
    param_str_start_date = ParameterString(name="str_start_date")
    param_str_end_date = ParameterString(name="str_end_date")
    param_str_cabin = ParameterString(name="str_cabin", default_value="All")
    param_str_haul = ParameterString(name="str_haul", default_value="All")
    param_str_model = ParameterString(name="str_model", default_value="All")
    param_is_last_date = ParameterString(name="is_last_date", default_value="1")
    param_trials = ParameterString(name="trials", default_value="1")
    param_df_targets = ParameterString(name="df_targets")
    param_use_type = ParameterString(name="use_type", default_value="Client")  # New parameter to determine type

    # Read the configuration file
    configuration = utils.read_config_data()

    # Initialize the processors used in the pipeline executions
    processors = utils.Processors(
        base_job_prefix=base_job_prefix,
        role=role,
        instance_count=processing_instance_count,
        instance_type="ml.r5.xlarge",  # Instance type for processing
        sagemaker_session=sagemaker_session,
    )

    # ETL Step - Extract, Transform, Load
    framework_processor = processors.framework()
    
    # Define the arguments for running a PySpark job for ETL
    etl_step_args = framework_processor.get_run_args(
        # Path to the preprocessing script
        code=path_join(BASE_DIR, "code", "etl.py"),

        # List of dependencies required by the preprocessing script
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "etl.txt")
        ],
        arguments=[  # Command line arguments to the framework script
            "--s3_bucket_nps",
            param_s3_bucket_nps,
            "--s3_bucket_lf",
            param_s3_bucket_lf,
            "--s3_path_read_nps",
            param_s3_path_read_nps,
            "--s3_path_read_lf",
            param_s3_path_read_lf,
            "--s3_path_write",
            param_s3_path_write,
            "--str_execution_date",
            param_str_execution_date,
            "--str_start_date",
            param_str_start_date,
            "--str_end_date",
            param_str_end_date,
            "--str_cabin",
            param_str_cabin,
            "--str_haul",
            param_str_haul,
            "--str_model",
            param_str_model,
            "--use_type",
            param_use_type,
        ],
    )

    # Create a processing step for ETL
    etl_step = ProcessingStep(
        name="etl_step",  # Name of the step
        processor=framework_processor,  # Processor to be used (framework in this case)
        inputs=etl_step_args.inputs,  # Inputs for the processor
        outputs=etl_step_args.outputs,  # Where to store the outputs
        job_arguments=etl_step_args.arguments,  # Arguments for the processor
        code=etl_step_args.code,  # Code to be executed
    )

    # Aggregated Preprocess Step (only if "use_type" is Aggregated)
    agg_preprocess_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "agg_preprocess.py"),  # Path to the preprocessing script for aggregation
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "agg_preprocess.txt")
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,  # S3 bucket for data storage
            "--s3_path_write",
            param_s3_path_write,  # Path for writing aggregated outputs
            "--str_execution_date",
            param_str_execution_date,  # Execution date for the aggregation job           
            "--df_targets",
            param_df_targets,
        ],
    )

    # Define the processing step for aggregated preprocessing
    agg_preprocess_step = ProcessingStep(
        name="agg_preprocess_step",  # Name of the step
        processor=framework_processor,  # Processor to be used for the step
        inputs=agg_preprocess_step_args.inputs,  # Inputs for the step
        outputs=agg_preprocess_step_args.outputs,  # Outputs for the step
        job_arguments=agg_preprocess_step_args.arguments,  # Arguments for the step
        code=agg_preprocess_step_args.code,  # Code to be executed
    )

    # Aggregated Predict and Explain Step (only if "use_type" is Aggregated)
    agg_predict_explain_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "agg_predict_explain.py"),  # Path to the predict and explain script for aggregation
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "agg_predict_explain.txt")
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,  # S3 bucket for data storage
            "--s3_path_write",
            param_s3_path_write,  # Path for writing prediction outputs
            "--str_execution_date",
            param_str_execution_date,  # Execution date for the prediction job

        ],
    )

    # Define the processing step for aggregated predict and explain
    agg_predict_explain_step = ProcessingStep(
        name="agg_predict_explain_step",  # Name of the step
        depends_on=["agg_preprocess_step"],  # This step depends on the aggregated preprocess step
        processor=framework_processor,  # Processor to be used for the step
        inputs=agg_predict_explain_step_args.inputs,  # Inputs for the step
        outputs=agg_predict_explain_step_args.outputs,  # Outputs for the step
        job_arguments=agg_predict_explain_step_args.arguments,  # Arguments for the step
        code=agg_predict_explain_step_args.code,  # Code to be executed
    )

    # Clusterize Step - Clustering data
    # Instantiate the processor for clustering
    framework_processor = processors.framework()

    # Configure the arguments for the processing step for clustering
    clusterize_step_args = framework_processor.get_run_args(
        # Path to the preprocessing script
        code=path_join(BASE_DIR, "code", "clusterize.py"),

        # List of dependencies required by the preprocessing script
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),
            path_join(BASE_DIR, "packages", "config.yml"),
            path_join(BASE_DIR, "packages", "requirements", "clusterize.txt")
        ],

        # Arguments to pass to the preprocessing script
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,  # S3 bucket for data storage
            "--s3_path_write",
            param_s3_path_write,  # Path for writing outputs
            "--str_execution_date",
            param_str_execution_date,  # Execution date for the clustering job
        ],
    )

    # Define the processing step for clustering
    clusterize_step = ProcessingStep(
        name="clusterize_step",  # Step name
        processor=framework_processor,  # Processor used for clustering
        inputs=clusterize_step_args.inputs,  # Inputs for clustering
        outputs=clusterize_step_args.outputs,  # Outputs for clustering
        job_arguments=clusterize_step_args.arguments,  # Job arguments for clustering
        code=clusterize_step_args.code,  # Code to execute for clustering
    )

    # Simulate Step - Simulation of model
    framework_processor = processors.framework()
    simulate_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "simulate.py"),  # Path to the simulating code
        dependencies=[
            path_join(BASE_DIR, "packages", "config.yml"),  # Dependencies required for simulating
            path_join(BASE_DIR, "packages", "requirements", "simulate.txt"),
            path_join(BASE_DIR, "packages", "utils.py"),
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,  # S3 bucket for data storage
            "--s3_path_write",
            param_s3_path_write,  # Path for writing outputs
            "--str_execution_date",
            param_str_execution_date,  # Execution date for the simulation job
            "--df_targets",
            param_df_targets,  # Dataframe targets for the simulation
        ],
    )

    # Define the processing step for simulation
    simulate_step = ProcessingStep(
        name="simulate_step",  # Name of the step
        # depends_on=["clusterize_step"],  # This step depends on the clusterize step
        # depends_on=["predict_explain_ori_step"],
        processor=framework_processor,  # Processor to use for the simulation
        inputs=simulate_step_args.inputs,  # Inputs for the simulation
        outputs=simulate_step_args.outputs,  # Outputs for the simulation
        job_arguments=simulate_step_args.arguments,  # Arguments for the simulation
        code=simulate_step_args.code,  # Code to execute for the simulation
    )
    
    
    # Predict and Explain Step - Used when "use_type" is Client
    framework_processor = processors.framework()
    predict_explain_ori_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "predict_explain_ori.py"),  # Path to the predict.py script
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),  # Path to the utils.py file
            path_join(BASE_DIR, "packages", "config.yml"),  # Path to the config.yml file
            path_join(BASE_DIR, "packages", "requirements", "predict_explain.txt"),  # Path to the predict.txt file
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,  # S3 bucket for data storage
            "--s3_path_write",
            param_s3_path_write,  # Path for writing prediction outputs
            "--str_execution_date",
            param_str_execution_date,  # Execution date for the prediction job
        ],
    )

    # Define the processing step for predict and explain
    predict_explain_ori_step = ProcessingStep(
        name="predict_explain_ori_step",  # Name of the step
        depends_on=["clusterize_step"],  # This step depends on the simulate step
        processor=framework_processor,  # Processor to use for the step
        inputs=predict_explain_ori_step_args.inputs,  # Input data for the step
        outputs=predict_explain_ori_step_args.outputs,  # Output data for the step
        job_arguments=predict_explain_ori_step_args.arguments,  # Additional job arguments
        code=predict_explain_ori_step_args.code,  # Code to execute for the step
    )

    # Predict and Explain Step - Used when "use_type" is Client
    framework_processor = processors.framework()
    predict_explain_step_args = framework_processor.get_run_args(
        code=path_join(BASE_DIR, "code", "predict_explain.py"),  # Path to the predict.py script
        dependencies=[
            path_join(BASE_DIR, "packages", "utils.py"),  # Path to the utils.py file
            path_join(BASE_DIR, "packages", "config.yml"),  # Path to the config.yml file
            path_join(BASE_DIR, "packages", "requirements", "predict_explain.txt"),  # Path to the predict.txt file
        ],
        arguments=[
            "--s3_bucket",
            param_s3_bucket_nps,  # S3 bucket for data storage
            "--s3_path_write",
            param_s3_path_write,  # Path for writing prediction outputs
            "--str_execution_date",
            param_str_execution_date,  # Execution date for the prediction job
        ],
    )

    # Define the processing step for predict and explain
    predict_explain_step = ProcessingStep(
        name="predict_explain_step",  # Name of the step
        depends_on=["simulate_step"],  # This step depends on the simulate step
        processor=framework_processor,  # Processor to use for the step
        inputs=predict_explain_step_args.inputs,  # Input data for the step
        outputs=predict_explain_step_args.outputs,  # Output data for the step
        job_arguments=predict_explain_step_args.arguments,  # Additional job arguments
        code=predict_explain_step_args.code,  # Code to execute for the step
    )

    # Condition to determine which set of steps to run based on "use_type"
    condition = ConditionEquals(left=param_use_type, right="Aggregated")

    # Define the condition step to choose between aggregated steps or client steps
    condition_step = ConditionStep(
        name="UseAggregatedCondition",  # Name of the condition step
        # depends_on = ["etl_step"],
        conditions=[condition],  # Condition to evaluate
        if_steps=[agg_preprocess_step, agg_predict_explain_step],  # Steps to execute if "use_type" is "Aggregated"
        # else_steps=[clusterize_step, predict_explain_ori_step, simulate_step]  # Steps to execute if "use_type" is "Client"
        else_steps=[simulate_step] 
    )

    # PIPELINE - Define the overall pipeline structure
    pipeline = Pipeline(
        name=pipeline_name,  # Name of the pipeline
        parameters=[
            processing_instance_count,  # Number of instances for data processing
            param_str_execution_date,  # Execution date as a string
            param_s3_bucket_nps,  # S3 bucket name
            param_s3_bucket_lf,  # S3 bucket name
            param_s3_path_read_nps,  # S3 path for reading the nps data
            param_s3_path_read_lf,  # S3 path for reading the lf data
            param_s3_path_write,  # S3 path for writing data
            param_is_last_date,  # Flag indicating if it is the last date for data processing
            param_str_start_date,  # Start date for processing
            param_str_end_date,  # End date for processing
            param_str_cabin,  # Cabin information
            param_str_haul,  # Haul information
            param_str_model,
            param_df_targets,  # Dataframe targets
            param_use_type,  # Use type to determine pipeline flow
        ],
        # steps=[etl_step, condition_step],  # First the ETL step, followed by the condition step which contains the branches
        steps=[condition_step],
        sagemaker_session=sagemaker_session,  # Sagemaker session object
    )
    return pipeline
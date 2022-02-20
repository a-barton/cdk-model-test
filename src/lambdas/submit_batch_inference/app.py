import boto3
import datetime


def find_latest_training_job(model_name):
    client = boto3.client("sagemaker")
    models = client.list_training_jobs(
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=100,
        StatusEquals="Completed",
        NameContains=model_name,
    )
    return models["TrainingJobSummaries"][0]["TrainingJobName"]


def make_model_from_training_job(model_name):
    client = boto3.client("sagemaker")
    job_details = client.describe_training_job(TrainingJobName=model_name)
    model = client.create_model(
        ModelName=model_name,
        PrimaryContainer=dict(
            Image=job_details["AlgorithmSpecification"]["TrainingImage"],
            ModelDataUrl=job_details["ModelArtifacts"]["S3ModelArtifacts"],
        ),
        ExecutionRoleArn=job_details["RoleArn"],
    )
    return model_name


def lambda_handler(event, context):

    model_name = event["model_name"]
    inference_data_uri = event["inference_data_uri"]
    resource_config = event["resource_config"]

    client = boto3.client("sagemaker")

    inference_job_name = (
        f"{model_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    latest_training_job_name = find_latest_training_job(model_name)
    try:
        sagemaker_model_name = make_model_from_training_job(latest_training_job_name)
    except:  # Most likely a result of the model already existing.
        sagemaker_model_name = latest_training_job_name

    response = client.create_transform_job(
        TransformJobName=inference_job_name,
        ModelName=sagemaker_model_name,
        BatchStrategy="MultiRecord",
        MaxPayloadInMB=6,
        TransformInput={
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"{inference_data_uri}",
                }
            },
            "ContentType": "application/jsonlines",
            "SplitType": "Line",
        },
        TransformOutput={"S3OutputPath": f"{inference_data_uri}/inference_output",},
        TransformResources={
            "InstanceType": resource_config["instance_type"],
            "InstanceCount": resource_config["instance_count"],
        },
    )

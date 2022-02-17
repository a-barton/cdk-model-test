import boto3
import datetime


def lambda_handler(event, context):

    model_name = event["model_name"]
    role_arn = event["role_arn"]
    train_data_uri = event["train_data_uri"]
    resource_config = event["resource_config"]
    use_spot_training = event["use_spot_training"]

    train_job_name = (
        f"{model_name}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    )

    client = boto3.client("sagemaker")

    response = client.create_training_job(
        TrainingJobName=train_job_name,
        AlgorithmSpecification={
            "AlgorithmName": model_name,
            "TrainingInputMode": "File",
            "RoleArn": role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{train_data_uri}/input",
                        }
                    },
                }
            ],
            "OutputDataConfig": {"S3OutputPath": f"s3://{train_data_uri}/output"},
            "ResourceConfig": {
                "InstanceType": resource_config["instance_type"],
                "InstanceCount": resource_config["instance_count"],
                "VolumeSizeInGB": resource_config["volume_size"],
            },
            "EnableManagedSpotTraining": use_spot_training,
            "StoppingCondition": {
                "MaxRuntimeInSeconds": 900,
                "MaxWaitTimeInSeconds": 1000,
            },
        },
    )

    return {"statusCode": 200}


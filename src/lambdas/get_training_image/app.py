import boto3


def lambda_handler(event, context):
    training_job_arn = event["TrainingJobArn"]
    training_job_name = training_job_arn.split("/")[-1]

    client = boto3.client("sagemaker")
    training_job_desc = client.describe_training_job(TrainingJobName=training_job_name)
    image_uri = training_job_desc["AlgorithmSpecification"]["TrainingImage"]
    model_artifacts_uri = training_job_desc["ModelArtifacts"]["S3ModelArtifacts"]

    return {"TrainingImage": image_uri, "S3ModelArtifactsURI": model_artifacts_uri}

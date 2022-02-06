import aws_cdk as cdk
from cdk_stacks.sagemaker_stack import SagemakerStack
from cdk_stacks.cdk_pipeline import PipelineStack

MODEL_NAME = "cdk-model-test"
ACCOUNT = "149167650712"
REGION = "ap-southeast-2"

app = cdk.App()
SagemakerStack(app, "SagemakerStack")
PipelineStack(app, "PipelineStack", env={"account": ACCOUNT, "region": REGION,})

app.synth()

import json
import aws_cdk as cdk
from cdk_stacks.sagemaker_stack import SagemakerStack
from cdk_stacks.cdk_pipeline import PipelineStack

config = json.load(open("config.json"))
ACCOUNT = config["ACCOUNT"]
REGION = config["REGION"]

app = cdk.App()
SagemakerStack(app, "CDKModelSagemakerStack", config=config)
PipelineStack(
    app,
    "CDKModelPipelineStack",
    config=config,
    env={"account": ACCOUNT, "region": REGION,},
)

app.synth()

import aws_cdk as cdk
from constructs import Construct
from .sagemaker_stack import SagemakerStack


class PipelineAppStage(cdk.Stage):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        lambdaStack = SagemakerStack(self, "SagemakerStack")

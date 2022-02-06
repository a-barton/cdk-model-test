import aws_cdk as cdk
from aws_cdk.pipelines import CodePipeline, CodePipelineSource, ShellStep
from .sagemaker_app_stage import PipelineAppStage

APP_ACCOUNT = "149167650712"
APP_REGION = "ap-southeast-2"


class PipelineStack(cdk.Stack):
    def __init__(self, scope, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        pipeline = CodePipeline(
            self,
            "Pipeline",
            pipeline_name="CDKModelPipeline",
            synth=ShellStep(
                "Synth",
                input=CodePipelineSource.git_hub(
                    "a-barton/cdk-model-test", "cdk-pipeline-test"
                ),
                commands=[
                    "npm install -g aws-cdk",
                    "python -m pip install -r build_requirements.txt",
                    "cdk synth",
                ],
            ),
        )

        pipeline.add_stage(
            PipelineAppStage(
                self,
                "test",
                env=cdk.Environment(account=APP_ACCOUNT, region=APP_REGION),
            )
        )


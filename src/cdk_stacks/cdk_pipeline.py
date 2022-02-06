import aws_cdk as cdk
from aws_cdk.pipelines import CodePipeline, CodePipelineSource, ShellStep

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
                input=CodePipelineSource.git_hub("a-barton/cdk-model-test", "main"),
                commands=[
                    "npm install -g aws-cdk",
                    "python -m pip install -r build_requirements.txt",
                    "cdk synth",
                ],
            ),
        )


import aws_cdk as cdk
from aws_cdk.pipelines import CodePipeline, CodePipelineSource, ShellStep


class CodePipelineStack(cdk.Stack):
    def __init__(self, scope, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        pipeline = CodePipeline(
            self,
            "Pipeline",
            pipeline_name="CodePipeline",
            synth=ShellStep(
                "Synth",
                input=CodePipelineSource.git_hub("a-barton/cdk-model-test", "main"),
                commands=[
                    "npm install -g aws-cdk",
                    # "python -m pip install -r requirements.txt",
                    "cdk synth",
                ],
            ),
        )


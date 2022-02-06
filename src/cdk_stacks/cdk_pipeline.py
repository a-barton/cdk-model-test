import aws_cdk as cdk
from aws_cdk.pipelines import CodePipeline, CodePipelineSource, ShellStep
from .sagemaker_app_stage import PipelineAppStage


class PipelineStack(cdk.Stack):
    def __init__(self, scope, id: str, config: dict, **kwargs):
        super().__init__(scope, id, **kwargs)

        ACCOUNT = config["ACCOUNT"]
        REGION = config["REGION"]
        GITHUB_OWNER = config["GITHUB_OWNER"]
        GITHUB_REPO = config["GITHUB_REPO"]

        pipeline = CodePipeline(
            self,
            "CDKModelPipeline",
            pipeline_name="CDKModelPipeline",
            synth=ShellStep(
                "Synth",
                input=CodePipelineSource.git_hub(GITHUB_OWNER, GITHUB_REPO),
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
                config=config,
                env=cdk.Environment(account=ACCOUNT, region=REGION),
            )
        )


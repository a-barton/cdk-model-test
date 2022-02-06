from aws_cdk import (
    CfnParameter,
    Stack,
    App,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
)
from aws_cdk.aws_ecr_assets import DockerImageAsset


class SagemakerStack(Stack):
    def __init__(self, app: App, id: str, config: dict) -> None:
        super().__init__(app, id)

        MODEL_NAME = config["MODEL_NAME"]

        # Define resource name parameters
        model_name = CfnParameter(
            self, "model", type="String", default=MODEL_NAME,
        ).value_as_string

        # Define Docker image for the model, referencing the local Dockerfile in the repo
        asset = DockerImageAsset(self, "MLInferenceImage", directory="src/container")
        primary_container_definition = sagemaker.CfnModel.ContainerDefinitionProperty(
            image=asset.image_uri,
        )

        # Execution role that SageMaker will assume to run the model
        sagemaker_execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess"
                )
            ],
        )

        sagemaker_model = sagemaker.CfnModel(
            self,
            "SagemakerModel",
            model_name=f'model-{model_name.replace("_","-").replace("/","--")}',
            execution_role_arn=sagemaker_execution_role.role_arn,
            primary_container=primary_container_definition,
        )

        model_endpoint_config = sagemaker.CfnEndpointConfig(
            self,
            "ModelEndpointConfig",
            endpoint_config_name=f'config-{model_name.replace("_","-").replace("/","--")}',
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    model_name=sagemaker_model.model_name,
                    initial_instance_count=1,
                    initial_variant_weight=1.0,
                    instance_type="ml.t2.medium",
                    variant_name="production-medium",
                )
            ],
        )

        model_endpoint = sagemaker.CfnEndpoint(
            self,
            "ModelEndpoint",
            endpoint_name=f'endpoint-{model_name.replace("_","-").replace("/","--")}',
            endpoint_config_name=model_endpoint_config.endpoint_config_name,
        )

        model_endpoint_config.node.add_dependency(sagemaker_model)
        model_endpoint.node.add_dependency(model_endpoint_config)

import os
from aws_cdk import (
    Stack,
    App,
    aws_s3 as s3,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    Environment,
)
from aws_cdk.aws_ecr_assets import DockerImageAsset


class SagemakerStack(Stack):
    def __init__(self, app: App, id: str) -> None:
        super().__init__(app, id)

        asset = DockerImageAsset(self, "MLInferenceImage", directory="src/container")
        primary_container_definition = sagemaker.CfnModel.ContainerDefinitionProperty(
            image=asset.image_uri,
        )

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

        sagemaker_vpc = ec2.Vpc(self, "Vpc", cidr="192.168.0.0/16")

        sagemaker_security_group = ec2.SecurityGroup(self, "SG", vpc=sagemaker_vpc)

        subnet_selection = sagemaker_vpc.select_subnets(
            subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT
        )

        vpc_config = sagemaker.CfnModel.VpcConfigProperty(
            security_group_ids=[sagemaker_security_group.security_group_id],
            subnets=subnet_selection.subnet_ids,
        )

        sagemaker_model = sagemaker.CfnModel(
            self,
            "SagemakerModel",
            execution_role_arn=sagemaker_execution_role.role_arn,
            primary_container=primary_container_definition,
            vpc_config=vpc_config,
        )


app = App()
SagemakerStack(app, "SagemakerStack")
app.synth()

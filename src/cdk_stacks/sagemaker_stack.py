import os
from aws_cdk import (
    CfnParameter,
    Stack,
    App,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_lambda as _lambda,
    aws_stepfunctions as _aws_stepfunctions,
    aws_stepfunctions_tasks as _aws_stepfunctions_tasks,
    aws_s3,
    aws_s3_deployment,
    RemovalPolicy,
)
from aws_cdk.aws_ecr_assets import DockerImageAsset


class SagemakerStack(Stack):
    def __init__(self, app: App, id: str, config: dict) -> None:
        super().__init__(app, id)

        MODEL_NAME = config["MODEL_NAME"]

        #########################
        ## Sagemaker resources ##
        #########################

        # Define Cloudformation parameters
        model_name = CfnParameter(
            self, "model", type="String", default=MODEL_NAME,
        ).value_as_string

        # Define Docker image for the model, referencing the local Dockerfile in the repo
        docker_image_asset = DockerImageAsset(
            self, "MLInferenceImage", directory="src/container"
        )
        primary_container_definition = sagemaker.CfnModel.ContainerDefinitionProperty(
            image=docker_image_asset.image_uri,
        )

        # Define S3 bucket for modelling data/artifacts
        bucket = aws_s3.Bucket(
            self,
            "S3Bucket",
            bucket_name=f"{model_name}-bucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
        )

        # Add training data to the S3 bucket
        __dirname = os.path.dirname(os.path.realpath(__file__))
        training_data_bucket_deployment = aws_s3_deployment.BucketDeployment(
            self,
            "TrainingData",
            sources=[
                aws_s3_deployment.Source.asset(
                    os.path.join(__dirname, "../../data/train/")
                )
            ],
            destination_bucket=bucket,
        )

        # Execution role that SageMaker will assume to run the model
        sagemaker_execution_role = iam.Role(
            self,
            "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            inline_policies=[
                iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["s3:*"],
                            resources=[
                                f"{bucket.bucket_arn}",
                                f"{bucket.bucket_arn}/*",
                            ],
                        ),
                        iam.PolicyStatement(
                            actions=[
                                "ecr:BatchCheckLayerAvailability",
                                "ecr:BatchGetImage",
                                "ecr:GetDownloadUrlForLayer",
                            ],
                            resources=["*"],
                        ),
                    ]
                ),
            ],
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "AmazonSageMakerFullAccess",
                ),
            ],
        )

        # Sagemaker model itself, referencing container definition from above and execution role
        sagemaker_model = sagemaker.CfnModel(
            self,
            "SagemakerModel",
            model_name=f'model-{model_name.replace("_","-").replace("/","--")}',
            execution_role_arn=sagemaker_execution_role.role_arn,
            primary_container=primary_container_definition,
        )

        # Endpoint configuration, referencing the model created above
        # model_endpoint_config = sagemaker.CfnEndpointConfig(
        #     self,
        #     "ModelEndpointConfig",
        #     endpoint_config_name=f'config-{model_name.replace("_","-").replace("/","--")}',
        #     production_variants=[
        #         sagemaker.CfnEndpointConfig.ProductionVariantProperty(
        #             model_name=sagemaker_model.model_name,
        #             initial_instance_count=1,
        #             initial_variant_weight=1.0,
        #             instance_type="ml.t2.medium",
        #             variant_name="production-medium",
        #         )
        #     ],
        # )

        # Endpoint, referencing the endpoint configuration created above
        # model_endpoint = sagemaker.CfnEndpoint(
        #     self,
        #     "ModelEndpoint",
        #     endpoint_name=f'endpoint-{model_name.replace("_","-").replace("/","--")}',
        #     endpoint_config_name=model_endpoint_config.endpoint_config_name,
        # )

        # model_endpoint_config.node.add_dependency(sagemaker_model)
        # model_endpoint.node.add_dependency(model_endpoint_config)

        ###########################
        ## Stepfunction workflow ##
        ###########################

        training_job_config = {
            "training_image": docker_image_asset.image_uri,
            "model_name": MODEL_NAME,
            "role_arn": sagemaker_execution_role.role_arn,
            "train_data_uri": bucket.s3_url_for_object(),
            "resource_config": {
                "instance_type": "ml.m5.large",
                "instance_count": 1,
                "volume_size": 10,
            },
            "use_spot_training": True,
        }

        training_submit_lambda_role = iam.Role(
            self,
            "submitTrainingLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )

        training_submit_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                ],
                resources=["*"],
            ),
        )
        training_submit_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[sagemaker_execution_role.role_arn],
            ),
        )

        training_submit_lambda = _lambda.Function(
            self,
            "submitTrainingLambda",
            handler="app.lambda_handler",
            runtime=_lambda.Runtime.PYTHON_3_9,
            code=_lambda.Code.from_asset("src/lambdas/submit_training/"),
            role=training_submit_lambda_role,
        )

        step_function = _aws_stepfunctions.StateMachine(
            self,
            "StepFunction",
            definition=_aws_stepfunctions_tasks.LambdaInvoke(
                self,
                "TrainingSubmitJob",
                lambda_function=training_submit_lambda,
                payload=_aws_stepfunctions.TaskInput.from_object(training_job_config),
            ).next(_aws_stepfunctions.Succeed(self, "Success")),
        )


import os
import datetime
from aws_cdk import (
    CfnParameter,
    Stack,
    App,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_lambda as _lambda,
    aws_stepfunctions as sfn,
    aws_stepfunctions_tasks as tasks,
    aws_s3,
    aws_s3_deployment,
    RemovalPolicy,
    aws_ec2 as ec2,
    Size,
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
        # docker_image_asset = DockerImageAsset(
        #     self, "MLInferenceImage", directory="src/container"
        # )
        # primary_container_definition = sagemaker.CfnModel.ContainerDefinitionProperty(
        #     image=docker_image_asset.image_uri,
        # )

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
            destination_key_prefix="train",
        )

        # Add test batch inference data to the S3 bucket
        inference_data_bucket_deployment = aws_s3_deployment.BucketDeployment(
            self,
            "InferenceData",
            sources=[
                aws_s3_deployment.Source.asset(
                    os.path.join(__dirname, "../../data/batch_inference/")
                )
            ],
            destination_bucket=bucket,
            destination_key_prefix="batch_inference",
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
                        iam.PolicyStatement(
                            actions=["sagemaker:CreateModel"], resources=["*"],
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
        # sagemaker_model = sagemaker.CfnModel(
        #     self,
        #     "SagemakerModel",
        #     model_name=f'model-{model_name.replace("_","-").replace("/","--")}',
        #     execution_role_arn=sagemaker_execution_role.role_arn,
        #     primary_container=primary_container_definition,
        # )

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

        ####################################
        ## Training Stepfunction workflow ##
        ####################################

        training_submit_task = tasks.SageMakerCreateTrainingJob(
            self,
            "submitTrainingTask",
            integration_pattern=sfn.IntegrationPattern.RUN_JOB,
            training_job_name=sfn.JsonPath.string_at("$.JobName"),
            role=sagemaker_execution_role,
            algorithm_specification=tasks.AlgorithmSpecification(
                training_image=tasks.DockerImage.from_asset(
                    self, "CDKModelImage", directory="src/container",
                ),
                training_input_mode=tasks.InputMode.FILE,
            ),
            input_data_config=[
                tasks.Channel(
                    channel_name="train",
                    data_source=tasks.DataSource(
                        s3_data_source=tasks.S3DataSource(
                            s3_data_type=tasks.S3DataType.S3_PREFIX,
                            s3_location=tasks.S3Location.from_bucket(
                                bucket, key_prefix="train"
                            ),
                        )
                    ),
                )
            ],
            output_data_config=tasks.OutputDataConfig(
                s3_output_location=tasks.S3Location.from_bucket(
                    bucket, key_prefix="training_output"
                ),
            ),
            resource_config=tasks.ResourceConfig(
                instance_type=ec2.InstanceType("m5.large"),
                instance_count=1,
                volume_size=Size.gibibytes(10),
            ),
            result_path="$.TrainingJobResponse",
        )

        get_training_image_lambda_role = iam.Role(
            self,
            "getTrainingImageLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )

        get_training_image_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["sagemaker:DescribeTrainingJob",],
                resources=["*"],
            ),
        )

        get_training_image_lambda = _lambda.Function(
            self,
            "getTrainingImageLambda",
            handler="app.lambda_handler",
            runtime=_lambda.Runtime.PYTHON_3_9,
            code=_lambda.Code.from_asset("src/lambdas/get_training_image/"),
            role=get_training_image_lambda_role,
        )

        get_training_image_task = tasks.LambdaInvoke(
            self,
            "getTrainingImageTask",
            lambda_function=get_training_image_lambda,
            payload=sfn.TaskInput.from_json_path_at("$.TrainingJobResponse"),
            result_path="$.TrainingJob.Details",
        )

        create_model_task = tasks.SageMakerCreateModel(
            self,
            "createModelTask",
            model_name=sfn.JsonPath.string_at("$.ModelName"),
            primary_container=tasks.ContainerDefinition(
                image=tasks.DockerImage.from_json_expression(
                    sfn.JsonPath.string_at(
                        "$.TrainingJob.Details.Payload.TrainingImage"
                    )
                ),
                mode=tasks.Mode.SINGLE_MODEL,
                model_s3_location=tasks.S3Location.from_json_expression(
                    "$.TrainingJob.Details.Payload.S3ModelArtifactsURI"
                ),
            ),
        )

        training_step_function_definition = (
            training_submit_task.next(get_training_image_task)
            .next(create_model_task)
            .next(sfn.Succeed(self, "TrainingSuccessStep"))
        )
        training_step_function = sfn.StateMachine(
            self, "TrainingStepFunction", definition=training_step_function_definition,
        )

        #####################################
        ## Inference Stepfunction workflow ##
        #####################################

        inference_job_config = {
            "model_name": MODEL_NAME,
            "inference_data_uri": bucket.s3_url_for_object(key="batch_inference"),
            "resource_config": {"instance_type": "ml.m5.large", "instance_count": 1,},
        }

        inference_submit_lambda_role = iam.Role(
            self,
            "submitBatchInferenceLambdaRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )

        inference_submit_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "sagemaker:CreateTransformJob",
                    "sagemaker:DescribeTransformJob",
                    "sagemaker:ListModels",
                    "sagemaker:ListTrainingJobs",
                    "sagemaker:CreateModel",
                    "sagemaker:DescribeTrainingJob",
                ],
                resources=["*"],
            ),
        )
        inference_submit_lambda_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["iam:PassRole"],
                resources=[sagemaker_execution_role.role_arn],
            ),
        )

        inference_submit_lambda = _lambda.Function(
            self,
            "submitBatchInferenceLambda",
            handler="app.lambda_handler",
            runtime=_lambda.Runtime.PYTHON_3_9,
            code=_lambda.Code.from_asset("src/lambdas/submit_batch_inference/"),
            role=inference_submit_lambda_role,
        )

        inference_step_function = sfn.StateMachine(
            self,
            "BatchInferenceStepFunction",
            definition=tasks.LambdaInvoke(
                self,
                "BatchInferenceSubmitJob",
                lambda_function=inference_submit_lambda,
                payload=sfn.TaskInput.from_object(inference_job_config),
            ).next(sfn.Succeed(self, "InferenceSuccessStep")),
        )


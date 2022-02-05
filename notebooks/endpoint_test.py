# %%

import pandas as pd
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# %%

endpoint_name = "endpoint-cdk-model-test"
predictor = sagemaker.predictor.Predictor(
    endpoint_name=endpoint_name,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

# %%

test_inference_data = pd.read_csv("../data/test_inference_input.csv")
inference_data_json = test_inference_data.to_json(orient="records")

# %%

result = predictor.predict(inference_data_json)
result

# %%

import joblib
import pandas as pd
import os


def predict(df):
    model, scaler, label_encoder = load_model()
    scaled = scaler.transform(df)
    predictions = model.predict(scaled)
    return pd.DataFrame(label_encoder.inverse_transform(predictions))


def load_model():
    print(f"Working dir right before loading model: {os.getcwd()}")
    print(f"Contents of working dir: {os.listdir()}")
    model = joblib.load("/opt/ml/model/model.joblib")
    scaler = joblib.load("/opt/ml/model/scaler.joblib")
    label_encoder = joblib.load("/opt/ml/model/label_encoder.joblib")
    return model, scaler, label_encoder

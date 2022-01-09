import joblib
import pandas as pd


def predict(df):
    model, scaler, label_encoder = load_model()
    scaled = scaler.transform(df)
    predictions = model.predict(scaled)
    return label_encoder.inverse_transform(predictions)


def load_model():
    model = joblib.load("/opt/ml/model/model.joblib")
    scaler = joblib.load("/opt/ml/model/scaler.joblib")
    label_encoder = joblib.load("/opt/ml/model/label_encoder.joblib")
    return model, scaler, label_encoder

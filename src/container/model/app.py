import json
import io
import pandas as pd
import flask
from inference import predict
import logging

logging.basicConfig(level=logging.DEBUG)

app = flask.Flask(__name__)


@app.route("/ping", methods=["GET", "POST"])
def ping():
    return flask.Response(
        response=json.dumps({"message": "Status okay"}),
        status=200,
        mimetype="application/json",
    )


@app.route("/invocations", methods=["GET", "POST"])
def predict():
    logging.debug("Request registered. Starting prediction.")
    if flask.request.content_type == "application/json":
        logging.debug("Parsing the body of the request.")
        data = flask.request.data.decode("utf-8")
        s = io.StringIO(data)
        df = pd.read_json(s, lines=True)
    else:
        logging.debug(
            f"Bad request. Received content of type '{flask.request.content_type}' when expected JSON Lines."
        )
        return flask.Response(
            response=json.dumps(
                {"message": "This predictor only supports JSON Lines data"}
            ),
            status=415,
            mimetype="application/json",
        )

    logging.debug("Making predictions on the data now.")
    result = predict(df)
    logging.debug("Prediction successful. Responding to request.")

    return flask.Response(
        response=result.to_json(orient="records", lines=True),
        status=200,
        mimetype="application/json",
    )

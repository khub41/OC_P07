import mlflow
import numpy as np
from flask import Flask, request, jsonify

model_path = 'best_model'
model = mlflow.sklearn.load_model(model_path)

app = Flask(__name__)


@app.route('/predict', methods=["POST"])
def prediction():
    data = request.get_json()
    prediction = np.array2string(model.predict_proba(data)[0,1])

    return jsonify(prediction)


def run_app():
    app.run(debug=True, port=12345)
import joblib
import numpy as np
import pandas as pd
import shap
from flask import Flask, request, jsonify

model_path = 'best_model/model.pkl'
model = joblib.load(model_path)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


@app.route('/predict', methods=["POST"])
def prediction():
    data = request.get_json()
    prediction = np.array2string(model.predict_proba(data)[0,1])

    return jsonify(prediction)


@app.route('/explain', methods=["POST"])
def explain():
    data_client = request.json
    data_client_values = np.array([list(data_client.values())])
    data_client_features = list(data_client.keys())
    explainer_shap = shap.TreeExplainer(model)
    shap_values_client = explainer_shap.shap_values(data_client_values)
    shap_values_client_serie = pd.Series(index=data_client_features, data=shap_values_client[1][0, :])

    return jsonify(shap_values_client_serie.to_dict())


if __name__ == '__main__':
    app.run(debug=True)
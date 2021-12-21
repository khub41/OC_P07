import joblib
import numpy as np
import shap
from flask import Flask, request, jsonify

model_path = 'best_model/model.pkl'
model = joblib.load(model_path)

app = Flask(__name__)

@app.route('/predict', methods=["POST"])
def prediction():
    data = request.get_json()
    prediction = np.array2string(model.predict_proba(data)[0,1])

    return jsonify(prediction)


@app.route('/explain', methods=["POST"])
def explain():
    data_client = request.json
    data_client = np.array([data_client["array"]])
    explainer_shap = shap.TreeExplainer(model)
    shap_values_client = explainer_shap.shap_values(data_client)
    print(shap_values_client)
    return jsonify(np.array2string(shap_values_client[1][0, :]))


if __name__ == '__main__':
    app.run(debug=True)
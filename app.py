import joblib
import numpy as np
from flask import Flask, request, jsonify

model_path = 'best_model/model.pkl'
model = joblib.load(model_path)

app = Flask(__name__)

@app.route("/")
def init_get():
    pass

@app.route('/predict', methods=["POST"])
def prediction():
    data = request.get_json()
    prediction = np.array2string(model.predict_proba(data)[0,1])

    return jsonify(prediction)


if __name__ == '__main__':
    app.run(debug=True)
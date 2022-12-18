import json

from flask import Flask, request, jsonify
from flask_cors import CORS

from helpers import predecir

app = Flask(__name__)


@app.route('/')
def hello_world():
    return {'user': 'Admin'}

CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})
@app.route('/predict', methods=['POST'])
def predict():
    if (request.method == 'POST'):
        data = request.get_json()
        imagen = data.get("image")
        predicciones = []
        predicciones = predecir(imagen)
        data = {
            "message": "Predictions made satisfactorily",
            "clase": predicciones["clase"],
            "probabilidades": predicciones["probabilidades"]

        }
        response = app.response_class(response=json.dumps(data),
                                      status=200,
                                      mimetype='application/json')
        return response


# We only need this for local development.
if __name__ == '__main__':
    app.run(host="0.0.0.0")

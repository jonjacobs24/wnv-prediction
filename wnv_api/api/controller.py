from flask import Blueprint, request, jsonify
from WNVPrediction.predict import make_prediction
from WNVPrediction import __version__ as _version

from api.config import get_logger
from api import __version__ as api_version

import pandas as pd
import json

_logger = get_logger(logger_name=__name__)


prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/health', methods=['GET'])
def health():
	if request.method == 'GET':
		_logger.info('health status OK')
		return 'ok'


@prediction_app.route('/version', methods=['GET'])
def version():
	if request.method == 'GET':
		return jsonify({'model_version': _version,
						'api_version': api_version})


@prediction_app.route('/v1/predict/wnvprediction', methods=['POST'])
def predict():
	if request.method == 'POST':
		# Step 1: Extract POST data from request body as JSON
		input_data = request.get_json()
		_logger.debug(f'Inputs: {input_data}')

		# Step 2: change to datapframe

		X = pd.DataFrame.from_dict(input_data).iloc[:,1:]

		# Step 3: Model prediction
		result = make_prediction(X=X)
		_logger.debug(f'Outputs: {result}')

		# Step 4: Convert numpy ndarray to list
		predictions = result.get('predictions').tolist()
		version = result.get('version')

		# Step 5: Return the response as JSON
		return jsonify({'predictions': predictions,
						'version': version})

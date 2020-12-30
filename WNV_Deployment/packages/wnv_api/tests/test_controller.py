from WNVPrediction.config import config as model_config
from WNVPrediction.processing.data_management import load_and_ready_data as lr
from WNVPrediction import __version__ as _version

from api.config import get_logger

import json
import math

from api import __version__ as api_version

_logger = get_logger(logger_name=__name__)


def test_health_endpoint_returns_200(flask_test_client):
	# When
	response = flask_test_client.get('/health')

	# Then
	assert response.status_code == 200


def test_version_endpoint_returns_version(flask_test_client):
	# When
	response = flask_test_client.get('/version')

	# Then
	assert response.status_code == 200
	response_json = json.loads(response.data)
	assert response_json['model_version'] == _version
	assert response_json['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):
	# Given
	# Load the test data from the WNV_Prediction package
	# This is important as it makes it harder for the test
	# data versions to get confused by not spreading it
	# across packages.

	test_data = lr(weather_path=model_config.RAW_WEATHER,mosquito_path=model_config.RAW_MOSQUITO, 
		spray_path=model_config.RAW_SPRAY,target_present=False)
	
	post_json = test_data.to_json(orient='records')

	# When
	response = flask_test_client.post('/v1/predict/wnvprediction',
									  json=json.loads(post_json))


	# Then
	assert response.status_code == 200
	response_json = json.loads(response.data)
	prediction = response_json['predictions']
	response_version = response_json['version']
	assert response_version == _version
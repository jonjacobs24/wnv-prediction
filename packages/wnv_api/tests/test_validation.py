
from WNVPrediction.config import config as model_config
from WNVPrediction.processing.data_management import load_and_ready_data as lr
from WNVPrediction import __version__ as _version

import json
import math


def test_prediction_endpoint_validation_200(flask_test_client):
    # Given
    # Load the test data from the regression_model package.
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

    # Check correct number of errors removed
   
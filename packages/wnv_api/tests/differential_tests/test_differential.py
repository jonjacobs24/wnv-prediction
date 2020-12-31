import math

from WNVPrediction.config import config as model_config
from WNVPrediction.predict import make_prediction
from WNVPrediction.processing.data_management import load_and_ready_data as lr
import pandas as pd
import pytest


from api import config

@pytest.mark.skip
@pytest.mark.differential
def test_model_prediction_differential(
        *,
        save_file: str = 'test_data_predictions.csv'):
    """
    This test compares the prediction result similarity of
    the current model with the previous model's results.
    """

    # Given
    # Load the saved previous model predictions
    previous_model_df = pd.read_csv(f'{config.PACKAGE_ROOT}/{save_file}')
    previous_model_predictions = previous_model_df.predictions.values

    test_data = lr(weather_path=model_config.RAW_WEATHER,mosquito_path=model_config.RAW_MOSQUITO, 
        spray_path=model_config.RAW_SPRAY,target_present=False)

    multiple_test_json = test_data.iloc[99:600,:]

    # When
    current_result = make_prediction(input_data=multiple_test_input)
    current_model_predictions = current_result.get('predictions')

    # Then
    # diff the current model vs. the old model
    assert len(previous_model_predictions) == len(
        current_model_predictions)

    # Perform the differential test
    for previous_value, current_value in zip(
            previous_model_predictions, current_model_predictions):

        # convert numpy float64 to Python float.
        previous_value = previous_value.item()
        current_value = current_value.item()

        # rel_tol is the relative tolerance – it is the maximum allowed
        # difference between a and b, relative to the larger absolute
        # value of a or b. For example, to set a tolerance of 5%, pass
        # rel_tol=0.05.
        assert math.isclose(previous_value,
                            current_value,
                            rel_tol=model_config.ACCEPTABLE_MODEL_DIFFERENCE)
"""
This script should only be run in CI.
Never run it locally or you will disrupt the
differential test versioning logic.
"""

import pandas as pd

from WNVPrediction.predict import make_prediction
from WNVPrediction.processing.data_management import load_and_ready_data as lr

from api import config


def capture_predictions(
        *,
        save_file: str = 'test_data_predictions.csv'):
    """Save the test data predictions to a CSV."""

    test_data = lr(weather_path=model_config.RAW_WEATHER,mosquito_path=model_config.RAW_MOSQUITO, 
        spray_path=model_config.RAW_SPRAY,target_present=False)

    # we take a slice with no input validation issues
    multiple_test_json = test_data.iloc[99:600,:]

    predictions = make_prediction(input_data=multiple_test_json)

    # save predictions for the test dataset
    predictions_df = pd.DataFrame(predictions)

    # hack here to save the file to the regression model
    # package of the repo, not the installed package
    predictions_df.to_csv(
        f'{config.PACKAGE_ROOT.parent}/'
        f'WNVPrediction/WNVPrediction/datasets/{save_file}')


if __name__ == '__main__':
    capture_predictions()
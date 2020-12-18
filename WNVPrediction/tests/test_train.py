import pytest
from WNVPrediction.trained_model import predict
from WNVPrediction.processing import data_management as dm
from WNVPrediction.config import config


def test_training_roc_score(roc_thresh):
	X,y = dm.load_and_ready_data(weather_path=config.RAW_WEATHER,
		mosquito_path=config.RAW_MOSQUITO,spray_path=config.RAW_SPRAY,target_present=True)
	
	score_train, score_test = predict.make_prediction_and_score(X=X,y=y)
	assert score_train >= roc_thresh

def test_testing_roc_score(roc_thresh):

	X,y = dm.load_and_ready_data(weather_path=config.RAW_WEATHER,
		mosquito_path=config.RAW_MOSQUITO,spray_path=config.RAW_SPRAY,target_present=True)
	
	score_train, score_test = predict.make_prediction_and_score(X=X,y=y)
	assert score_test >= roc_thresh
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from WNVPrediction.config import config
from WNVPrediction.processing import preprocessors as pp
from WNVPrediction.processing import data_management as dm
from WNVPrediction import __version__ as _version

import json

import logging
_logger = logging.getLogger(__name__)

pipeline_file_name = f"{config.PIPELINE_NAME}{_version}.pkl"
pipe = dm.load_pipeline(path=pipeline_file_name)


def make_prediction(*,X:pd.DataFrame) -> pd.Series:
	"""makes  a prediction based on data and a loaded pipeline """
	
	y_pred = pipe.predict_proba(X)[:,1]
	
	_logger.info(
		f'Making predictions with model version: {_version} '
		f'Input Data: {X}'
		f'Predictions: {y_pred}'
	)
	results = {"predictions": y_pred, "version": _version}
	return results


def make_prediction_and_score(*,X:pd.DataFrame,y:pd.Series) -> (float, float):
	"""makes  a prediction based on data and a loaded pipeline """

	#train, test split
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)

	y_pred_test = pipe.predict_proba(X_test)[:,1]
	roc_test = roc_auc_score(y_test,y_pred_test)

	y_pred_train = pipe.predict_proba(X_train)[:,1]
	roc_train = roc_auc_score(y_train,y_pred_train)

	_logger.info(
		f'Making predictions with model version: {_version} '
		f'Train ROC_AUC: {roc_train}'
		f'Test ROC_AUC: {roc_test}'
	)
	
	return roc_train, roc_test
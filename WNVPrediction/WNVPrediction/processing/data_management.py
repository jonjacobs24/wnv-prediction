import pandas as pd
import joblib
from WNVPrediction.config import config
from sklearn.pipeline import Pipeline
from WNVPrediction.processing import preprocessors as pp
from WNVPrediction.processing import data_management as dm

from WNVPrediction import __version__ as _version

import logging

_logger = logging.getLogger(__name__)


def save_pipeline(*,pipeline_to_persist) -> None:
	"""persist the pipeline """
	save_path = config.PIPELINE_NAME
	joblib.dump(pipeline_to_persist, save_path)
	print('pipeline saved')


def load_pipeline(*,path:str) -> Pipeline:
	"""load a persisted pipeline  """
	pipe = joblib.load(filename=path)
	return pipe

def load_and_ready_data(target_present=False,*,weather_path:str,mosquito_path:str,spray_path:str) -> (pd.DataFrame, pd.Series):
	"""loads and preprocesses raw data"""

	# read training data
	w = pd.read_csv(weather_path)
	m = pd.read_csv(mosquito_path)
	s = pd.read_csv(spray_path)
	
	#processing into one dataset
	X = pp.prepare_raw_data(weather_data=w,mosquito_data=m,spray_data=s) 
	
	#separating target variable
	if target_present:
		y = X[config.TARGET]
		X = X.drop(config.TARGET,axis=1)

		#lagging features
		lagger = pp.FeatureLagger(variables=config.FEAT_TO_LAG)
		X = lagger.transform(X)

		return X,y

	if ~target_present:

		#lagging features
		lagger = pp.FeatureLagger(variables=config.FEAT_TO_LAG)
		X = lagger.transform(X)

		return X
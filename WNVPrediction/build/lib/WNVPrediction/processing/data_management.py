import pandas as pd
import joblib
from WNVPrediction.config import config
from sklearn.pipeline import Pipeline
from WNVPrediction.processing import preprocessors as pp
from WNVPrediction.processing import data_management as dm

from WNVPrediction import __version__ as _version
import logging
import typing as t

_logger = logging.getLogger(__name__)


def save_pipeline(*,pipeline_to_persist) -> None:
	"""persist the pipeline """
	

	save_file_name = f"{config.PIPELINE_NAME}{_version}.pkl"
	save_path = config.TRAINED_MODEL_DIR / save_file_name

	joblib.dump(pipeline_to_persist, save_path)
	_logger.info(f"saved pipeline: {save_file_name}")

	remove_old_pipelines(files_to_keep=[save_file_name])



def load_pipeline(*,path:str) -> Pipeline:
	"""load a persisted pipeline  """
	file_path = config.TRAINED_MODEL_DIR / path
	pipe = joblib.load(filename=file_path)
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

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
	"""
	Remove old model pipelines.
	This is to ensure there is a simple one-to-one
	mapping between the package version and the model
	version to be imported and used by other applications.
	However, we do also include the immediate previous
	pipeline version for differential testing purposes.
	"""
	do_not_delete = files_to_keep + ['__init__.py','__pycache__']
	for model_file in config.TRAINED_MODEL_DIR.iterdir():
		if model_file.name not in do_not_delete:
			model_file.unlink()
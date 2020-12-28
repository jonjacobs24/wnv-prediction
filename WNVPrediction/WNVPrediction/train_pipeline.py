import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate

import joblib

import pipeline
from WNVPrediction.config import config
from WNVPrediction.processing import preprocessors as pp
from WNVPrediction.processing import data_management as dm

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

import logging
_logger = logging.getLogger(__name__)

def run_training():
	"""Train the model."""


	X,y = dm.load_and_ready_data(weather_path=config.RAW_WEATHER,mosquito_path=config.RAW_MOSQUITO,
		spray_path=config.RAW_SPRAY,target_present=True)


	# divide train and test
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)  

	params = {'randomforestclassifier__n_estimators': np.arange(config.N_ESTIMATORS,config.N_ESTIMATORS+50,50),
			'randomforestclassifier__max_depth' :  np.arange(config.MAX_DEPTH,config.MAX_DEPTH+1)}

	rf = GridSearchCV(pipeline.pipe,param_grid=params,cv=5, scoring = 'roc_auc',verbose=1)
	rf.fit(X_train,y_train)

	clf = rf.best_estimator_ 

	dm.save_pipeline(pipeline_to_persist=clf)

	y_pred = clf.predict_proba(X_test)[:,1]
	

if __name__ == '__main__':
	run_training()





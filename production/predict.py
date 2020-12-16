import pandas as pd

import joblib
import config
import preprocessors as pp


def make_prediction(input_data):
	
	_pipe = joblib.load(filename=config.PIPELINE_NAME)
	
	results = _pipe.predict_proba(input_data)[:,1]

	return results
   





if __name__ == '__main__':
	
	w = pd.read_csv(config.RAW_WEATHER)
	m = pd.read_csv(config.RAW_MOSQUITO)
	s = pd.read_csv(config.RAW_SPRAY)

	w = pp.prepare_weather(w)
	X = pp.unite(m,w,s)
	
	y = X[config.TARGET]
	X = X.drop(config.TARGET,axis=1)

	# test pipeline
	import numpy as np
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import roc_auc_score


	X_train, X_test, y_train, y_test = train_test_split(
		X,y, 
		test_size=0.2,
		random_state=42)

	y_pred_test = make_prediction(X_test)
	roc_test = roc_auc_score(y_test,y_pred_test)

	y_pred_train = make_prediction(X_train)
	roc_train = roc_auc_score(y_train,y_pred_train)

	print('TRAIN ROC_AUC: {}'.format(roc_train))
	print('TEST ROC_AUC: {}'.format(roc_test))
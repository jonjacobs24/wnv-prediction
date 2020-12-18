import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate

import joblib

import pipeline
import config
import preprocessors as pp

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def run_training():
	"""Train the model."""


	# read training data
	w = pd.read_csv(config.RAW_WEATHER)
	m = pd.read_csv(config.RAW_MOSQUITO)
	s = pd.read_csv(config.RAW_SPRAY)

	w = pp.prepare_weather(w)
	X = pp.unite(m,w,s)
	
	y = X[config.TARGET]
	X = X.drop(config.TARGET,axis=1)

	lagger = pp.FeatureLagger(variables=config.FEAT_TO_LAG)
	X = lagger.transform(X)


	# divide train and test
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=123)  

	params = {'randomforestclassifier__n_estimators': np.arange(config.N_ESTIMATORS-50,config.N_ESTIMATORS+100,50),
			'randomforestclassifier__max_depth' :  np.arange(config.MAX_DEPTH-1,config.MAX_DEPTH+2)}

	rf = GridSearchCV(pipeline.pipe,param_grid=params,cv=5, n_jobs=-1, scoring = 'roc_auc',verbose=1)
	rf.fit(X_train,y_train)

	clf = rf.best_estimator_ 


	joblib.dump(clf, config.PIPELINE_NAME)

	y_pred = clf.predict_proba(X_test)[:,1]
	return roc_auc_score(y_test,y_pred)

if __name__ == '__main__':
	run_training()





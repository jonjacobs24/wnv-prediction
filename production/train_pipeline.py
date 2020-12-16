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

	# lagger = pp.FeatureLagger(variables=config.FEAT_TO_LAG)

	# X = lagger.transform(X)

	#print(X.describe())

	# divide train and test
	X_train, X_test, y_train, y_test = train_test_split(
		X,y,
		test_size=0.2,
		random_state=42)  

	params = {'randomforestclassifier__n_estimators': np.arange(config.N_ESTIMATORS-50,config.N_ESTIMATORS+100,50),
			'randomforestclassifier__max_depth' :  np.arange(config.MAX_DEPTH-1,config.MAX_DEPTH+2)}

	rf = GridSearchCV(pipeline.pipe,param_grid=params,cv=5, n_jobs=-1, scoring = 'roc_auc',verbose=10)
	rf.fit(X_train,y_train)

	clf = rf.best_estimator_ 
	# cv = cross_validate(pipeline.pipe, X_train, y_train, scoring = 'roc_auc', cv=5, 
	# 	return_estimator = True)

	# # index = np.where(cv['test_score'] == np.max(cv['test_score']))
	# # index = index[0][0]
	# # clf = cv['estimator'][index] 

	# clf = pp.Cv_Avg(cv['estimator'])


	y_pred = clf.predict_proba(X_train)[:,1]
	print(roc_auc_score(y_train,y_pred))

	joblib.dump(clf, config.PIPELINE_NAME)


if __name__ == '__main__':
	run_training()





pipe_rf = make_pipeline( 
	StandardScaler(), 
	RandomForestClassifier(random_state=42)
)

params = {'randomforestclassifier__n_estimators': np.arange(750,2000,50),
		  'randomforestclassifier__max_depth' :  np.arange(6,11)}

rf = GridSearchCV(pipe,param_grid=params,cv=5, n_jobs=-1, scoring = 'roc_auc',verbose=10)

rf.fit(X_train,y_train)
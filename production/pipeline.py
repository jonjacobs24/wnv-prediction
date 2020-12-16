from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import preprocessors as pp
import config


pipe = Pipeline(
	[
		('feature_lagger', pp.FeatureLagger(variables=config.FEAT_TO_LAG)),
		
		('standard_scaler', StandardScaler()),
		
		('randomforrestclassifier', RandomForestClassifier(random_state=42))
	]
)
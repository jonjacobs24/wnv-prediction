from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from WNVPrediction.processing import preprocessors as pp
from WNVPrediction.config import config

import logging
_logger = logging.getLogger(__name__)

pipe = Pipeline(
	[
		#('feature_lagger', pp.FeatureLagger(variables=config.FEAT_TO_LAG)),
		
		('standard_scaler', StandardScaler()),
		
		('randomforestclassifier', RandomForestClassifier(random_state=42))
	]
)
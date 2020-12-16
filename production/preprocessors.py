import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import config



def prepare_weather(df):

	w = df.copy()

	#subsetting weather based on necessary columns
	w = w[config.FROM_WEATHER]

	#initial replacing and filling missing or trace calues
	w = w.replace(['M','-'],np.nan)
	w = w.replace(['T','  T'], 0.01)

	#filling missing values with 0 where relevant
	w['PrecipTotal'] = w['PrecipTotal'].fillna(0)

	#filling missing values with backfill and then forward fill to catch last NAN 
	w = w.fillna(method='ffill')
	w = w.fillna(method='bfill')

	#correcting dtypea
	as_float = config.WEATHER_FLOATS
	w.loc[:,as_float] = w.loc[:,as_float].astype('float')
	
	#engineering two features
	w['sunrise_diff'] = w['Sunrise'].diff().fillna(0)
	w['dev_65'] = w['Cool'] - w['Heat']
	w = w.drop(['Cool','Heat','Sunrise'],axis=1)

	#handling weather codes
	#for each code, creating a column where value is 1 if happens, 0 if doesn't, then drop codeSum
	codes = []
	for code in w.CodeSum.unique():
		for c in code.split(' '):
			codes.append(c)
	codes = pd.Series(codes)
	codes = codes.unique()[1:]

	#for each code, creating a column where value is 1 if happens, 0 if doesn't, then drop codeSum
	for c in codes:
		col = [1 if c in s else 0 for s in w.CodeSum]
		
		if np.sum(col)>(len(col)/100): 
			w[c] = col

	w = w.drop('CodeSum',axis=1)
	w = w.drop(config.WEATHER_CODES_DROP, axis=1)


	#breaking out two weather stations
	w_1 = w[w['Station']==1].drop('Station',axis=1)
	w_2 = w[w['Station']==2].drop('Station',axis=1)
	w = (w_1.set_index('Date') + w_2.set_index('Date')) / 2
	w = w.reset_index()

	return w

def unite(df_m,df_w,df_sp):


	w = df_w.copy()
	sp = df_sp.copy()
	df = df_m.copy()


	#integrating spray data into mosquito data
	mi_per_deg_lat = 364000/5280 #36400 ft per degree lat
	mi_per_deg_long = 288200/5280 #288200 ft per degree long

	#chaning date columns to pandas datetime for easy handling
	sp['Date'] = pd.to_datetime(sp['Date'])
	df['Date'] = pd.to_datetime(df['Date'])
	w['Date'] =  pd.to_datetime(w['Date'])

	def sprayed(dist, time, traps, sprays):
		"""Returns wether or not a mosquito trap locations was sprayed within a certain 
		distance and time frame. Distance is in miles, time is 0 for year, 1 for month, and 2 for day. d is dataset"""
		s = []
		#for each trap, find the distances to all sprays within the timeframe, if the miniumum is below 
		# the threshold, than it was sprayed during that time period
		period = {0:'y',1:'m',2:'d'}
		
		for i,r  in traps.iterrows():

		#creating a mask to select relevant spray locations based on date
			mask = sprays['Date'].dt.to_period(period[time]) == r[0].to_period(period[time]) 
			
			#passing the loop if there are no sprays during the right window
			if mask.sum() == 0:
				s.append(0)
				continue
			spray = sprays[mask]

			#finding euclidian distance based on lat/long converted to miles
			lat_d = (spray.iloc[:,2]-r[7]) * mi_per_deg_lat
			long_d = (spray.iloc[:,3]-r[8])* mi_per_deg_long
			d = np.sqrt(lat_d**2 + long_d**2)

			#if the closest spray in the time period is within the cutoff distance, assign 1, otherwise 0
			if d.min() <= dist:
				s.append(1) 
			else: 
				s.append(0)

		return s

	df['spray_day'] = sprayed(1,2,df,sp)

	#dropping irrelevant columns
	df = df.drop(config.FEAT_DROP,axis=1)

	df_avg = df.merge(w,on='Date').drop('Date',axis=1)

	return df_avg

class FeatureLagger(BaseEstimator, TransformerMixin):

	def __init__(self, variables=None):
		if not isinstance(variables, list):
			self.variables = [variables]
		else:
			self.variables = variables

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		
		#lagging functions
		def lagged(df,n):
			return df.rolling(n, min_periods=1).mean().dropna()
		def exp_lag(df,n):
			return df.ewm(span=n,min_periods=1).mean().dropna()

		X = X.copy()

		for lag in self.variables:
			feature = lag[0]
			n = lag[1]
			kind = lag[2]

			if kind == 'lag':
				name = feature + ' ' + str(n) + ' ' + kind
				X[name] = lagged(X[feature],n)
				X = X.drop(feature,axis=1)

			elif kind == 'exp':
				name = feature + ' ' + str(n) + ' ' + kind
				X[name] = exp_lag(X[feature],n)
				X = X.drop(feature,axis=1)

		return X


class Cv_Avg(BaseEstimator):

	def __init__(self,estimators):
		self.estimators = estimators


	def predict_proba(self,X): 
		prob = np.zeros(X.shape[0])
		for e in self.estimators:
			prob += e.predict_proba(X)[:,1]

		return prob/len(self.estimators)







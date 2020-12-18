#data source locations
RAW_WEATHER = '../data/raw/weather.csv.zip'
RAW_SPRAY = '../data/raw/spray.csv.zip'
RAW_MOSQUITO = '../data/raw/train.csv.zip'

#target variable for analysis
TARGET = 'WnvPresent'


#features to select from weather data
FROM_WEATHER = ['Date', 'Station', 'Depart', 'PrecipTotal','ResultSpeed','ResultDir','Heat','Cool',
	'CodeSum','Sunrise']

#weather features that should be cast as floats
WEATHER_FLOATS = ['Depart','Heat','Cool','PrecipTotal','Sunrise']

#weather codes to drop in the analysis
WEATHER_CODES_DROP = ['RA', 'TSRA', 'DZ']


#features to drop from the united data set
FEAT_DROP = ['Address','Block','Street','AddressNumberAndStreet','AddressAccuracy','Species',
	'Trap','Latitude','Longitude','NumMosquitos']


#Features to lag, their lag duration, and type of lag
FEAT_TO_LAG = [('spray_day', 4, 'lag'), ('Depart', 14, 'exp'), ('PrecipTotal', 2, 'exp'),
			('ResultSpeed', 58, 'lag'), ('ResultDir', 6, 'exp'), ('dev_65', 58, 'lag'), 
       		('BR', 50, 'exp'), ('HZ', 10, 'exp'), ('TS', 38, 'exp')]


#random forest classifier parameters

PIPELINE_NAME = 'RandomForest.pkl'
TRAINED_MODEL_DIR = './'
N_ESTIMATORS = 1050
MAX_DEPTH = 9

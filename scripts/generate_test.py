import numpy as np
import pandas as pd



def generate_test():
	"""Train the model."""
	X = pd.read_pickle('../packages/WNVPrediction/WNVPrediction/data/processed/X2.pkl')
	test_X = X.iloc[300:400,:]
	test_X.to_json('test_data.json',orient='records')

if __name__ == '__main__':
	generate_test()





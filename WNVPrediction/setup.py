import setuptools

setuptools.setup(
   name='WNVPrediction',
   version='0.1.0',
   description='A precitive tool for the City of Chicagos management of the West Nile Virus',
   author='Jonathan Jacobs',
   author_email='jonjacobs24@gmail.com',
   packages=setuptools.find_packages(),  #same as name
   #install_requires=['numpy', 'scikit-learn','pandas','joblib'] #external packages as dependencies
)

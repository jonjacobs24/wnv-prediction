# West Nile Virus Prediction

Relying on publicly available climatological data, as well as mosquito and public health monitoring data from the City of Chicago Department of Public Health, I predict the presence of West Nile Virus (WNV) carrying mosquitoes for the city of Chicago and surrounding areas. A Random Forest Classification algorithm implementation in Python predicts the presence of West Nile Virus carrying mosquitoes in the Chicago area with a final ROC AUC of 0.82. 

Feature analysis with the SHAP library largely confirms existing knowledge about the interactions between mosquito populations and climatological factors. Yet, there are interesting relationships between human factors and mosquito populations that are previously unreported and advance the model's ability to predict the presence of the WNV. Associations between WNV and haze, seasonality, and mist are prevalent factors in predicting the WNV according to this analysis and modeling. 

```
├── README.md 
│
├── 01 Cleaning and EDA.ipynb            <- Notebook that imports, cleans, and analyses raw data
│
├── 02 Modeling and Prediction.ipynb     <- ML and feature analysis on the data cleaned in the previous notebook
│
├── WNVFinalReport [JonathanJacobs].pdf  <- Detailed report on the entire process and findings
│
├── plots                                <- A selection of plots refereced in the report
│
├── data
│   ├── train.csv.zip                    <- Data from the City of Chicago Department of Public Health
│   ├── spray.csv.zip                    <- Insecticide spray data
│   ├── weather.csv.zip                  <- NOAA weather data at two weather stations in Chicago
│   ├── X2.pkl, y.pkl                    <- Cleaned and explored data ready for analysis
│   └── noaa_..._documentation.pdf.      <- Document explaining the form and types of data observed
│
├── Old Notebooks                        <- notebooks of a priliminary analysis

```

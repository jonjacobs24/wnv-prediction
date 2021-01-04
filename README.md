# West Nile Virus Prediction

Relying on publicly available climatological data, as well as mosquito and public health monitoring data from the City of Chicago Department of Public Health, I predict the presence of West Nile Virus (WNV) carrying mosquitoes for the City of Chicago and surrounding areas. A Random Forest Classification algorithm implementation in Python predicts the presence of West Nile Virus carrying mosquitoes in the Chicago area with a final ROC AUC of 0.82. 

Feature analysis with the SHAP library largely confirms existing knowledge about the interactions between mosquito populations and climatological factors. Yet, there are interesting relationships between human factors and mosquito populations that are previously unreported and advance the model's ability to predict the presence of the WNV. Associations between WNV and dry airborne particulate, seasonality, and mist are prevalent factors in predicting the WNV according to this analysis and modeling. 

```
pip install WNVPrediction                   <- Install this package

```

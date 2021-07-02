from dl_model import dataset_dl_prediction
from linear_ml_models import dataset_ml_prediction
from dataset_prepare import DataSet

import pandas as pd
import numpy as np


def run_rmr_predictions(epochs, lr, h_units, opt_name, dropout, weights=None):
	# Run classic machine learning models
	ml_models_summary = dataset_ml_prediction(DataSet)

	# Run deep learning model
	dl_model_summary = dataset_dl_prediction(epochs=epochs,
	                                         lr=lr,
	                                         h_units=h_units,
	                                         opt_name=opt_name,
	                                         dropout=dropout,
	                                         weights_file=weights)
	summary(ml_models_summary, dl_model_summary)


def summary(ml_summary, dl_summary):
	all_models_summary = [ml_summary, dl_summary]
	# get all models results
	all_models_summary = list(np.round(all_models_summary, decimals=4))
	models_summary = pd.DataFrame(
		{'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Support Vector Machines', 'Deep Learning'],
		 'R-squared Score': all_models_summary})

	models_summary = models_summary.sort_values(by='R-squared Score', ascending=False)

	# print results by format
	print("\n")
	print('*' * 125)
	print("\n\n", models_summary)
	print(
		f"\nWe've found out that the best model to predict the RMR is: {models_summary.iloc[0][0]} with R-2 Accuracy of {np.round(models_summary.iloc[0][1], 3)}%")
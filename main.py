from dataset_prepare import DataSet
from data_analysis import DataAnalyst
from linear_ml_models import DataMLPredictor
from dl_model import train, preprocess, run_optuna_experiment

import pandas as pd
import numpy as np
import datetime
import sys
import os

columns = ['age', 'RMR', 'Hight(m)', ' Weight(kg)', 'BMI', 'BF(%)', 'FM(kg)', 'FFM(kg)', 'gender']

path_to_dataset = 'dataset/15_6_RMR_ID_cmmon_equations_noID.csv'

# TODO:
# Check how to complete the missing data in the dataset - data completion. Maybe put some means or min-max or median. Use Tal's feature selection tutorial if nessecary  https://nbviewer.jupyter.org/github/taldatech/cs236756-intro-to-ml/blob/master/cs236756_tutorial_04_pca_feature_selection.ipynb https://scikit-learn.org/stable/modules/feature_selection.html
# At presentation, explain data
# Categorial data --> one hot vector
# Add small noise (0.005)
# relevant paper - https://arxiv.org/abs/2106.11189
# Don`t normalize target feautres ---> normalizing because if not DL is not working
# Manipulate data


# FINISHED:
# Use StandarScaler at ML models
# Example:
# scaler = StandardScaler()
# x_train_S = scaler.fit_Transform(X_train)
# x_test_s = scaler.tranform(X_test)
# add R-2 square to dl
# Use Optuna for hyper-parameters search in our deep learning net


def create_dataset():
	prepared_dataset = DataSet(path=path_to_dataset)
	prepared_dataset.CreateDataFrame()
	prepared_dataset.CreateSubSetDataFrame(columns=columns)
	prepared_dataset.PrintDataDescription()
	prepared_dataset.DropMissingRows()
	# prepared_dataset.ShowAllPlots()

	return prepared_dataset


def dataset_analysis():
	data_analysis = DataAnalyst(DataSet.df)
	data_analysis.dim_reduction()
	pass


def dataset_ml_prediction():
	data_predictor = DataMLPredictor(DataSet.df)
	return data_predictor.CompareModels()


def dataset_dl_prediction():
	X_train, X_test, y_train, y_test = preprocess(DataSet.df)
	return train(X_train, X_test, y_train, y_test)


def run_optuna(df):
	run_optuna_experiment(df)


def make_log_file():
	if not os.path.isdir('logs'):
		os.mkdir('logs')
	log_file = open('logs/MetabolicRate_' + str(str(datetime.datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")), 'w')
	sys.stdout = log_file
	return log_file


if __name__ == '__main__':
	log_file = make_log_file()

	DataSet = create_dataset()
	# run_optuna(DataSet.df)

	ml_models_summary = dataset_ml_prediction()
	dl_model_summary = dataset_dl_prediction()
	ml_models_summary.append(dl_model_summary)
	all_models_summary = ml_models_summary
	all_models_summary = list(np.round(all_models_summary, decimals=4))
	models_summary = pd.DataFrame({'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Support Vector Machines', 'Deep Learning'], 'R-squared Score': all_models_summary})
	models_summary.sort_values(by='R-squared Score', ascending=False)
	print("\n")
	print('*' * 125)
	print("\n\n", models_summary)
	print(
		f"\nWe've found out that the best model to predict the RMR is: {models_summary.iloc[0][0]} with R-2 Accuracy of {np.round(models_summary.iloc[0][1], 3)}%")
	log_file.close()

from optuna_search import OptunaRunStudy
from dl_model import use_model, preprocess
from linear_ml_models import DataMLPredictor
from dataset_prepare import DataSet

import pandas as pd
import numpy as np
import argparse
import datetime
import sys
import os

# 'Height_Squared', 'Weight_Squared', 'FM_Squared', 'FMM_Squared'
columns = ['RMR', 'age', 'Height(m)', 'Weight(kg)', 'BMI', 'BF(%)', 'FM(kg)', 'FFM(kg)', 'abdominal(cm)', 'Thyroidism', 'BS(yes/no)', 'diabetes', 'exercise', 'gender', 'No_P_diets', 'Few_P_Diets', 'Alot_P_diets']

path_to_dataset = 'dataset/15_6_RMR_ID_common_equations_noID_AddNoise.csv'


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def make_log_file():
	if not os.path.isdir('logs'):
		os.mkdir('logs')
	log_file = open(
		'logs/MetabolicRate_' + str(str(datetime.datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")) + str(".log"),
		'w')
	sys.stdout = log_file
	return log_file


parser = argparse.ArgumentParser(prog="Detector App", description="Starting Captain's Eye Main Algorithm App!")
parser.add_argument('-o', '--optuna', action="store_true",
                    help='Run Optuna optimization for detecting best DL model parameters')

parser.add_argument('--study-name', type=str, default='RMR-predict',
                    help='Run Optuna optimization for detecting best DL model parameters')

parser.add_argument('-t', '--trials', type=int, default=100,
                    help='Number of epoch for Deep Learning Model')

parser.add_argument('-p', '--path', type=str, required=False, default='dataset/15_6_RMR_ID_cmmon_equations_noID.csv',
                    help='Enter path to dataset')

parser.add_argument('-log', '--log', type=str2bool, default=True,
                    help='Write output to new log file at logs/ directory')

parser.add_argument('-e', '--epochs', type=int, default=5000,
                    help='Number of epoch for Deep Learning Model')

parser.add_argument('-lr', '--learning-rate', type=float, default=0.001,
                    help='Step size for the optimizer which trains the DL model')
parser.add_argument('-hu', '--hidden-units', type=int, default=750,
                    help='Number of hidden units in the hidden layer of the DL model')
parser.add_argument('-opt', '--optimizer-name', type=str, choices=["Adam", "RMSprop", "SGD"], default='Adam',
                    help='Optimizer for training the DL model')
parser.add_argument('-d', '--dropout', type=float, default=0.45,
                    help='Probability of dropout layer for turning off neurons in the DL model')

parser.add_argument('-w', '--weights_file', type=str, default="",
                    help='Path to weight file, if exist')
args = parser.parse_args()


def create_dataset():
	prepared_dataset = DataSet(path=path_to_dataset)
	prepared_dataset.CreateDataFrame()
	prepared_dataset.CountNaNInColumns()
	prepared_dataset.CreateSubSetDataFrame(columns=columns)
	prepared_dataset.PrintDataDescription()
	prepared_dataset.DropMissingRows()
	prepared_dataset.ShowAllPlots()
	exit(0)

	return prepared_dataset


def dataset_ml_prediction():
	data_predictor = DataMLPredictor(DataSet.df)
	return data_predictor.CompareModels()


def dataset_dl_prediction(epochs, lr, h_units, opt_name, dropout, weights_file):
	# start deep learning model prediction
	print("\n")
	print('*' * 125)
	print("Starting Deep Learning algorithm for prediction of the 'RMR' feature")
	print('*' * 125)
	print("\n")

	# preprocess
	X_train, X_test, y_train, y_test = preprocess(DataSet.df)

	# train and test
	dl_prediction = use_model(X_train, X_test, y_train, y_test, epochs, lr, h_units, opt_name, dropout, weights_file)
	return dl_prediction


def run_rmr_predictions(epochs, lr, h_units, opt_name, dropout, weights=None):
	# Run classic machine learning models
	ml_models_summary = dataset_ml_prediction()

	# Run deep learning model
	dl_model_summary = dataset_dl_prediction(epochs=epochs,
	                                         lr=lr,
	                                         h_units=h_units,
	                                         opt_name=opt_name,
	                                         dropout=dropout,
	                                         weights_file=weights)
	# get all models results
	ml_models_summary.append(dl_model_summary)
	all_models_summary = list(np.round(ml_models_summary, decimals=4))
	models_summary = pd.DataFrame({'Model': ['Linear Regression', 'Random Forest', 'XGBoost', 'Support Vector Machines', 'Deep Learning'],
	                               'R-squared Score': all_models_summary})

	models_summary = models_summary.sort_values(by='R-squared Score', ascending=False)

	# print results by format
	print("\n")
	print('*' * 125)
	print("\n\n", models_summary)
	print(f"\nWe've found out that the best model to predict the RMR is: {models_summary.iloc[0][0]} with R-2 Accuracy of {np.round(models_summary.iloc[0][1], 3)}%")


if __name__ == '__main__':

	if args.log:
		log_file = make_log_file()

	DataSet = create_dataset()
	if args.optuna:
		OptunaRunStudy(data_frame=DataSet.df,
		               epochs=args.epochs,
		               n_trials=args.trials,
		               study_name=args.study_name)
	else:
		run_rmr_predictions(epochs=args.epochs,
		                    lr=args.learning_rate,
		                    h_units=args.hidden_units,
		                    opt_name=args.optimizer_name,
		                    dropout=args.dropout,
		                    weights=args.weights_file)

	if args.log:
		log_file.close()

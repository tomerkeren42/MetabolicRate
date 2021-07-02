from rmr_predictor import run_rmr_predictions
from dataset_prepare import create_dataset
from optuna_search import OptunaRunStudy
import argparse
import datetime
import sys
import os

# Parameters for our own convenient. Should put all in arguments when running the program
# 'Height_Squared', 'Weight_Squared', 'FM_Squared', 'FMM_Squared'
subset_columns = ['RMR', 'age', 'Height(m)', 'Weight(kg)', 'BMI', 'BF(%)', 'FM(kg)', 'FFM(kg)', 'abdominal(cm)', 'Thyroidism', 'BS(yes/no)', 'diabetes', 'exercise', 'gender', 'No_P_diets', 'Few_P_Diets', 'Alot_P_diets']


def str2bool(v):
	"""
	takes string and check it's boolean value
	:param v: input str taken from argument
	:return: boolean interpretation of v
	"""
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


def make_log_file():
	"""
	if activated, create new log file named MetablicRate_<current time>.log in the logs/ directory
	:return: log file
	"""
	if not os.path.isdir('logs'):
		os.mkdir('logs')
	log_file = open(
		'logs/MetabolicRate_' + str(str(datetime.datetime.now()).split(".")[0].replace(":", "-").replace(" ", "_")) + str(".log"),
		'w')
	sys.stdout = log_file
	return log_file


def parse_arguments():
	# Parser arguments
	parser = argparse.ArgumentParser(prog="RMR Predictor",
	                                 description="Starting RMR predictor tool - trainable deep learning net, which is compared to other ML algorithms")

	parser.add_argument('-o', '--optuna', action="store_true",
	                    help='Run Optuna optimization for detecting best DL model parameters')
	parser.add_argument('-sn', '--study-name', type=str, default='RMR-predict',
	                    help='Run Optuna optimization for detecting best DL model parameters')
	parser.add_argument('--trials', type=int, default=100, nargs=1,
	                    help='Number of epoch for Deep Learning Model')

	parser.add_argument('-p', '--path', type=str, required=True, default='dataset/15_6_RMR_ID_common_equations_noID_AddNoise.csv',
	                    help='Enter path to dataset')

	parser.add_argument('--log', type=str2bool, default=True,
	                    help='Write output to new log file at logs/ directory')
	parser.add_argument('--epochs', type=int, default=5000,
	                    help='Number of epoch for Deep Learning Model')
	parser.add_argument('--learning-rate', type=float, default=0.001,
	                    help='Step size for the optimizer which trains the DL model')
	parser.add_argument('--hidden-units', type=int, default=750,
	                    help='Number of hidden units in the hidden layer of the DL model')
	parser.add_argument('--optimizer-name', type=str, choices=["Adam", "RMSprop", "SGD"], default='Adam',
	                    help='Optimizer for training the DL model')
	parser.add_argument('--dropout', type=float, default=0.45,
	                    help='Probability of dropout layer for turning off neurons in the DL model')
	parser.add_argument('--weights_file', type=str, default="",
	                    help='Path to weight file, if exist and do not want to train new net')

	return parser.parse_args()


if __name__ == '__main__':
	args = parse_arguments()
	if args.log:
		log_file = make_log_file()

	DataSet = create_dataset(args, subset_columns)
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

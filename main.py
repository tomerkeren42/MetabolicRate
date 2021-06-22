from dataset_prepare import DataSet
from data_analysis import DataAnalyst
from linear_ml_models import DataPredictor
from dl_model import train, preprocess
columns = ['age', 'RMR', 'Hight(m)', ' Weight(kg)', 'BMI', 'BF(%)', 'FM(kg)', 'FFM(kg)', 'neck']

path_to_dataset = 'dataset/15_6_RMR_ID_cmmon_equations_noID.csv'


def create_dataset():
	prepared_dataset = DataSet(path=path_to_dataset)
	prepared_dataset.CreateDataFrame()
	# prepared_dataset.CountNaNInColumns()
	prepared_dataset.CreateSubSetDataFrame(columns=columns)
	prepared_dataset.DropMissingRows()
	# prepared_dataset.PrintDataDescription()
	# prepared_dataset.ShowAllPlots()

	return prepared_dataset


def dataset_analysis():
	data_analysis = DataAnalyst(DataSet.df)
	data_analysis.dim_reduction()
	pass


def dataset_ml_prediction():
	data_predictor = DataPredictor(DataSet.df)
	# data_predictor.ComputeMSELinearApproaches()
	data_predictor.CompareModels()


def dataset_dl_prediction():
	X_train, X_test, y_train, y_test = preprocess(DataSet.df)
	train(X_train, X_test, y_train, y_test)


if __name__ == '__main__':
	DataSet = create_dataset()
	# dataset_analysis()
	# dataset_ml_prediction()
	dataset_dl_prediction()

from dataset_prepare import DataSet
from data_analysis import DataAnalyst
from linear_model import DataPredictor

all_columns = ['N', 'age', 'gender', 'RMR', 'Hight (m)', ' Weight (kg)', 'BMI', 'BF(%)', 'FM(kg)', 'FFM(kg)', 'abdominal (cm)', 'neck', 'exercise', 'exercise- kind', 'diabetes', 'Thyroidism', 'P_ diets', 'BS', 'GLU', 'HbA1C(%)', 'TG', 'HDL', 'LDL', 't.CHO', 'DATE_2', 'RMR_2', ' Weight (kg)_2', 'FM(%)_2', 'FM', 'FFM', 'weight(Dexa)', 'BMI(DEXA)']
subset_columns = ["age", "RMR", "Hight (m)", " Weight (kg)", "BMI", "BF(%)", "FM(kg)"]

path_to_dataset = 'dataset/16_3_RMR.csv'


def create_dataset():
	prepared_dataset = DataSet(path=path_to_dataset)
	prepared_dataset.CreateDataFrame()
	prepared_dataset.CountNaNInColumns()
	prepared_dataset.CreateSubSetDataFrame(columns=subset_columns)
	prepared_dataset.DropMissingRows()
	# prepared_dataset.PrintDataDescription()
	# prepared_dataset.ShowAllPlots()

	return prepared_dataset


def dataset_analysis():
	data_analysis = DataAnalyst(DataSet.df)
	data_analysis.dim_reduction()
	pass


def dataset_prediction():
	data_predictor = DataPredictor(DataSet.df)
	data_predictor.preprocess()
	# data_predictor.ComputeMSELinearApproaches()
	data_predictor.CompareModels()


if __name__ == '__main__':
	DataSet = create_dataset()
	# dataset_analysis()
	dataset_prediction()

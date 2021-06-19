from dataset_prepare import DataSet
from data_analysis import DataAnalyst
all_columns = ['N', 'age', 'gender', 'RMR', 'Hight (m)', ' Weight (kg)', 'BMI', 'BF(%)', 'FM(kg)', 'FFM(kg)', 'abdominal (cm)', 'neck', 'exercise', 'exercise- kind', 'diabetes', 'Thyroidism', 'P_ diets', 'BS', 'GLU', 'HbA1C(%)', 'TG', 'HDL', 'LDL', 't.CHO', 'DATE_2', 'RMR_2', ' Weight (kg)_2', 'FM(%)_2', 'FM', 'FFM', 'weight(Dexa)', 'BMI(DEXA)']
subset_columns = ["age", "RMR", "Hight (m)", " Weight (kg)", "BMI", "BF(%)", "FM(kg)", "abdominal (cm)", "neck"]

path_to_dataset = 'dataset/16_3_RMR.csv'

if __name__ == '__main__':
	DataSet = DataSet(path=path_to_dataset)
	DataSet.CreateDataFrame()
	DataSet.CreateSubSetDataFrame(columns=subset_columns)
	DataSet.ReplaceMissingRows(replacer=-1)
	# DataSet.DataSetSort(column="age", ascending=False)
	DataAnalyst = DataAnalyst(DataSet.to_use_df)
	DataAnalyst.dim_reduction()

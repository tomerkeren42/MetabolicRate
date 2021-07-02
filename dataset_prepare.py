import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class DataSet:
	def __init__(self, path):
		self.dataset_path = path
		self.df = None
		self.columns = []
		print('*' * 125)
		print("Preparing DataSet for prediction of the 'RMR' feature")
		print('*' * 125)

	def CreateDataFrame(self):
		self.df = pd.read_csv(self.dataset_path, na_values=['.', '#NUM!', '#VALUE!', '#VALUE!'])
		self.columns = list(self.df.columns.values)

	def CreateSubSetDataFrame(self, columns):
		self.columns = columns
		self.df = pd.DataFrame(data=self.df, columns=columns)
		print(f"\nCreated DataFrame out of columns: {self.columns}")

	def DataSetSort(self, column, ascending=False):
		self.df = self.df.sort_values(by=column, ascending=ascending)

	def DropMissingRows(self):
		self.df = self.df.dropna(how="any")

	def CountNaNInColumns(self):
		nan_dict = {}
		for col in self.columns:
			nan_dict[str(col)] = self.df[col].isna().sum()
		print(f"\nNaN in DataFrame summary: {nan_dict}")

	def ReplaceMissingRows(self, replacer=0):
		self.df = self.df.fillna(value=replacer)
		self.df.replace({".": str(replacer), "#NUM!": str(replacer), '#VALUE!': str(replacer)}, inplace=True)

	def CountOutLiers(self):
		print(f"\nDefined OutLiers as not between first and third quantile")
		outlier_dict = {}
		for k, v in self.df.items():
			q1 = v.quantile(0.25)
			q3 = v.quantile(0.75)
			irq = q3 - q1
			v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
			outlier_dict[str(k)] = np.shape(v_col)[0] * 100.0 / np.shape(self.df)[0]
		print(f"OutLiers in DataFrame summary: {outlier_dict}")

	def PrintDataDescription(self):
		print(f"\nData shape: {self.df.shape}")
		self.CountNaNInColumns()
		self.CountOutLiers()
		print("\n", self.df.describe())

	def ShowBoxPlot(self):
		fig, axs = plt.subplots(ncols=6, nrows=3, figsize=(20, 15))
		index = 0
		axs = axs.flatten()
		for k, v in self.df.items():
			sns.boxplot(y=k, data=self.df, ax=axs[index])
			index += 1
		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

	def ShowDistPlot(self):
		fig, axs = plt.subplots(ncols=6, nrows=3, figsize=(20, 15))
		index = 0
		axs = axs.flatten()
		for k, v in self.df.items():
			sns.histplot(v, ax=axs[index])
			index += 1
		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

	def ShowHeatMap(self):
		plt.figure(figsize=(20, 10))
		sns.heatmap(self.df.corr().abs(), annot=True)

	def ShowAllPlots(self):
		self.ShowHeatMap()
		self.ShowDistPlot()
		self.ShowBoxPlot()
		plt.show()


def create_dataset(args, subset_columns):
	"""
	Method for controlling the dataset created
	:return: the dataset object itself
	"""
	prepared_dataset = DataSet(path=args.path)
	prepared_dataset.CreateDataFrame()
	prepared_dataset.CountNaNInColumns()
	prepared_dataset.CreateSubSetDataFrame(columns=subset_columns)
	prepared_dataset.PrintDataDescription()
	prepared_dataset.DropMissingRows()
	# prepared_dataset.ShowAllPlots()

	return prepared_dataset
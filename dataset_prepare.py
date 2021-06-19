import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


class DataSet:
	def __init__(self, path):
		self.dataset_path = path
		self.df = None
		self.columns = []

	def CreateDataFrame(self):
		self.df = pd.read_csv(self.dataset_path, na_values=['.', '#NUM!', '#VALUE!'])
		self.columns = list(self.df.columns.values)
		print(f"\nCreated DataFrame out of columns: {self.columns}\n")

	def CreateSubSetDataFrame(self, columns):
		self.columns = columns
		self.df = pd.DataFrame(data=self.df, columns=columns)
		print(f"\nCreated DataFrame out of columns: {self.columns}\n")

	def DataSetSort(self, column, ascending=False):
		self.df = self.df.sort_values(by=column, ascending=ascending)

	def DropMissingRows(self):
		self.df = self.df.dropna(how="any")

	def CountNaNInColumns(self):
		for col in self.columns:
			print(f"The number of NaN in col: {col} is: {self.df[col].isna().sum()}")

	def ReplaceMissingRows(self, replacer=0):
		self.df = self.df.fillna(value=replacer)
		self.df.replace({".": str(replacer), "#NUM!": str(replacer), '#VALUE!': str(replacer)}, inplace=True)

	def CountOutLiers(self):
		print(f"Defined OutLiers as not between first and third quantile")
		for k, v in self.df.items():
			q1 = v.quantile(0.25)
			q3 = v.quantile(0.75)
			irq = q3 - q1
			v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
			perc = np.shape(v_col)[0] * 100.0 / np.shape(self.df)[0]
			print("Column %s outliers = %.2f%%" % (k, perc))

	def PrintDataDescription(self):
		print(f"Data shape: {self.df.shape}\n")
		print(self.CountOutLiers())
		print("\n", self.df.describe())

	def ShowBoxPlot(self):
		fig, axs = plt.subplots(ncols=7, nrows=1, figsize=(20, 10))
		index = 0
		axs = axs.flatten()
		for k, v in self.df.items():
			sns.boxplot(y=k, data=self.df, ax=axs[index])
			index += 1
		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

	def ShowDistPlot(self):
		fig, axs = plt.subplots(ncols=7, nrows=1, figsize=(20, 10))
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

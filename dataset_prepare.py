import pandas as pd


class DataSet:
	def __init__(self, path):
		self.dataset_path = path
		self.df = None

	def CreateDataFrame(self):
		self.df = pd.read_csv(self.dataset_path)

	def CreateSubSetDataFrame(self, columns):
		self.df = pd.DataFrame(data=self.df, columns=columns)

	def DataSetSort(self, column, ascending=False):
		self.df = self.df.sort_values(by=column, ascending=ascending)

	def DropMissingRows(self):
		self.df = self.df.dropna(how="any")

	def ReplaceMissingRows(self, replacer=0):
		self.df = self.df.fillna(value=replacer)
		self.df.replace({".": str(replacer), "#NUM!": str(replacer), '#VALUE!': str(replacer)}, inplace=True)


import pandas as pd


class DataSet:
	def __init__(self, path):
		self.dataset_path = path
		self.df = None
		self.to_use_df = None

	def CreateDataFrame(self):
		self.df = pd.read_csv(self.dataset_path)

	def CreateSubSetDataFrame(self, columns):
		self.to_use_df = pd.DataFrame(data=self.df, columns=columns)

	def DataSetSort(self, column, ascending=False):
		self.to_use_df = self.to_use_df.sort_values(by=column, ascending=ascending)

	def DropMissingRows(self):
		self.to_use_df = self.to_use_df.dropna(how="any")

	def ReplaceMissingRows(self, replacer=0):
		self.to_use_df = self.to_use_df.fillna(value=replacer)
		self.to_use_df.replace({".": str(replacer), "#NUM!": str(replacer), '#VALUE!': str(replacer)}, inplace=True)


import numpy as np


class DataAnalyst:
	def __init__(self, data_frame):
		self.data = data_frame.to_numpy()

	def dim_reduction(self):
		print(self.data)
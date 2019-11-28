from collections import defaultdict

class event_tree(object):
	def __init__(self, dataframe):
		self.dataframe = dataframe
		self.variables = list(self.dataframe.columns)
		self.paths = defaultdict(int)

	def unique_path_counts(self):
		for variable_number in range(0, len(self.variables)):
			dataframe_upto_variable = self.dataframe.loc[:, self.variables[0:variable_number+1]]
			for row in dataframe_upto_variable.itertuples():
				row = row[1:]
				self.paths[row] += 1
		return self.paths

	





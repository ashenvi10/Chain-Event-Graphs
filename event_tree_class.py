from collections import defaultdict

class event_tree(object):
	def __init__(self, dataframe):
		self.dataframe = dataframe
		self.variables = list(self.dataframe.columns)
		self.paths = defaultdict(int) #dict entry gives the path counts
		self.situations = self.nodes()
		self.root = self.nodes()[0]
		self.edge_information = defaultdict() 
		self.edge_labels = self.edge_labels_creation()
		self.edges = self.edge_creation()
		self.leaves = self.get_leaves()

	def counts_for_unique_path_counts(self):
		for variable_number in range(0, len(self.variables)):
			dataframe_upto_variable = self.dataframe.loc[:, self.variables[0:variable_number+1]]
			for row in dataframe_upto_variable.itertuples():
				row = row[1:]
				self.paths[row] += 1
		return self.paths

	def nodes(self):
		if len(list(self.paths.keys())) == 0:
			self.counts_for_unique_path_counts()
		node_list = ['s0'] #root node 
		vertex_number = 1 
		for path in list(self.paths.keys()):
			node_list.append('s%d' %vertex_number)
			vertex_number += 1
		return node_list

	def edges_labels_counts(self):
		edge_labels_list = ['root']
		edges_list = []
		for path in list(self.paths.keys()):
			path = list(path)
			edge_labels_list.append(path)
			if path[:-1] in edge_labels_list:
				path_edge_comes_from = edge_labels_list.index(path[:-1])
				edges_list.append([self.situations[path_edge_comes_from], self.situations[edge_labels_list.index(path)]])
			else:
				edges_list.append([self.situations[0], self.situations[edge_labels_list.index(path)]])
			self.edge_information[((*path,),(*edges_list[-1],))] = self.paths[tuple(path)]
		return self.edge_information

	def edge_labels_creation(self):
		if len(list(self.edge_information.keys())) == 0:
			self.edges_labels_counts()
		return [x[0] for x in list(self.edge_information.keys())]

	def edge_creation(self):
		if len(list(self.edge_information.keys())) == 0:
			self.edges_labels_counts()
		return [x[1] for x in list(self.edge_information.keys())]

	def get_leaves(self):
		situations_where_edges_start = [x[0] for x in self.edges]
		return [x[1] for x in self.edges if x[1] not in situations_where_edges_start]

	




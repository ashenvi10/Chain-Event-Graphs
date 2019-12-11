from collections import defaultdict

class event_tree(object):
	def __init__(self, dataframe):
		self.dataframe = dataframe
		self.variables = list(self.dataframe.columns)
		self.paths = defaultdict(int) #dict entry gives the path counts
		self.situations = self._nodes()
		self.root = self._nodes()[0]
		self.edge_information = defaultdict() 
		self.edge_labels = self._edge_labels_creation()
		self.edges = self._edge_creation()
		self.emanating_nodes = self._emanating_nodes()
		self.terminating_nodes = self._terminating_nodes()
		self.leaves = self._get_leaves()

	def _counts_for_unique_path_counts(self):
		for variable_number in range(0, len(self.variables)):
			dataframe_upto_variable = self.dataframe.loc[:, self.variables[0:variable_number+1]]
			for row in dataframe_upto_variable.itertuples():
				row = row[1:]
				self.paths[row] += 1
		return self.paths

	def _number_of_categories_per_variable(self):
		categories_of_variable = []
		for variable in self.variables:
			categories_of_variable.append(len(self.dataframe[variable].unique().tolist()))
		return categories_of_variable

	def _nodes(self):
		if len(list(self.paths.keys())) == 0:
			self._counts_for_unique_path_counts()
		node_list = ['s0'] #root node 
		vertex_number = 1 
		for path in list(self.paths.keys()):
			node_list.append('s%d' %vertex_number)
			vertex_number += 1
		return node_list

	def _edges_labels_counts(self):
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

	def _edge_labels_creation(self):
		if len(list(self.edge_information.keys())) == 0:
			self._edges_labels_counts()
		return [x[0] for x in list(self.edge_information.keys())]

	def _edge_creation(self):
		if len(list(self.edge_information.keys())) == 0:
			self._edges_labels_counts()
		return [edge_info[1] for edge_info in list(self.edge_information.keys())]

	def _get_leaves(self):
		return [edge_pair[1] for edge_pair in self.edges if edge_pair[1] not in self.emanating_nodes]

	def _emanating_nodes(self): #situations where edges start
		return [edge_pair[0] for edge_pair in self.edges]

	def _terminating_nodes(self): #situations where edges terminate
		return [edge_pair[1] for edge_pair in self.edges]

	def equivalent_sample_size(self):
		alpha = max(self._number_of_categories_per_variable()) - 1
		self.prior(alpha)

	def prior(self, equivalent_sample_size):
		prior = [0] *len(self.edges)
		sample_size_at_node = dict(float)
		sample_size_at_node[self.roof] = equivalent_sample_size
		assigned_nodes = list(self.root)
		while (all(prior) !=0) == False:
			for node in assigned_nodes:
				if node in self.emanating_nodes:
					number_of_occurences = self.emanating_nodes.count(node)
					equal_distribution_of_sample = sample_size_at_node[node]/number_of_occurences
					indices_of_relevant_terminating_nodes = [self.edges.index(edge_pair) for edge_pair in self.edges if edge_pair[0] == node]
					for index in indices_of_relevant_terminating_nodes:
						prior[index] = equal_distribution_of_sample
						sample_size_at_node[self.terminating_nodes[index]] = equal_distribution_of_sample
						assigned_nodes.append(self.terminating_nodes[index])
					assigned_nodes.remove(node)
		return prior

	def hyperstage(self):
		hyperstage = []
		number_of_edges = []
		for node in self.emanating_nodes:
			number_of_edges.append(len([edge_pair for edge_pair in self.edges if edge_pair[0]==node]))
		for value in set(number_of_edges):
			situations_with_value_edges = []
			for index in range(0, self.emanating_nodes):
				if number_of_edges[index] == value:
					situations_with_value_edges.append(self.emanating_nodes[index])
			hyperstage = hyperstage + [situations_with_value_edges]
		return hyperstage

	#Bayesian Agglommerative Hierarchical Clustering algorithm implementation
	#def AHC(self, prior = self.prior(alpha), hyperstage = self.hyperstage(), alpha = self.equivalent_sample_size()):





	




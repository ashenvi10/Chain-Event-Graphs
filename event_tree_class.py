from collections import defaultdict
import scipy.special as special
from operator import add

class event_tree(object):
	def __init__(self, dataframe):
		self.dataframe = dataframe
		self.variables = list(self.dataframe.columns)
		self.paths = defaultdict(int) #dict entry gives the path counts
		self.situations = self._nodes()
		self.root = self._nodes()[0]
		self.edge_information = defaultdict() 
		self.edge_counts = self._edge_counts()
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

	def _edge_counts(self):
		if len(list(self.edge_information.keys())) == 0:
			self._edges_labels_counts()
		return [self.edge_information[x] for x in list(self.edge_information.keys())]

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

	def default_equivalent_sample_size(self):
		alpha = max(self._number_of_categories_per_variable()) - 1
		self.default_prior(alpha)

	def default_prior(self, equivalent_sample_size):
		default_prior = [0] *len(self.edges)
		sample_size_at_node = dict(float)
		sample_size_at_node[self.roof] = equivalent_sample_size
		assigned_nodes = list(self.root)
		while (all(default_prior) !=0) == False:
			for node in assigned_nodes:
				if node in self.emanating_nodes:
					number_of_occurences = self.emanating_nodes.count(node)
					equal_distribution_of_sample = sample_size_at_node[node]/number_of_occurences
					indices_of_relevant_terminating_nodes = [self.edges.index(edge_pair) for edge_pair in self.edges if edge_pair[0] == node]
					for index in indices_of_relevant_terminating_nodes:
						default_prior[index] = equal_distribution_of_sample
						sample_size_at_node[self.terminating_nodes[index]] = equal_distribution_of_sample
						assigned_nodes.append(self.terminating_nodes[index])
					assigned_nodes.remove(node)
		return default_prior

	def default_hyperstage(self):
		default_hyperstage = []
		number_of_edges = []
		for node in self.emanating_nodes:
			number_of_edges.append(len([edge_pair for edge_pair in self.edges if edge_pair[0]==node]))
		for value in set(number_of_edges):
			situations_with_value_edges = []
			for index in range(0, self.emanating_nodes):
				if number_of_edges[index] == value:
					situations_with_value_edges.append(self.emanating_nodes[index])
			default_hyperstage = default_hyperstage + [situations_with_value_edges]
		return default_hyperstage

	def posterior(self, prior):
		return list(map(add, prior, self.edge_counts))

	def _function1(self, array):
		return special.gammaln(sum(array))

	def _function2(self, array):
		return sum([special.gammaln(number) for number in array])

	def _loglikehood(self, prior, posterior):
		prior_contribution = [(self._function1(element) - self._function2(element)) for element in prior]
		posterior_contribution = [(self._function2(element) - self._function1(element)) for element in posterior]
		return (sum(prior_contribution) + sum(posterior_contribution))

	def _bayesfactor(self, prior1, posterior1, prior2, posterior2):
		new_prior = list(map(add, prior1, prior2))
		new_posterior = list(map(add, posterior1, posterior2))
		return (self._function1(new_prior) - self._function1(new_posterior) + self._function2(new_posterior) - self._function2(new_prior) 
			+ self._function1(posterior1) + self._function1(posterior2) - self._function1(prior1) - self._function1(prior2) 
			+ self._function2(prior1) + self._function2(prior2) - self._function2(posterior1) - self._function2(posterior2))

	def _issubset(self, item, hyperstage):
		if any(set(item).issubset(element) for element in hyperstage) == True:
			return 1
		else:
			return 0 

	def _sort_list(self, a_list_of_lists):
		for index1 in range(0, len(a_list_of_lists)):
			for index2 in range(index1+1, len(a_list_of_lists)):
				array1 = a_list_of_lists[index1]
				array2 = a_list_of_lists[index2]
				if len(set(array1) & set(array2)) != 0:
					a_list_of_lists.append(list(set(array1) | set(array2)))
					a_list_of_lists[index1] = []
					a_list_of_lists[index2] = []
		new_list_of_lists = [list for list in a_list_of_lists if list != []]
		if new_list_of_lists == a_list_of_lists:
			return new_list_of_lists
		else:
			return self._sort_list(new_list_of_lists)

	#Bayesian Agglommerative Hierarchical Clustering algorithm implementation
	def AHC_transitions(self, prior = None, hyperstage = None, alpha = None):
		if alpha is None:
			alpha = self.default_equivalent_sample_size()
		if prior is None:
			prior = self.default_prior(alpha)
		if hyperstage is None:
			hyperstage = self.default_hyperstage()
		prior = prior.copy()
		hyperstage = hyperstage.copy()
		posterior = self.posterior(prior).copy()
		length = len(prior)
		likelihood = self._loglikehood(prior, posterior)

		posterior_conditional_probabilities = posterior.copy()

		merged_situation_list = []

		bayesfactor_score = 1
		while bayesfactor_score > 0:
			local_merges = []
			local_scores = []

			for situation1 in range(0, length):
				if all(items ==0 for items in posterior[situation1]) == False: #as we will set one of the merged situations/stages as 0 vectors later to retain indices
					model1 = [prior[situation1], posterior[situation1]]
					for situation2 in range(situation1 +1, length):
						if sub([situation1, situation2], hyperstage) == 1 and all(items ==0 for items in posterior[situation2]) == False:
							model2 = [prior[situation2], posterior[situation2]]
							local_scores.append(self._bayesfactor(*model1, *model2))
							local_merges.append([situation1,situation2])
			if max(local_scores) > 0:
				bayesfactor_score = max(local_scores)
				merged_situation_list.append(local_merges[local_scores.index(bayesfactor_score)])

				make_changes_to = merged_situation_list[-1]

				prior[make_changes_to[0]] = list(map(add, prior[make_changes_to[0]], prior[make_changes_to[1]]))
				posterior[make_changes_to[0]] = list(map(add, posterior[make_changes_to[0]], posterior[make_changes_to[1]]))

				prior[make_changes_to[1]] = [0] *len(prior[make_changes_to[0]])
				posterior[make_changes_to[1]] = [0] *len(prior[make_changes_to[0]])


				posterior_conditional_probabilities[make_changes_to[0]] = posterior[make_changes_to[0]]
				posterior_conditional_probabilities[make_changes_to[1]] = posterior[make_changes_to[0]]

				likelihood += bayesfactor_score

			else: 
				bayesfactor_score = 0

		mean_posterior_conditional_probabilities = []
		for array in posterior_conditional_probabilities:
			total = sum(array)
			mean_posterior_conditional_probabilities.append([round(element/total, 3) for element in array])


		return (self._sort_list(merged_situation_list), likelihood, mean_posterior_conditional_probabilities)
















	




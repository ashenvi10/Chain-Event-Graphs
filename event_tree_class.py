from collections import defaultdict
import scipy.special as special
from operator import add
import pydotplus as ptp
from IPython.display import Image
import random

class event_tree(object):
	def __init__(self, params):
		self.dataframe = params.get('dataframe')
		self.variables = list(self.dataframe.columns)
		self.sampling_zero_paths = params.get('sampling_zero_paths')
		self.paths = defaultdict(int) #dict entry gives the path counts
		self._dummy_paths = defaultdict(int)
		self.nodes = self._nodes() #contains leaves
		self.root = self._nodes()[0]
		self.edge_information = defaultdict() 
		self.edge_counts = self._edge_counts()
		self.edge_labels = self._edge_labels_creation()
		self.edges = self._edge_creation()
		self.emanating_nodes = self._emanating_nodes()
		self.terminating_nodes = self._terminating_nodes()
		self.leaves = self._get_leaves()
		self.situations = self._get_situations()
		self.edge_countset = self._edge_countset()

	def _counts_for_unique_path_counts(self):
		self._dummy_paths = defaultdict(int)
		for variable_number in range(0, len(self.variables)):
			dataframe_upto_variable = self.dataframe.loc[:, self.variables[0:variable_number+1]]
			for row in dataframe_upto_variable.itertuples():
				row = row[1:]
				self._dummy_paths[row] += 1

		if self.sampling_zero_paths != None:
			self.sampling_zeros(self.sampling_zero_paths)	

		depth = len(max(list(self._dummy_paths.keys()), key=len))
		keys_of_list = list(self._dummy_paths.keys())
		sorted_keys = []
		for deep in range(0,depth+1):
		    unsorted_mini_list = [key for key in keys_of_list if len(key) == deep]
		    sorted_keys = sorted_keys + sorted(unsorted_mini_list)

		for key in sorted_keys:
			self.paths[key] = self._dummy_paths[key]
	
		return self.paths

	def sampling_zeros(self, paths):
		#The paths must be tuples in a list in order
		#i.e. path[:-1] should already be a key in self.paths
		for path in paths:
			if (path[:-1] in list(self._dummy_paths.keys())) or len(path) == 1:
				self._dummy_paths[path] = 0
			else:
				raise ValueError("The path up to it's last edge should be added first. Ensure the tuple ends with a comma.")

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
				edges_list.append([self.nodes[path_edge_comes_from], self.nodes[edge_labels_list.index(path)]])
			else:
				edges_list.append([self.nodes[0], self.nodes[edge_labels_list.index(path)]])
			self.edge_information[((*path,),(*edges_list[-1],))] = self.paths[tuple(path)]
		return self.edge_information

	def _edge_counts(self):
		if len(list(self.edge_information.keys())) == 0:
			self._edges_labels_counts()
		return [self.edge_information[x] for x in list(self.edge_information.keys())]

	def _edge_countset(self):
		edge_countset = []
		for node in self.situations:
			edgeset = [edge_pair[1] for edge_pair in self.edges if edge_pair[0] == node]
			edge_countset.append([self.edge_counts[self.terminating_nodes.index(vertex)] for vertex in edgeset])
		return edge_countset

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

	def _get_situations(self):
		return [node for node in self.nodes if node not in self.leaves]

	def _emanating_nodes(self): #situations where edges start
		return [edge_pair[0] for edge_pair in self.edges]

	def _terminating_nodes(self): #situations where edges terminate
		return [edge_pair[1] for edge_pair in self.edges]

	def default_equivalent_sample_size(self):
		alpha = max(self._number_of_categories_per_variable()) 
		return alpha

	def default_prior(self, equivalent_sample_size):
		default_prior = [0] *len(self.situations)
		sample_size_at_node = dict()
		sample_size_at_node[self.root] = equivalent_sample_size
		to_assign_nodes = self.situations.copy()
		for node in to_assign_nodes:
			number_of_occurences = self.emanating_nodes.count(node)
			equal_distribution_of_sample = sample_size_at_node[node]/number_of_occurences
			default_prior[self.situations.index(node)] = [equal_distribution_of_sample] *number_of_occurences
			relevant_terminating_nodes = [self.terminating_nodes[self.edges.index(edge_pair)] for edge_pair in self.edges if edge_pair[0] == node]
			for terminating_node in relevant_terminating_nodes:
				sample_size_at_node[terminating_node] = equal_distribution_of_sample
		return default_prior

	def default_hyperstage(self):
		default_hyperstage = []
		number_of_edges = []
		for node in self.situations:
			number_of_edges.append(self.emanating_nodes.count(node))
		for value in set(number_of_edges):
			situations_with_value_edges = []
			for index in range(0, len(self.situations)):
				if number_of_edges[index] == value:
					situations_with_value_edges.append(self.situations[index])
			default_hyperstage = default_hyperstage + [situations_with_value_edges]
		return default_hyperstage

	def posterior(self, prior):
		posterior = []
		for index in range(0, len(prior)):
			posterior.append(list(map(add, prior[index], self.edge_countset[index])))
		return posterior

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
						if self._issubset([self.situations[situation1], self.situations[situation2]], hyperstage) == 1 and all(items ==0 for items in posterior[situation2]) == False:
							model2 = [prior[situation2], posterior[situation2]]
							local_scores.append(self._bayesfactor(*model1, *model2))
							local_merges.append([situation1,situation2])
			if local_scores != [] and max(local_scores) > 0:
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
			elif max(local_scores) <= 0:
				bayesfactor_score = 0

		mean_posterior_conditional_probabilities = []
		for array in posterior_conditional_probabilities:
			total = sum(array)
			mean_posterior_conditional_probabilities.append([round(element/total, 3) for element in array])

		list_of_merged_situations = self._sort_list(merged_situation_list)

		merged_situations = []
		for stage in list_of_merged_situations:
			merged_situations.append([self.situations[index] for index in stage])
		self._merged_situations = merged_situations
		self._mean_posterior_conditional_probabilities = mean_posterior_conditional_probabilities
		return (merged_situations, likelihood, mean_posterior_conditional_probabilities)

	def event_tree_figure(self, filename):
		nodes_for_event_tree = [(node, str(node)) for node in self.nodes]
		event_tree_graph = ptp.Dot(graph_type = 'digraph')
		for edge in self.edges:
			edge_index = self.edges.index(edge)
			edge_details = str(self.edge_labels[edge_index][-1]) + '\n' + str(self.edge_counts[edge_index])
			event_tree_graph.add_edge(ptp.Edge(edge[0], edge[1], label = edge_details, labelfontcolor="#009933", fontsize="10.0", color="black" ))
		for node in nodes_for_event_tree:
			event_tree_graph.add_node(ptp.Node(name = node[0], label = node[1], style = "filled"))
		event_tree_graph.write_png(str(filename) + '.png')
		return Image(event_tree_graph.create_png())

	def _generate_colours(self, number):
		_HEX = '0123456789ABCDEF'
		def startcolor():
			return '#' + ''.join(random.choice(_HEX) for _ in range(6))
		colours = []
		for index in range(0, number):
			newcolour = startcolor()
			while newcolour in colours:
				newcolour = startcolor()
			colours.append(newcolour)
		return colours

	def staged_tree_figure(self, filename):
		try:
			self._merged_situations
			self._mean_posterior_conditional_probabilities
		except NameError:
			print ("First run self.AHC_transitions()")
		else:
			number_of_stages = len(self._merged_situations)
			colours_for_tree = self._generate_colours(number_of_stages)
			staged_tree_graph = ptp.Dot(graph_type = 'digraph')
			for edge in self.edges:
				edge_index = self.edges.index(edge)
				edge_details = str(self.edge_labels[edge_index][-1])
				staged_tree_graph.add_edge(ptp.Edge(edge[0], edge[1], label = edge_details, labelfontcolor="#009933", fontsize="10.0", color="black" ))
			colour_index = 0
			for stage in self._merged_situations:
				for situation in stage:
					staged_tree_graph.add_node(ptp.Node(name = situation, label = situation, style = "filled", fillcolor = colours_for_tree[colour_index]))
				colour_index += 1
			staged_tree_graph.write_png(str(filename) + '.png')
			print("Number of stages is %s." %len(self._merged_situations))
			return Image(staged_tree_graph.create_png())









	




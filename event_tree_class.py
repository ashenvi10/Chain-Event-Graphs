from collections import defaultdict
import scipy.special as special
from operator import add
import pydotplus as ptp
from IPython.display import Image
import random
import itertools


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
		self._merged_situations = None
		self._mean_posterior_conditional_probabilities = None
		self._stage_colours = None #list of colours for the stages
		self._position_colours = None #pairs of (position, colour)

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
		number_of_stages = len(self._merged_situations)
		stage_colours = self._generate_colours(number_of_stages)
		colours_for_situations = []
		for node in self.nodes:
			stage_logic_values = [(node in stage) for stage in self._merged_situations] 
			if all(value == (False) for value in stage_logic_values):
				colours_for_situations.append((node, 'lightgrey'))
			else:
				colour_index = stage_logic_values.index((True))
				colours_for_situations.append((node, stage_colours[colour_index]))
		self._stage_colours = colours_for_situations

		return (merged_situations, likelihood, mean_posterior_conditional_probabilities)

	def _ceg_positions_edges(self):
		if self._merged_situations == None:
			raise ValueError("Run AHC transitions first.")
		if self._mean_posterior_conditional_probabilities == None:
			raise ValueError("Run AHC transitions first.")

		ceg_edge_labels = self.edge_labels.copy()
		ceg_edges = self.edges.copy()
		ceg_edge_counts = self.edge_counts.copy()

		ceg_positions = self.nodes.copy()
		cut_vertices = []

		for edge in self.edges:
			if edge[1] in self.leaves:
				edge_index = ceg_edges.index(edge)
				ceg_edges.pop(edge_index)
				ceg_edge_labels.append(ceg_edge_labels[edge_index])
				ceg_edge_counts.append(ceg_edge_counts[edge_index])
				ceg_edge_labels.pop(edge_index)
				ceg_edge_counts.pop(edge_index)
				ceg_edges.append((edge[0], 'w_inf'))
				cut_vertices.append(edge[0])
		ceg_positions = [node for node in ceg_positions if node not in self.leaves]
		ceg_positions.append('w_inf')
		cut_vertices = list(set(cut_vertices))

		while cut_vertices != ['s0']:
			cut_stages = []
			while cut_vertices != []:
				for vertex_1 in cut_vertices:
					cut_stage_1 = [vertex_1]
					vertex_1_info = []
					for edge in ceg_edges:
						if edge[0] == vertex_1:
							edge_index = ceg_edges.index(edge)
							vertex_1_info.append((ceg_edge_labels[edge_index][-1], edge[1]))
					vertex_1_info.sort(key = lambda tup: tup[0])
					for vertex_2 in cut_vertices:
						if vertex_1 != vertex_2:
							logic_values = [(vertex_1 in stage, vertex_2 in stage) for stage in self._merged_situations]
							if any(value == (True, True) for value in logic_values):
								vertex_2_info = []
								for edge in ceg_edges:
									if edge[0]== vertex_2:
										edge_index = ceg_edges.index(edge)
										vertex_2_info.append((ceg_edge_labels[edge_index][-1], edge[1]))
								#checking if the terminating nodes and edge labels match
								vertex_2_info.sort(key = lambda tup: tup[0])
								if vertex_1_info == vertex_2_info:
									cut_stage_1.append(vertex_2)
					cut_stages.append(cut_stage_1)
					for node in cut_stage_1:
						cut_vertices.remove(node)

			#new vertex set
			add_vertices = [x[0] for x in cut_stages]
			remove_vertices = list(itertools.chain(*cut_stages))
			remove_vertices = [node for node in remove_vertices if node not in add_vertices]
			replacement_nodes = []
			for node in remove_vertices:
				replace_with = [x[0] for x in cut_stages if node in x][0]
				replacement_nodes.append(replace_with)
			ceg_positions = [node for node in ceg_positions if node not in remove_vertices]

			#new edge set
			edges_to_remove = []
			edges_to_adapt = []

			for edge_index in range(0, len(ceg_edges)):
				edge = ceg_edges[edge_index]
				label  = ceg_edge_labels[edge_index][-1]
				if edge[0] in remove_vertices:
					replace_index = []
					edges_to_remove.append(edge)
					replace_node = replacement_nodes[remove_vertices.index(edge[0])]
					for index_2 in range(0, len(ceg_edges)):
						replace_edge = ceg_edges[index_2]
						if replace_edge == (replace_node, edge[1]) and ceg_edge_labels[index_2][-1] == label:
							replace_index.append(index_2)
					ceg_edge_counts[replace_index[0]] += ceg_edge_counts[edge_index]
				elif edge[1] in remove_vertices:
					edges_to_adapt.append(edge)

			# for edge in ceg_edges:
			# 	if edge[0] in remove_vertices:
			# 		edge_index = ceg_edges.index(edge)
			# 		edges_to_remove.append(edge)
			# 		replace_node = replacement_nodes[remove_vertices.index(edge[0])] 
			# 		replace_index = ceg_edges.index((replace_node, edge[1]))
			# 		ceg_edge_counts[replace_index] += ceg_edge_counts[edge_index]
			# 	elif edge[1] in remove_vertices:
			# 		edges_to_adapt.append(edge)

			for edge in edges_to_remove:
				edge_index = ceg_edges.index(edge)
				ceg_edges.pop(edge_index)
				ceg_edge_labels.pop(edge_index)
				ceg_edge_counts.pop(edge_index)

			for edge in edges_to_adapt:
				edge_index = ceg_edges.index(edge)
				ceg_edges.pop(edge_index)
				replace_node = replacement_nodes[remove_vertices.index(edge[1])]
				ceg_edges.insert(edge_index, (edge[0], replace_node))

			cut_vertices = [edge[0] for edge in ceg_edges if edge[1] in add_vertices]
			cut_vertices = list(set(cut_vertices))

		colours_for_positions = []
		for position in ceg_positions:
			position_colour = [pair for pair in self._stage_colours if pair[0] == position]
			if len(position_colour) == 0:
				colours_for_positions.append((position, 'lightgrey'))
			else:
				colours_for_positions.append(position_colour[0])
		self._position_colours = colours_for_positions

		return (ceg_positions, ceg_edges, ceg_edge_labels, ceg_edge_counts)

	def ceg_figure(self, filename):
		ceg_positions, ceg_edges, ceg_edge_labels, ceg_edge_counts = self._ceg_positions_edges() 
		nodes_for_ceg = [(node, str(node)) for node in ceg_positions]
		ceg_graph = ptp.Dot(graph_type = 'digraph')
		for edge_index in range(0, len(ceg_edges)):
			edge = ceg_edges[edge_index]
			edge_details = str(ceg_edge_labels[edge_index][-1]) + '\n' + str(ceg_edge_counts[edge_index])
			ceg_graph.add_edge(ptp.Edge(edge[0], edge[1], label = edge_details, labelfontcolor="#009933", fontsize="10.0", color="black" ))
		for node in nodes_for_ceg:
			fill_colour = [pair[1] for pair in self._position_colours if pair[0] == node[0]][0]
			ceg_graph.add_node(ptp.Node(name = node[0], label = node[1], style = "filled", fillcolor = fill_colour))
		ceg_graph.write_png(str(filename) + '.png')
		return Image(ceg_graph.create_png())

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
		random.seed(12345)
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
			self._stage_colours
		except ValueError:
			print ("First run self.AHC_transitions()")
		else:
			nodes_for_staged_tree = [(node, str(node)) for node in self.nodes]
			staged_tree_graph = ptp.Dot(graph_type = 'digraph')
			for edge in self.edges:
				edge_index = self.edges.index(edge)
				edge_details = str(self.edge_labels[edge_index][-1]) + '\n' + str(self.edge_counts[edge_index])
				staged_tree_graph.add_edge(ptp.Edge(edge[0], edge[1], label = edge_details, labelfontcolor="#009933", fontsize="10.0", color="black" ))
			for node in nodes_for_staged_tree:
				fill_colour = [pair[1] for pair in self._stage_colours if pair[0] == node[0]][0]
				staged_tree_graph.add_node(ptp.Node(name = node[0], label = node[1], style = "filled", fillcolor = fill_colour))
			staged_tree_graph.write_png(str(filename) + '.png')
			return Image(staged_tree_graph.create_png())









	




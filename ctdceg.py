from collections import defaultdict
import scipy.special as special
from operator import add
import pydotplus as ptp
from IPython.display import Image
import random
import itertools
import time
import functools
import datetime
import networkx as nx
import numpy as np
import math
from event_tree_class import ceg

class CtDceg(ceg):
	def __init__(self, params):
		'''holding time columns should be column names
		user needs to edit repeating_tree_paths
		user needs to edit known shape parameters:
		here they are arranged accoridng to the alphabetic ordering of 
		the keys of holding_time_information dictionary'''
		self.df = params.get('dataframe_with_ht')

		# self.df = self.df.replace(np.nan, '', regex=True) #seems to be breaking things!

		self.holding_time_columns = params.get('holding_time_columns')
		self.sample_size = params.get('sample_size')
		self.shape_parameters = params.get('shape_parameters')

		self.ceg_dataframe = self._ceg_dataframe()
		
		super().__init__({'dataframe': self.ceg_dataframe, 
						'sampling_zero_paths':params.get('sampling_zero_paths')})

		#a dict with keys as (walk of edge labels) and entry
		# as list of holding times for that walk
		# self.holding_time_information = defaultdict(list)
		# self._dummy_holding_time_information = defaultdict(list)
		# self.holding_times = self._holding_times()
		# self.holding_time_edges = self._holding_time_edges()

		# #adding the repeated tree values to the orginal tree
		# self.paths = self.repeating_tree_paths()
		# #ensuring this change is propagated to self.edge_information,
		# #self.edge_counts and self.edge_countset from the ceg class
		# self.edge_information = self._edges_labels_counts()
		# self.edge_counts = self._edge_counts()
		# self.edge_countset = self._edge_countset()

		#Used in the AHC method to store
		# - the list of edges merged into a single cluster
		# - the mean posterior holding time
		self._merged_edges = None
		self._mean_holding_times = None

		#list of colours for edges as (edge, colour)
		self._edge_colours = None
		self._edge_colours_optimal = None


	def _ceg_dataframe(self):
		'''takes the values from the first passage slice + the associated
		set of cyclic edges and saves that as the "original df" which is 
		passed on to the ceg class'''
		ceg_dataframe = self.df.copy()
		ceg_dataframe = ceg_dataframe.iloc[:self.sample_size]

		for col in self.holding_time_columns:
			ceg_dataframe.drop(col, axis = 1, inplace = True)
		return ceg_dataframe

	def _holding_times(self):
		'''saving the holding time values in a dictionary where the key is the
		sequence of edge labels until the edge for which the holding time is
		observed. The entry in the dictionary is the holding time'''
		for col in self.holding_time_columns:
			col_index = self.df.columns.get_loc(col)
			df_upto_col = self.df.iloc[:, 0:col_index+1].copy()
			for row in df_upto_col.itertuples():
				labelrow = row[1:-1]
				labelrow = [edge_label for edge_label in labelrow if type(edge_label) == str and
							edge_label != '']
				print(labelrow)
				if labelrow[0] == 'Community low loop':
					labelrow[0] = 'Community'
				elif labelrow[0] == 'Communal low loop':
					labelrow[0] = 'Communal'

				labelrow = tuple(labelrow)
				print(labelrow)
				edge = [key[1] for key in self.edge_information if key[0] == labelrow]
				keylabel  = (labelrow, edge[0])

				if row[-1] != np.nan and row[-1] != 'NaN' and row[-1] != '' and row[-1] != 'nan':
					self._dummy_holding_time_information[keylabel].append(row[-1])

		holding_times = []
		for key in sorted(self._dummy_holding_time_information.keys()):
			self.holding_time_information[key] = self._dummy_holding_time_information[key]
			holding_times.append(self._dummy_holding_time_information[key])

		return holding_times

	def _holding_time_edges(self):
		holdingtime_edges = []
		for key in sorted(self.holding_time_information.keys()):
			holdingtime_edges.append(key[1])
		return holdingtime_edges
			
	'''def repeating_tree_paths(self):
		e.g. conditions: 
		if 'communal low loop' or 'community low loop' then
			repeat from row[1]
		else	
			row[1] == 'Low' then repeating from row[1]
			row[2] == 'High' then repeating from row[3] 
		Note that conditions have to be for df minus holding time columns
		
		df_repeating = self.df.copy()
		df_repeating = df_repeating.iloc[self.sample_size:]

		#conditions:
		for variable_number in range(1, len(self.variables)):
			dataframe_upto_variable = df_repeating.loc[:, self.variables[0:variable_number+1]]
			for row in dataframe_upto_variable.itertuples():
				row = row[1:]
				row = [edge_label for edge_label in row if edge_label != np.nan and
				edge_label != 'NaN' and edge_label != 'nan' and edge_label != '']
				
				if row[0] == 'Communal low loop' and len(row) >= 2:
					row[0] = 'Communal'
					row = tuple(row)
					self.paths[row] += 1
				elif row[0] == 'Community low loop' and len(row) >= 2:
					row[0] = 'Community'
					row = tuple(row)
					self.paths[row] += 1

				elif (row[0] == 'Communal' or row[0] == 'Community') and row[1] == 'High' and len(row) >= 4:
					row = tuple(row)
					self.paths[row] += 1

		return self.paths'''

	def _holdingtime_loglikelihood(self, prior_beta, prior_gamma, known_shape, holdingtimes):
		'''calculating log likelihood given a inv gamma priors, known Weibull shape
		parameter and the holding times'''

		likelihood = 0
		n = len(holdingtimes)
		posterior_beta = prior_beta + n
		posterior_gamma = (prior_gamma + sum([time** known_shape for time in holdingtimes]))

		if all(items != 0 for items in holdingtimes) == True:
			likelihood += (prior_beta * math.log(prior_gamma) + special.gammaln(posterior_beta)
				+ n * math.log(known_shape) + 
				(known_shape - 1) * sum([math.log(time) for time in holdingtimes]) 
				- special.gammaln(prior_beta) - posterior_beta * math.log(posterior_gamma))
		else:
			pass

		return likelihood

	def _holdingtime_bayesfactor(self, prior_beta1, prior_gamma1, known_shape1, holdingtimes1,
									prior_beta2, prior_gamma2, known_shape2, holdingtimes2):
		newprior_beta = prior_beta1 + prior_beta2
		newprior_gamma = prior_gamma1 + prior_gamma2
		newholdingtimes = holdingtimes1 + holdingtimes2
		newshape = known_shape1

		return (self._holdingtime_loglikelihood(newprior_beta, newprior_gamma, 
			newshape, newholdingtimes) -
		self._holdingtime_loglikelihood(prior_beta1, prior_gamma1, known_shape1, holdingtimes1)
		-self._holdingtime_loglikelihood(prior_beta2, prior_gamma2, known_shape2, holdingtimes2))

	def default_prior_beta(self):
		alpha = self.default_equivalent_sample_size()
		default_prior = self.default_prior(alpha)
		prior_beta = []
		for key in sorted(self.holding_time_information.keys()):
			situation = key[1][0]
			situation_index = self.situations.index(situation)
			prior_beta.append(default_prior[situation_index][0])
		return prior_beta

	def default_prior_gamma(self, phantom_holding_time = 2):
		prior_gamma = [phantom_holding_time**shape for shape in self.shape_parameters]
		return prior_gamma

	def default_holdingtime_hyperstage(self):
		default_hyperstage = []
		for shape in set(self.shape_parameters):
			hyper = []
			for index, value in enumerate(self.shape_parameters):
				if value == shape:
					hyper.append(self.holding_time_edges[index])
			default_hyperstage.append(hyper)

		return default_hyperstage

	def AHC_holding_times(self, phantom_holding_time = 2, prior_beta = None,
						prior_gamma = None, hyperstage = None):
		''' Bayesian Agglommerative Hierachical Clustering algorithm implementation. 
		It returns a list of lists of edges which have been merged together, 
		the likelihood of the final model and the mean posterior 
		conditional probabilities of the stages.'''
		if prior_beta is None:
			prior_beta = self.default_prior_beta()
		if prior_gamma is None:
			prior_gamma = self.default_prior_gamma(phantom_holding_time)
		if hyperstage is None:
			hyperstage = self.default_holdingtime_hyperstage()
		length = len(prior_beta)
		shape_parameters = self.shape_parameters.copy()
		holding_times = self.holding_times.copy()

		likelihood = 0
		for i in range(len(prior_beta)):
			likelihood += self._holdingtime_loglikelihood(prior_beta[i],
						prior_gamma[i], shape_parameters[i], holding_times[i])
		
		merged_edges_list = []

		bayes_factor_score = 1
		while bayes_factor_score > 0:
			local_merges = []
			local_scores = []

			for edge1 in range(length):
				if all(items ==0 for items in holding_times[edge1]) == False: #as we will set one of the merged situations/stages as 0 vectors later to retain indices	
					model1 = [prior_beta[edge1], prior_gamma[edge1],
							  shape_parameters[edge1], holding_times[edge1]]
					for edge2 in range(edge1 +1, length):
						if self._issubset([self.holding_time_edges[edge1], self.holding_time_edges[edge2]], 
							hyperstage) ==1 and all(items ==0 for items in holding_times[edge2]) == False:
							model2 =  [prior_beta[edge2], prior_gamma[edge2],
							  		   shape_parameters[edge2], holding_times[edge2]]
							local_scores.append(self._holdingtime_bayesfactor(*model1, *model2))
							local_merges.append([edge1, edge2])
			if local_scores != [] and max(local_scores) >0:
				bayes_factor_score = max(local_scores)
				merged_edges_list.append(local_merges[local_scores.index(bayes_factor_score)])

				make_changes_to = merged_edges_list[-1]

				prior_beta[make_changes_to[0]] = prior_beta[make_changes_to[0]] + prior_beta[make_changes_to[1]]
				prior_gamma[make_changes_to[0]] = prior_gamma[make_changes_to[0]] + prior_gamma[make_changes_to[1]]
				holding_times[make_changes_to[0]] = holding_times[make_changes_to[0]] + holding_times[make_changes_to[1]]

				prior_beta[make_changes_to[1]] = 0
				prior_gamma[make_changes_to[1]] = 0
				holding_times[make_changes_to[1]] = [0]

				likelihood += bayes_factor_score
			elif max(local_scores) <= 0 or local_scores == []:
				bayes_factor_score = 0

		list_of_merged_edges = self._sort_list(merged_edges_list)

		merged_edges = []
		for stage in list_of_merged_edges:
			merged_edges.append([self.holding_time_edges[index] for index in stage])
		self._merged_edges = merged_edges
		self._no_holding_time()
		
		mean_holding_times = []
		for index, edge in enumerate(self.holding_time_edges):
			posterior_beta = prior_beta[index] + len(holding_times[index])
			posterior_gamma = (prior_gamma[index] + 
						sum([time**shape_parameters[index] for time in holding_times[index]]))
			
			# prior_beta being zero means its part of a cluster but is not the representative
			# holding times being zero means same as above OR the edge has no holding time
			if prior_beta[index] == 0 or all(item == 0 for item in holding_times[index]) == True:
				mean_holding_times.append(0)
			elif posterior_beta > 1:
				mean_holding_times.append(round(posterior_gamma/(posterior_beta -1),2)) #inverse gamma mean
			else:
				raise ValueError("check posterior beta")

		new_mean_holding_times = self._distribute_mean_holding_time(mean_holding_times)
		self._mean_holding_times = new_mean_holding_times

		number_of_holdingtime_stages = len(self._merged_edges)
		stage_colours = self._generate_colours(number_of_holdingtime_stages)
		colours_for_edges = []

		for edge in self.holding_time_edges:
			stage_logic_values = [(edge in stage) for stage in self._merged_edges] 
			if all(value == (False) for value in stage_logic_values):
				colours_for_edges.append((edge, 'black'))
			else:
				colour_index = stage_logic_values.index((True))
				colours_for_edges.append((edge, stage_colours[colour_index]))
		self._edge_colours = colours_for_edges

		return (merged_edges, likelihood, new_mean_holding_times)

	def _no_holding_time(self):
		zero_holding_times = []
		for index, value in enumerate(self.holding_times):
			if all(items == 0 for items in value) == True:
				zero_holding_times.append(self.holding_time_edges[index])
		self._merged_edges.append(zero_holding_times)

		

	def _distribute_mean_holding_time(self, mean_holding_times):

		merged_edges = self._merged_edges.copy()

		for edge in self.holding_time_edges:
			which_cluster = [cluster for cluster in merged_edges if edge in cluster]
			if which_cluster == []:
				merged_edges.append([edge])

		new_mean = []
		cluster_means = []
		for cluster in merged_edges:
			not_zero = 0
			for edge in cluster:
				index = self.holding_time_edges.index(edge)
				if mean_holding_times[index] != 0:
					not_zero += 1
					cluster_means.append(mean_holding_times[index])
			if not_zero == 0:
				cluster_means.append(0)

		for edge in self.holding_time_edges:
			which_cluster = [merged_edges.index(cluster) for cluster in merged_edges if edge in cluster]
			index = which_cluster[0]
			new_mean.append(cluster_means[index])

		return new_mean

	def ctdceg_staged_tree_figure(self, filename):
		'''function to draw the staged tree for the process described by the dataset.'''
		try:
			self._merged_situations
			self._mean_posterior_conditional_probabilities
			self._stage_colours
			self._edge_colours
		except ValueError:
			print ("First run self.AHC_transitions() and self.AHC_holding_times()")
		else:
			nodes_for_staged_tree = [(node, str(node)) for node in self.nodes]
			staged_tree_graph = ptp.Dot(graph_type = 'digraph', rankdir = 'LR')
			for edge in self.edges:
				edge_colour = [pair[1] for pair in self._edge_colours if pair[0] == edge]
				if edge_colour == []:
					edge_colour = ["black"]
				edge_index = self.edges.index(edge)
				edge_details = str(self.edge_labels[edge_index][-1]) + '\n' + str(self.edge_counts[edge_index])
				staged_tree_graph.add_edge(ptp.Edge(edge[0], edge[1], label = edge_details, labelfontcolor="#009933", fontsize="10.0", color= edge_colour[0]))
			for node in nodes_for_staged_tree:
				fill_colour = [pair[1] for pair in self._stage_colours if pair[0] == node[0]][0]
				staged_tree_graph.add_node(ptp.Node(name = node[0], label = node[1], style = "filled", fillcolor = fill_colour))
			staged_tree_graph.write_png(str(filename) + '.png')
			return Image(staged_tree_graph.create_png())












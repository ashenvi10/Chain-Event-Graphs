import pandas as pd
from ctdceg import CtDceg
import random
import numpy as np
from data_generation_CTDCEG import data_generation
from scipy.stats import entropy
import math

ht_col = ['Fall time', 'Outcome time']
all_paths = [('Community',),
			 ('Communal',),
			 ('Community', 'Low'),
			 ('Community', 'High'),
			 ('Communal', 'Low'),
			 ('Communal', 'High'),
             ('Community', 'Low', 'Fall'),
             ('Community', 'High', 'Not treated'),
             ('Community', 'High', 'Treated'),
             ('Communal', 'Low', 'Fall'), 
             ('Communal', 'High', 'Not treated'),
             ('Communal', 'High', 'Treated'),
             ('Community', 'High', 'Not treated', 'Fall'), 
             ('Community', 'High', 'Treated', 'Fall'),
             ('Communal', 'High', 'Not treated', 'Fall'),
             ('Communal', 'High', 'Treated', 'Fall'),
             ('Community', 'Low', 'No fall'),
             ('Communal', 'Low', 'No fall'),
             ('Community', 'Low', 'Fall', 'Loop'), 
             ('Community', 'High', 'Not treated', 'No fall'),
             ('Community', 'High', 'Treated', 'No fall'),
             ('Communal', 'Low', 'Fall', 'Loop'),
             ('Communal', 'High', 'Not treated', 'No fall'),
             ('Communal', 'High', 'Treated', 'No fall'),
             ('Community', 'High', 'Not treated', 'Fall', 'Loop'),
             ('Community', 'High', 'Not treated', 'Fall', 'Move'),
             ('Community', 'High', 'Not treated', 'Fall', 'Complications'),
             ('Community', 'High', 'Treated', 'Fall', 'Loop'),
             ('Community', 'High', 'Treated', 'Fall', 'Move'),
             ('Community', 'High', 'Treated', 'Fall', 'Complications'),
             ('Communal', 'High', 'Not treated', 'Fall', 'Loop'),
             ('Communal', 'High', 'Not treated', 'Fall', 'Complications'),
             ('Communal', 'High', 'Treated', 'Fall', 'Loop'),
             ('Communal', 'High', 'Treated', 'Fall', 'Complications'),
             ]

shape_parameters = [2.2, 0.5, 1, 0, 
			  1.6, 0.5, 1, 0,
			  1.2, 1, 0,
			  2.2, 0.5, 1, 1.6, 0,
			  1.6, 0.5, 1, 1.6, 0,
			  1.2, 1, 0 ]

#generating stage clusters: total 11
gt_situation_clusters = [['s0'], ['s1'], ['s2'], ['s3', 's5'],
                        ['s4', 's6'], ['s7'], ['s8', 's12'], 
                        ['s11'], ['s9', 's13'],
                        ['s15', 's17'], ['s20', 's22']]

#generating edge clusters: total 9
gt_edge_clusters = [[('s15', 's25'), ('s17', 's27'), ('s20', 's29'), ('s22', 's32')],
                    [('s15', 's26'), ('s17', 's28'), ('s20', 's30'), ('s22', 's33')],
                     [('s9', 's19'), ('s13', 's24')], 
                     [('s4', 's9'), ('s6', 's13')],
                     [ ('s20', 's31'), ('s22', 's34')], 
                     [('s7', 's15')],
                     [('s11', 's20')], 
                     [('s8', 's17'), ('s12', 's22')],
                     [('s7', 's16'), ('s8', 's18'), ('s4', 's10'), 
                     ('s11', 's21'), ('s12', 's23'), ('s6', 's14')]]

#generating conditional transition probabilities
#indexed same as self.situations
gt_transition_probabilities = [[0.35, 0.65], [0.64, 0.36], [0.5, 0.5], [0.3, 0.7],
                              [0.25, 0.75], [0.3, 0.7], [0.25, 0.75], [0.78, 0.22], 
                              [0.51, 0.49], [1], [0.7, 0.3], [0.51, 0.49], [1],
                              [0.3, 0.7], [0.3, 0.7], [0.25, 0.5, 0.25], [0.25, 0.5, 0.25]]

#generating holding time Weibull scale parameters
#indexed same as self.holding_time_edges
gt_scale_parameters = [224.69, 5.4, 25.3, 0, 
                       390.05, 5.4, 25.3, 0, 
                       433.97, 45.48, 0,
                       290.19, 5.4, 25.3, 170.14, 0,
                       390.05, 5.4, 25.3, 170.14, 0, 
                       433.97, 45.48, 0]

population_scale  = [1, 3, 5, 10, 15, 20]
population = [500*scale for scale in population_scale]

random.seed(123456)
seeds = np.random.randint(0,100000,1)

df_situation_analysis = pd.DataFrame(columns=['Population', 'Imaginary_sample_size', 'Number_of_situation_clusters', 
                              'Situation_likelihood', 'Situation_error'])
df_edge_analysis = pd.DataFrame(columns=['Population', 'Phantom_holding_time', 
                              'Number_of_edge_clusters', 'Edge_likelihood', 'Edge_error'])

# KL divergence for two Weibull distributions with same shape parameter
def KL_Weibull(shape, scale1, scale2):
      KL = shape* (math.log(scale2) - math.log(scale1)) + (scale1/scale2)**shape - 1
      return KL

for seed in seeds:
      data_generation(seed)
      for i in range(1,7):
            filename = '/Users/Aditi/Dropbox/PhD stuff/Papers/CT-DCEG/data/ctdceg_modified_data%s.xlsx' %i
            df_ht = pd.read_excel(filename)
            columns = list(df_ht.columns)
            remove_indices = columns.index('Residence')
            df_ht = df_ht.drop(df_ht.columns[:remove_indices], axis = 1)

            df1 = CtDceg({'dataframe_with_ht': df_ht, 'holding_time_columns': ht_col,
			  'sample_size': population[i-1], 'shape_parameters': shape_parameters}) 

            #checking for sampling zeros
            sampling_zeros = [path for path in all_paths if path not in df1.paths]
            if sampling_zeros != []:
                  raise Error('sampling zeros exist')

            # values for the situation clusters
            iss_list = []
            situation_likelihood = []
            number_of_situation_clusters = []
            situational_error = []

            for j in range(1, 2): 
                  #setting imaginary sample size
                  iss = j * 0.25
                  iss_list.append(iss)
                  situation_likelihood.append(df1.AHC_transitions(alpha = iss)[1])

                  merged_situations = df1._merged_situations.copy()

                  for situation in df1.situations:
                        which_cluster = [cluster for cluster in merged_situations if situation in cluster]
                        if which_cluster == []:
                              merged_situations.append([situation])

                  number_of_situation_clusters.append(len(merged_situations))

                  # Computing KL divergence
                  KL_div = 0
                  for probs1, probs2 in zip(gt_transition_probabilities, df1._mean_posterior_conditional_probabilities):
                        KL_div += entropy(probs1, probs2)

                  situational_error.append(KL_div)


            df_situations = pd.DataFrame({'Population': [population[i-1]]*len(iss_list), 
                                          'Imaginary_sample_size': iss_list,
                                          'Number_of_situation_clusters': number_of_situation_clusters, 
                                          'Situation_likelihood': situation_likelihood, 
                                          'Situation_error': situational_error})
            df_situation_analysis = df_situation_analysis.append(df_situations,
                                                                  ignore_index = True,
                                                                  sort = False)

            #values for the edge clusters
            # for a fixed iss of 4
            
            phantom_holding_time_list = []
            edge_likelihood = []
            number_of_edge_clusters = []
            edge_error = []

            for k in range(1,2): 
                  phantom_ht = k
                  phantom_holding_time_list.append(phantom_ht)
                  edge_likelihood.append(df1.AHC_holding_times(phantom_holding_time = phantom_ht)[1])
                  
                  merged_edges = df1._merged_edges.copy()

                  for edge in df1.holding_time_edges:
                        which_cluster = [cluster for cluster in merged_edges if edge in cluster]
                        if which_cluster == []:
                              merged_edges.append([edge])

                  number_of_edge_clusters.append(len(merged_edges))

                  temp_mean_holding_times = df1._mean_holding_times.copy()
                  # mean lambda! not holding time
                  mean_holding_times = [time**(1/shape) if time != 0 else 0 for time,shape in zip(temp_mean_holding_times, shape_parameters)]

                  # Computing KL divergence
                  KL_div = 0
                  for shape, scale1, scale2 in zip(shape_parameters, gt_scale_parameters, mean_holding_times):
                        if shape ==0:
                              pass
                        else:
                              KL_div += KL_Weibull(shape, scale1, scale2)

                  edge_error.append(KL_div)

            df_edges = pd.DataFrame({'Population': [population[i-1]]*len(phantom_holding_time_list), 
                                    'Phantom_holding_time': phantom_holding_time_list, 
                                    'Number_of_edge_clusters': number_of_edge_clusters, 
                                    'Edge_likelihood': edge_likelihood, 
                                    'Edge_error': edge_error})
            df_edge_analysis = df_edge_analysis.append(df_edges,
                                                      ignore_index = True,
                                                      sort = False)

df_situation_analysis.Population = df_situation_analysis.Population.astype('category')
df_situation_analysis.Imaginary_sample_size = df_situation_analysis.Imaginary_sample_size.astype('float')
df_situation_analysis.Number_of_situation_clusters = df_situation_analysis.Number_of_situation_clusters.astype('int')

df_edge_analysis.Population = df_edge_analysis.Population.astype('category')
df_edge_analysis.Phantom_holding_time = df_edge_analysis.Phantom_holding_time.astype('float')
df_edge_analysis.Number_of_edge_clusters = df_edge_analysis.Number_of_edge_clusters.astype('int')

df_situation_analysis.to_excel('/Users/Aditi/Dropbox/PhD stuff/Papers/CT-DCEG/data/situation_analysis.xlsx')
df_edge_analysis.to_excel('/Users/Aditi/Dropbox/PhD stuff/Papers/CT-DCEG/data/edge_analysis.xlsx')



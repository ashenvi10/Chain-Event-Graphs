import pandas as pd
import numpy as np 
from event_tree_class import event_tree
from collections import defaultdict 
import csv
#uncomment (remove the hash) before the @timeit decorator before _ceg_positions_edges(self) and
# _ceg_positions_edges_optimal(self)

datasets_1 = ['CHDS.latentexample1.xlsx', 'falls.xlsx', 'rdceg_falls.xlsx', 'hayes-roth.xlsx','balance_scale.xlsx', 'glass.xlsx',
'iris.xlsx','tic-tac-toe.xlsx', 'breast-cancer.xlsx', 'nursery.xlsx']

dataset_dict = defaultdict(list)

i = 0
for dataname in datasets_1:
	filename = 'Datasets/' + dataname 
	data = pd.read_excel(filename)
	df = event_tree({'dataframe' : data})
	information = []
	information.append(len(df.situations))
	information.append(max(df.shortest_path))
	df.AHC_transitions()
	information.append(df._ceg_positions_edges())
	information.append(df._ceg_positions_edges_optimal())
	dataset_dict[i] = information
	i += 1
	print(information)

with open("Datasets/experiments.csv", "w") as f:
    writer = csv.writer(f)
    for entry in dataset_dict:
      writer.writerow([entry, dataset_dict[entry]])
f.close() 
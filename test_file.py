import pandas as pd
from event_tree_class import event_tree

df = pd.read_excel('CHDS.latentexample1.xlsx')

dataframe1 = event_tree(df)
#print(dataframe1.counts_for_unique_path_counts())
#print(dataframe1.situations)
#print(dataframe1.edge_labels)
print(dataframe1.leaves)


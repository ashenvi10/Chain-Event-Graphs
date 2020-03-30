import pandas as pd
from event_tree_class import event_tree

df = pd.read_excel('CHDS.latentexample1.xlsx')

dataframe1 = event_tree(df)
#print(dataframe1.default_hyperstage())
#dataframe1.event_tree_figure('event_tree')
#print(dataframe1.AHC_transitions(alpha = 3, hyperstage = [['s0'], ['s1', 's2'], ['s3', 's4', 's5', 's6'],['s7', 's8' ,'s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18']]))
print(dataframe1.AHC_transitions())
#print(dataframe1.edge_countset)

dataframe1.staged_tree_figure('staged_tree')



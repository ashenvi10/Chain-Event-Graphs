import pandas as pd
from event_tree_class import event_tree

df = pd.read_excel('CHDS.latentexample1.xlsx')

dataframe1 = event_tree(df)
print(dataframe1.unique_path_counts())
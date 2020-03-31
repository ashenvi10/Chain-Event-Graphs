import pandas as pd
from event_tree_class import event_tree

df = pd.read_excel('CHDS.latentexample1.xlsx')

dataframe1 = event_tree(df)

nodes = []
for i in range(0, 43):
    nodes.append('s%d' %i)

def test_nodes():
    assert dataframe1.nodes == nodes

def test_situations():
    assert dataframe1.situations == nodes[0:19]

def test_leaves():
    assert dataframe1.leaves == nodes[19:]

def test_default_ess():
    assert dataframe1.default_equivalent_sample_size() == 3

def test_default_prior():
    assert dataframe1.default_prior(60) == [[30,30],[15,15],[15,15],
    [5,5,5],[5,5,5],[5,5,5],[5,5,5],[2.5,2.5],[2.5,2.5],[2.5,2.5],
    [2.5,2.5],[2.5,2.5],[2.5,2.5],[2.5,2.5],[2.5,2.5],[2.5,2.5],
    [2.5,2.5],[2.5,2.5],[2.5,2.5]]

hyperstage1 = ['s3', 's4', 's5', 's6'] 
hyperstage2 = [x for x in nodes[0:19] if x not in hyperstage1]
def test_default_hyperstage():
    assert dataframe1.default_hyperstage() == [hyperstage2, hyperstage1]

#checking whether we get the same stages as in Fig 7.5a
# in Collazo, Gorgen and Smith CEG book
ceg_book_hyperstage = [['s0'], ['s1', 's2'], ['s3', 's4', 's5', 's6'],
['s7', 's8' ,'s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18']]
def test_ceg_in_book_stages():
    assert dataframe1.AHC_transitions(alpha =3, hyperstage = ceg_book_hyperstage)[0] == [['s3', 's5', 's6'],
    ['s7','s11','s15','s16','s17'],['s8','s10','s12','s13','s18'],['s9','s14']]

#print(dataframe1.AHC_transitions(alpha = 3, hyperstage = [['s0'], ['s1', 's2'], ['s3', 's4', 's5', 's6'],['s7', 's8' ,'s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18']]))
print(dataframe1.edge_countset)
#[[507, 383], [237, 270], [337, 46], [86, 119, 32], [158, 105, 74], [90, 65, 115], [14, 21, 11], [70, 16], [117, 41], [109, 10], [75, 30], [76, 14], [8, 24], [16, 49], [11, 104], [59, 15], [12, 2], [18, 3], [8, 3]]


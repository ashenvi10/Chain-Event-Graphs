import pandas as pd
from event_tree_class import event_tree
import pytest

df = pd.read_excel('CHDS.latentexample1.xlsx')

dataframe1 = event_tree({'dataframe' : df})

nodes = []
for i in range(0, 43):
    nodes.append('s%d' %i)

def test_nodes():
    assert dataframe1.nodes == nodes

def test_situations():
    assert dataframe1.situations == nodes[0:19]

def test_leaves():
    assert dataframe1.leaves == nodes[19:]

def test_edge_countset():
    assert dataframe1.edge_countset == [[507, 383], [237, 270], 
    [46, 337], [86, 32, 119], [90, 65, 115], [14, 11, 21], 
    [105, 158, 74], [70, 16], [24, 8], [109, 10], [76, 14], 
    [49, 16], [104, 11], [12, 2], [8, 3], [18, 3], [75, 30], 
    [117, 41], [59, 15]]

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
    assert dataframe1.AHC_transitions(alpha =3, hyperstage = ceg_book_hyperstage)[0] == [['s9', 's12'], 
    ['s3', 's4', 's5'], ['s16', 's17', 's8', 's11', 's14'], 
    ['s18', 's7', 's10', 's13', 's15']]

def test_ceg_in_book_loglikelihood():
    assert dataframe1.AHC_transitions(alpha =3, hyperstage = ceg_book_hyperstage)[1] == pytest.approx(-2478.49, rel = 0.02)

dataframe2 = event_tree({'dataframe': df, 'sampling_zero_paths': [('Average',), ('Average','High',)]})
def test_sampling_stages():
    assert dataframe2.AHC_transitions(alpha=3)[0] == [['s11', 's14'], ['s5', 's6', 's7'], 
    ['s18', 's19', 's10', 's13', 's16'], ['s17', 's20', 's9', 's12', 's15']]

def test_sampling_likelihood():
    assert dataframe2.AHC_transitions(alpha=3)[1] == pytest.approx(-2486.89, 0.02)

def test_sampling_error1():
    with pytest.raises(ValueError) as excinfo:   
        dataframe3 = event_tree({'dataframe': df, 'sampling_zero_paths': [('Average','High',),('Average',)]})
        assert "The path up to it's last edge should be added first. Ensure the tuple ends with a comma." in str(excinfo.value)

def test_sampling_error2():
    with pytest.raises(ValueError) as excinfo:   
        dataframe4 = event_tree({'dataframe': df, 'sampling_zero_paths': [('Average'),('Average','High',)]})
        assert "The path up to it's last edge should be added first. Ensure the tuple ends with a comma." in str(excinfo.value)


dataframe1.AHC_transitions(alpha=3, hyperstage=ceg_book_hyperstage)
dataframe1.staged_tree_figure("staged_tree")
dataframe1.ceg_figure("ceg1")

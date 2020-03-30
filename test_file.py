import pandas as pd
from event_tree_class import event_tree

df = pd.read_excel('CHDS.latentexample1.xlsx')

dataframe1 = event_tree(df)
#print(dataframe1.default_hyperstage())
#dataframe1.event_tree_figure('event_tree')
#print(dataframe1.AHC_transitions(alpha = 3, hyperstage = [['s0'], ['s1', 's2'], ['s3', 's4', 's5', 's6'],['s7', 's8' ,'s9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18']]))
print(dataframe1.AHC_transitions())
#print(dataframe1.edge_countset)

#dataframe1.staged_tree_figure('staged_tree')


'''
    master_merger = [] #main array indicating what should be merged
    #pairwise comparison of all situations/stages for each round of the AHC
    maxscore = 1 #maxscore is the highest positive bayesfactor score at each round of the AHC
    while maxscore > 0:
        merger = [0] #array to indicate what may be merged
        score = [0] #array to store positive bayesfactor scores at each round of the AHC
        
        for i in range(0, length):
            if all(x ==0 for x in posterior[i]) == False: #as we will set one of the merged situations/stages as 0 vectors later to retain indices
                for j in range(i+1, length):
                    if len(prior[i]) == len(prior[j]) and all(x==0 for x in posterior[j]) == False and sub([i,j], hyperstage) == 1:
                        newscore = bayesfactor(prior[i], posterior[i], prior[j], posterior[j])
                        if newscore > 0:
                            score.append(newscore) #adding all the positive scores to an array
                            merger.append([i,j]) #noting indices of the situations/stages which should be merged
        maxscore = max(score) #highest positive newscore
        master_merger.append(merger[score.index(maxscore)])
        
        #updating the prior and posterior based on what's been merged
        if maxscore >0:
            changes = master_merger[-1] #indices for whats been most recently merged
            
            #making one of these the prior and posterior for the combined stage
            prior[changes[0]] = list(map(add, prior[changes[0]], prior[changes[1]]))
            posterior[changes[0]] = list(map(add, posterior[changes[0]], prior[changes[1]]))
            
            #giving the other one an empty vector for the prior and posterior to retain indexing
            prior[changes[1]] = [0]* len(prior[changes[0]])
            posterior[changes[1]] = [0] * len(prior[changes[0]])
            
            cond_probs[changes[0]] = posterior[changes[0]]
            cond_probs[changes[1]] = posterior[changes[0]]
            
            likelihood += maxscore
        
    single_cells = []
    for i in range(0, length):
        if len(prior[i]) == 1:
            single_cells.append(i)
    
    master_merger.append(single_cells)
    
    for x in cond_probs:
        total = sum(x)
        cond_probs[cond_probs.index(x)] = [round(y/total, 2) for y in x]
    

    #now we sort out the master_merger list
    master_merger = sortmaster(master_merger)
    master_merger.append(0)
    master_merger = sortmaster(master_merger)
    return (master_merger, likelihood, cond_probs, posterior)  '''
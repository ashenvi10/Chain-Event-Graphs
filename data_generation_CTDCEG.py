import numpy as np
import random 
from pandas import ExcelWriter
import pandas as pd
import xlrd
import xlsxwriter

def community(a):
    '''generating data for an individual in the community setting for risk and treatment variables'''
    val = [] 
    if a[0] <0.5: #low risk
        val= val + ['Low', '']
    else: 
        val.append('High') #high risk
        if a[1] <0.7: #treated
            val.append('Treated')
        else: #not treated
            val.append('Not treated')
    return val
        
def community_high_treated(a):
    '''generating data for high risk, treated individuals in the community setting for fall and outcome variables'''
    val = [] 
    d1 = a[0]
    d2 = a[1]
    if d1 < 0.49: #don't fall
        val = val + ['No fall', 0, '', '']
    else: #fall
        val.append('Fall')
        val.append(float((390.05* np.random.weibull(1.6, 1)).round(3)))
        if d2 < 0.25:  #severe complications
            val.append('Complications')
            val.append(float((5.4* np.random.weibull(0.5, 1)).round(3)))
        elif d2 >= 0.25 and d2 < 0.75: #loop
            val.append('Loop')
            val.append(float((25.3* np.random.weibull(1, 1)).round(3)))
        else: #move to communal establishment
            val.append('Move')
            val.append(float((170.14* np.random.weibull(1.6, 1)).round(3)))
    return val

def community_high_nottreated(a):
    '''generating data for high risk, not treated individuals in the community setting for fall and outcome variables'''
    val = [] 
    d1 = a[0]
    d2 = a[1]
    if d1 < 0.30: #don't fall
        val = val + ['No fall', 0, '', '']
    else: #fall
        val.append('Fall')
        val.append(float((290.19* np.random.weibull(2.2, 1)).round(3))) 
        if d2 < 0.25: #severe complications
            val.append('Complications')
            val.append(float((5.4* np.random.weibull(0.5, 1)).round(3)))
        elif d2 >= 0.25 and d2 < 0.75: #loop
            val.append('Loop')
            val.append(float((25.3* np.random.weibull(1, 1)).round(3)))
        else: #move to communal establishment
            val.append('Move')
            val.append(float((170.14* np.random.weibull(1.6, 1)).round(3)))
    return val

            
def low(a):
    '''generating data for low risk individuals in both settings for fall and outcome variables'''
    val = [] 
    d0 = a[0]
    d1 = a[1]
    if d0 < 0.75: #don't fall
        val = val + ['No fall', 0, '', '']
    else: #fall
        val.append('Fall')
        val.append(float((433.97* np.random.weibull(1.2, 1)).round(3)))
        val.append('Loop') #loop
        val.append(float((45.48* np.random.weibull(1, 1)).round(3)))
    return val

def communal(a):
    '''generating data for low risk individuals in the communal setting for risk and treatment variables'''
    val = [] #for transitions
    if a[0] <0.36: #low risk
        val = val + ['Low', '']
    else: #high risk
        val.append('High')
        if a[1] < 0.70:  #treated
            val.append('Treated')
        else: #not treated
            val.append('Not treated')
    return val
            
def communal_high_treated(a):
    '''generating data for high risk, treated individuals in the communal setting for fall and outcome variables'''
    val = [] #for transitions
    d1 = a[0]
    d2 = a[1]
    if d1 < 0.49: #don't fall
        val = val + ['No fall', 0, '', '']
    else: #fall
        val.append('Fall')
        val.append(float((390.05* np.random.weibull(1.6, 1)).round(3)))
        if d2 < 0.30: #severe complications
            val.append('Complications')
            val.append(float((5.4* np.random.weibull(0.5, 1)).round(3)))
        else: #loop
            val.append('Loop')
            val.append(float((25.3* np.random.weibull(1, 1)).round(3)))
    return val


def communal_high_nottreated(a):
    '''generating data for high risk, not treated individuals in the communal setting for fall and outcome variables'''
    val = [] #for transitions
    d1 = a[0]
    d2 = a[1]
    if d1 < 0.22: #don't fall
        val = val + ['No fall', 0, '', '']
    else: #fall
        val.append('Fall')
        val.append(float((224.69* np.random.weibull(2.2, 1)).round(3))) 
        if d2 < 0.30:  #severe complications
            val.append('Complications')
            val.append(float((5.4* np.random.weibull(0.5, 1)).round(3)))
        else: #loop
            val.append('Loop')
            val.append(float((25.3* np.random.weibull(1, 1)).round(3)))
    return val
            

#generating the data
def data_generation(seed):
    random.seed(seed)
    i = 0 #number of distinct datasamples
    for num in (1,3,5,10,15,20): #creating datasets of varying sizes
        i += 1 
        pop = 500 *num  #population
        filename = '/Users/Aditi/Dropbox/PhD stuff/Papers/CT-DCEG/data/ctdceg_data%s.xlsx' %i 
        data = xlsxwriter.Workbook(filename)
        bold = data.add_format({'bold': True})    
        ws = data.add_worksheet() 
        ws.write_row(0, 0 , ['Residence', 'Risk', 'Treatment', 'Fall', 'Fall time', 'Outcome', 'Outcome time']*5) #followed for five falls per person
    

        for k in range(1,pop+1):
            ivec = np.random.rand(5,1)  
            ival = [] # data vector  
            if ivec[0] < 0.65:  #community residents observed
                ival.append('Community')
                ival = ival + community(ivec[1:3])            
                if ival[-1] == 'Not treated' and ival[-2] == 'High':
                    ival = ival + community_high_nottreated(ivec[3:5]) 
                elif ival[-1] == 'Treated':
                    ival = ival + community_high_treated(ivec[3:5])
                elif ival[-2] == 'Low':
                    ival = ival + low(ivec[3:5])
            else: #communal establishment residents observed
                ival.append('Communal')
                ival = ival + communal(ivec[1:3])
                if ival[-1] == 'Not treated' and ival[-2] == 'High':
                    ival = ival + communal_high_nottreated(ivec[3:5]) 
                elif ival[-1] == 'Treated':
                    ival = ival + communal_high_treated(ivec[3:5])
                elif ival[-2] == 'Low':
                    ival = ival + low(ivec[3:5])
    
            iters = 4 #max number of passage-slices 
            n = iters
            while n > 0: 
                if ival[-2] == 'Complications': #individual has severe complications
                    break
                elif (ival[-7] == 'Community' or ival[-7] == 'Community low loop') and ival[-6] == 'High': #high-risk community individual
                    randomvalues = np.random.rand(2,1)
                    if ival[-2] == 'Loop' and ival[-5] == 'Treated': #looped back at last passage-slice
                        ival = ival + ['Community', 'High', 'Treated'] + community_high_treated(randomvalues) 
                    elif ival[-2] == 'Loop' and ival[-5] == 'Not treated': #looped back at last passage-slice
                        ival = ival + ['Community', 'High', 'Not treated'] + community_high_nottreated(randomvalues) 
                    elif ival [-2] == 'Move' and ival[-5] == 'Treated': #move to communal establishment
                        ival = ival + ['Communal', 'High', 'Treated'] + communal_high_treated(randomvalues)
                    elif ival[-2] == 'Move' and ival[-5] == 'Not treated': #move to communal establishment
                        ival = ival + ['Communal', 'High', 'Not treated'] + communal_high_nottreated(randomvalues)
                                        
                #low-risk community individual who looped back at last passage-slice    
                elif (ival[-7] == 'Community' or ival[-7] == 'Community low loop') and ival[-6] == 'Low' and ival[-2] == 'Loop': 
                    randomvalues = np.random.rand(2,1)
                    ival = ival + ['Community low loop'] + community(randomvalues)
                    randomvalues = np.random.rand(2,1)
                    if ival[-2] == 'Low':
                        ival = ival + low(randomvalues)
                    elif ival[-1] == 'Not treated' and ival[-2] == 'High':
                        ival = ival + community_high_nottreated(randomvalues)
                    elif ival[-1] == 'Treated':
                        ival = ival + community_high_treated(randomvalues) 
                    
                #-----  
                elif (ival[-7] == 'Communal' or ival[-7] == 'Communal for loop') and ival[-6] == 'High': #high-risk communal establishment individual
                    randomvalues = np.random.rand(2,1)
                    if ival[-2] == 'Loop' and ival[-5] == 'Treated': #looped back at last passage-slice
                        ival = ival + ['Communal', 'High', 'Treated'] + communal_high_treated(randomvalues)
                    elif ival[-1] == 'Loop' and ival[-3] == 'Not treated': #looped back at last passage-slice
                        ival = ival + ['Communal', 'High', 'Not treated'] + communal_high_nottreated(randomvalues) 
            
                #low-risk communal establishment individual who looped back at last passage-slice
                elif (ival[-7] == 'Communal' or ival[-7] == 'Communal for loop') and ival[-6] == 'Low' and ival[-2] == 'Loop': 
                    randomvalues = np.random.rand(2,1)
                    ival = ival + ['Communal low loop'] + communal(randomvalues)
                    randomvalues = np.random.rand(2,1)
                    if ival[-2] == 'Low':
                        ival = ival + low(randomvalues)
                    elif ival[-1] == 'Not treated' and ival[-2] == 'High':
                        ival = ival + communal_high_nottreated(randomvalues)
                    elif ival[-1] == 'Treated':
                        ival = ival + communal_high_treated(randomvalues)
                n = n - 1

            ws.write_row(k, 0 ,ival)
        data.close()

    #preparing data for analysis
    for i in range(1,7): 
        filename = '/Users/Aditi/Dropbox/PhD stuff/Papers/CT-DCEG/data/ctdceg_data%s.xlsx' %i 
        df = pd.read_excel(filename, sheet_name = 'Sheet1')
        pop = len(df) #population
        #data from different passage-slice for each individual treated as independent due to model assumptions
        filename_mod = '/Users/Aditi/Dropbox/PhD stuff/Papers/CT-DCEG/data/ctdceg_modified_data%s.xlsx' %i 
        data = xlsxwriter.Workbook(filename_mod)
        bold = data.add_format({'bold': True})    
        ws = data.add_worksheet() #for modified data
        ws.write_row(0, 0 , ['Residence', 'Risk', 'Treatment', 'Fall', 'Fall time', 'Outcome', 'Outcome time'])

        col = df.columns
        df_list = df.values.tolist() 

        for i in range(0,6): #to write each passage-slice as a different row
            df1 = df[col[i*7:(i*7)+7]] 
            df1 = df1.values.tolist() 
            for k in range(i*pop, (i*pop)+pop): 
                j = k%pop
                new = df1[j]
                new = [x if str(x) != 'nan' else 'NaN' for x in new]
                ws.write_row(k+1, 0 ,new)
                
        data.close()

    #Remove all the blank lines from the datasets
    for i in range(1,7): 
        filename = '/Users/Aditi/Dropbox/PhD stuff/Papers/CT-DCEG/data/ctdceg_modified_data%s.xlsx' %i 
        df = pd.ExcelWriter(filename)
        df1 = pd.read_excel(filename, sheet_name = 'Sheet1',keep_default_na = False) 
        df1 = df1.loc[df1['Residence'] != 'NaN']
        df1 = df1.reset_index(drop=True)
        df1.to_excel(df, 'Sheet1')
        df.save()
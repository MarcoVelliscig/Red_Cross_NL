def find_best_match_user_input( poss_matches , name_to_match, score_threshold, use_tricks=False):

        if use_tricks :
                poss_matches_trim = [poss_matches[i].replace('(CAPITAL)','').replace('CITY','').replace('OF','') for i in range(len(poss_matches))]
                regex = re.compile(".*?\((.*?)\)")
                poss_matches_trim = [re.sub("[\(\[].*?[\)\]]", "", poss_matches_trim[i]) for i in range(len(poss_matches))]
                name_to_match_trim = name_to_match.replace('CITY','').replace('OF','').strip()


        else:
              poss_matches_trim = poss_matches
              name_to_match_trim= name_to_match
              
        ratio = [(difflib.SequenceMatcher(None,poss_matches_trim[i], name_to_match_trim)).ratio() \
                 for i in range(len(poss_matches))]

        vec_poss = np.array(zip(poss_matches, ratio))
        vec_poss_sorted = np.array(sorted(vec_poss ,key=lambda x: x[1], reverse=True))
        most_prob_name_match = vec_poss_sorted[0,0]
        best_ratio = vec_poss_sorted[0,1]
        if float(best_ratio) < score_threshold: 
            #ask if the possible match is right
            print 'is ' , most_prob_name_match , 'the right match for ' , name_to_match , '(score:',best_ratio , ')'
            respond = raw_input('Return for yes, everything else for no : ')

            if respond != '' : 
                sorted_prob_name_match =vec_poss_sorted[:,0]
                sorted_prob_name_match_numbered = np.array(zip(sorted_prob_name_match, range(len(sorted_prob_name_match))))
                print '\n select from the best match for ' ,name_to_match ,' from this list: \n',  sorted_prob_name_match_numbered
                selected_index = raw_input('select the right choice by number, write -1 for none of the above : ')
                if int(selected_index) == -1 :
                    most_prob_name_match  = 'Not found'

                else:
                    most_prob_name_match = sorted_prob_name_match_numbered[int(selected_index),0]


            print '==' , most_prob_name_match , 'is the right match for ' , name_to_match, '\n'
        best_match=most_prob_name_match

        return best_match 





import numpy as np
from scipy import optimize
from pylab import scatter, show, legend, xlabel, ylabel
from sklearn import linear_model, datasets
from sklearn.ensemble import RandomForestClassifier 
import csv as csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cross_validation
from sklearn.dummy import DummyClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import string
import difflib
from nltk.metrics import edit_distance
from nltk.metrics import masi_distance
from nltk.metrics import jaccard_distance
import operator
import re
#add percentages of exact matches 



#########################################################################
# read the Pcodes from template files
# for provinces , municipalities and barangays
df_pcodes_pro = pd.read_excel('template.xlsx',sheetname='Province', 
                              skiprows = 1 , skip_footer = 1,header=None,encoding='utf-8')
df_pcodes_pro.columns = ['Pcode_province',
                         'name_province']

df_pcodes_mun = pd.read_excel('template.xlsx',sheetname='Municipality', 
                              skiprows = 1 , skip_footer = 21,header=None,encoding='utf-8')
df_pcodes_mun.columns = ['Pcode_province',
                         'name_province', 
                         'Pcode_municipality',
                         'name_municipality']


df_pcodes_bar = pd.read_excel('template.xlsx',sheetname='Barangay', 
                              skiprows = 1 , skip_footer = 21,header=None,encoding='utf-8')
df_pcodes_bar.columns = ['Pcode_municipality',
                         'name_municipality', 
                         'Pcode_barangay',
                         'name_barangay']

#build a template file with the 3 admin levels

# create a template dataframe
#first create a template for L1 L2 L3 admin level names and codes
df_template =  pd.merge(df_pcodes_mun, df_pcodes_bar, 
                         left_on ='Pcode_municipality', 
                         right_on='Pcode_municipality', 
                         how = 'inner')

df_template = df_template[[u'Pcode_province', u'name_province', u'Pcode_municipality',
       u'name_municipality_x',  u'Pcode_barangay', u'name_barangay']]



#produce a list of names that need to be pcoded




df_template.columns = ['L1_code',
                       'L1_name', 
                       'L2_code',
                       'L2_name',
                       'L3_code',
                       'L3_name']



#########################################################################






#########################################################################
option =0


if option == 0 :
        # INPUT
        #specify the file you want to pcode
        df_raw = pd.read_csv('barangay.csv')
        sav_name = 'barangay_pcoded_080716.csv'
        #specify which columns correspont do what
        dict_raw = {'NAME_1': 'L1_name', 
                    'NAME_2': 'L2_name', 
                    'NAME_3': 'L3_name'}

        #specify the levels in the file
        level_tag = ['L1', 'L2' , 'L3']

        #different confidence level for different admin levels
        #threshold of 0.0 means there is no user imput
        threshold = {'L1':0.9, 'L2':0.8, 'L3':0.7}

        #if True it removes indications of city, capital and (names in parethesis)
        name_tricks = True



if option == 1 :
        # INPUT
        #specify the file you want to pcode
        df_raw = pd.read_csv('ndhrhis_v1.csv')
        sav_name = 'ndhrhis_v1_pcoded_080716.csv'
        #specify which columns correspont do what
        dict_raw = {'municipality':'L2_name' , 
                    'province'  : 'L1_name'}


        #specify the levels in the file
        level_tag = ['L1', 'L2' ]
        #different confidence level for different admin levels
        threshold = {'L1':0.9, 'L2':0.7}

        #if True it removes indications of city, capital and (names in parethesis)
        name_tricks = True





df = df_raw[dict_raw.keys()]
df.columns = [dict_raw.values()]
df = df.drop_duplicates()

#########################################################################
name_per_tag = ['_name', '_code']

#keep from the template only the relevant admin levels
columns_template = [x + y for x in level_tag for y in name_per_tag ]

df_template = df_template[columns_template]
df_template = df_template.drop_duplicates()


for admin_level in level_tag: 
        df_template[admin_level+'_name']=df_template[admin_level+'_name'].str.upper()
        df[admin_level+'_name']=df[admin_level+'_name'].str.upper()
        df[admin_level+'_code'] = np.NaN 




exception = 'TOTAL'

verbose = True
verbose = False
n_perfect_matches =0 
n_no_matches =0 
counter = 0 
for index in df.index : 
#for index in df.index[0:10] : 
        df_template_matches = df_template
        for admin_level in level_tag:
                if verbose : 
                        print 'len template dataframe level', admin_level\
                                , len(df_template_matches)
                        print df_template_matches.describe()
                        
                #gets the name of the admin level for the index entry
                name_admin_level = df.loc[index][admin_level+'_name']
                if name_admin_level  == exception : continue
                #it tries to get a perfect match straight away
                n_matches_current_level = sum(df_template_matches[admin_level+'_name']==
                                              name_admin_level)
                if verbose : print 'num matches', admin_level ,  n_matches_current_level
                
                if (n_matches_current_level) > 0 :
                        if verbose : print ''
                             
                elif (n_matches_current_level) == 0 :
                        print "perc completed " , ((float(counter)/len(df.index))*100),'\n'
                        poss_matches = (df_template_matches[admin_level+'_name'].drop_duplicates()).values
                        score_threshold=threshold[admin_level]                     
                        best_match  = find_best_match_user_input( poss_matches , name_admin_level, score_threshold, use_tricks = name_tricks) 
                        if best_match == 'Not found' :  
                                n_no_matches +=1 
                                break 
                        #print 'admin ' , admin_level , name_admin_level ,  'bestmatch ' , best_match , score_m , 'edit dist' , edit_distance(best_match , name_admin_level), '\n'
                        name_admin_level = best_match
                        n_matches_current_level = sum(df_template_matches[admin_level+'_name']==
                                              name_admin_level)
                        
                df_template_matches = df_template_matches.loc[
                        df_template_matches[admin_level+'_name']==name_admin_level]
                
                if (n_matches_current_level) == 0 & (admin_level== level_tag[-1]):
                        n_no_matches +=1 
                if n_matches_current_level == 1 :
                        
                        n_perfect_matches +=1
                        if verbose : print df_template_matches
                        for admin_level_tag in level_tag: 
                                df.loc[index,admin_level_tag+'_code']=df_template_matches[admin_level_tag+'_code'].values
                #add dictionary with known matches
        counter+=1

print 'perfect match found  ' , n_perfect_matches
print 'no match found  ' , n_no_matches






df_pcoded = pd.merge(df_raw, df, 
                     left_on =dict_raw.keys(), 
                     right_on=dict_raw.values(), 
                     how = 'inner')
df_pcoded.to_csv(sav_name,encoding='utf-8' )



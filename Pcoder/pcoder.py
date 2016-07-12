# Code developed by Marco Velliscig (marco.velliscig AT gmail.com)
# for the dutch red cross
# released under GNU GENERAL PUBLIC LICENSE Version 3


def find_best_match_user_input( poss_matches , name_to_match,  upper_level , score_threshold, known_matches , use_tricks=False):
        known_match_tag = name_to_match+ ' ' + upper_level
        try :
                #known matches should have the previus level if available
                #known matches are coded with the previus level to avois 
                #same names in different admin level to be treated the same
                best_match = known_matches[known_match_tag]
        except:
                if use_tricks :
                        poss_matches_trim = [poss_matches[i].replace('CITY','').replace('OF','').strip() for i in range(len(poss_matches))]
                        regex = re.compile(".*?\((.*?)\)")
                        poss_matches_trim = [re.sub("[\(\[].*?[\)\]]", "", poss_matches_trim[i]) for i in range(len(poss_matches))]
                        poss_matches_trim = [poss_matches_trim[i].strip() for i in range(len(poss_matches))]
                        name_to_match_trim = name_to_match.replace('CITY','').replace('OF','').strip()
                        name_to_match_trim = re.sub("[\(\[].*?[\)\]]", "", name_to_match_trim)
                        name_to_match_trim =      name_to_match_trim.strip()               

                else:
                      poss_matches_trim = poss_matches
                      name_to_match_trim= name_to_match

                ratio = [(difflib.SequenceMatcher(None,poss_matches_trim[i], name_to_match_trim)).ratio() \
                         for i in range(len(poss_matches))]

                vec_poss = np.array(zip(poss_matches, ratio))
                vec_poss_sorted = np.array(sorted(vec_poss ,key=lambda x: x[1], reverse=True))
                try: 
                        most_prob_name_match = vec_poss_sorted[0,0]

                except:
                        print 'error'
                        print 'name to match ', name_to_match
                        print 'poss matches' , poss_matches
                        most_prob_name_match  = 'error'
                        return most_prob_name_match

                best_ratio = vec_poss_sorted[0,1]
                if float(best_ratio) < score_threshold: 
                    #ask if the possible match is right
                    print 'is ' , most_prob_name_match , 'the right match for ' , name_to_match , '(score:',best_ratio , ')'
                    respond = raw_input('press return for yes, everything else for no : ')

                    if respond != '' : 
                        sorted_prob_name_match =vec_poss_sorted[:,0]
                        sorted_prob_name_match_numbered = np.array(zip(sorted_prob_name_match, range(len(sorted_prob_name_match))))
                        print '\n select from the best match for ' ,name_to_match ,' from this list: \n',  sorted_prob_name_match_numbered
                        selected_index = raw_input('select the right choice by number, press return for not found : ')
                        if selected_index == '' :
                            most_prob_name_match  = 'Not found'

                        else:
                            most_prob_name_match = sorted_prob_name_match_numbered[int(selected_index),0]
                
                known_matches[known_match_tag] = most_prob_name_match 
                print '==' , most_prob_name_match , 'is the right match for ' , name_to_match , best_ratio , '\n'
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
#this part can be commented out if a template has already been produced


#building template file for the philippines
#this part is philippines specific
# a template should have a name and pcode for every admin level

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

df_template.to_csv('pcode_template_philippines.csv',encoding='utf-8')

##########################################################################


#INPUT
# Read the template csv file and assign admin levels to columns
filename_template = 'pcode_template_philippines.csv'

#INPUT
#fill in the assignation of columns
#match columns names with admin levels
dict_raw_template  = {'Pcode_province':     'L1_code',
                      'name_province':      'L1_name', 
                      'Pcode_municipality': 'L2_code',
                      'name_municipality_x':'L2_name',
                      'Pcode_barangay':     'L3_code',
                      'name_barangay':      'L3_name'}



df_raw_template = pd.read_csv(filename_template)

df_template = df_raw_template[dict_raw_template.keys()]
df_template.columns = [dict_raw_template.values()]
df_template = df_template.drop_duplicates()
#########################################################################






#########################################################################
option =0

#preset input options for philippines
if option == 0 :
        # INPUT
        #specify the file you want to pcode
        filename = 'barangay.csv'
        
        #specify the output file name
        sav_name = 'barangay_pcoded_080716.csv'
        #specify which columns correspont do what
        dict_raw = {'NAME_1': 'L1_name', 
                    'NAME_2': 'L2_name', 
                    'NAME_3': 'L3_name'}

        #specify the levels in the file
        level_tag = ['L1', 'L2' , 'L3']

        #different confidence level for different admin levels
        #threshold of 0.0 means there is no user imput but less safe results
        #threshold of 1.0 means that the code will always ask user input
        #this threshold should be higher for higher admin levels
        threshold = {'L1':0.9, 'L2':0.7, 'L3':0.7}

        #if True it removes indications of city, capital and (names in parethesis)
        name_tricks = True



if option == 1 :
        # INPUT
        #specify the file you want to pcode
        filename = 'ndhrhis_v1.csv'
        #specify the output file name
        sav_name = 'ndhrhis_v1_pcoded_110716.csv'
        #specify which columns correspont do what
        dict_raw = {'municipality':'L2_name' , 
                    'province'  : 'L1_name'}


        #specify the levels in the file
        level_tag = ['L1', 'L2' ]
        #different confidence level for different admin levels
        threshold = {'L1':0.9, 'L2':0.7}

        #if True it removes indications of city, capital and (names in parethesis)
        name_tricks = True




df_raw = pd.read_csv(filename)
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
        df[admin_level+'_best_match_name'] = np.NaN 




exception = 'TOTAL'

verbose = True
verbose = False
n_perfect_matches =0 
n_no_matches =0 
counter = 0 
known_matches = {}
for index in df.index : 
#for index in df.index[0:10] : 
#for index in [2251] : 

        df_template_matches = df_template
        upper_level = ''
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
                        best_match  = find_best_match_user_input( poss_matches , name_admin_level,  upper_level , score_threshold, known_matches ,  use_tricks = name_tricks) 
                        if best_match == 'Not found' :  
                               n_no_matches +=1 
                               print '************* Not found'
                               print df.loc[index]
                               break 
                        elif best_match == 'error' :         
                               n_no_matches +=1 
                               print '************* error admin ' , admin_level , name_admin_level
                               print df.loc[index]

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
                                df.loc[index,admin_level_tag+'_best_match_name']=df_template_matches[admin_level_tag+'_name'].values
                upper_level += admin_level + df.loc[index][admin_level+'_name']
                #add dictionary with known matches
        counter+=1

print 'perfect match found  ' , n_perfect_matches
print 'no match found  ' , n_no_matches






df_pcoded = pd.merge(df_raw, df, 
                     left_on =dict_raw.keys(), 
                     right_on=dict_raw.values(), 
                     how = 'inner')
df_pcoded.to_csv(sav_name,encoding='utf-8' )



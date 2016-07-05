
def find_best_match( poss_matches , name_to_match ):

        ratio = [(difflib.SequenceMatcher(None,poss_matches[i], name_to_match)).ratio() 
                 for i in range(len(poss_matches))]
        dict =  {k: v for k, v in zip(poss_matches, ratio)}
        ##SOMINOT DON MARIANO MARCOS
#sominot
error
        #print zip(poss_matches , cost_edit)
        #print zip(poss_matches , cost_masi)
        #print zip(poss_matches , cos_jaccardt)

        best_match  = max(dict.iteritems(), key=operator.itemgetter(1))[0]
        if dict[best_match] < 0.6 : print dict
        if dict[best_match] < 0.6 : print best_match,  dict[best_match] 

        return best_match , dict[best_match] 


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

#add percentages of exact matches 

#read the Pcodes for provinces , municipalities and barangays
df_pcodes_pro = pd.read_excel('../PCodes/PCodes/data.xlsx',sheetname='Province', 
                              skiprows = 1 , skip_footer = 1,header=None,encoding='utf-8')
df_pcodes_pro.columns = ['Pcode_province',
                         'name_province']

df_pcodes_mun = pd.read_excel('../PCodes/PCodes/data.xlsx',sheetname='Municipality', 
                              skiprows = 1 , skip_footer = 21,header=None,encoding='utf-8')
df_pcodes_mun.columns = ['Pcode_province',
                         'name_province', 
                         'Pcode_municipality',
                         'name_municipality']


df_pcodes_bar = pd.read_excel('../PCodes/PCodes/data.xlsx',sheetname='Barangay', 
                              skiprows = 1 , skip_footer = 21,header=None,encoding='utf-8')
df_pcodes_bar.columns = ['Pcode_municipality',
                         'name_municipality', 
                         'Pcode_barangay',
                         'name_barangay']


#read the scraped ndhrhis database for doctor count per municipality
df_ndh = pd.read_csv('ndhrhis_v1.csv')

#jannis code at the barangay level that need to be Pcoded 
df_bar = pd.read_csv('barangay.csv')
#df_ndh[['municipality','province']]

#strip name for unwanted feautures
#first we have to remove from the official Pcodes name the CITY and CAPITAL string
df_pcodes_mun['name_municipality_trim_pcodes'] = df_pcodes_mun['name_municipality']\
    .str.replace('CITY', '').str.replace('OF', '').str.strip(' ')\
    .str.replace('CAPITAL', '').str.strip(' ').str.replace('-', '')\
    .str.replace('(', '').str.replace(')', '').str.strip(' ')

#we do the same for the ndhrhis database
df_ndh['name_municipality_trim_ndh'] = df_ndh['municipality']\
    .str.replace('CITY', '').str.replace('OF', '').str.strip(' ')\
    .str.replace('\(CAPITAL\)', '').str.strip(' ').str.replace('-', '').str.strip(' ')



#finding municipalities with for loop
# this should be vectorized (use match groupby concat multiindex)


df_ndh['pcode'] = np.NaN
exception = 'TOTAL'
for index in range(len(df_ndh)) : 
    name_m = df_ndh.iloc[index]['municipality']
    name_p = df_ndh.iloc[index]['province']
    #print 'prov and mun ', name_p , ' - ',  name_m
    if  df_ndh.iloc[index]['municipality'] == exception : continue
    try:
        pcode  = df_pcodes_mun.loc[
            (df_pcodes_mun.name_municipality_trim_pcodes == df_ndh.iloc[index]['name_municipality_trim_ndh']) &
            (df_pcodes_mun.name_province == df_ndh.iloc[index]['province']) ,
            'Pcode_municipality']
        df_ndh.loc[index,'pcode']=  pcode.values
        #print 'pcode retrived ' , pcode.values
    except:
        #print '\n'        
        #print 'no exact match found'
        #print 'prov and mun ', name_p , ' - ',  name_m
        #print 'pcode retrived ' , pcode.values
        
        
        poss_matches  = df_pcodes_mun.loc[df_pcodes_mun.name_province == 
                                          name_p, 
                                          'name_municipality_trim_pcodes'].values
        score_p =1.
        
        if len(poss_matches) == 0 :
            name_p, score_p = find_best_match( df_pcodes_mun.name_province.values, name_p )
            poss_matches = df_pcodes_mun.loc[df_pcodes_mun.name_province == 
                                          name_p, 
                                          'name_municipality_trim_pcodes'].values
        #poss_matches = [unicode.encode(poss_matches[i]) for i in range(len(poss_matches))]
        #poss_matches = [poss_matches[i].encode('ascii','ignore') for i in range(len(poss_matches))]

        #print poss_matches
        name_to_match = df_ndh.iloc[index]['name_municipality_trim_ndh']
        best_match , score_m = find_best_match( poss_matches , name_to_match )
        

        pcode  = df_pcodes_mun.loc[
            (df_pcodes_mun.name_municipality_trim_pcodes == best_match) &
            (df_pcodes_mun.name_province ==  name_p),
            'Pcode_municipality']
        df_ndh.loc[index,'pcode']=  pcode.values
    
        if (score_m < 0.6) |(score_p < 0.6) : 
            print 'pcode ',  df_pcodes_mun.loc[df_pcodes_mun.Pcode_municipality==pcode.values[0],['name_province','name_municipality','Pcode_municipality']]

            print 'ndhis ' ,  df_ndh.iloc[index][['province','municipality','pcode']]
            print '\n'
    

df_ndh_pcoded = pd.merge(df_ndh, df_pcodes_mun, 
                         left_on ='pcode', 
                         right_on='Pcode_municipality', 
                         how = 'inner')
df_ndh_pcoded.to_csv('ndhrhis_v1_w_pcode_v2.csv',encoding='utf-8' )

stop()




df_ndh_notot  =df_ndh[df_ndh.municipality != 'TOTAL']
print ' unique number of municipalities in the ndhrhis database' , len(df_ndh.loc[df_ndh.municipality != 'TOTAL' , 'municipality'].unique())

#we merge the two datasets by keeping only the succesful metches
result = pd.merge(df_ndh_notot, df_pcodes_mun, 
                  left_on ='name_municipality_trim_ndh', 
                  right_on='name_municipality_trim_pcodes', 
                  how = 'outer')
#count the succesful matches
print 'entries in the ndhrhis database with pcode assignation' , 
len(result.loc[(result.municipality != 'TOTAL')  & ( result.municipality != np.NaN ), 'municipality'].unique())

result.loc[ result.municipality.notnull()]
#result_outer
stop()

#we merge the two datasets by keeping only the succesful metches
df_ndh_pcoded = pd.merge(df_ndh, df_pcodes_mun, 
                         left_on ='name_municipality_trim_ndh', 
                         right_on='name_municipality_trim_pcodes', 
                         how = 'inner')



result_notot  = result[result.municipality != 'TOTAL']
sum(result_notot.Pcode_municipality.isnull())


for i in result[['municipality','name_municipality_trim_ndh','name_municipality_trim_pcodes','name_municipality']]: print i
result_notot.to_csv('ndhrhis_v1_w_pcode.csv',encoding='utf-8' )
#merge right df_ndh and df_pcodes_mun.columns


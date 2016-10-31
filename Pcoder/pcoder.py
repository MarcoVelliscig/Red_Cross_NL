# Code developed by Marco Velliscig (marco.velliscig AT gmail.com)
# for the dutch red cross
# released under GNU GENERAL PUBLIC LICENSE Version 3


# to do
# find the most common words (city of capital) that need to be cleaned out
# to be provided as a list
# add extra word metric as an option
# add support for multiple words (different lenguages?)


def construct_known_match_tag(name , upper_level):
        """ Function that creates a tag with the name and the previus level
        to be added to the list of known matches
        """ 
        return name+ ' ' + upper_level

def find_best_match_user_input( poss_matches , 
                                name_to_match,  
                                upper_level , 
                                score_threshold, 
                                reject_threshold, 
                                known_matches , 
                                use_tricks=False):
        """ record linkage function that selects from a list of candidates name
        the best match for a given name applying string metrics 
        a list of known matches is also passed
        thresholds can be specified
        """
        
        known_match_tag = construct_known_match_tag(name_to_match , upper_level)
        #try first if the target is in the known matches dictionary
        try :
                best_match = known_matches[known_match_tag]
        except:
                if use_tricks :
                        #trim the strings from words like city and capital that can reduce 
                        # the accuarcy of the match
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
                #vector containing all possibilities with their score
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

                if float(best_ratio) <= reject_threshold:
                        most_prob_name_match  = 'Not found'
                        
                elif (float(best_ratio) > reject_threshold) & \
                     (float(best_ratio) < score_threshold): 
                    #ask if the possible match is right
                        print 'is ' , most_prob_name_match , 'the right match for ' , name_to_match , '(score:',best_ratio , ')'
                        respond = raw_input('press return for yes, everything else for no : ')

                        if respond != '' : 
                                sorted_prob_name_match =vec_poss_sorted[:,0]
                                sorted_prob_name_match_numbered = np.array(zip(sorted_prob_name_match, range(len(sorted_prob_name_match))))
                                print '\n select from the best match for ' ,name_to_match ,' from this list: \n',  sorted_prob_name_match_numbered

                                while True : 
                                        selected_index = raw_input('select the right choice by number, press return for not found : ')
                                        if selected_index == '' :
                                                most_prob_name_match  = 'Not found'
                                                break

                                        elif selected_index.isdigit():
                                                most_prob_name_match = sorted_prob_name_match_numbered[int(selected_index),0]
                                                break
                                        else:
                                                continue
                #update the known matched dictionary
                known_matches[known_match_tag] = most_prob_name_match 
                print '==' , most_prob_name_match , 'is the right match for ' , name_to_match , best_ratio , '\n'
                best_match=most_prob_name_match

        return best_match 


def match_against_template(df , df_template, level_tag ,ask_below_score, reject_below_score,exception = [] , reverse = False, verbose = False):
        
        

        # the code usually does the matching from the shallower to the deeper level
        # but it can also go the other way around even if it is less efficient this way
        # if you combine the 2 approaches you should account for most cases
        known_matches={}
        counter =0
        n_perfect_matches =0 
        n_no_matches =0 
        if reverse : 
                level_tag_use = list(reversed(level_tag))
        else:
                level_tag_use = level_tag 
        # do the search only for those line where the deepest 
        # admin level is null
        for index in  df.loc[df[level_tag[-1]+'_code'].isnull()].index :

                df_template_matches = df_template
                upper_level = ''
                for admin_level in level_tag_use :
                        if verbose : 
                                print 'len template dataframe level', admin_level\
                                        , len(df_template_matches)
                                print df_template_matches.describe()
                        
                        #gets the name of the admin level for the index entry
                        name_admin_level = df.loc[index][admin_level+'_name']
                        if name_admin_level in  exception : continue
                        # it tries to get a perfect match straight away
                        # !!!! this is not needed if a match is made by merge first
                        
                        n_matches_current_level = sum(df_template_matches[admin_level+'_name']==
                                              name_admin_level)
                        if verbose : print 'num matches', admin_level ,  n_matches_current_level
                


                        if (n_matches_current_level) > 0 :
                                if verbose : print ''

                        elif (n_matches_current_level) == 0 :
                                print "perc completed " , ((float(counter)/len(df.index))*100),'\n'
                                poss_matches = (df_template_matches[admin_level+'_name'].drop_duplicates()).values
                                score_threshold=ask_below_score[admin_level]     
                                reject_threshold=reject_below_score[admin_level]       

                                best_match  = find_best_match_user_input( poss_matches , name_admin_level,  upper_level , score_threshold, reject_threshold, known_matches ,  use_tricks = name_tricks) 
                                if best_match == 'Not found' :  
                                       n_no_matches +=1 
                                       #print '************* Not found, doing full search **********'
                                       print df.loc[index]
                                       #add here the full search instead 
                                       
                                       #break 
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

                                        df.loc[index,admin_level_tag+'_code']=(df_template_matches[admin_level_tag+'_code'].values[0])
                                        df.loc[index,admin_level_tag+'_best_match_name']=(df_template_matches[admin_level_tag+'_name'].values[0])
                        upper_level += admin_level + df.loc[index][admin_level+'_name']
                        #add dictionary with known matches
                counter+=1

        return df



import numpy as np
import pandas as pd
import difflib
# alternative metrics
# from nltk.metrics import edit_distance
# from nltk.metrics import masi_distance
# from nltk.metrics import jaccard_distance
# import operator
import re


region = 'philippines'



#it can produce the template from a file with all pcodes
# and different admin levels
produce_template=False

if region == 'malawi' : 




        #########################################################################
        #this part can be skipped if a template has already been produced
        if produce_template :

                df_pcodes = pd.read_csv('malawi_template.csv',encoding='utf-8',index_col=0)




                
                dict_template  = {'P_CODE_REGION':     'L1_code',
                                  'REGION':            'L1_name', 
                                  'P_CODE_DISTRICT':   'L2_code',
                                  'DISTRICT':          'L2_name',
                                  'P_CODE_TA':         'L3_code',
                                  'TRAD_AUTH':         'L3_name'}
                
                df_template = df_pcodes[dict_template.keys()]
                df_template.columns = [dict_template.values()]
                df_template = df_template.drop_duplicates()
        

                df_template.to_csv('pcode_template_malawi.csv',encoding='utf-8')

        ##########################################################################
        ##########################################################################


        #INPUT
        # Read the template csv file and assign admin levels to columns
        filename_template = 'pcode_template_malawi.csv'

        df_template = pd.read_csv(filename_template)

        
        #########################################################################


if region == 'philippines' :        

        produce_template =False
        # template for the philippines
        #add a string in front of the pcode
        preappend_string = 'PH'

        #########################################################################
        #this part can be skipped if a template has already been produced
        if produce_template :

                #building template file for the philippines
                #this part is philippines specific
                # a template should have a name and pcode for every admin level

                # read the Pcodes from template files
                # for provinces , municipalities and barangays
                df_pcodes_pro = pd.read_excel('template.xlsx',sheetname='Province', 
                                              skiprows = 1 , skip_footer = 1,header=None,encoding='utf-8',converters={0:str})

                df_pcodes_pro.columns = ['Pcode_province',
                                         'name_province']

                df_pcodes_mun = pd.read_excel('template.xlsx',sheetname='Municipality', 
                                              skiprows = 1 , skip_footer = 1,header=None,encoding='utf-8',converters={0:str,2:str})
                df_pcodes_mun.columns = ['Pcode_province',
                                         'name_province', 
                                         'Pcode_municipality',
                                         'name_municipality']


                df_pcodes_bar = pd.read_excel('template.xlsx',sheetname='Barangay', 
                                              skiprows = 1 , skip_footer = 1,header=None,encoding='utf-8',converters={0:str,2:str})
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


                for col in [u'Pcode_province', u'Pcode_municipality', u'Pcode_barangay']:
                        df_template[col] = df_template[col].apply(lambda x : preappend_string + str(x).strip())


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

#select one of the options below or create a new one
option =4


#option template (substitute with desired values)
if option == 'template' :
        # INPUT
        #specify the file you want to pcode
        #it can be a csv or a .xlsx file
        filename = './haima/filename.xlsx'

        #inizialize the list of known matches
        # in the future it could be loaded
        known_matches = {}
        #specify the output file name
        # this is a csv file
        sav_name = './haima/filename_pcoded.csv'

        # if excel file it is possible to specify the sheet name
        # put 0 if only one sheet, otherwise specify sheet name
        sheet_excel_name = 0
        #specify which columns correspont to what
        dict_raw = {'Province': 'L1_name', 
                    'Municipality': 'L2_name'}

        #specify the levels in the file
        level_tag = ['L1', 'L2' ]


        # ------ thresholds for word linkage ----
        # 0.8 is usually safe
        #different confidence level for different admin levels
        #threshold of 0.0 means there is no user imput (but less safe results)
        #threshold of 1.0 means that the code will always ask user input
        #this threshold should be higher for higher admin levels
        ask_below_score =  {'L1':0.9, 'L2':0.8}
        # if the score is below the reject level it is considered not found 
        reject_below_score= {'L1':0.55, 'L2':0.55, 'L3':0.55}

        #if True it removes indications of city, capital and (names in parethesis)
        # this is tailored to the philippines right now and english
        # in the future it can be determined automatically
        name_tricks = True





if option == 4 :
        # INPUT
        #specify the file you want to pcode
        filename = './haima/haima_houses_damaged.xlsx'

        known_matches = {}
        #specify the output file name
        sav_name = './haima/haima_houses_damaged_pcoded.csv'

        #0 if only one sheet, otherwise specify sheet name
        sheet_excel_name = 0
        #specify which columns correspont to what
        dict_raw = {'Province': 'L1_name', 
                    'Municipality': 'L2_name'}

        #specify the levels in the file
        level_tag = ['L1', 'L2' ]

        #different confidence level for different admin levels
        #threshold of 0.0 means there is no user imput but less safe results
        #threshold of 1.0 means that the code will always ask user input
        #this threshold should be higher for higher admin levels
        ask_below_score =  {'L1':0.9, 'L2':0.8}
        reject_below_score= {'L1':0.55, 'L2':0.55, 'L3':0.55}
        #if True it removes indications of city, capital and (names in parethesis)
        name_tricks = True



if option == 3 :
        # INPUT
        #specify the file you want to pcode
        filename = 'Typhoon Yolanda - Casualties.xlsx'

        known_matches = {}
        #specify the output file name
        sav_name = 'Typhoon Yolanda - Casualties_pcoded.csv'
        sheet_excel_name = 'Summary'
        #specify which columns correspont to what
        dict_raw = {'Province': 'L1_name', 
                    'Municipality': 'L2_name'}

        #specify the levels in the file
        level_tag = ['L1', 'L2' ]

        #different confidence level for different admin levels
        #threshold of 0.0 means there is no user imput but less safe results
        #threshold of 1.0 means that the code will always ask user input
        #this threshold should be higher for higher admin levels
        ask_below_score =  {'L1':0.9, 'L2':0.7}
        reject_below_score= ask_below_score #{'L1':0.55, 'L2':0.55, 'L3':0.55}
        #if True it removes indications of city, capital and (names in parethesis)
        name_tricks = True
        





#preset input options for philippines
if option == 0 :
        # INPUT
        #specify the file you want to pcode
        filename = 'barangay.csv'

        known_matches = {}
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
        ask_below_score =  {'L1':0.9, 'L2':0.7, 'L3':0.7}
        reject_below_score= ask_below_score #{'L1':0.55, 'L2':0.55, 'L3':0.55}
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

        known_matches = {}
        #specify the levels in the file
        level_tag = ['L1', 'L2' ]
        #different confidence level for different admin levels
        ask_below_score= {'L1':0.9, 'L2':0.7}
        reject_below_score=  {'L1':0.6, 'L2':0.6}
        #if True it removes indications of city, capital and (names in parethesis)
        name_tricks = True

if option == 2 :
        # INPUT
        #specify the file you want to pcode
        filename = 'governance_index.xlsx'
        
        #specify the output file name
        sav_name = 'governance_index_pcoded_250716.csv'
        #specify which columns correspont do what
        dict_raw = {'municipality':'L2_name' , 
                    'province'  : 'L1_name'}

        known_matches = {}
        #specify the levels in the file
        level_tag = ['L1', 'L2' ]
        #different confidence level for different admin levels
        ask_below_score= {'L1':0.9, 'L2':0.7}
        reject_below_score=  {'L1':0., 'L2':0.}
        #if True it removes indications of city, capital and (names in parethesis)
        name_tricks = True


        
#read the file to pcode
if filename.split(".")[-1] == 'csv' :
        df_raw = pd.read_csv(filename,encoding='utf-8')
elif filename.split(".")[-1] == 'xlsx' :
        df_raw = pd.read_excel(filename,encoding='utf-8', sheetname=sheet_excel_name)

# make the df_raw uppercase for merging pourposes 
# see the end of the code
for i in range(len(dict_raw)) : df_raw[dict_raw.keys()[i]]=df_raw[dict_raw.keys()[i]].str.upper()

#rename columns
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
        #df[admin_level+'_code'] = np.NaN 
        df[admin_level+'_best_match_name'] = np.NaN 

level_tag_name = [ ] 
for tag in level_tag : level_tag_name.append(tag+'_name') 


#### a simple join is tried first for exact matches

df = pd.merge(df , df_template , on = level_tag_name   , how = 'left')

### 
#forward and backward pass
df = match_against_template(df , df_template, level_tag,ask_below_score, reject_below_score, reverse = False)
df = match_against_template(df , df_template, level_tag,ask_below_score, reject_below_score, reverse = True)



# saving the known matches so they can be loaded and modified
# later version the known matches file can be specified in the options
# warm_start

df_known_matches = pd.DataFrame.from_dict(known_matches, orient='index')
#df_known_matches.reset_index()
#df_known_matches.columns = [ 'name_raw' , 'name_match' ] 
name_km_sav = 'known_matches_' + sav_name
df_known_matches.to_csv(name_km_sav,encoding='utf-8' )
df_dummy = pd.read_csv(name_km_sav,encoding='utf-8')


#merge it back to the original file
df_pcoded = pd.merge(df_raw, df, 
                     left_on =dict_raw.keys(), 
                     right_on=dict_raw.values(), 
                     how = 'inner')
df_pcoded.to_csv(sav_name,encoding='utf-8' )

print ' list of no matches' , sum(df[level_tag[-1]+'_code'].isnull())
print ' list of matches ' , sum(df[level_tag[-1]+'_code'].notnull())




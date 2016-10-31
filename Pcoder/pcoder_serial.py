import pandas as pd
from os import listdir
from os.path import isfile, join
from pcoder import find_best_match_user_input



level_tag = ['L1', 'L2' , 'L3']



filename_template = 'pcode_template_malawi.csv'

df_template = pd.read_csv(filename_template)
df_template_serial = pd.DataFrame(columns =['name' , 'code'] )


for name_level_tag in level_tag:
    col = [name_level_tag + '_name' , name_level_tag + '_code']
    df_temp = df_template[col].drop_duplicates()
    df_temp.columns = ['name' , 'code']
    df_template_serial = df_template_serial.append(df_temp)

df_template_serial.to_csv('pcode_template_malawi_serial.csv')

path = './malawi_files/'

files_in_dir = [f for f in listdir(path) if (isfile(join(path, f)) and (not f.startswith('pcoded')) and (f.endswith('csv')))]
for filename in files_in_dir :
    print filename
    df_to_pcode = pd.read_csv(path+filename)

    df = pd.merge(df_to_pcode ,  df_template_serial , left_on =df_to_pcode.columns[0], right_on = 'name',  how='left')
    
    known_matches={}
    for index in  df.loc[df.code.isnull()].index :
        name_admin_level = df.ix[index,0]
        #print name_admin_level
        poss_matches = df_template_serial.name.values
        best_match  = find_best_match_user_input( poss_matches , name_admin_level,  '' , 0.8, 0.8 , known_matches,  use_tricks = False)
        #print best_match
        #print df.ix[index,'name'] , df.ix[index,'code']
        if best_match !=  'Not found':
            df.ix[index,'name'] = df_template_serial.loc[df_template_serial.name == best_match, 'name'].values[0]

            df.ix[index,'code'] = df_template_serial.loc[df_template_serial.name == best_match, 'code'].values[0]


            print 'new match' , df.ix[index,'name'] , df.ix[index,'code']
    
    df.to_csv(path+'pcoded_'+filename)

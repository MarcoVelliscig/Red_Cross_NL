def norm_column(df , name_vec , new_name_vec , norm_type = '0mean1std', rename = False):
    """ Normalize column across differnt train and test dataset  """
    #the min should be the min of the min to avoid negative params
    #the max should be the max of the max to avoid bigger than one params
    for name,  new_name in zip(name_vec, new_name_vec):
        min =df[name].min() 
        max = df[name].max() 

        if norm_type == 'm_min_div_minmax':
            df[new_name] = (df[name] - min )/ (max-min)
        if norm_type == 'div_max':
            df[new_name] = (df[name])/ (max)  
      
        print df[new_name].describe()


def apply_transform(df , name_vec , new_name_vec, trans = 'log', drop = False, rename = False):
    """ Normalize column across differnt train and test dataset  """
    #the min should be the min of the min to avoid negative params
    #the max should be the max of the max to avoid bigger than one params
    for name,  new_name in zip(name_vec, new_name_vec):


        if trans == 'log' : df[new_name] = np.log10(df[name])
        if trans == 'log_p1' : df[new_name] = np.log10(df[name]+1)
        if trans == 'log_div_max_log' : df[new_name] = np.log10(df[name]) / np.log10(df[name].max()) 

        if trans == 'div_max' : df[new_name] = (df[name]) /    df[name].max()
        print df[new_name].describe()
        if drop : df = df.drop([name], axis=1)




def fill_null(df_vec,name_vec , groupby_col = 'all' , method = 'fill_median' , sample = 'first', drop = False, rename = False):
    """ fill missing values using only the train dataset to compute medians for each group """
    if method == 'fill_median':


        if sample == 'first' : 
            df_sample = df_vec[0]
        else:
            df_sample =  df_vec[0].append(df_vec[1])



        if groupby_col == 'all' : 
            med = df_sample.median()
        else : 
            med = df_sample.groupby(groupby_col).median()
        
        print med

        for df in df_vec :
            for name in name_vec:
                if  rename == True :
                    new_name = name+method+groupby_col
                    df[new_name] = df[name]
                else:
                    new_name=name
                unique_values = (df[groupby_col].unique())
                for i in range(len(unique_values)) : 
                    df.loc[ (df[name].isnull())&(df[groupby_col]==unique_values[i]), new_name] = med[name].iloc[i] 
                if drop : df = df.drop([name], axis=1)


def one_hot(df , name_vec , drop = False) :
    """transform one column in one hot encoding
    possible to drop the original column if wanted"""


    for name in name_vec:
        class_one_hot =pd.get_dummies(df[name])

        #dat1.join(dat2)
        df = df.join( class_one_hot)
        #df_vec[i] = pd.concat([df_vec[i], class_one_hot], axis=1)
        if drop : df = df.drop([name], axis=1)
        print df.columns
    return df




# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.5, random_state=0)

# # Set the parameters by cross-validation
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# scores = ['precision', 'recall']

# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()

#     clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
#                        scoring='%s_weighted' % score)
#     clf.fit(X_train, y_train)

#     print("Best parameters set found on development set:")
#     print()
#     print(clf.best_params_)
#     print()
#     print("Grid scores on development set:")
#     print()
#     for params, mean_score, scores in clf.grid_scores_:
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean_score, scores.std() * 2, params))
#     print()

#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(X_test)
#     print(classification_report(y_true, y_pred))
#     print()








def model_pred(X,Y, maximize='accuracy' , model_type='logreg' ,limits = [-3 , 1 , 0.5] ):


    #learn Logistic regression
    best_lambda = 0
    best_score = -1
    n_cv_sets = 5
    X_train, X_test , Y_train , Y_test= train_test_split(X,Y, test_size = 1/float(n_cv_sets))

    for i in np.arange(limits[0],limits[1] , limits[2]):
        lambd = 10**i
        print lambd
        if model_type=='linreg' :
            model = linear_model.Ridge(alpha =  lambd )
        if model_type=='lasso' :
            model = linear_model.Lasso(alpha =  lambd )
        elif model_type=='randomforest' :
            model = RandomForestRegressor(n_estimators =int(lambd), n_jobs=2)
        elif model_type=='GBT' :
            model = GradientBoostingClassifier(n_estimators=int(lambd), learning_rate=1.0,max_depth=1, random_state=0)
        r2 = cross_validation.cross_val_score(model, X_train, Y_train, cv=n_cv_sets, scoring='r2')
        #print 'accuracy split' , accuracy
        print("r2: %0.2f (+/- %0.2f)" % (r2.mean(), r2.std() * 2))
        

        if maximize=='r2': score_to_max = r2.mean()


        if  score_to_max > best_score :
            best_lambda = lambd
            best_score = score_to_max
            print 'best score ' , best_score , 'best lambd ' , best_lambda

    print 'done searching'

    print '===========  best score ' , best_score , 'best lambd ' , best_lambda
    #use the best lambda to fir the model
    if model_type=='linreg' :
        model = linear_model.Ridge(alpha =  best_lambda )
        model.fit(X_train, Y_train)
        print  zip(X_train.columns , model.coef_)
    elif model_type=='lasso' :    
        model = linear_model.Lasso(alpha =  best_lambda )
        model.fit(X_train, Y_train)        
    elif model_type=='randomforest' :
        model = RandomForestRegressor(n_estimators =int(best_lambda), n_jobs=2)
        model.fit(X_train, Y_train)
        print zip(X_train.columns , model.feature_importances_)
    elif model_type=='GBT' :
        model = GradientBoostingClassifier(n_estimators=int(best_lambda), learning_rate=1.0,max_depth=1, random_state=0)

    elif model_type=='NN' :
        model = GradientBoostingClassifier(n_estimators=int(best_lambda), learning_rate=1.0,max_depth=1, random_state=0)

        model.fit(X_train, Y_train)

    print '===params best model' ,model.get_params()
    score_test = model.score(X_test, Y_test)
    print 'score test' ,score_test

    #sklearn.cross_validation.cross_val_predict(estimator, X, y=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')[source]
    Y_test_pred = (model.predict(X_test))
    print ' diff pred and ground ' , np.mean(abs(Y_test_pred -Y_test ))
    #Y_sub_pred= pd.Series(model.predict(X))
    Y_sub_pred=cross_validation.cross_val_predict(model, X, y=Y, cv=n_cv_sets, n_jobs=2)
    #Y_sub_pred=(model.predict(X))
    

    
    return Y_sub_pred , best_score


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
from sklearn.ensemble import RandomForestRegressor
import string
#df = pd.read_excel('Haiyan_Overview.xlsx')
#df = pd.read_excel('150707_Philippines_MultiplicModel_and_DEA(1I_2O_CCR_O)output_combined.xlsx',sheetname='DataWithDEA')

# Index([u'OBJECTID', u'ISO', u'P_Code', u'M_Code', u'Municipality',
#        u'Area (km2)', u'Avg. Elevation (m)', u'Perimeter (m)',
#        u'Coastline Length (m)', u'C/P Ratio',
#        u'Distance to coastline (m, mean)', u'Distance to coastline (m, stdev)',
#        u'Ruggedness index (mean)', u'Ruggedness index (stdev)',
#        u'Slope (mean)', u'Slope (stdev)', u'Landuse (Most common)',
#        u'Population 2010 census', u'Population 2013 est.',
#        u'Pop. Density 2010 per km2', u'Poverty (%)', u'Avg. Windspeed (km/h)',
#        u'Distance from typhoon path (km)', u'Area Flooded (%)', u'Rainfallme',
#        u'Surge Height int. (m)', u'Surge risk class (REACH)',
#        u'People affected', u'People affected (% 2010)', u'Deaths',
#        u'Houses damaged (REACH)', u'Houses damaged (% Shelter Cluster)',
#        u'Emergency shelter (% planned & reached Shelter Cluster)',
#        u'Support to Self Recovery (% planned & reached Shelter Cluster)',
#        u'OCHA CCCM', u'OCHA Education', u'OCHA Early Recovery & Livelyhoods',
#        u'OCHA Food security and Agriculture', u'OCHA Nutrition',
#        u'OCHA Protection', u'OCHA WASH'],
#       dtype='object')

new_f= [ 'Perimeter (m)','Coastline length (m)', 'C/P Ratio',
         'Distance to coastline (m, mean)', 'Distance to coastline (m, stdev)',
         'Ruggedness index (mean)', 'Ruggedness index (stdev)','Slope (mean)', 
         'Slope (stdev)', 'Landuse (Most common)', 'Rainfallme']
#df[new_f].hist()
#plt.show()
#df = pd.read_csv('Haiyan_Overview.csv')
df = pd.read_csv('20160623_Haiyan_Overview.csv')
for col in df.columns : print df[col].describe() , '\n'
for col in df.columns : print df.loc[df['People affected']>350000,col]
#df['People affected'] = df['People affected (gov report)']
#df = df[df['People affected']>2000]
len(df[df['People affected (% 2010)']==0.5])
improved_feauture_list = ['area_log' , 
                          'elevation_log', 
                          'poverty_frac', 
                          'pop13_est_log',
                          'pop_density_log' ,
                          'wind_speed_log', 
                          'dist_norm', 
                          'surge_height_log', 
                          'C/P Ratio',
                          'rugged_index',
                          'coastline_length',
                          'dist_coast',
                          'dist_coast_std',
                          'rugged_index_std'
                          'slope_mean', 
                          'slope_std' ]


df['Coastline length (m)'] = df['Coastline length (m)'] +1.0

#----------------------------
df['poverty_frac'] = df['Poverty (%)']/100.

transformation = 'log_div_max_log'
print  'transform as ' , transformation , '\n'
name_old =['Area (km2)', 
           'Avg. Elevation (m)', 
           'Population 2013 est.',
           'Population 2010 census',
           'Pop. Density 2010 per km2',
           'Avg. Windspeed (km/h)',
           'Surge Height int. (m)',
           'Coastline length (m)',
           'Distance to coastline (m, mean)', 
           'Distance to coastline (m, stdev)']


name_new=  ['area_log', 
            'elevation_log', 
            'pop13_est_log', 
            'pop10_log',
            'pop_density_log',
            'wind_speed_log',
            'surge_height_log',
            'coastline_length',
            'dist_coast',
            'dist_coast_std']

print  'varaiables to transform ' ,  '\n' , zip(name_old , name_new)
apply_transform(df ,name_old ,name_new, trans = transformation)




#----------------------------
transformation = 'div_max'
print  'transform as ' , transformation , '\n'
name_old =['Distance from typhoon path (km)', 
           'Ruggedness index (mean)' ,
           'Ruggedness index (stdev)',
           'Slope (mean)', 
           'Slope (stdev)',
           'Rainfallme']

name_new=  ['dist_norm',
            'rugged_index',
            'rugged_index_std',
            'slope_mean', 
            'slope_std',
            'rainfall']

print  'varaiables to transform ' ,  '\n' , zip(name_old , name_new)
apply_transform(df ,name_old ,name_new, trans = transformation)
#----------------------------

#one hot encoding
df  = one_hot(df , [ 'Landuse (Most common)'], drop = False)





df['perc_people_affected']=(df['People affected']/df['Population 2010 census'])

df = df.dropna(subset=[ 'surge_height_log','pop10_log','perc_people_affected','poverty_frac' ])





improved_feauture_list = ['area_log' , 
                          'elevation_log', 
                          'poverty_frac', 
                          'pop10_log',
                          'pop_density_log' ,
                          'wind_speed_log', 
                          'dist_norm', 
                          'surge_height_log', 
                          'C/P Ratio',
                          'rugged_index',
                          'coastline_length',
                          'dist_coast',
                          'dist_coast_std',
                          'rugged_index_std',
                          'slope_mean', 
                          'slope_std', 
                          'rainfall',
                          #14, 20, 40,130,190, 210
]
X = df[improved_feauture_list]
ids = df['M_Code']

X.hist()
plt.show()


#$$$$$$$$$$$$$$$$$$$$$$   try to predict the pop % people affected
predict_on = 'perc_people_affected'
Y = df[predict_on]


# model_name = 'linreg'
# print '&&&&&&&& predict ' ,predict_on , ' with ' ,model_name 
# #for i in range(len(X)): print X.iloc[i] 
# Y_pred_linreg , best_linreg = model_pred(X,Y, maximize='r2' , model_type=model_name  ,limits = [-6 , 1 , 0.5] )
# print model_name , ' mean perc  pop affected error' , np.mean(abs(Y_pred_linreg - Y))
# print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_linreg - Y)*(df['Population 2010 census'])  )  )
# print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_linreg - Y)*(df['Population 2010 census']) - df['People affected']  ) /  (df['People affected'] +1))






# model_name = 'lasso'
# print '&&&&&&&& predict ' ,predict_on , ' with ' ,model_name
# Y_pred_lasso , best_lasso  = model_pred(X,Y, maximize='r2' , model_type=model_name ,limits = [-6 , 1 , 0.5] )
# print model_name , ' mean perc  pop affected error' , np.mean(abs(Y_pred_lasso - Y))
# print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_lasso - Y)*(df['Population 2010 census'])  )  )
# print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_lasso - Y)*(df['Population 2010 census']) - df['People affected']  ) / ( df['People affected']+1) )

# model_name='randomforest'
# print '&&&&&&&& predict ' ,predict_on , ' with ' ,model_name
# Y_pred_randomforest , best_randomforest = model_pred(X,Y, maximize='r2' , model_type=model_name ,limits = [1 , 3.5 , 0.5] )
# print model_name , ' mean perc  pop affected error' , np.mean(abs(Y_pred_randomforest - Y))
# print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_randomforest - Y)*(df['Population 2010 census'])  )  )
# print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_randomforest - Y)*(df['Population 2010 census']) - df['People affected']  ) /  (df['People affected']+1) )

# Y_df = pd.DataFrame(Y.values )
# Y_df.columns = [predict_on] 
# Y_df['pred_'+predict_on] = Y_pred_randomforest




# Y_df.describe()
# Y_df['pred_'+predict_on].hist(bins=20,alpha=0.5)
# Y_df[predict_on].hist(bins=40,alpha=0.5)
# plt.show()
# Y_df.plot()
# plt.show()


#stop()

# predictions_file = open("predict_perc_pop.csv", "wb")
# open_file_object = csv.writer(predictions_file)
# open_file_object.writerow(["pcode","perc_affec_pred" , "perc_true"])
# open_file_object.writerows(zip(ids, Y_pred_randomforest, Y.values))
# predictions_file.close()


df = df.dropna(subset=[ 'surge_height_log','pop10_log','perc_people_affected','poverty_frac' ,'Houses damaged (REACH)']).reset_index()

X = df[improved_feauture_list]
ids = df['M_Code']

predict_on = 'Houses damaged (REACH)'
Y = df[predict_on]/df['Population 2010 census']

model_name='linreg'
print '@@@@@@@@@@@@@@@@@  predict ' ,predict_on , ' with ' ,model_name  

Y_pred_linreg , best = model_pred(X,Y, maximize='r2' , model_type=model_name ,limits = [-6 , 1 , 0.5] )
print model_name , ' mean house damage  error' , np.mean(abs(Y_pred_linreg - Y)*df['Population 2010 census'])
#print 'linreg mean  house damage error' , np.mean(abs(Y_pred_linreg - Y))*df['People affected'].mean()

model_name='lasso'
Y_pred_lasso , best = model_pred(X,Y, maximize='r2' , model_type=model_name ,limits = [-6 , 1 , 0.5] )
print model_name , ' mean house damage error' , np.mean(abs(Y_pred_lasso - Y)*df['Population 2010 census'])
#print 'lasso mean  pop affected error' , np.mean(abs(Y_pred_lasso - Y))*df['People affected'].mean()


model_name='randomforest'
Y_pred_randomforest , best = model_pred(X,Y, maximize='r2' , model_type=model_name ,limits = [1 , 3.5 , 0.5] )
print model_name , 'mean house damage error' , np.mean(abs(Y_pred_randomforest - Y)*df['Population 2010 census'])
#print 'randomforest mean pop affected error' , np.mean(abs(Y_pred_randomforest - Y))*df['People affected'].mean()

predict_on_tag = 'houses'
Y_df_h = pd.DataFrame(Y.values )
Y_df_h.columns = ['perc_'+predict_on_tag+'_true'] 
Y_df_h['perc_'+predict_on_tag+'_pred'] = Y_pred_randomforest
Y_df_h['M_Code'] = df['M_Code']
Y_df_h['Municipality'] = df['Municipality']
Y_df_h['num_'+predict_on_tag+'_pred'] = Y_df_h['perc_'+predict_on_tag+'_pred']*df['Population 2010 census']
Y_df_h['num_'+predict_on_tag+'_true'] = Y_df_h['perc_'+predict_on_tag+'_true']*df['Population 2010 census']
Y_df_h['num_'+predict_on_tag+'_error'] = abs(Y_df_h['num_'+predict_on_tag+'_pred'] - Y_df_h['num_'+predict_on_tag+'_true'])
Y_df_h['perc_'+predict_on_tag+'_error'] = abs(Y_df_h['perc_'+predict_on_tag+'_pred'] - Y_df_h['perc_'+predict_on_tag+'_true'])
Y_df_h['rel_num_'+predict_on_tag+'_error'] = Y_df_h['num_'+predict_on_tag+'_error'] / Y_df_h['num_'+predict_on_tag+'_true'] 


Y_df_h.to_csv("house_d_29062016.csv")

Y_df_h[[u'num_houses_pred',u'num_houses_true',]].plot()
Y_df_h['num_houses_pred'].hist(bins=20,alpha=0.5)
Y_df_h['num_houses_true'].hist(bins=30,alpha=0.5)

Y_df_h.describe()
Y_df_h['pred_'+predict_on_tag].hist(bins=20,alpha=0.5)
Y_df_h[predict_on].hist(bins=40,alpha=0.5)
plt.show()
Y_df_h.plot()
plt.show()


pred_house_perc  = Y_pred_randomforest
pred_house_num = Y_pred_randomforest*df['Population 2010 census']
true_house_perc = Y.values
true_house_num

predictions_file = open("predict_perc_house_damaged.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["pcode","perc_affec_pred" , 'perc_true' ])
open_file_object.writerows(zip(ids, Y_pred_randomforest , Y.values))
predictions_file.close()



#Learning old feautures

stop()
print '0000000000000000000000000000000 old feautures 0000000000000000000000000000000'
feauture_list = ['area_log' , 'elevation_log', 'poverty_frac', 'pop10_log','pop_density_log' ,'wind_speed_log', 'dist_norm', 'surge_height_log']
X = df[feauture_list]
ids = df['M_Code']





#$$$$$$$$$$$$$$$$$$$$$$   try to predict the pop % people affected
predict_on = 'perc_people_affected'
Y = df[predict_on]


model_name = 'linreg'
print '&&&&&&&& predict ' ,predict_on , ' with ' ,model_name 
#for i in range(len(X)): print X.iloc[i] 
Y_pred_linreg , best_linreg = model_pred(X,Y, maximize='r2' , model_type=model_name  ,limits = [-6 , 1 , 0.5] )
print model_name , ' mean perc  pop affected error' , np.mean(abs(Y_pred_linreg - Y))
print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_linreg - Y)*(df['Population 2013 est.'])  )  )

model_name = 'lasso'
print '&&&&&&&& predict ' ,predict_on , ' with ' ,model_name
Y_pred_lasso , best_lasso  = model_pred(X,Y, maximize='r2' , model_type=model_name ,limits = [-6 , 1 , 0.5] )
print model_name , ' mean perc  pop affected error' , np.mean(abs(Y_pred_lasso - Y))
print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_lasso - Y)*(df['Population 2013 est.'])  )  )

model_name='randomforest'
print '&&&&&&&& predict ' ,predict_on , ' with ' ,model_name
Y_pred_randomforest , best_randomforest = model_pred(X,Y, maximize='r2' , model_type=model_name ,limits = [1 , 3.5 , 0.5] )
print model_name , ' mean perc  pop affected error' , np.mean(abs(Y_pred_randomforest - Y))
print model_name , '  pop affected error' , np.mean(  abs(  (Y_pred_randomforest - Y)*(df['Population 2013 est.'])  )  )





stop()

#fairly evenly distrib
df['dist_norm'] = df['Distance from typhoon path (m)']/df['Distance from typhoon path (m)'].max()
#put a floor in the distance estimation
df.loc[df.dist_norm==0, 'dist_norm']=0.001
df['people_affected_log']=np.log10(df['People affected'])

df['people_affected_norm'] = df['People affected']/df['People affected'].max()
df['people_affected_lognorm'] = df['people_affected_log']/df['people_affected_log'].max()



#df['perc_people_affected']=(df['People affected']/df['Population 2010 census'])
#df['perc_people_affected_log']=np.log10(df['perc_people_affected'])
#df['perc_people_affected_lognorm']=(df['perc_people_affected_log'])/df['perc_people_affected_log'].max()


df['pop13_est'] =df['Population 2013 est.'] / df['Population 2013 est.'].max()




df['surge_risk']= df['Surge risk class (REACH)'].map( {'inland': 0, 'low': 0.3 , 'medium':0.6 ,'high':1.0})



feauture_list = ['area_log' , 'elevation_log', 'poverty_frac', 'pop13_est_log','pop_density_log' ,'wind_speed_log', 'dist_norm', 'surge_height_log']












Y = df['people_affected_lognorm']
                                    
#for i in range(len(X)): print X.iloc[i] 
Y_pred_linreg , best = model_pred(X,Y, maximize='r2' , model_type='linreg' ,limits = [-6 , 1 , 0.5] )
print 'linreg mean pop affected error' , np.mean(abs(10**((Y.values)*df['people_affected_log'].max()) - 10**(Y_pred_linreg*df['people_affected_log'].max())))

Y_pred_lasso , best = model_pred(X,Y, maximize='r2' , model_type='lasso' ,limits = [-6 , 1 , 0.5] )
print 'lasso mean pop affected error' , np.mean(abs(10**((Y.values)*df['people_affected_log'].max()) - 10**(Y_pred_lasso*df['people_affected_log'].max())))



Y_pred_randomforest , best = model_pred(X,Y, maximize='r2' , model_type='randomforest' ,limits = [1 , 3.5 , 0.5] )
print 'randomforest mean pop affected error' , np.mean(abs(10**((Y.values)*df['people_affected_log'].max()) - 10**(Y_pred_randomforest*df['people_affected_log'].max())))









predictions_file = open("predict_perc_pop.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["pcode","perc_affec_pred" ])
open_file_object.writerows(zip(ids, Y_pred_randomforest))
predictions_file.close()




df = df.dropna(subset=[ 'surge_height_log','pop13_est_log','people_affected_norm', 'Houses damaged (REACH)' ])

X = df[feauture_list]


Y = df['Houses damaged (REACH)']
                                    
#for i in range(len(X)): print X.iloc[i] 
Y_pred_linreg , best = model_pred(X,Y, maximize='r2' , model_type='linreg' ,limits = [-6 , 1 , 0.5] )
print 'linreg mean house damage  error' , np.mean(abs(Y_pred_linreg - Y))
#print 'linreg mean  house damage error' , np.mean(abs(Y_pred_linreg - Y))*df['People affected'].mean()


Y_pred_lasso , best = model_pred(X,Y, maximize='r2' , model_type='lasso' ,limits = [-6 , 1 , 0.5] )
print 'lasso mean house damage error' , np.mean(abs(Y_pred_lasso - Y))
#print 'lasso mean  pop affected error' , np.mean(abs(Y_pred_lasso - Y))*df['People affected'].mean()



Y_pred_randomforest , best = model_pred(X,Y, maximize='r2' , model_type='randomforest' ,limits = [1 , 3.5 , 0.5] )
print 'randomforest mean house damage error' , np.mean(abs(Y_pred_randomforest - Y))
#print 'randomforest mean pop affected error' , np.mean(abs(Y_pred_randomforest - Y))*df['People affected'].mean()





predictions_file = open("predict_house_damage.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["pcode","perc_affec_pred" ])
open_file_object.writerows(zip(ids, Y_pred_randomforest))
predictions_file.close()


Y = df['Houses damaged (REACH)'] / df['People affected']
                                    
#for i in range(len(X)): print X.iloc[i] 
Y_pred_linreg , best = model_pred(X,Y, maximize='r2' , model_type='linreg' ,limits = [-6 , 1 , 0.5] )
print 'linreg mean house perc damage  error' , np.mean(abs(Y_pred_linreg - Y))
print 'linreg mean  houseperc damage error' , np.mean(abs(Y_pred_linreg - Y))*df['People affected'].mean()


Y_pred_lasso , best = model_pred(X,Y, maximize='r2' , model_type='lasso' ,limits = [-6 , 1 , 0.5] )
print 'lasso mean house perc damage error' , np.mean(abs(Y_pred_lasso - Y))
print 'lasso mean house affected error' , np.mean(abs(Y_pred_lasso - Y))*df['People affected'].mean()



Y_pred_randomforest , best = model_pred(X,Y, maximize='r2' , model_type='randomforest' ,limits = [1 , 3.5 , 0.5] )
print 'randomforest mean house perc damage error' , np.mean(abs(Y_pred_randomforest - Y))
print 'randomforest mean pop affected error' , np.mean(abs(Y_pred_randomforest - Y))*df['People affected'].mean()





predictions_file = open("predict_perc_house_damage.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["pcode","perc_affec_pred" ])
open_file_object.writerows(zip(ids, Y_pred_randomforest))
predictions_file.close()



TODO()
#try to predict the pop density 

Y = df['perc_people_affected']
                                    
#for i in range(len(X)): print X.iloc[i] 
Y_pred_linreg , best = model_pred(X,Y, maximize='r2' , model_type='linreg' ,limits = [-6 , 1 , 0.5] )
print 'mean perc  pop affected error' , np.mean(abs(Y_pred_linreg - Y))
print 'mean  pop affected error' , np.mean(abs(Y_pred_linreg - Y))*df['People affected'].mean()

Y_pred_lasso , best = model_pred(X,Y, maximize='r2' , model_type='lasso' ,limits = [-6 , 1 , 0.5] )
print 'mean perc  pop affected error' , np.mean(abs(Y_pred_lasso - Y))
print 'mean  pop affected error' , np.mean(abs(Y_pred_lasso - Y))*df['People affected'].mean()



Y_pred_randomforest , best = model_pred(X,Y, maximize='r2' , model_type='randomforest' ,limits = [1 , 3.5 , 0.5] )
print 'mean perc pop affected error' , np.mean(abs(Y_pred_randomforest - Y))
print 'mean pop affected error' , np.mean(abs(Y_pred_randomforest - Y))*df['People affected'].mean()

Y.reset_index().plot()
Y_pred_randomforest.plot()
df_sort['People affected'].reset_index().plot()


Y.reset_index()['perc_people_affected'].plot()
Y_pred_randomforest.reset_index()[0].plot()
abs(Y.reset_index()['perc_people_affected']-Y_pred_randomforest.reset_index()[0]).plot()
Y['pred'] = Y_pred_randomforest
Y_s=Y.sort('perc_people_affected')
Y_s['perc_people_affected'].reset_index().plot()

Y_s['pred'.reset_index()].plot()

# automatic selection of best feautures/ combination of feautures
# grid parameter search
# neural network

# Utility function to report best scores
def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print 'cross validation scores', score.cv_validation_scores
        print("Parameters: {0}".format(score.parameters))
        print("")



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

def model_pred(X,Y,
               hyperparams ,
               maximize='accuracy' ,
               model_type='logreg' ,
               n_iter_search = 30,
               n_cv_sets = 10 ,
               limits = [-3 , 1 , 0.5] ):



    X_train, X_test , Y_train , Y_test= train_test_split(X,Y, test_size = 0.3)

    param_dist = hyperparams
    
    if model_type=='linreg' :
        model = linear_model.ElasticNet()
    elif model_type=='lasso' :
        model = linear_model.Lasso()
    elif model_type=='randomforest' :
        model = RandomForestRegressor()
    elif model_type=='GBT' :
        model = GradientBoostingRegressor()
    elif model_type == 'NN':
        model = MLPRegressor()

    #how to decide the score
    random_search = RandomizedSearchCV(model, param_distributions=param_dist,n_iter=n_iter_search, cv = n_cv_sets, )#scoring=
    #scorer(estimator, X, y)

    start = time()
    random_search.fit(X_train, Y_train)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)
    #random_search.fit(X_train, Y_train)
    Y_test_pred= random_search.predict(X_test)
    print 'score test set' , random_search.score(X_test, Y_test)




    print '===params best model' ,random_search.best_params_ 



    print ' diff pred and ground ' , np.mean(abs(Y_test_pred -Y_test ))
    


    model =random_search.best_estimator_    
    Y_sub_pred=cross_validation.cross_val_predict(model, X, y=Y, cv=n_cv_sets, n_jobs=2)
    #Y_sub_pred=(model.predict(X))
    best_score=random_search.best_score_ 
    print ' score on train set' , best_score

    if model_type=='linreg' :
        print zip(X_train.columns , model.coef_)
    elif model_type=='lasso' :
        model = linear_model.Lasso()
    elif model_type=='randomforest' :
        print zip(X_train.columns , model.feature_importances_)
    elif model_type=='GBT' :
        print zip(X_train.columns , model.feature_importances_)         
    elif model_type == 'NN':
        model = MLPRegressor()


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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import string
from time import time
from scipy.stats import randint as sp_randint
from operator import itemgetter
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
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
         'Slope (stdev)', 'Landuse (Most common)', 'Rainfallme','Surge_Hint']
#df[new_f].hist()
#plt.show()
#df = pd.read_csv('Haiyan_Overview.csv')
df = pd.read_csv('20160623_Haiyan_Overview.csv')
for col in df.columns : print df[col].describe() , '\n'
for col in df.columns : print df.loc[df['People affected']>350000,col]
#df['People affected'] = df['People affected (gov report)']
#df = df[df['People affected']>2000]
len(df[df['People affected (% 2010)']==0.5])


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

#improved_feauture_list = ['dist_norm']








df = df.dropna(subset=[ 'surge_height_log','pop10_log','perc_people_affected','poverty_frac' ,'Houses damaged (REACH)']).reset_index()

#df = df.loc[df['Houses damaged (REACH)']>10]

X = df[improved_feauture_list]
ids = df['M_Code']

X.hist()
plt.show()


#=============================

predict_on = 'Houses damaged (REACH)'
Y = np.log(df[predict_on]/df['Population 2010 census'])



hyperparams_dict ={}
hyperparams_dict['GBT'] = {"loss" : ['ls', 'lad', 'huber'],
                           "n_estimators" : [100,500,700,1000],
                           "learning_rate": [0.01, 0.05, 0.1, 0.2],
                           "max_depth": [3, 2,None],
                           'max_features': ['auto', 'sqrt', 'log2'],
                           "min_samples_split": sp_randint(1, 11),
                           "min_samples_leaf": sp_randint(1, 11),
                           "criterion": ["mse", "friedman_mse"]}



hyperparams_dict['linreg'] = {"alpha" : 10.0 ** np.arange(-4, 1, 0.2),
                              "l1_ratio": np.arange(0. , 1. , 0.1),
                              "fit_intercept":[False]}

hyperparams_dict['NN']= {"hidden_layer_sizes":[(5, ),(10, ),(5,3 ),(10,3),(200, 10),(500, 100,20)],
                         "activation" : ['identity', 'logistic', 'tanh', 'relu'],
                         "alpha" : 10.0 ** -np.arange(1, 7),
                         "max_iter" :  [int(10**x) for x in np.arange(2, 4,0.5)]}

hyperparams_dict['randomforest']= {"n_estimators" : [10,50,100,500,700,1000],
                                   "max_depth": [3, 2, 4 ,None],
                                   'max_features': ['auto', 'sqrt', 'log2'],
                                   "min_samples_split": sp_randint(1, 11),
                                   "min_samples_leaf": sp_randint(1, 11),
                                   "bootstrap": [True, False],
                                   "criterion": ["mse"]}


models_vector = [ 'linreg', 'NN', 'GBT', 'randomforest']

for model_name in models_vector:

#====================

    hyperparams = hyperparams_dict[model_name]



    Y_pred , best = model_pred(X,Y,hyperparams , model_type=model_name , n_iter_search = 40, n_cv_sets = 10 )
    print model_name , ' mean house damage  error' , np.mean(abs(np.exp(Y_pred) - np.exp(Y))*df['Population 2010 census'])
    print model_name , ' median house damage  error' , np.median(abs(np.exp(Y_pred) - np.exp(Y))*df['Population 2010 census'])
#====================



    predict_on_tag = 'houses'
    Y_df_h = pd.DataFrame(np.exp(Y.values) )
    Y_df_h.columns = ['perc_'+predict_on_tag+'_true'] 
    Y_df_h['perc_'+predict_on_tag+'_pred'] = np.exp(Y_pred)
    Y_df_h['M_Code'] = df['M_Code']
    Y_df_h['Municipality'] = df['Municipality']
    Y_df_h['num_'+predict_on_tag+'_pred'] = Y_df_h['perc_'+predict_on_tag+'_pred']*df['Population 2010 census']
    Y_df_h['num_'+predict_on_tag+'_true'] = Y_df_h['perc_'+predict_on_tag+'_true']*df['Population 2010 census']
    Y_df_h['num_'+predict_on_tag+'_error'] = abs(Y_df_h['num_'+predict_on_tag+'_pred'] - Y_df_h['num_'+predict_on_tag+'_true'])
    Y_df_h['num_'+predict_on_tag+'_error_noabs'] = (Y_df_h['num_'+predict_on_tag+'_pred'] - Y_df_h['num_'+predict_on_tag+'_true'])

    Y_df_h['perc_'+predict_on_tag+'_error'] = abs(Y_df_h['perc_'+predict_on_tag+'_pred'] - Y_df_h['perc_'+predict_on_tag+'_true'])
    Y_df_h['rel_num_'+predict_on_tag+'_error'] = Y_df_h['num_'+predict_on_tag+'_error'] / Y_df_h['num_'+predict_on_tag+'_true'] 

    Y_df_h_sort = Y_df_h.sort_values('num_houses_true', axis=0)
    Y_df_h_sort = Y_df_h_sort.reset_index()

    Y_df_h_sort['num_houses_true'].plot()
    Y_df_h_sort['num_houses_pred'].plot()
    Y_df_h_sort['num_houses_error_noabs'].plot()
    plt.show()
     
    Y_df_h_sort['num_houses_error_noabs'].hist()
    plt.show()

    Y_df_h.to_csv("house_damaged_pred_" + model_name + ".csv")











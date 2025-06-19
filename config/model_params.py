from scipy.stats import randint, uniform
#from lightgbm import LGBMClassifier
#from sklearn.model_selection import RandomizedSearchCV

LIGHTGBM_PARAMS={
    'n_estimators':randint(100,500),
    'max_depth':randint(5,50),
    'learning_rate':uniform(0.01,0.2),
    'num_leaves':randint(20,100),
    'boosting_type':['gbdt' , 'dart' , 'goss']
}
#LIGHTGBM_PARAMS
## * Could go GridSearchCV but it takes a long time so will do RandomizedSearchCV
## randint will be needed to give parameters in ranges

RANDOM_SEARCH_PARAMS = {
    'n_iter':2,
    'cv':2,
    'n_jobs':-1,
    'verbose':2,
    'random_state':42,
    'scoring':'accuracy'
}


##from config.model_params import *
##LIGHTGBM_PARAMS
##RANDOM_SEARCH_PARAMS

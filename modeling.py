import re

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import train_test_split

from udf import launch_models_CV, loss_score

# Import Data

data_appli = pd.read_csv('data/data_full.csv').set_index('SK_ID_CURR')
data_appli = data_appli[['AMT_CREDIT']]

data_train = pd.read_csv('data/data_train_scaled_resampled_5.csv', index_col=[0])
data_train = data_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
# data_train = data_train.sample(frac=0.05, random_state=41)
labels_train = data_train['TARGET'].copy()
data_train.drop(columns=['TARGET'], inplace=True)



# CV validation
models_params = {
    # 'LogisticRegression': {
    #     'params': {
    #         'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    #         'class_weight': ["balanced", None],
    #         'solver': ['saga']
    #     },
    #     'estimator': LogisticRegression(),
    # },
    #
    # 'RandomForestClassifier': {
    #     'params': {
    #         'max_depth': [5, 10, 20],
    #         'max_features': ['auto', 'log2', 'sqrt'],
    #         'min_samples_leaf': [1, 2, 4],
    #         'min_samples_split': [2, 5, 10],
    #         'random_state': [41]
    #     },
    #     'estimator': RandomForestClassifier()
    # },
    'LGBMClassifier': {
        'params': {
            'num_leaves': [10, 20, 50],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 250, 500],
            'class_weight': ['balanced', None],
            'reg_alpha': [0.5, 1],
            'reg_lambda': [0.5, 1],
            'random_state': [41]
        },
        'estimator': LGBMClassifier()
    },
}

scorings_gridsearch = {'f1': make_scorer(f1_score,
                                         average='binary',
                                         pos_label=1),
                       'matthews': make_scorer(matthews_corrcoef),
                       "roc_auc": make_scorer(roc_auc_score),
                       "price_loss": make_scorer(loss_score,
                                                 data_appli_test=data_appli)}

# Train

best_scores = launch_models_CV(
    data_train,
    labels_train,
    models_params,
    default_mode=False,
    folds=3,
    scorings=scorings_gridsearch,
    favorite_scoring='f1',
    comment='only lgbm only train data resampled',
    n_jobs=3)



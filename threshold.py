


import mlflow
import re

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from udf import launch_models_CV, loss_score, determine_threshold

# Import Data

data_appli = pd.read_csv('data/data_full.csv').set_index('SK_ID_CURR')
data_appli = data_appli[['AMT_CREDIT']]

data_train = pd.read_csv('data/data_train_scaled_resampled_5.csv', index_col=[0])
data_train = data_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
# data_train = data_train.sample(frac=0.05, random_state=41)
labels_train = data_train['TARGET'].copy()
data_train.drop(columns=['TARGET'], inplace=True)

data_test = pd.read_csv("data/data_test_scaled.csv", index_col=[0])
labels_test = data_test['TARGET'].copy()
data_test.drop(columns=['TARGET'], inplace=True)


scorings_gridsearch = {'f1': make_scorer(f1_score,
                                         average='binary',
                                         pos_label=1),
                       'matthews': make_scorer(matthews_corrcoef),
                       "roc_auc": make_scorer(roc_auc_score),
                       "price_loss": make_scorer(loss_score,
                                                 data_appli_test=data_appli)}


print(labels_train.value_counts())

logged_model = 'runs:/a4a40c98e82c4913a6e66558e4665c94/best_model'

best_model_mlflow = mlflow.sklearn.load_model(logged_model)


labels_preds_df = pd.DataFrame(best_model_mlflow.predict_proba(data_test), index=data_test.index)
labels_preds_df['label_true'] = labels_test.values
labels_preds_df['pred_thresh_5'] = best_model_mlflow.predict(data_test)


confusion = confusion_matrix(labels_preds_df['label_true'], labels_preds_df['pred_thresh_5'])

baseline_score = loss_score(labels_preds_df['label_true'], labels_preds_df['pred_thresh_5'], data_appli_test=data_appli)
f1_baseline = f1_score(labels_preds_df['label_true'], labels_preds_df['pred_thresh_5'], pos_label=1.0)
matthews_baseline = matthews_corrcoef(labels_preds_df['label_true'], labels_preds_df['pred_thresh_5'])
roc_auc_baseline = roc_auc_score(labels_preds_df['label_true'], labels_preds_df['pred_thresh_5'])

losses, conf_matrix = determine_threshold(best_model_mlflow, data_test, labels_test, np.linspace(0,1,20,endpoint=True), data_appli)
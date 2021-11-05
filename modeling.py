import pandas as pd
import numpy as np
import mlflow
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Import Data
data = pd.read_csv('data/data_scale_filled_balanced.csv', index_col=[0])
data = data.sample(frac=0.2, random_state=41)
labels = data['TARGET'].copy()
data.drop(columns=['TARGET'], inplace=True)

# Split data
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.33, random_state=41)

# Train
# Baseline
dum_clf = DummyClassifier(strategy='uniform')
# TODO : RECUPERER FONCTION CV DU PROJET ENERGIE



# CV validation
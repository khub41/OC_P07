import time

import pandas as pd
import numpy as np
from udf import scale_data, scree_plot, reduce_dim_pca, tuning_kmeans, timer, train_dbscan, train_tsne, over_sample
from sklearn.impute import KNNImputer, SimpleImputer


def handle_infinite_values(data):
    """
    for the columns containing inf data we replace it by the highest value in the column that is not inf
    :param data:
    :return: data with no inf values
    """
    data_local = data.copy()
    maxes = data_local.max()
    id_inf_cols = maxes[maxes == np.inf].index
    for id_inf_col in id_inf_cols:
        inf_col = data_local[id_inf_col]
        max_except_inf = inf_col[(inf_col.notnull()) & (inf_col != np.inf)].max()
        try:
            data_local[id_inf_col].replace(np.inf, max_except_inf, inplace=True)
        except KeyError as e:
            pass

    return data_local

# Let's handle the missing values
def handle_missing_values(data, mode=None):
    if mode is None:
        mode = 'knn'
    if mode == 'knn':
        imputer = KNNImputer(n_neighbors=3)
        data_filled = pd.DataFrame(imputer.fit_transform(data),
                     index=data.index,
                     columns=data.columns)
    elif mode == 'simple':
        imputer = SimpleImputer(strategy='median')
        data_filled = pd.DataFrame(imputer.fit_transform(data),
                                   index=data.index,
                                   columns=data.columns)
    else :
        print("Mode is not conform, returning unfilled dataset")
        return data
    return data_filled


with timer('import data'):
    data_full = pd.read_csv("data/data_full.csv", index_col=[0])

# data_full = data_full.sample(1000)


with timer('getting good clients'):
    data_full = data_full[data_full.TARGET == 0]
    print(data_full.shape)



# Let's try to understand better the problem here:
# Only 8% of the training data set has a positive target
# ML models need to have more balanced data to be able to get scores
# There are two solutions:
# Create fake postive targets that look like the positive targets (make the dataset bigger)
# Deleting negative targets (interesting to save calculation time)


with timer('inf values'):
    data_full = handle_infinite_values(data_full)

with timer('missing values'):
    data_full = handle_missing_values(data_full, mode='simple')

with timer('scaling'):
    data_full_scale, scaler = scale_data(data_full.drop(columns=['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index'],
                                                        errors='ignore'))
with timer('computing pca opti'):
    scree_plot(data_full_scale, data_full_scale.shape[1], savefig='scree_plot')

with timer('reducing dim with pca'):
    data_full_scale, pca_fitted = reduce_dim_pca(data_full_scale, 500)

with timer('train KMEANS'):
    tuning_kmeans(data_full_scale.sample(int(50e3), random_state=41),
                  list(range(2, 10)),
                  'kmeans_balance',
                  3,
                  100,
                  run_name='tuning sample 50k 400comp')

with timer('train dbscan'):
    train_dbscan(data_full_scale, 'full rows 400comp')

with timer('tsne'):
    train_tsne(data_full_scale.sample(frac=0.5, random_state=41),
               pd.Series(["'Good' clients"] * len(data_full_scale),
                         name='label'),
               savefig='tsne_negatives_50pct_rs41')
#


import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from udf import scale_data, timer, re_sample, handle_infinite_values, handle_missing_values

with timer('import data'):
    data_full = pd.read_csv("data/data_full.csv", index_col=[0])


# data_full = data_full.sample(50000)

with timer('getting training set'):
    data_full = data_full[data_full.TARGET.isin([1, 0])]
    data_full = data_full.set_index('SK_ID_CURR')
    labels_full = data_full.TARGET
    data_full.drop(columns=['TARGET'], inplace=True)

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
    data_full_scale, scaler = scale_data(data_full.drop(columns=['SK_ID_BUREAU', 'SK_ID_PREV', 'index'],
                                                        errors='ignore'))


# with timer('computing pca opti'):
#     scree_plot(data_full_scale, data_full_scale.shape[1], savefig='scree_plot')

# with timer('reducing dim with pca'):
#     data_full_scale, pca_fitted = reduce_dim_pca(data_full_scale, 500)

with timer('split data'):
    data_train, data_test, labels_train, labels_test = train_test_split(data_full_scale, labels_full, test_size=0.30,
                                                                        random_state=41)

with timer('curve fit'):
    # plot_perf_balancing_strategy(data_train, labels_train, data_test, labels_test, savefig='undersampling_strategies')
    pass

with timer('Re sampling'):
    # Under sampling
    data_train_resampled, labels_train_resampled = re_sample(data_train,
                                                             labels_train,
                                                             RandomUnderSampler,
                                                             params={'sampling_strategy': 0.5,
                                                                     'random_state': 41})
data_train_resampled['TARGET'] = labels_train_resampled
data_train_resampled.to_csv('data/data_train_scaled_resampled_5.csv')

data_test['TARGET'] = labels_test
# data_test.to_csv("data/data_test_scaled.csv")


import copy
import json
import random
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.2f}s".format(title, time.time() - t0))


def scale_data(raw_features):
    scaler = StandardScaler()

    data_scale = pd.DataFrame(scaler.fit_transform(raw_features),
                              index=raw_features.index,
                              columns=raw_features.columns)

    return data_scale, scaler


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
    else:
        print("Mode is not conform, returning unfilled dataset")
        return data
    return data_filled


def scree_plot(data_scale, max_comp, savefig=False):
    pca_scree = PCA(n_components=max_comp)
    pca_scree.fit(data_scale)
    plt.style.use('fivethirtyeight')
    plt.xlabel('Nb de Composantes Principales')
    plt.ylabel('Pourcentage de variance expliquée (cumulée)')
    plt.title('Scree plot PCA')
    plt.plot(np.arange(1, pca_scree.n_components_ + 1), pca_scree.explained_variance_ratio_.cumsum() * 100,
             color='#8c704d')
    if savefig:
        plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)
    plt.show()


def reduce_dim_pca(data_scale, n_comp):
    pca = PCA(n_components=n_comp)
    data_scale_decomp = pd.DataFrame(pca.fit_transform(data_scale))
    return data_scale_decomp, pca


def tuning_kmeans(data, range_n_clust, experiment, n_init, max_iter, run_name='tuning'):
    slh_scores = {}
    db_scores = {}
    mlflow.set_experiment(experiment)
    for n_clust in range_n_clust:
        with mlflow.start_run(run_name=f"{run_name} ({n_clust} clusters)"):
            start = time.time()
            model_k_means = KMeans(n_clusters=n_clust, n_init=n_init, max_iter=max_iter)
            print(f"{model_k_means} start training")
            labels = model_k_means.fit_predict(data)
            slh = silhouette_score(data, labels)
            db = davies_bouldin_score(data, labels)
            elapsed = time.time() - start
            mlflow.log_param("n_clusters", n_clust)
            mlflow.log_param('max_iter', max_iter)
            mlflow.log_param('n_init', n_init)
            mlflow.log_metric('train time', elapsed)
            mlflow.log_metric("silh score", slh)
            mlflow.log_metric("DB score", db)
            print(f"{model_k_means} trained in {elapsed}s")
            print(f"Slh score : {slh}\nDB score : {db}")
            slh_scores[n_clust] = slh
            db_scores[n_clust] = db
            mlflow.end_run()


def train_dbscan(data, run_name):
    data_train = data.copy()
    mlflow.set_experiment('dbscan_descriptors')
    with mlflow.start_run(run_name=run_name):
        model = DBSCAN(n_jobs=-1)
        print(f"{model} start training and predicting")
        start = time.time()
        labels = model.fit_predict(data_train)
        elapsed = time.time() - start
        print(f"{model} trained in {elapsed}s")

        data_train = pd.DataFrame(data_train)
        data_train['predicted_label'] = labels

        slh = silhouette_score(data_train, labels)
        db = davies_bouldin_score(data_train, labels)
        mlflow.log_metric('train time', elapsed)
        mlflow.log_metric("silh score", slh)
        mlflow.log_metric("DB score", db)
        mlflow.log_metric('n_clusters', len(set(model.labels_)))
        print(f"Slh score : {slh}\nDB score : {db}")

        mlflow.end_run()
    return data_train


def train_tsne(data, labels, perplexity=50, learning_rate=200, show=True, savefig=False):
    tsne = TSNE(init='random', random_state=41, n_jobs=-1, perplexity=perplexity, learning_rate=learning_rate)
    data_tsne = tsne.fit_transform(data)
    df_data_tsne_labels = pd.DataFrame(data_tsne, columns=['x', 'y']).merge(labels, left_index=True, right_index=True)
    # colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    if len(labels.unique()) == 1:
        colors = ['b']
    else:
        number_of_colors = len(labels.unique())

        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                  for i in range(number_of_colors)]
    if show:
        plt.style.use('seaborn')
        plt.figure(figsize=(8, 8))
        plt.axis('equal')
        for label, color in zip(labels.unique(), colors):
            group = df_data_tsne_labels[df_data_tsne_labels[labels.name] == label]
            plt.scatter(group.x,
                        group.y,
                        label=label,
                        color=color,
                        s=6)
        plt.legend()
        plt.show = ()
        if savefig:
            plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)

    return data_tsne


def re_sample(data, labels, sampler, params=None):
    if params is None:
        params = {'random_state': 41}
    sampler = sampler(**params)
    data_res, labels_res = sampler.fit_resample(data, labels)
    # Getting the indexes
    index_samples = data.iloc[sampler.sample_indices_].index
    data_res.index = index_samples
    labels_res.index = index_samples
    return data_res, labels_res


def plot_perf_balancing_strategy(data_train,
                                 labels_train,
                                 data_test,
                                 labels_test,
                                 strategies=np.arange(0.3, 1.1, 0.1),
                                 savefig=False):
    strategies[-1] = 1
    f1_positives = []
    f1_negatives = []
    for strategy in strategies:
        with timer('Re sampling'):
            # Under sampling
            data_resampled, labels_resampled = re_sample(data_train,
                                                         labels_train,
                                                         RandomUnderSampler,
                                                         params={'sampling_strategy': strategy,
                                                                 'random_state': 41})
        with timer("training"):
            clf = RandomForestClassifier()
            clf.fit(data_resampled, labels_resampled)
        with timer('predicting'):
            labels_predict = clf.predict(data_test)
        with timer('scoring'):
            f1_positives.append(f1_score(labels_test, labels_predict))
            f1_negatives.append(f1_score(labels_test, labels_predict, pos_label=0))

    plt.style.use('seaborn')
    plt.title('F1 scores')
    plt.plot(strategies, f1_positives, label='positives', color='#8c704d', linewidth=3)
    plt.plot(strategies, f1_negatives, label='negatives', color='#376643', linewidth=3)
    plt.grid(axis='x')
    plt.xlabel('Sampling strategies (QttyPositives/QttyNegatives)')
    plt.legend()
    if savefig:
        plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)
    plt.show()


# def under_sample(data, labels, random_state=41):
#     sampler = ClusterCentroids(random_state=random_state)
#     data_res, labels_res = sampler.fit_resample(data, labels)
#     return data_res, labels_res
# # MODELING FUNCTIONS

def launch_models_CV(
        data_train,
        labels,
        models_params,
        default_mode=False,
        folds=3,
        scorings='neg_root_mean_squared_error',
        favorite_scoring=None,
        comment=None,
        n_jobs=-1):
    # Iterate on model types

    best_estimator_favourite_scores = {}
    for model_type in models_params.keys():

        mlflow.set_experiment(model_type)

        with mlflow.start_run(run_name=comment):

            # Prepare training
            estimator = models_params[model_type]['estimator']

            if not default_mode:

                # Get parameters grid

                params = models_params[model_type]['params']
                gscv = GridSearchCV(
                    estimator=estimator,
                    param_grid=params,
                    scoring=scorings,
                    n_jobs=n_jobs,
                    cv=folds,
                    refit=favorite_scoring
                )

                # Saving gridsearch params in MlFlow Artifact
                # Deleting estimators object from the artifact (json incompatible)
                models_params_artifact = copy.deepcopy(models_params)
                for model_type_artifact in models_params_artifact.keys():
                    del models_params_artifact[model_type_artifact]['estimator']
                with open('models_params.json', 'w') as fp:
                    json.dump(models_params_artifact, fp)

                mlflow.log_artifact('models_params.json')

                # Training
                print('Start training of {} with GridSearch and CV'.format(model_type))
                start = time.time()

                gscv.fit(
                    data_train,
                    labels)

                elapsed = time.time() - start

                best_estimator = gscv.best_estimator_
                results = pd.DataFrame(gscv.cv_results_)

                mlflow.sklearn.log_model(best_estimator, "best_model")

                # Get results
                results_best_estimator = results[results['rank_test_{}'.format(favorite_scoring)] == 1]
                for scoring in scorings:
                    mlflow.log_metric(scoring.strip('neg_'),
                                      abs(results_best_estimator['mean_test_{}'.format(scoring)].values[0]))
                    print(scoring.strip('neg_') + ' : ',
                          abs(results_best_estimator['mean_test_{}'.format(scoring)].values[0]))
                mlflow.log_metric('training_time', elapsed)
                best_estimator_favourite_scores[model_type] = abs(
                    results_best_estimator[f'mean_test_{favorite_scoring}'].values[0])

                for param, value in gscv.best_params_.items():
                    mlflow.log_param(param, value)
                mlflow.log_param("folds", folds)

                print('Successful training and logging of {}, in {} seconds\n\n'.format(model_type, round(elapsed)))

            # Default mode
            else:
                print('Start training of {} (default mode)'.format(model_type))
                start = time.time()
                results = cross_validate(
                    estimator,
                    data_train,
                    labels,
                    scoring=scorings,
                    cv=folds,
                    n_jobs=n_jobs,
                    return_estimator=True
                )
                elapsed = time.time() - start

                for scoring in scorings:
                    print(scoring.strip('neg_'), abs(results['test_{}'.format(scoring)].mean()))
                    mlflow.log_metric(scoring.strip('neg_'), abs(results['test_{}'.format(scoring)].mean()))
                mlflow.log_metric('training_time', elapsed)

                for param, value in results['estimator'][0].get_params().items():
                    if param in models_params[model_type]['params'].keys():
                        mlflow.log_param(param, value)
                mlflow.log_param("folds", folds)
                print('Successful training and logging of {}, in {} seconds (Default Mode)\n\n'.format(model_type,
                                                                                                       round(elapsed)))

            mlflow.end_run()
    return best_estimator_favourite_scores


def qualify_decision(row):
    if row.label_true == 0:
        if row.label_pred == 0:
            # Bon Payeur Accepté
            return 'BPA'
        else:
            # Bon Payeur Ecarté
            return 'BPE'
    else:
        if row.label_pred == 0:
            # Mauvais Payeur Accepté
            return 'MPA'
        else:
            # Mauvais Payeur Ecarté
            return 'MPE'


def compute_loss(row, data_appli):
    """
    This loss function is a prototype that we'll use for educational purposes

    - Accepting a good client brings a negative loss of the amount of the interests
    - Not accepting a good client costs the interests
    - We assume that a default costs half the price of the credit
    - Say no to a bad client brings a null loss

    We'll assume the interests are 1.5%

    This approach is naive, this function needs to be tuned with an expert in Home credit and banking administration


    :param row: pandas.Series
    :param data_appli: pandas.DataFrame. WARNING: This df needs to be filled with the raw data and not the scaled data
    :return: float
    """
    amount_credit = data_appli.loc[row.name].AMT_CREDIT

    if row.decision_quali == 'BPA':
        return - 0.15 * amount_credit
    elif row.decision_quali == 'BPE':
        return 0
    elif row.decision_quali == 'MPA':
        return 0.8 * amount_credit
    elif row.decision_quali == 'MPE':
        return 0
    else:
        raise f'{row.decision_quali} is not an accepted format'


def loss_score(labels_true, labels_pred, data_appli_test=None):
    labels_df = pd.DataFrame(data={'label_true': labels_true, 'label_pred': labels_pred}, index=labels_true.index)

    labels_df['decision_quali'] = labels_df.apply(qualify_decision, axis=1)

    labels_df['loss'] = labels_df.apply(compute_loss, data_appli=data_appli_test, axis=1)

    return labels_df.loss.sum()

def determine_threshold(model, data_test, labels_test, range_thresh, data_appli):

    labels_preds_df = pd.DataFrame(model.predict_proba(data_test), index=data_test.index)
    labels_preds_df['label_true'] = labels_test.values
    losses = {}
    confusion_matrices = {}
    for threshold in range_thresh:

        labels_preds_df[f'pred_thresh_{threshold}'] = labels_preds_df[1] > threshold
        labels_preds_df[f'pred_thresh_{threshold}'].replace(True, 1.0)
        labels_preds_df[f'pred_thresh_{threshold}'].replace(False, 0.0)
        losses[threshold] = loss_score(labels_preds_df['label_true'],
                                       labels_preds_df[f'pred_thresh_{threshold}'],
                                       data_appli_test=data_appli)
        conf_mat = confusion_matrix(labels_preds_df['label_true'],
                                       labels_preds_df[f'pred_thresh_{threshold}'], normalize='true')
        confusion_matrices[threshold] = conf_mat
    return pd.Series(losses), confusion_matrices

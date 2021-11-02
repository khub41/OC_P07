import time
import random

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from imblearn.over_sampling import SMOTE
import pandas as pd

from contextlib import contextmanager

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
        plt.show=()
        if savefig:
            plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)

    return data_tsne

def over_sample(data, labels, random_state=41):
    smote_sampler = SMOTE(random_state=random_state)
    data_res, labels_res = smote_sampler.fit_resample(data, labels)
    return data_res, labels_res




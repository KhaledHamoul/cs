import io
import base64
from app.templatetags.template_filters import get_item
from api.optimumClustersNumber import elbow, silhouette, gapStatistic
from api.clustering import kmeans, hierarchical, spectral, mini_batch_kmeans, birch, mean_shift, dbscan, optics
import pandas as pd
import numpy as np
import requests
import json
import operator
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from matplotlib.collections import LineCollection
from scipy.cluster.hierarchy import dendrogram
from pandas.plotting import parallel_coordinates
import seaborn as sns
from sklearn.manifold import TSNE


palette = sns.color_palette("bright", 10)

# get image element from matplotlib plt


def getPltImage(plt):
    f = io.BytesIO()
    plt.savefig(f, format="png", facecolor=(0.95, 0.95, 0.95))
    encoded_img = base64.b64encode(
        f.getvalue()).decode('utf-8').replace('\n', '')
    f.close()
    return '<img id="result-visual" src="data:image/png;base64,%s" />' % encoded_img

# build dataset matrix from dataset records


def buildDatasetMatrix(dataset):
    datasetMatrix = []
    for record in dataset.records.all():
        temp = []
        for attribute in dataset.attributes.all():
            temp.append(ord(get_item(record.data, attribute.name)[0]) if type(get_item(
                record.data, attribute.name)) == str else get_item(record.data, attribute.name))

        datasetMatrix.append(temp)

    return datasetMatrix


def buildDataFrame(dataset):
    dataDict = {}
    index = 0
    for record in dataset.records.all():
        temp = {}
        for attribute in dataset.attributes.all():
            temp[attribute.name] = ord(get_item(record.data, attribute.name)[0] if type(get_item(
                record.data, attribute.name)) == str else get_item(record.data, attribute.name))

        dataDict[index] = temp
        index += 1

    datasetDf = pd.DataFrame.from_dict(dataDict, orient='index')

    return datasetDf

# methods switcher


def switcher(method, args):
    # Optimum clusters number algorithms
    if method == 'elbow':
        return elbow(args)

    if method == 'silhouette':
        return silhouette(args)

    if method == 'gap_statistic':
        return gapStatistic(args)

    # clustering algorithms
    if method == 'kmeans':
        return kmeans(args)

    if method == 'hierarchical':
        return hierarchical(args)

    if method == 'spectral':
        return spectral(args)

    if method == 'mini_batch_kmeans':
        return mini_batch_kmeans(args)

    if method == 'birch':
        return birch(args)

    if method == 'mean_shift':
        return mean_shift(args)

    if method == 'dbscan':
        return dbscan(args)

    if method == 'optics':
        return optics(args)

    else:
        return "Invalid method"


def indexes(datasetDf, labels, num_clusters):
    dbs = davies_bouldin_score(datasetDf, labels)
    cs = calinski_harabasz_score(datasetDf, labels)
    ss = silhouetteScore(datasetDf, labels, num_clusters)

    clusters = []
    tempDf = np.append(datasetDf, labels.reshape(-1, 1), axis=1)
    for i in range(0, num_clusters):
        l = tempDf[tempDf[:, datasetDf.shape[1]] == i]
        l = np.delete(l, datasetDf.shape[1], axis=1)
        clusters.append(l)

    di = dunn(clusters)

    dbs = float("{:.4f}".format(dbs))
    cs = float("{:.4f}".format(cs))
    ss = float("{:.4f}".format(ss))
    di = float("{:.4f}".format(di))

    return {'dbs': dbs, 'cs': cs, 'ss': ss, 'di': di}

# Dunn index for clustering
# https://gist.github.com/douglasrizzo/cd7e792ff3a2dcaf27f6


def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000

    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])

    return np.min(values)


def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])

    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])

    return np.max(values)


def dunn(k_list):
    """ Dunn index [CVI]

    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))

    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])

        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di


def plotPca3d(datasetDf, labels=None, sample_points=None):
    # plt.style.use('seaborn-whitegrid')

    if labels is None:
        labels = [1 for i in range(datasetDf.shape[0])]

    pca = PCA(n_components=3, svd_solver='full')
    pcaTempDataset = pca.fit_transform(datasetDf)

    pcaTempDataset = np.append(pcaTempDataset, labels.reshape(-1, 1), axis=1)

    if sample_points is None:
        reducedDataset = pcaTempDataset
    else:
        reducedDataset = pcaTempDataset[np.random.choice(
            pcaTempDataset.shape[0], sample_points, replace=False)]

    fig = px.scatter_3d(
        reducedDataset, x=0, y=1, z=2, color=reducedDataset[:, 3],
        title=f' ',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.write_html('core/templates/3d_pca.html', auto_open=False)

    fig = plt.figure(1, figsize=(10, 6))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()

    ax.scatter(reducedDataset[:, 0], reducedDataset[:, 1], reducedDataset[:, 2],
               c=reducedDataset[:, 3], cmap=plt.cm.nipy_spectral, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])

    return plt


def tsne(datasetDf, labels, sample_points=None):
    perplexity = 10
    tsne = TSNE(n_components=3, perplexity=perplexity)
    tsneTempDf = pd.DataFrame(tsne.fit_transform(datasetDf))

    tsneTempDf = np.append(tsneTempDf, labels.reshape(-1, 1), axis=1)

    if sample_points is None:
        reducedDataset = tsneTempDf
    else:
        reducedDataset = tsneTempDf[np.random.choice(
            tsneTempDf.shape[0], sample_points, replace=False)]

    fig = px.scatter_3d(
        reducedDataset, x=0, y=1, z=2, color=reducedDataset[:, 3],
        title=f'Total Explained Variance (T-SNE): ',
        labels={'0': 'PC 1', '1': 'PC 2', '2': 'PC 3'}
    )
    fig.write_html('core/templates/3d_tsne.html', auto_open=False)


def parallelCoordinates(datasetDf, labels, num_clusters):
    X_clustered = pd.DataFrame(datasetDf)
    X_clustered["cluster"] = labels

    df = X_clustered

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.cluster == i])

    # Create the plot
    fig = plt.figure(figsize=(24, 30))
    title = fig.suptitle(
        "Parallel Coordinates Plot for the Clusters", fontsize=18)
    fig.subplots_adjust(top=0.95, wspace=0)

    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters
    for i in range(num_clusters):
        plt.subplot(num_clusters, 1, i + 1)
        for j, c in enumerate(cluster_points):
            if i != j:
                pc = parallel_coordinates(c, 'cluster', color=[(
                    palette[j][0], palette[j][1], palette[j][2], 0.2)])

        pc = parallel_coordinates(cluster_points[i], 'cluster',
                                  color=[(palette[i][0], palette[i][1], palette[i][2], 0.5)])

        # Stagger the axes
        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks()[1::2]:
            tick.set_pad(20)

    return plt


def displayParallelCoordinatesCentroids(df, num_clusters):
    # Create the plot
    fig = plt.figure(figsize=(24, 10))
    title = fig.suptitle(
        "Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'cluster', color=palette)

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

    return plt


def silhouetteScore(datasetDf, labels, n_clusters):
    from sklearn.metrics import silhouette_samples, silhouette_score

    silhouette_avg = silhouette_score(datasetDf, labels)
    # sample_silhouette_values = silhouette_samples(datasetDf, labels)

    return silhouette_avg

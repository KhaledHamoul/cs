import hashlib
import time
import json
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans, Birch, MeanShift, DBSCAN, OPTICS
from sklearn import metrics, preprocessing
import scipy
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.feature_extraction.image import grid_to_graph
import matplotlib
import seaborn as sns
from math import ceil
from sklearn.neighbors import NearestCentroid
from distutils.version import LooseVersion
from scipy.ndimage.filters import gaussian_filter
from scipy import sparse
import scipy.cluster.hierarchy as sch
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, inconsistent
import skimage
from skimage.transform import rescale
from django.conf import settings as djangoSettings
import os
import zipfile

matplotlib.use('Agg')

# Methods


def kmeans(args):
    print('==== K-Means ====')

    clustersNumber = int(args.get('clustersNumber'))
    datasetDf = args.get('datasetDf')
    samplingPoints = args.get('samplingPoints')

    model = KMeans(n_clusters=clustersNumber)
    algoResult = model.fit(datasetDf)

    labels = algoResult.labels_

    result = prepareResult(datasetDf=datasetDf, algoResult=algoResult, model=model,
                           labels=labels, clustersNumber=clustersNumber, samplingPoints=samplingPoints)

    exportCsv(datasetDf, labels, clustersNumber)

    return result


def exportCsv(datasetDf, labels, clustersNumber):

    datasetDf['clusters'] = labels
    for i in range(0, clustersNumber - 1):
        datasetDf[datasetDf['clusters'] == i].to_csv(djangoSettings.STATIC_ROOT + '/clusters/cluster' + str(i+1) + '.csv', index=False)
        

def hierarchical(args):
    print('==== Hierarchical ====')

    clustersNumber = int(args.get('clustersNumber'))
    datasetDf = args.get('datasetDf')
    samplingPoints = args.get('samplingPoints')

    model = AgglomerativeClustering(
        n_clusters=clustersNumber, affinity='euclidean')
    algoResult = model.fit(datasetDf)

    labels = algoResult.labels_

    result = prepareResult(datasetDf=datasetDf, algoResult=algoResult, model=model,
                           labels=labels, clustersNumber=clustersNumber, samplingPoints=samplingPoints)

    return result


def spectral(args):
    print('==== spectral ====')

    clustersNumber = int(args.get('clustersNumber'))
    datasetDf = args.get('datasetDf')
    samplingPoints = args.get('samplingPoints')

    # D = pairwise_distances(datasetDf)  # Distance matrix
    # S = np.max(D) - D  # Similarity matrix
    # S = sparse.coo_matrix(S)

    model = SpectralClustering(
        n_clusters=clustersNumber, affinity='precomputed')

    model = KMeans(n_clusters=clustersNumber)
    algoResult = model.fit(datasetDf)

    labels = algoResult.labels_

    result = prepareResult(datasetDf=datasetDf, algoResult=algoResult, model=model,
                           labels=labels, clustersNumber=clustersNumber, samplingPoints=samplingPoints)

    return result


def mini_batch_kmeans(args):
    print('==== MiniBatch K-Means ====')

    clustersNumber = int(args.get('clustersNumber'))
    datasetDf = args.get('datasetDf')
    samplingPoints = args.get('samplingPoints')

    model = MiniBatchKMeans(n_clusters=clustersNumber)
    algoResult = model.fit(datasetDf)

    labels = algoResult.labels_

    result = prepareResult(datasetDf=datasetDf, algoResult=algoResult, model=model,
                           labels=labels, clustersNumber=clustersNumber, samplingPoints=samplingPoints)

    return result


def birch(args):
    print('==== MiniBatch K-Means ====')

    clustersNumber = int(args.get('clustersNumber'))
    datasetDf = args.get('datasetDf')
    samplingPoints = args.get('samplingPoints')

    model = Birch(n_clusters=clustersNumber)
    algoResult = model.fit(datasetDf)

    labels = algoResult.labels_

    result = prepareResult(datasetDf=datasetDf, algoResult=algoResult, model=model,
                           labels=labels, clustersNumber=clustersNumber, samplingPoints=samplingPoints)

    return result


def mean_shift(args):
    print('==== Mean Shift ====')

    clustersNumber = int(args.get('clustersNumber'))
    datasetDf = args.get('datasetDf')
    samplingPoints = args.get('samplingPoints')

    model = MeanShift()
    algoResult = model.fit(datasetDf)

    labels = algoResult.labels_

    result = prepareResult(datasetDf=datasetDf, algoResult=algoResult, model=model,
                           labels=labels, clustersNumber=clustersNumber, samplingPoints=samplingPoints)

    return result


def dbscan(args):
    print('==== DBSCAN ====')

    clustersNumber = int(args.get('clustersNumber'))
    datasetDf = args.get('datasetDf')
    samplingPoints = args.get('samplingPoints')

    model = DBSCAN()
    algoResult = model.fit(datasetDf)

    labels = algoResult.labels_

    result = prepareResult(datasetDf=datasetDf, algoResult=algoResult, model=model,
                           labels=labels, clustersNumber=clustersNumber, samplingPoints=samplingPoints)

    return result


def optics(args):
    print('==== OPTICS ====')

    clustersNumber = int(args.get('clustersNumber'))
    datasetDf = args.get('datasetDf')
    samplingPoints = args.get('samplingPoints')

    model = OPTICS()
    algoResult = model.fit(datasetDf)

    labels = algoResult.labels_

    result = prepareResult(datasetDf=datasetDf, algoResult=algoResult, model=model,
                           labels=labels, clustersNumber=clustersNumber, samplingPoints=samplingPoints)

    return result


# private
def variablesSprendingPlot(model, datasetDf, plotingColumns):
    labels = pd.DataFrame(model.labels_)
    labeledDataset = pd.concat((datasetDf, labels), axis=1)
    labeledDataset = labeledDataset.rename({0: 'labels'}, axis=1)

    labeledDataset['Constant'] = "Data"

    featuresNumber = len(list(labeledDataset))
    plotColumns = plotingColumns
    plotRows = ceil((featuresNumber - 2) / plotColumns)

    f, axes = plt.subplots(plotRows, plotColumns, sharex=False)
    f.set_size_inches(18, 9 * plotRows)
    f.subplots_adjust(hspace=0.2, wspace=0.7)

    for i in range(0, plotRows):
        for j in range(0, plotColumns):
            featuresNumber -= 1
            if featuresNumber > 1:
                col = labeledDataset.columns[j + (i * plotColumns)]

                try:
                    ax = sns.swarmplot(
                        x=labeledDataset['Constant'], y=labeledDataset[col].values, hue=labeledDataset['labels'], ax=axes[i, j])
                    ax.set_title(col)
                except Exception:
                    ax = sns.swarmplot(
                        x=labeledDataset['Constant'], y=labeledDataset[col].values, hue=labeledDataset['labels'], ax=axes[j])
                    ax.set_title(col)

    # avoid circular imports (error)
    from api.helpers import getPltImage

    visual = '<div class="col-sm-12">' + getPltImage(plt) + '</div>'
    visual = '<div class="row">' + visual + '</div>'

    plt.close("all")

    return visual, labeledDataset


def prepareResult(datasetDf, algoResult, model, labels, clustersNumber, samplingPoints):
    from api.helpers import plotPca3d, parallelCoordinates, displayParallelCoordinatesCentroids, tsne, getPltImage, indexes
    samplingPoints = int(samplingPoints) if samplingPoints != '' else None

    # 3d PCA plot
    plt1 = plotPca3d(datasetDf=datasetDf, labels=labels,
                     sample_points=samplingPoints)
    visual = '<div class="col-sm-12">' + getPltImage(plt1) + '</div>'
    pca3d = '<div class="row">' + visual + '</div>'

    # Parallel coordinates plot
    plt2 = parallelCoordinates(
        datasetDf=datasetDf, labels=labels, num_clusters=clustersNumber)
    visual = '<div class="col-sm-12">' + getPltImage(plt2) + '</div>'
    parallelCoord = '<div class="row">' + visual + '</div>'

    # parallel cooridnates centoirds plot
    try:
        centroids = pd.DataFrame(algoResult.cluster_centers_)
        pass
    except Exception as e:
        clf = NearestCentroid()
        clf.fit(datasetDf, labels)
        centroids = pd.DataFrame(clf.centroids_)
        print(e)
        pass

    centroids['cluster'] = centroids.index
    plt3 = displayParallelCoordinatesCentroids(
        centroids, num_clusters=clustersNumber)
    visual = '<div class="col-sm-12">' + getPltImage(plt3) + '</div>'
    parallelCentroids = '<div class="row">' + visual + '</div>'

    # t-SNE plot
    tsne(datasetDf=datasetDf, labels=labels, sample_points=samplingPoints)

    plt.close("all")

    labels = pd.DataFrame(model.labels_)
    labeledDataset = pd.concat((datasetDf, labels), axis=1)
    labeledDataset = labeledDataset.rename({0: 'labels'}, axis=1)

    labeledDataset['Constant'] = "Data"

    indexes = indexes(datasetDf=datasetDf,
                      labels=algoResult.labels_, num_clusters=clustersNumber)

    return {
        'visuals': {
            'pca3d': pca3d,
            'parallelCoord': parallelCoord,
            'parallelCentroids': parallelCentroids
        },
        # 'data': {
        #     'labeledDataset': labeledDataset.drop(['Constant'], axis=1).to_json(),
        #     'centers': centers
        # },
        'indexes': indexes
    }


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        # plt.title('Hierarchical Clustering Dendrogram (truncated)')
        # plt.xlabel('sample index or (cluster size)')
        # plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -1),
                             textcoords='offset points',
                             fontsize=50,
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, linewidth=7, color='#f00')
    return ddata

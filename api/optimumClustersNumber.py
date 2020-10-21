import hashlib
import time
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
import scipy
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib
matplotlib.use('Agg')

# Methods


def elbow(args):
    maxIterationNumber = args.get('maxIterationsNumeber')
    datasetMatrix = args.get('datasetMatrix')

    distortions = []
    inertias = []
    distortionsResult = {}
    inertiaResult = {}
    K = range(1, int(maxIterationNumber) + 1)

    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(datasetMatrix)

        ssw = sum(np.min(cdist(datasetMatrix, kmeanModel.cluster_centers_,
                               'euclidean'), axis=1)) / len(datasetMatrix)
        distortions.append(ssw)
        inertias.append(kmeanModel.inertia_)

        distortionsResult[k] = sum(np.min(cdist(
            datasetMatrix, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / len(datasetMatrix)
        inertiaResult[k] = kmeanModel.inertia_

    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using the Distortions')

    # avoid circular imports (error)
    from api.helpers import getPltImage

    visual = '<div class="col-sm-6">' + getPltImage(plt) + '</div>'

    plt.close()

    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using the Inertia')

    visual += '<div class="col-sm-6">' + getPltImage(plt) + '</div>'
    visual = '<div class="row">' + visual + '</div>'

    plt.close("all")

    return {'visual': visual, 'data': {'distortions': distortionsResult, 'inertia': inertiaResult}}


def silhouette(args):
    maxIterationNumber = args.get('maxIterationsNumeber')
    datasetMatrix = args.get('datasetMatrix')

    silhouetteAvgs = {}
    avgs = []

    range_n_clusters = range(2, int(maxIterationNumber) + 1)

    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax3) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    for n_clusters in range_n_clusters:
        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(datasetMatrix) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters)
        cluster_labels = clusterer.fit_predict(datasetMatrix)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(datasetMatrix, cluster_labels)

        silhouetteAvgs[n_clusters] = silhouette_avg
        avgs.append(silhouette_avg)

        if int(maxIterationNumber) == n_clusters:
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(
                datasetMatrix, cluster_labels)

            y_lower = 10
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # 2nd Plot showing the actual clusters formed
            # colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
            # ax2.scatter(np.array(datasetDf)[:, 0], np.array(datasetDf)[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            #             c=colors, edgecolor='k')

            # # Labeling the clusters
            # centers = clusterer.cluster_centers_
            # # Draw white circles at cluster centers
            # ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            #             c="white", alpha=1, s=200, edgecolor='k')

            # for i, c in enumerate(centers):
            #     ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
            #                 s=50, edgecolor='k')

            # ax2.set_title("The visualization of the cludunnstered data.")
            # ax2.set_xlabel("Feature space dunnfor the 1st feature")
            # ax2.set_ylabel("Feature space for the 2nd feature")

            # plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
            #               "with n_clusters = %d" % n_clusters),
            #              fontsize=14, fontweight='bold')

    # avoid circular imports (error)
    from api.helpers import getPltImage

    ax3.set_title("The Silhouette Method avgs")
    ax3.set_xlabel("Cluster label")
    ax3.set_ylabel("The silhouette coefficient values")
    ax3.plot(range_n_clusters, avgs, 'bx-')

    ax3.set_axisbelow(True)
    ax3.minorticks_on()
    ax3.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax3.grid(which='minor', linestyle=':', linewidth='0.2', color='black')

    visual = '<div class="col-sm-12">' + getPltImage(plt) + '</div>'

    plt.close("all")

    return {'visual': visual, 'data': {'silhouette_avgs': silhouetteAvgs}}


def gapStatistic(args):
    maxIterationNumber = args.get('maxIterationsNumeber')
    datasetMatrix = args.get('datasetMatrix')
    nrefs = 5

    gaps = np.zeros((len(range(1, int(maxIterationNumber))),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, int(maxIterationNumber))):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            km = KMeans(k)
            km.fit(datasetMatrix)

            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(datasetMatrix)

        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append(
            {'clusterCount': k, 'gap': gap}, ignore_index=True)

    # # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    # return (gaps.argmax() + 1, resultsdf)

    plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')  
    plt.plot(resultsdf.clusterCount, resultsdf.gap, linewidth=3)
    plt.scatter(resultsdf[resultsdf.clusterCount == (gaps.argmax() + 1)].clusterCount,
                resultsdf[resultsdf.clusterCount == (gaps.argmax() + 1)].gap, s=250, c='r')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Gap statisitc score')
    plt.title('Gap statistic scores by Clusters number')

    # avoid circular imports (error)
    from api.helpers import getPltImage

    visual = '<div class="col-sm-12">' + getPltImage(plt) + '</div>'

    plt.close("all")

    return {'visual': visual, 'data': {'optimal': int(gaps.argmax() + 1)}}

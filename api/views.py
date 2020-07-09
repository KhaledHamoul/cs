from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import csv
from app.models import Dataset, Record, Attribute
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from api.helpers import getPltImage
from api.optimumClustersNumber import switcher

# upload & validate dataset


@csrf_exempt
def upload_dataset(request):
    csvfile = request.FILES['dataset']
    decoded_file = csvfile.read().decode('utf-8').splitlines()
    reader = csv.reader(decoded_file, delimiter=',')

    attributes = {}
    attributes = next(reader)

    reader = csv.DictReader(decoded_file, delimiter=',')
    records = []
    for row in reader:
        records.append(row)

    # create the dataset
    dataset = Dataset(title='Dataset_name')
    dataset.save()

    # save the attributes
    for attributeName in attributes:
        attributeInstance = Attribute(name=attributeName, label=attributeName)
        attributeInstance.save()
        dataset.attributes.add(attributeInstance)

    # save the records
    for data in records:
        record = Record(data=data)
        record.save()
        dataset.records.add(record)

    return JsonResponse({'attributes': attributes, 'records_count': len(records)})

# optimum_clusters_number


@csrf_exempt
def optimum_clusters_number(request):
    datasetId = request.POST.get('datasetId')
    dataset = Dataset.objects.get(id=datasetId)
    
    method = request.POST.get('method')
    maxIterationsNumeber = request.POST.get('maxIterationsNumeber')

    result = switcher(method=method, args={'maxIterationsNumeber': maxIterationsNumeber})

    return JsonResponse({'plot': getPltImage(plt), 'result': result})
    # return JsonResponse({'status': dataset.toJSON()})


def calculate_WSS(points, kmax):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + \
                (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse

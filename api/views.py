from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import csv
import json
from app.models import Dataset, Record, Attribute
from api.models import Result
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from api.helpers import switcher, buildDatasetMatrix, buildDataFrame
from api.pre_processing import pre_processing
import pandas as pd
from sklearn.decomposition import PCA

# upload & validate dataset


@csrf_exempt
def upload_dataset(request):
    csvfile = request.FILES['dataset']
    datasetName = request.POST.get('dataset_name')
    datasetDescription = request.POST.get('dataset_description') if request.POST.get('dataset_description') != None else ''
    allowMissinValues = request.POST.get('Allow_missing_values')

    print(allowMissinValues)
    # link : https://www.kaggle.com/smohubal/market-customer-segmentation-clustering/notebook
    try:
        if allowMissinValues == 'false':
            tmpCsvfile = pd.read_csv(csvfile)
            datasetInstance = pre_processing(tmpCsvfile)
            visual, data = datasetInstance.missing_percent()
            return JsonResponse({'data': data, 'visual': visual}, status=400)
        else:
            csvfile.seek(0)
    except Exception:
        csvfile.seek(0)

    decoded_file = csvfile.read().decode('utf-8').splitlines()
    reader = csv.reader(decoded_file, delimiter=',')

    attributes = {}
    attributes = next(reader)

    reader = csv.DictReader(decoded_file, delimiter=',')
    records = []
    for row in reader:
        records.append(row)

    # create the dataset
    dataset = Dataset(title=datasetName, description=datasetDescription)
    dataset.save()

    # save the attributes
    for attributeName in attributes:
        try:
            attributeInstance = Attribute.objects.get(name=attributeName)
        except Attribute.DoesNotExist:
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
    method = request.POST.get('method')
    maxIterationsNumeber = request.POST.get('maxIterationsNumeber')
    pcaComponents = request.POST.get('pcaComponents')

    dataset = Dataset.objects.get(id=datasetId)
    datasetMatrix = buildDatasetMatrix(dataset=dataset)

    if pcaComponents != "" and pcaComponents != 0:
        pca = PCA(n_components=float(pcaComponents))
        datasetMatrix = pca.fit_transform(datasetMatrix)

    result = switcher(method=method, args={
                      'datasetMatrix': datasetMatrix, 'maxIterationsNumeber': maxIterationsNumeber})

    return JsonResponse(result)

# optimum_clusters_number


@csrf_exempt
def clustering(request):
    datasetId = request.POST.get('datasetId')
    method = request.POST.get('method')
    clustersNumber = request.POST.get('clustersNumber')
    plotingColumns = request.POST.get('plotingColumns')
    linkageMethod = request.POST.get('linkageMethod')

    dataset = Dataset.objects.get(id=datasetId)
    datasetMatrix = buildDatasetMatrix(dataset=dataset)
    datasetDf = buildDataFrame(dataset=dataset)

    args = {
        'datasetMatrix': datasetMatrix,
        'datasetDf': datasetDf,
        'clustersNumber': clustersNumber,
        'plotingColumns': plotingColumns,
        'linkageMethod': linkageMethod
    }

    result = switcher(method=method, args=args)

    # used for saving later on
    result['method'] = method
    result['datasetId'] = datasetId

    return JsonResponse(result)


@csrf_exempt
def save_result(request):
    datasetId = int(request.POST.get('datasetId'))
    method = request.POST.get('method')
    visual = request.POST.get('visual')
    data = request.POST.get('data[labeledDataset]')

    try:
        result = Result(data=data, visual=visual, dataset_id=datasetId, method=method)
        result.save()
        return JsonResponse({'message': 'Saved successfuly'})
    except Exception as e:
        return JsonResponse({'message': str(e)}, status=400)


from django.contrib.auth.decorators import login_required
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
from django.template.response import TemplateResponse
from pprint import pprint
from logging import warning
from app.models import Dataset
from django.db.models import Count
import json


@login_required(login_url="/login/")
def index(request):
    return render(request, "index.html")


@login_required(login_url="/login/")
def pages(request):
    context = {}
    try:
        load_template = request.path[1:]
        html_template = loader.get_template(load_template + '.html')
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        try:
            load_template = request.path.split('/')[-1]
            html_template = loader.get_template(load_template)
            return HttpResponse(html_template.render(context, request))

        except template.TemplateDoesNotExist:
            html_template = loader.get_template('errors/error-404.html')
            return HttpResponse(html_template.render(context, request))

        except:

            html_template = loader.get_template('errors/error-500.html')
            return HttpResponse(html_template.render(context, request))

    except:

        html_template = loader.get_template('errors/error-500.html')
        return HttpResponse(html_template.render(context, request))

# index


@login_required(login_url="/login/")
def data_index(request):
    datasets = Dataset.objects.annotate(
        records_count=Count("records")).filter(deleted=False)
    return render(request, "data/index.html", {'datasets': datasets})

# view


@login_required(login_url="/login/")
def data_view(request, id):
    try:
        dataset = Dataset.objects.get(id=id)
        return render(request, "data/view.html", {'dataset': dataset})
    except Dataset.DoesNotExist:
        print('Dataset #{id} does not exist !')
        return redirect('data_index')

# update


@login_required(login_url="/login/")
def data_update(request, id):
    try:
        dataset = Dataset.objects.get(id=id)
        return render(request, "data/update.html", {'dataset': dataset, 'fields': ['name', 'label', 'description', 'attribute_type']})
    except Dataset.DoesNotExist:
        print('Dataset #{id} does not exist !')
        return redirect('data_index')

# delete


@login_required(login_url="/login/")
def data_delete(request, id):
    try:
        dataset = Dataset.objects.get(id=id)
        dataset.deleted = True
        dataset.save()
    except Dataset.DoesNotExist:
        print('Dataset #{id} does not exist !')

    return redirect('data_index')

# delete


@login_required(login_url="/login/")
def optimum_clusters_number(request):
    datasets = Dataset.objects.annotate(
        records_count=Count("records")).filter(deleted=False)
    algorithms = [
        {
            "label": 'Elbow',
            "mathod": 'elbow'
        },
        {
            "label": 'Silhouette',
            "mathod": 'silhouette'
        }
    ]
    return render(request, "analysis/optimum_clusters_number.html", {'datasets': datasets, 'algorithms': algorithms})

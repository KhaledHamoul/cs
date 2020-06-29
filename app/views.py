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


@login_required(login_url="/login/")
def data_index(request):
    datasets = Dataset.objects.annotate(records_count=Count("records")).all()
    return render(request, "data/index.html", {'datasets': datasets})

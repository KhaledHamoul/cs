from django.db import models
from django.contrib.auth.models import User
import json

class Attribute(models.Model):
    name = models.CharField(max_length=200, unique=True)
    label = models.CharField(max_length=200)
    description = models.TextField()
    attribute_type = models.CharField(max_length=30)

class Record(models.Model):
    data = models.TextField()

class Dataset(models.Model):
    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    title = models.CharField(max_length=200)
    description = models.TextField(max_length=200)
    deleted = models.BooleanField(default=False)
    # foreign keys
    parent = models.ForeignKey('Dataset', on_delete=models.DO_NOTHING, null=True)
    attributes = models.ManyToManyField(Attribute)
    records = models.ManyToManyField(Record)




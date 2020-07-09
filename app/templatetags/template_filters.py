from django.template.defaulttags import register
from app.models import *
import json


@register.filter
def get_item(item, key):
    itemType = type(item)
    if itemType == str:
        item = json.loads(item.replace("'", "\""))
        return item.get(key)

    if itemType == Attribute or itemType == Dataset or itemType == Record:
        return getattr(item, key)
    

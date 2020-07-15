import io
import base64
from app.templatetags.template_filters import get_item

# get image element from matplotlib plt
def getPltImage(plt):
    f = io.BytesIO()
    plt.savefig(f, format="png", facecolor=(0.95, 0.95, 0.95))
    encoded_img = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
    f.close()
    return '<img src="data:image/png;base64,%s" />' % encoded_img

# build csv file from dataset
def buildDatasetMatrix(dataset):
    datasetMatrix = []
    for record in dataset.records.all():
        temp = []
        for attribute in dataset.attributes.all():
            temp.append(ord(get_item(record.data, attribute.name)[0]) if type(get_item(record.data, attribute.name)) == str else get_item(record.data, attribute.name))

        datasetMatrix.append(temp)

    return datasetMatrix
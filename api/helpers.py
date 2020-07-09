import io
import base64

# get image element from matplotlib plt
def getPltImage(plt):
    f = io.BytesIO()
    plt.savefig(f, format="png", facecolor=(0.95, 0.95, 0.95))
    encoded_img = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
    f.close()
    return '<img src="data:image/png;base64,%s" />' % encoded_img

# build csv file from dataset
def buildCsv(dataset):
    return dataset
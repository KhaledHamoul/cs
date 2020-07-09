# method switcher
def switcher(method, args):
    print(method)
    switcher = {
        'elbow': elbow(args.get('maxIterationsNumeber')),
        'silhouette': silhouette(args.get('maxIterationsNumeber'))
    }

    return switcher.get(method, "Invalid method")

# Methods
def elbow(maxIterationsNumeber):
    print(maxIterationsNumeber)
    return 'This is elbow method with {maxIterationsNumeber} iterations'

def silhouette(maxIterationsNumeber):
    return 'This is silhouette method {maxIterationsNumeber} iterations'
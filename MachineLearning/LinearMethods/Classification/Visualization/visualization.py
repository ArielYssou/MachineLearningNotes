import matplotlib.pyplot as plt

colors = ['#4273e5', '#f7811b']
class dataSet(object):
    def __init__(self, label, color):
        self.label = label
        self.color = color
        self.xs = []
        self.ys = []

def plot_training_data(data, labels, axes):

    sets = [dataSet(i, colors[i]) for i in range(2)]
    for index in range(len(labels)):
        sets[labels[index]].xs.append(data[index][0])
        sets[labels[index]].ys.append(data[index][1])

    for ds in sets:
        axes.scatter(
                ds.xs, ds.ys,
                s=80, c="#ffffff",
                linewidths=1.5,
                edgecolors=ds.color,
                label = r'True $\omega_{}$'.format(ds.label)
                )
    return axes

def plot_classified(data, labels, axes)

    sets = [dataSet(i, colors[i]) for i in range(2)]
    for index in range(len(labels)):
        sets[labels[index]].xs.append(data[index][0])
        sets[labels[index]].ys.append(data[index][1])

    for ds in sets:
        axes.scatter(
                ds.xs, ds.ys,
                s=20, c = ds.color,
                label = r'Classified $\omega_{}$'.format(ds.label)
                )
    return axes

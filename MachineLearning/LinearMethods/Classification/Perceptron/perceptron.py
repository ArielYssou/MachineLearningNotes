import numpy as np
import matplotlib.pyplot as plt
from Visualization.visualization import *

def lin(x, a, b):
    return a * x + b

class Perceptron(object):
    def __init__(self, epochs = 3, learning_rate = 0.1):
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.weights = []
        self.bias = 0

    def activation(self, inputs):
        return sum(self.weights * inputs) + self.bias

    def predict(self, inputs): 
        return 1 if self.activation(inputs) >= 0 else 0

    def train(self, train, labels):
        self.weights = [ 0 for num in range(len(train[0])) ]

        for epoch in range(self.epochs):
            for row, label in zip(train, labels):
                prediction = self.predict(row)
                err = (label - prediction)
                print(err)

                self.bias += self.learning_rate * err
                self.weights += self.learning_rate * err * row

training_inputs = np.asarray([[2.7810836,2.550537003],
	[1.465489372,2.362125076],
	[3.396561688,4.400293529],
	[1.38807019,1.850220317],
	[3.06407232,3.005305973],
	[7.627531214,2.759262235],
	[5.332441248,2.088626775],
	[6.922596716,1.77106367],
	[8.675418651,-0.242068655],
	[7.673756466,3.508563011]])
labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

perceptron = Perceptron()
perceptron.train(training_inputs, labels)
print(perceptron.weights)
print(perceptron.bias)

fig, axes = plt.subplots(1)
plot_training_data(training_inputs, labels, axes)

classifications = [perceptron.predict(point) for point in training_inputs]
plot_classified(training_inputs, classifications, axes)

axes.legend()

axes.plot([lin(x, -perceptron.weights[1] / perceptron.weights[0], perceptron.bias) for x in np.linspace(0,9,2)], np.linspace(0,9,2), color='black')
plt.show()

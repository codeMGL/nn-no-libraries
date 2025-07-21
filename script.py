import numpy as np
from numpy.random import default_rng


def log(elt):
    print("TYPE", type(elt))
    if type(elt) == "<class 'list'>":
        print("Printing list")
        for i in elt:
            print(type(i))
        return type(elt)
    else:
        print(type(elt), elt)


def size(matrix):
    matrixSize = str(len(matrix)) + "x" + str(len(matrix[0]))
    print(matrixSize)
    return matrixSize


def printSize(matrix):
    return str(len(matrix)) + "x" + str(len(matrix[0]))


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            layer = Layer(i, layers[i], layers[i + 1])
            self.layers.append(layer)

        # Video notation
        self.z = [None] * (len(layers) - 1)
        self.a = [None] * (len(layers) - 1)

    def feedForward(self, inputs):
        self.a[0] = inputs

        # self.z[1] = np.dot(self.layers[0].weights, self.a[0])
        # self.z[1] = np.add(self.z[1], self.layers[0].biases)
        # print(self.z[1])
        # self.a[1] = self.ReLU(self.z[1])

        # self.z[2] = np.dot(self.layers[1].weights, self.a[1])
        # self.z[3] = np.dot(self.layers[2].weights, self.a[2])

        # Feeding forward each layer
        # A[i] = activation_fn( w[i - 1] * A[i - 1] + b[i - 1] )
        out = len(self.layers) - 1  # Output layer index
        for i in range(1, len(self.layers)):
            self.z[i] = np.dot(self.layers[i - 1].weights, self.a[i - 1])
            self.z[i] = np.add(self.z[i], self.layers[i - 1].biases)
            if i != out:
                self.a[i] = self.ReLU(self.z[i], i)
                print("a", i, size(self.a[i]), self.a[i])

        # Applying softmax to the output layer
        print("out", self.z[out])
        self.a[out] = self.softmax(self.z[out])
        print(self.a[out])
        # print("OUTPUT:", size(self.a[out]), self.a[out])

    def ReLU(self, matrix, i):
        # Using numpy
        # return np.maximum(0, matrix)
        print("relu", size(matrix), i)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                matrix[i][j] = 0 if elt < 0 else elt
        return matrix

    def softmax(self, matrix):
        e_x = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class Layer:

    def __init__(self, index, inputs, outputs):
        # Inputs, outputs, biases, weights
        self.index = index
        self.inputs = inputs  # Inputs
        self.outputs = outputs  # Outputs
        # print(self.__repr__())
        # Initializing weights and biases
        self.initParams(self.outputs, self.inputs)

    def __repr__(self):
        return f"Layer(index={self.index}, neurons={self.inputs})"

    def initParams(self, rows, cols):
        self.weights = np.random.rand(rows, cols)
        self.biases = np.random.rand(rows, 1)
        print("index", self.index, "weights", len(self.weights),
              "x", len(self.weights[0]), "biases", len(self.biases),
              "x", len(self.biases[0]))


# Using the video's notation - Transposing the data
data = np.array([list(range(20)),
                 list(range(5, 25)),
                 list(range(10, 30))])
x = data.copy()
x = np.transpose(x)

# print(data)
# print(x)

# Number of neurons in each layer
inputSize = 20
outputSize = 2
nn = NeuralNetwork([inputSize, 10,  5, outputSize])
# Transposed data. Size: inputs x m
nn.feedForward(x)

# print("Weights:")
# print(0, "size", len(nn.weights[0]), len(nn.weights[0][0]), nn.weights[0])
# print(1, nn.weights[1])
# print(2, nn.weights[2])


# class NeuralNetwork:
#     def __init__(self, layers):
#         self.layers = []
#         self.weights = []
#         self.biases = []
#         print("Layers:", self.layers)
#         self.initParams()

#     def initParams(self):
#         # Weights and biases between 0 and 1
#         for i in range(len(self.layers) - 1):
#             print(i)
#             print("l", self.layers[i + 1], self.layers[0])
#             # self.weights.append(default_rng(42).random(
#             #     self.layers[i + 1], self.layers[0]))
#             # self.weights.append(np.array(self.layers[i+1], self.layers[i]))
#             self.weights.append(np.random.rand(
#                 self.layers[i+1], self.layers[i]))
#             print("weight", i, self.weights[i])

#         # # Input layer
#         # self.a0 = np.zeros((self.input_size))

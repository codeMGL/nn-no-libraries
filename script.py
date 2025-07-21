# TO DO:
# Merge NeuralNetwork and Layer classes OR add more properties to Layer (w, b, dw, db...)
# [Optional] Make the code readable: Remove commented prints, add comments...

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


def size(matrix, message="Size:"):
    # matrix.shape also works
    matrixSize = str(len(matrix)) + "x" + str(len(matrix[0]))
    print(message, matrixSize, matrix)
    return [message, matrixSize, matrix]


def printSize(matrix):
    return str(len(matrix)) + "x" + str(len(matrix[0]))


class NeuralNetwork:

    def __init__(self, layers):
        self.layers = []
        for i in range(len(layers) - 1):
            layer = Layer(i, layers[i], layers[i + 1])
            self.layers.append(layer)

        print("Number of layers:", len(self.layers))
        print(self.layers)
        self.len = len(self.layers)
        # Video notation:
        # Raw values (without first input layer)
        self.z = [None] * (len(layers))
        self.dz = [None] * (len(layers))
        # Value after activation function
        self.a = [None] * (len(layers))
        self.da = [None] * (len(layers))
        # Derivative of the weights layer?
        self.dw = [None] * (len(layers))
        self.db = [None] * (len(layers))

    # Forwards propagation
    def feedForward(self, inputs):
        self.a[0] = inputs

        # Starting at index 1, looping throughout the whole list length
        for i in range(1, self.len + 1):
            # print("index", i)
            self.z[i] = np.dot(self.layers[i - 1].w, self.a[i - 1])
            self.z[i] = np.add(self.z[i], self.layers[i - 1].b)
            if i != self.len:
                self.a[i] = self.ReLU(self.z[i], i)
                # size(self.a[i], f"a{i}")
            else:
                # Applying softmax to the output layer
                self.a[i] = self.softmax(self.z[i])
                # size(self.a[i], "OUTPUT after softmax:")

        print("////////////////////////////////////////////////////")
        print("////////////////////////////////////////////////////")
        self.backwardsPropagation()

    def backwardsPropagation(self):
        # Calculating the error, based on the expected output (outputMatrix or "y")
        size(outputMatrix, "Results matrix:")
        size(self.a[self.len], f"a[{self.len}]:")

        # Calculating dw[3] and db[3]
        self.dz[self.len] = self.a[self.len] - outputMatrix
        size(self.dz[self.len], "dz[3]")
        self.dw[self.len] = (1 / m) * np.dot(
            self.dz[self.len], np.transpose(self.a[self.len - 1]))
        size(self.dw[self.len], f"dw{self.len}")
        # Bias gradient
        self.db[self.len] = (1 / m) * \
            np.sum(self.dz[self.len], axis=1, keepdims=True)  # (2, 1)
        size(self.db[self.len])

        #
        w_T = np.transpose(self.w[self.len])
        self.dz[self.len - 1] = np.dot(w_T,
                                       self.dz[self.len])
        self.dz[self.len - 1] *= self.ReLU_prime(self.z[self.len - 1])
        size(self.dz[self.len - 1], f"dz{self.len - 1}")

    def ReLU(self, matrix, i):
        # Using numpy: return np.maximum(0, matrix)
        # print("relu", size(matrix), i)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                matrix[i][j] = 0 if elt < 0 else elt
        return matrix

    def ReLU_prime(self, matrix):
        # As ReLU outputs an straight line, the derivative is 0
        # Using numpy: return (Z > 0).astype(float)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                matrix[i][j] = 0 if elt <= 0 else 1
        return matrix

    def softmax(self, matrix):
        e_x = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


class Layer:

    def __init__(self, index, inputs, outputs):
        # Includes inputs, outputs, biases, weights
        self.index = index
        self.inputs = inputs  # Inputs
        self.outputs = outputs  # Outputs
        # print(self.__repr__())
        # Initializing weights and biases
        self.initParams(self.outputs, self.inputs)

    def __repr__(self):
        return f"i={self.index}, neurons={self.inputs}, outputs={self.outputs})"

    def initParams(self, rows, cols):
        self.w = np.random.rand(rows, cols)
        self.b = np.random.rand(rows, 1)
        # print("index", self.index, "weights", len(self.weights),
        #       "x", len(self.weights[0]), "biases", len(self.biases),
        #       "x", len(self.biases[0]))


# Number of neurons in each layer
inputSize = 20
outputSize = 2

# Using the video's notation - Transposing the data
data = np.array([list(range(20)),
                 list(range(5, 25)),
                 list(range(10, 30))])
# Number of samples
m = len(data)
x = data.copy()
x = np.transpose(x)
# Outputs - Expected answer's index position
y = np.array([0, 1, 1])

outputMatrix_T = np.full((m, outputSize), 0)  # m x OutputSize (3x2)
# print("out matrix")
for i in range(m):
    outputMatrix_T[i][y[i]] = 1
    # print("i", i, y[i])
# size(outputMatrix_T)
outputMatrix = np.transpose(outputMatrix_T)


# print(data)
# print(x)


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

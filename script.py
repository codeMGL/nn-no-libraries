# TO DO:
# Merge NeuralNetwork and Layer classes OR add more properties to Layer (w, b, dw, db...)
# [Optional] Make the code readable: Remove commented prints, add comments...

import numpy as np
from numpy.random import default_rng


# Debugging DEPRECATED
def log(elt):
    print("TYPE", type(elt))
    if type(elt) == "<class 'list'>":
        print("Printing list")
        for i in elt:
            print(type(i))
        return type(elt)
    else:
        print(type(elt), elt)

# Debugging


def size(matrix, message="Size:", showMatrix=False):
    # matrix.shape also works
    matrixSize = str(len(matrix)) + "x" + str(len(matrix[0]))
    if showMatrix:
        print(message, matrixSize, matrix)
        return [message, matrixSize, matrix]
    else:
        print(message, matrixSize)
        return [message, matrixSize]


def printSize(matrix):
    return str(len(matrix)) + "x" + str(len(matrix[0]))


class NeuralNetwork:

    def __init__(self, layers, learningRate=0.2):
        self.layers = []
        self.w = []
        self.b = []
        self.lr = learningRate
        for i in range(len(layers) - 1):
            # layer = Layer(i, layers[i], layers[i + 1])
            # self.layers.append(layer)
            # Pasted
            self.w.append(np.random.rand(layers[i + 1], layers[i]))
            self.b.append(np.random.rand(layers[i + 1], 1))
            self.layers.append(
                "Layer " + str(i) + ". Weight: (" + str(layers[i + 1]) + "x" + str(layers[i]) + ")")

        print("Number of layers:", len(self.layers), self.layers)
        self.len = len(layers)
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

    def train(self, steps=100):
        if steps <= 0:
            return
        self.feedForward(x)
        self.backwardsPropagation()
        steps -= 1
        self.train(steps)

    # Forwards propagation
    def feedForward(self, inputs):
        print("Forwards propagation")
        self.a[0] = inputs

        # Starting at index 1, looping throughout the whole list length
        # NO, now just looponing (1, len)
        # size(self.w[0])
        # size(self.a[0])
        # size(self.w[0], "w0")
        for i in range(1, self.len):
            # print("index", i)
            self.z[i] = np.dot(self.w[i - 1], self.a[i - 1])
            self.z[i] = np.add(self.z[i], self.b[i - 1])
            if i != self.len:
                self.a[i] = self.ReLU(self.z[i], i)
                # size(self.a[i], f"relu a{i}")
            else:
                # Applying softmax to the output layer
                self.a[i] = self.softmax(self.z[i])
                size(self.a[i], "OUTPUT after softmax:")

    def backwardsPropagation(self):
        print("Backwards propagation")
        # Calculating the error, based on the expected output (outputMatrix or "y")
        last = self.len - 1
        print("last = ", last)
        size(outputMatrix, "Results matrix:")
        size(self.a[last], f"a[{last}]:")

        # Calculating dw[3] and db[3] // last = 2

        # Loop (last, 1) to create dz, dw, db
        for i in range(last, 0, -1):
            print("Layer ", i)
            if (i == last):
                self.dz[last] = self.a[last] - outputMatrix
            else:
                # Weird? Should be index = last
                w_T = np.transpose(self.w[i])
                self.dz[i] = np.dot(w_T, self.dz[i + 1])
                self.dz[i] *= self.ReLU_prime(self.z[i])
            size(self.dz[i], f"dz{i}", False)
            # -1 to be at the same pos as w. Weird
            self.dw[i] = (1 / m) * np.dot(
                self.dz[i], np.transpose(self.a[i - 1]))
            size(self.dw[i], f"dw{i}", False)
            # Bias gradient
            self.db[i] = (1 / m) * \
                np.sum(self.dz[i], axis=1, keepdims=True)
            size(self.db[i], f"db{i}", False)

        # Updating parameters
        for n in range(1, last):
            print("n", n)
            self.w[n] = self.w[n] - self.lr * self.dw[n + 1]
            self.b[n] -= self.lr * self.db[n + 1]
        # for w in range(len(self.w)):
        #     size(self.w[w])
        # for b in range(len(self.b)):
        #     size(self.b[b])

          # size(self.dz[last], f"dz[{last}]")
          # self.dw[last] = (1 / m) * np.dot(
          #     self.dz[last], np.transpose(self.a[last - 1]))
          # size(self.dw[last], f"dw{last}")
          # # Bias gradient
          # self.db[last] = (1 / m) * \
          #     np.sum(self.dz[last], axis=1, keepdims=True)  # (2, 1)
          # size(self.db[last])

          # # size(self.w[last - 1], "test")
          # # size(self.dz[last], f"dz{last}")
          # # Weird? Should be index = last
          # w_T = np.transpose(self.w[last - 1])
          # self.dz[last - 1] = np.dot(w_T,
          #                            self.dz[last])
          # self.dz[last - 1] *= self.ReLU_prime(self.z[last - 1])
          # # size(self.dz[last - 1], f"dz{last - 1}")
          # # data = np.transpose(x)
          # self.dw[last - 1] = (1 / m) * np.dot(self.dz[last - 1], data)
          # size(self.dw[last - 1])
          # self.db[last - 1] = (1 / m) * \
          #     np.sum(self.dz[last - 1], axis=1, keepdims=True)
          # size(self.db[last - 1], f"db{last - 1}")

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
outputSize = 10

# Using the video's notation - Transposing the data
data = np.array([list(range(inputSize)),
                 list(range(5, 5 + inputSize)),
                 list(range(10, 10 + inputSize)),
                 list(range(15, 15 + inputSize)),
                 list(range(20, 20 + inputSize)),
                 list(range(25, 25 + inputSize)),
                 list(range(30, 30 + inputSize)),
                 ])
# Number of samples
m = len(data)
x = np.transpose(data)
# Outputs - Expected answer's index position
y = np.array([0, 1, 1, 0, 0, 1, 0, 1])

outputMatrix_T = np.full((m, outputSize), 0)  # m x OutputSize (3x2)
# print("out matrix")
for i in range(m):
    outputMatrix_T[i][y[i]] = 1
    # print("i", i, y[i])
# size(outputMatrix_T)
outputMatrix = np.transpose(outputMatrix_T)


# print(data)
# print(x)


nn = NeuralNetwork([inputSize, 10, 5, outputSize])
# Transposed data. Size: inputs x m

trainingSteps = 10
nn.train(trainingSteps)

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

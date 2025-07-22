import numpy as np


def size(matrix, message="Size:", showMatrix=False):
    # matrix.shape[0] and matrix.shape[1] also works
    matrixSize = str(len(matrix)) + "x" + str(len(matrix[0]))
    if not showMatrix:
        matrix = " "
    print(message, matrixSize, matrix)
    return [message, matrixSize, matrix]


class NeuralNetwork:

    def __init__(self, layers, learningRate=0.2):
        # Weights and biases connect a layer and the next one
        self.w = []
        self.b = []
        self.lr = learningRate
        for i in range(len(layers) - 1):
            self.w.append(np.random.rand(layers[i + 1], layers[i]))
            self.b.append(np.random.rand(layers[i + 1], 1))

        self.len = len(layers)
        # Neuron values before activation function (raw)
        self.z = [None] * (len(layers))
        self.dz = [None] * (len(layers))
        # Value after activation function (ReLU and softmax)
        self.a = [None] * (len(layers))
        self.da = [None] * (len(layers))
        # Derivative of the weights and biases layers to calculate the error
        self.dw = [None] * (len(layers))
        self.db = [None] * (len(layers))

    def train(self, dataSamples, outputsMatrix, steps=100):
        # print("TRAINING...")
        if steps <= 0:
            # print("///////////////////////////////")
            print("Finished training")
            return
        self.feedForward(dataSamples)
        self.backwardsPropagation(outputsMatrix)
        steps -= 1
        self.train(dataSamples, outputsMatrix, steps)

    # Forwards propagation
    def feedForward(self, inputs):
        print("Forwards propagation")
        # Feed the inputs to the NN
        self.a[0] = inputs

        # Feeding forward each layer
        for i in range(1, self.len):
            self.z[i] = np.dot(self.w[i - 1], self.a[i - 1])
            self.z[i] = np.add(self.z[i], self.b[i - 1])
            # Applying the activation function
            if i != self.len:
                # ReLU for intermediate layers
                self.a[i] = self.ReLU(self.z[i], i)
            else:
                # Applying softmax to the output layer
                self.a[i] = self.softmax(self.z[i])
                # size(self.a[i], "OUTPUT after softmax:")

    def backwardsPropagation(self, outMatrix):
        print("Backwards propagation")
        last = self.len - 1
        # Calculating the error, based on the expected output (outMatrix)

        # Loop to create dz, dw, db
        for i in range(last, 0, -1):
            # Substracting the output minus the correct data (outMatrix)
            # or the output of each layer (z) applying the inverse act. fn
            if (i == last):
                self.dz[last] = self.a[last] - outMatrix
            else:
                w_T = np.transpose(self.w[i])
                self.dz[i] = np.dot(w_T, self.dz[i + 1])
                self.dz[i] *= self.ReLU_prime(self.z[i])
            # size(self.dz[i], f"dz{i}")
            self.dw[i] = (1 / m) * np.dot(
                self.dz[i], np.transpose(self.a[i - 1]))
            # size(self.dw[i], f"dw{i}")
            # Bias gradient
            self.db[i] = (1 / m) * \
                np.sum(self.dz[i], axis=1, keepdims=True)
            # size(self.db[i], f"db{i}")

        # Updating parameters based on the error (dw, db) and the learning rate
        for n in range(1, last):
            self.w[n] -= self.lr * self.dw[n + 1]
            self.b[n] -= self.lr * self.db[n + 1]

    def ReLU(self, matrix, i):
        # Using numpy: return np.maximum(0, matrix)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                matrix[i][j] = 0 if elt < 0 else elt
        return matrix

    def ReLU_prime(self, matrix):
        # As ReLU outputs a straight line, its derivative is 1 if x > 0
        # Using numpy: return (Z > 0).astype(float)
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                elt = matrix[i][j]
                matrix[i][j] = 0 if elt <= 0 else 1
        return matrix

    def softmax(self, matrix):
        e_x = np.exp(matrix - np.max(matrix, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)


# Number of neurons in each layer
inputSize = 20
outputSize = 10

data = np.array([])
# Number of samples
m = len(data)
x = np.transpose(data)
# Outputs - Expected answer's index position
y = np.array([])

outputMatrix_T = np.full((m, outputSize), 0)
for i in range(m):
    outputMatrix_T[i][y[i]] = 1
outputsMatrix = np.transpose(outputMatrix_T)


nn = NeuralNetwork([inputSize, 10, 5, outputSize])
trainingSteps = 2
nn.train(x, outputsMatrix, trainingSteps)

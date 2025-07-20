import numpy as np
from numpy.random import default_rng


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        print("Layers:", self.layers)
        self.initParams()

    def initParams(self):
        # Weights and biases between 0 and 1
        for i in range(len(self.layers) - 1):
            print(i)
            print("l", self.layers[i + 1], self.layers[0])
            # self.weights.append(default_rng(42).random(
            #     self.layers[i + 1], self.layers[0]))
            # self.weights.append(np.array(self.layers[i+1], self.layers[i]))
            self.weights.append(np.random.rand(
                self.layers[i+1], self.layers[i]))
            print("weight", i, self.weights[i])

        # # Input layer
        # self.a0 = np.zeros((self.input_size))


# Using the video's notation
test = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = test.copy()
x = np.transpose(x)

# print(test)
# print(x)

# Number of neurons in each layer
nn = NeuralNetwork([10, 5, 2, 2])

print(nn)
print("Weights:")
print(0, "size", len(nn.weights[0]), len(nn.weights[0][0]), nn.weights[0])
print(1, nn.weights[1])
print(2, nn.weights[2])

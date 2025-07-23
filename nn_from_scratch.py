import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import kagglehub


# Download latest version
path = kagglehub.dataset_download("altavish/boston-housing-dataset")

# print("Path to dataset files:", path)

# os.listdir(path)


df = pd.read_csv(path + "/HousingData.csv")
# df.head()  # the Y is MEDV

"""## Data Explore"""


# df.info()

df.describe()

df.shape

df.hist(figsize=(20, 15))

df.boxplot(figsize=(20, 15))

# looping for every col and get it's unique values
# for col in df.columns:
# print(f"COLUMN : {col} Unique Values : ")
# print(df[col].unique())

"""## Data Cleaning"""

df.isnull().sum()

df.isnull().sum().sum()

# df.head()


plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Boston Housing Data')
# plt.show()

low_corr = []
high_corr = []
for col in df.columns:
    if abs(df[col].corr(df["MEDV"])) >= 0.2:
        # print(f"COlUMN : {col}")
        # print(df[col].corr(df["MEDV"]))
        high_corr.append(col)
    else:
        low_corr.append(col)

# print(f"Low Corr : {low_corr}, High Corr: {high_corr}")

# so we are gonna drop "CHAS" Column cause it has very weak correlation for out target varible

df.drop("CHAS", axis=1, inplace=True)

df.sample(2)

"""# I will fill the null values with the KNNimpute but without using sklearn


## ✅ Manual KNN Imputer Plan (No Sklearn)

### 🔹 Step 1: Preprocessing
- [ ] Normalize data (Z-score or Min-Max)
  - Z-score: `(df - df.mean()) / df.std()`
  - Min-Max: `(df - df.min()) / (df.max() - df.min())`

---

### 🔹 Step 2: Detect Missing Values
- [ ] Loop through the DataFrame to find all `(row, col)` positions where data is missing
  Use: `np.argwhere(np.isnan(data))`

---

### 🔹 Step 3: For Each Missing Value
- [ ] Identify rows with **valid value in this column**
- [ ] Use only features that are **non-NaN** in current row
- [ ] Calculate **distance** to valid rows using known features
  - Use: `np.linalg.norm(row1 - row2)`
- [ ] Sort distances and select the **k nearest neighbors**
- [ ] Take the **mean** of the k values in the missing column
- [ ] Impute the missing value with that mean

---

### 🔹 Step 4: Finish
- [ ] Replace missing values in the original data
- [ ] Reverse normalization if needed:
  - `restored = (normalized * std) + mean`
- [ ] Merge with non-numeric columns (if any were dropped)

---

"""


def knn_impute_manual(df, k=5):
    df = df.copy()
    df_normalized = (df - df.mean()) / df.std()
    data = df_normalized.to_numpy()

    poses_nan = np.argwhere(np.isnan(data))

    for idx_row, idx_col in poses_nan:
        valid_rows = ~np.isnan(data[:, idx_col])
        complete_rows = ~np.isnan(data).any(axis=1)
        valid_mask = valid_rows & complete_rows

        if not np.any(valid_mask):
            continue

        known_cols = ~np.isnan(data[idx_row])
        distances = np.linalg.norm(
            data[valid_mask][:, known_cols] - data[idx_row, known_cols],
            axis=1
        )

        k_indices = np.argsort(distances)[:k]
        col_values = data[valid_mask][:, idx_col][k_indices]
        data[idx_row, idx_col] = np.mean(col_values)

    df_imputed = pd.DataFrame(data, columns=df.columns, index=df.index)
    return df_imputed


df = knn_impute_manual(df, k=5)

df.isnull().sum()

"""## Normalize Data"""

# so data Normalization is simply Scaling to a range between (0 to 1)

# Normalized data = (data - min(data)) / (max(data) - min(data))


def data_Normalization(df):
    Norm_df = df.copy()

    for col in Norm_df.columns:
        col_vlaues = Norm_df[col].values
        def Norm_rule(x): return (x - min(x)) / (max(x) - min(x))
        normed_values = Norm_rule(col_vlaues)
        Norm_df[col] = normed_values
    return Norm_df


Normed_data = data_Normalization(df)

# Normed_data.head()
# print("Normed_data", Normed_data)

# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25, 000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable(1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration(parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10, 000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63) ^ 2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's (OUTPUT)

data = np.zeros((Normed_data.shape[0], Normed_data.shape[1] - 1))
outputsMatrix = np.zeros((Normed_data.shape[0], 1))

length = len(Normed_data.iloc[0].values)
for i in range(Normed_data.shape[0]):
    # Getting every column except the last one (MEDV)
    data[i] = np.delete(Normed_data.iloc[i].values, length - 1, 0)
    # Adding MEDV data to the expected outputs matrix
    outputsMatrix[i] = Normed_data.iloc[i].values[length - 1]


# Each column is an example, that is (12, m) or (12, 506)
data = np.transpose(data)
# Each column should match every example, that is (1, m) or (1, 506)
outputsMatrix = np.transpose(outputsMatrix)
# Delete last element to test the NN

# Code
x = Normed_data.drop("MEDV", axis=1)
y = Normed_data["MEDV"]
x_train = x[:404].values
x_test = x[404:].values
y_train = y[:404].values
y_test = y[404:].values

x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T
##
# data = np.delete(data, len(data), 1)
# outputsMatrix = np.delete(outputsMatrix, len(outputsMatrix), 1)
print("xs", x_train.shape)
print("ys", y_train.shape)

""""## Transforming data """


"""## Neural Network"""


class NeuralNetwork:

    def __init__(self, layers, learningRate=0.2):
        # Weights and biases connect a layer and the next one
        self.w = []
        self.b = []
        self.lr = learningRate
        for i in range(len(layers) - 1):
            self.w.append(np.random.rand(layers[i + 1], layers[i]))
            self.b.append(np.random.rand(layers[i + 1], 1))

        print("w")
        for i in range(len(self.w)):
            print(self.w[i].shape)
        print("b")
        for j in range(len(self.b)):
            print(self.b[j].shape)

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
        for step in range(steps):
            self.feedForward(dataSamples)
            self.backwardsPropagation(outputsMatrix)
            a = self.feedForward(dataSamples)
            error = np.mean((a - outputsMatrix) ** 2)
            print("Error:", error)

        print("Finished training!")
        self.predict(x_test, y_test)
        return

    # Forwards propagation
    def feedForward(self, inputs):
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

        return self.a[self.len - 1]

    def backwardsPropagation(self, outMatrix):
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
            self.dw[i] = (1 / m) * np.dot(
                self.dz[i], np.transpose(self.a[i - 1]))
            # Bias gradient
            self.db[i] = (1 / m) * \
                np.sum(self.dz[i], axis=1, keepdims=True)

        # Updating parameters based on the error (dw, db) and the learning rate
        for n in range(1, last):  # last+1 ?????????
            self.w[n] -= self.lr * self.dw[n + 1]
            self.b[n] -= self.lr * self.db[n + 1]

    def predict(self, data, expectedOuts):
        print("Test data", data.shape)
        print("Outputs", expectedOuts.shape)
        a = self.feedForward(data)
        error = np.mean((a - expectedOuts) ** 2)
        print("Error after prediction:", error)

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


# Running the NN
layers = [x_train.shape[0], 6, 1]
trainingSteps = 2000
learningRate = 0.12
m = x_train.shape[1]
nn = NeuralNetwork(layers, learningRate)
nn.train(x_train, y_train, trainingSteps)
# Final squared mean error in test dataset: 0.0733

"""
Author: Chi-An Chen
Date: 2025-07-01
Description:
    - Implements a feedforward neural network
    - Supports XOR and linearly separable data
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue

        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21,1)

def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0]>pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n,1)

def show_result(x,y,pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground truth',fontsize = 18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result',fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()

# tanh 激活
def tanh(x):
    return np.tanh(x)

def tanh_deriv(a):
    return 1 - a**2

# 損失函數
def binary_cross_entropy(y_true, y_pred):
    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_deriv(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

# 神經網路模型
class FourLayerNN:
    def __init__(self, input_size=2, hidden_sizes=[16, 16, 16, 16], output_size=1, lr=0.1, init='kaiming'):
        self.lr = lr
        layer_sizes = [input_size] + hidden_sizes + [output_size]

        self.weights = []
        self.biases = []
        if init == 'xavier':
            # Xavier 初始化
            for i in range(len(layer_sizes) - 1):
                limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
                self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1])))
                self.biases.append(np.zeros((1, layer_sizes[i+1])))
        if init == 'kaiming':
            # Kaiming 初始化
            for i in range(len(layer_sizes) - 1):
                std = np.sqrt(2 / layer_sizes[i])
                self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * std)
                self.biases.append(np.zeros((1, layer_sizes[i+1])))

    def forward(self, X):
        self.zs = []
        self.activations = [X]

        for i in range(len(self.weights) - 1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            a = tanh(z)
            self.zs.append(z)
            self.activations.append(a)

        # output layer 使用 sigmoid
        z = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        a = 1 / (1 + np.exp(-z))
        self.zs.append(z)
        self.activations.append(a)

        return self.activations[-1]

    def backward(self, y_true):
        deltas = [binary_cross_entropy_deriv(y_true, self.activations[-1]) * (self.activations[-1] * (1 - self.activations[-1]))]

        for i in reversed(range(len(self.weights) - 1)):
            delta = (deltas[0] @ self.weights[i+1].T) * tanh_deriv(self.activations[i+1])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            dW = self.activations[i].T @ deltas[i]
            dB = np.sum(deltas[i], axis=0, keepdims=True)
            self.weights[i] -= self.lr * dW
            self.biases[i] -= self.lr * dB

    def train(self, X, y, epochs=100000, print_every=10000):
        loss_history = []
        for epoch in range(1, epochs + 1):
            pred_y = self.forward(X)
            loss = binary_cross_entropy(y, pred_y)
            self.backward(y)
            loss_history.append(loss)
            if epoch % print_every == 0:
                print(f"Epoch {epoch:>6}, Loss: {loss:.6f}")
        loss_diagram(np.array(loss_history))

    def predict(self, X):
        pred = self.forward(X)
        return pred, (pred > 0.5).astype(int)

def loss_diagram(loss):
    plt.title('Learning curve',fontsize = 18)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(np.arange(loss.shape[0]),loss,linewidth=1.5)
    plt.show()

def accuracy(y_pred,y_true):
    return 100 * np.sum(y_pred == y_true)/y_pred.shape[0]

# 主程式
if __name__ == "__main__":
    data_type = input("Choose dataset (1.XOR / 2.linear): ")
    if data_type == '1':
        x, y = generate_XOR_easy()
        x_test, y_test = generate_XOR_easy()
    elif data_type == '2':
        x, y = generate_linear(n=100)
        x_test, y_test = generate_linear(n=100)
    
    print("==========   Training   ==========")
    model = FourLayerNN(lr=0.1)
    model.train(x, y, epochs=100000, print_every=10000)

    print("==========  Predicting  ==========")
    pred_y, pred_type = model.predict(x_test)

    acc = accuracy(pred_type, y_test)
    for i in range(len(x_test)):
        print(f"Iter{i:>2}| Ground Truth: {y_test[i]}, Predicted: {pred_y[i]}")

    print(f"Accuracy: {acc:.2f}%")

    show_result(x_test, y_test, pred_type)

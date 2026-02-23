import sys
import os
import pickle
import numpy as np

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

from softmax import softmax
from sigmoid import sigmoid


def get_data():

    (x_train, t_train), (x_test, t_test) = load_mnist(
        normalize=True, flatten=True, one_hot_label=False
    )

    return x_test, t_test


# networkの重みを元々用意してあるsample_weightにする
def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(x, W1) + b1  # 線型結合
    z1 = sigmoid(a1)  # 活性化関数

    a2 = np.dot(z1, W2) + b2  # 線型結合
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)  # 最後の活性化関数はsoftmax

    return y


if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):  # len(x)はデータ数
        y = predict(network, x[i])  # x[i]は784次元のベクトル
        p = np.argmax(y)  # 配列中の最大要素のインデックスが返る。
        if p == t[i]:  # t[i]は正解ラベル
            accuracy_cnt += 1

    print(f"Accuracy:{float(accuracy_cnt) / len(x)}")

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

    batch_size = 100  # バッチ数

    accuracy_cnt = 0
    # iはバッチの先頭のインデックス
    for i in range(0, len(x), batch_size):  # len(x)はデータ数

        x_batch = x[i : i + batch_size]  # x_batchは784次元のベクトルが100行ある行列
        y_batch = predict(network, x_batch)  # predict関数自体は変えなくていいことに注意

        # argmaxのaxis=1は列方向を畳む（1列に潰す）という意味。よって各行の最大インデックスを並べることで1列に潰しているイメージ
        p = np.argmax(y_batch, axis=1)

        # 最大値のインデックスを並べた配列と正解ラベル配列の比較でboolean配列を作成し、総和で正解数をカウント
        accuracy_cnt += np.sum(p == t[i : i + batch_size])

    print(f"Accuracy:{float(accuracy_cnt) / len(x)}")

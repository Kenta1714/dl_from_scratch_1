import numpy as np
from sigmoid import sigmoid


# 恒等関数
def identity_function(x):
    return x


# パラメータの情報を格納する辞書を返す
def init_network():
    network = {}
    network["W1"] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network["b1"] = np.array([0.1, 0.2, 0.3])
    network["W2"] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network["b2"] = np.array([0.1, 0.2])
    network["W3"] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network["b3"] = np.array([0.1, 0.2])

    return network


# パラメータの情報を辞書から読み、そのパラメータを使用して入力から出力を計算する
def forward(network, X):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    b1, b2, b3 = network["b1"], network["b2"], network["b3"]

    a1 = np.dot(X, W1) + b1  # 線型結合
    z1 = sigmoid(a1)  # 活性化関数

    a2 = np.dot(z1, W2) + b2  # 線型結合
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)  # 最後の活性化関数は恒等関数

    return y


if __name__ == "__main__":
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)  # ノートブックと同じ結果になる

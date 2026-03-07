import sys
import os

sys.path.append(os.pardir)
import numpy as np
from ch03.softmax import softmax
from ch03.sigmoid import sigmoid
from ch04.loss_function import cross_entropy_error
from ch04.gradient import numerical_gradient


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        # 横ベクトルを前提としている
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

    def predict(self, x):

        W1, W2 = self.params["W1"], self.params["W2"]
        b1, b2 = self.params["b1"], self.params["b2"]

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])  # x.shape[0]はデータ数を表す

        return accuracy

    def numerical_gradient(self, x, t):
        def loss_W(_):
            """
            この関数は見た目ではWを使用していないが、勾配を計算できることに注意
            また、外部関数のnumerical_gradientの内部の実装で第一引数の関数に第二引数を渡す処理があるため、loss_Wは引数を一つsetしておかなければならない。
            ただし、あくまで形式的に渡されるだけでloss_Wに渡された引数が直接は使われないことに注意。


            """
            return self.loss(x, t)

        grads = {}
        # self.numerical_gradientではないので、再帰ではなく単に外部関数のnumerical_gradientを呼び出しているだけ。

        # 例えば下記の時、self.params["W1"]は配列そのものを指していることに注意。
        # つまり外部関数のx[idx] = tmp_val + hはself.params['W1'][idx] = tmp_val + hと同じ。
        # よってその直後のf(x)は、すでにW1のある要素が微小変化した状態のfがつかわれる。

        # lossの中のpredictはself.params["W1"]などを使用して出力を計算するが、このself.params["W1"]は外部関数の引数としても渡されており微小変化させられている。
        grads["W1"] = numerical_gradient(loss_W, self.params["W1"])
        grads["b1"] = numerical_gradient(loss_W, self.params["b1"])
        grads["W2"] = numerical_gradient(loss_W, self.params["W2"])
        grads["b2"] = numerical_gradient(loss_W, self.params["b2"])

        return grads

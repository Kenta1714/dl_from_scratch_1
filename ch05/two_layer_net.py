import sys
import os

sys.path.append(os.pardir)
import numpy as np
from ch03.softmax import softmax
from ch03.sigmoid import sigmoid
from ch04.loss_function import cross_entropy_error
from ch04.gradient import numerical_gradient

from ch05.layers import Relu, Affine, SoftmaxWithLoss

from collections import (
    OrderedDict,
)  # 昔はdictが順番を保証しなかったため保証されるorderdictというのがあった。現在はほぼ同じで、便利なメソッドがあるくらい。


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        # 横ベクトルを前提としている
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params["b1"] = np.zeros(hidden_size)
        self.params["W2"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b2"] = np.zeros(output_size)

        # レイヤの生成（前回から追加）
        self.layers = OrderedDict()
        self.layers["Affine1"] = Affine(self.params["W1"], self.params["b1"])
        self.layers["Relu1"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W2"], self.params["b2"])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):

        # 今まではparams辞書の値を代入し、直接パラを持つ変数を作成してそれを直接dotの引数に渡すなどして順伝播を記述していた。
        # 今回はlayers辞書の各value(つまりレイヤークラスのインスタンス)を順番に取り出してそのforwardメソッドをxに作用させて上書きしていくことで順伝播を記述
        # また、softmaxを入れないのは、特に予測時はスコアの大小のみ見れば良いため

        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        # softmaxwithlossの手前までを計算
        y = self.predict(x)
        # softmaxwithlossまで計算
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        # 各サンプルのmaxのindex。つまり、yは予測ラベルを並べたベクトル
        y = np.argmax(y, axis=1)
        # tがベクトルになっていない（つまり正解ラベルの配列でない場合）は、そのように直す（例えばone-hot行ベクトルが並んでいた場合、下記で正解ラベルのベクトルになる。）
        if t.ndim != 1:
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

    # BPを使用したgradientの関数
    def gradient(self, x, t):
        """
        各レイヤーを部品としてうまく分離して実装しているため、単純なfor文でforwardとbackwardを行うことができる。
        """

        # forward
        # lossを呼ぶことでその中でpredictが呼ばれ、順伝播が走る。
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        # 入出力のshapeは対応するようになっているので下記のような単純なfor文でok
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        # backwardを走らせた後なので、各レイヤーのインスタンス変数は勾配を保持している。
        grads["W1"], grads["b1"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W2"], grads["b2"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

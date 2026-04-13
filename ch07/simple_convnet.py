import sys
import os
import pickle
import numpy as np
from collections import OrderedDict

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

from ch04.gradient import numerical_gradient
from common.layers import Affine, SoftmaxWithLoss, Convolution, Pooling, Relu


# 主にtwo_layerやmulti_layer_net.pyからの変更点をメモする
class SimpleConvNet:

    def __init__(
        self,
        input_dim=(
            1,  # mnistは白黒画像なのでチャネル数は1
            28,
            28,
        ),  # 今までは入力（ただしバッチではなく一つのデータ）はベクトルだったのでinput_sizeだったが、一つのデータが3次元テンソルなのでそのdim
        conv_param={"filter_num": 30, "filter_size": 5, "pad": 0, "stride": 1},
        hidden_size=100,  # 今回は全結合層の隠れ層は一層だけなのでlistではなく普通に持つ
        output_size=10,
        weight_init_std=0.01,
    ):
        # 引数にdictがあるので、一旦それぞれに対して変数を割り当てる
        filter_num = conv_param["filter_num"]
        filter_size = conv_param["filter_size"]
        filter_pad = conv_param["pad"]
        filter_stride = conv_param["stride"]
        input_size = input_dim[
            1
        ]  # 画像の一辺の長さ。正方形画像を前提としていることに注意。
        # 畳み込み後の一辺の長さを求めている
        conv_output_size = (
            input_size + 2 * filter_pad - filter_size
        ) / filter_stride + 1
        # プーリング後の1サンプルあたりの要素数を求めている。後ろが全結合層でありflatten前提なのでフィルターの数分だけ足している。
        # プーリングは2×2のstride=2で固定なので同様にpool後のサイズも
        pool_output_size = int(
            filter_num * (conv_output_size / 2) * conv_output_size / 2
        )  # 畳み込み後の一辺の半分の長さの正方形のチャネルがチャネル数分できる

        # 重みの初期化
        # two_layer_netと似たような形の初期化をしている
        # 全結合層では(入力次元, 出力次元)の重み配列を用意するだけだったが畳み込み層はチャネル数分のフィルターをfilter_num個用意する
        self.params = {}
        self.params["W1"] = weight_init_std * np.random.randn(
            filter_num, input_dim[0], filter_size, filter_size
        )
        self.params["b1"] = np.zeros(filter_num)

        # 第二層で畳み込みの出力をflattenしたベクトルを全結合で隠れ層の次元数分のベクトルに変換する。
        self.params["W2"] = weight_init_std * np.random.randn(
            pool_output_size, hidden_size
        )
        self.params["b2"] = np.zeros(hidden_size)

        # 第三層はMLPと同じように、hiddenからoutputの次元に変換
        self.params["W3"] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(
            self.params["W1"],
            self.params["b1"],
            conv_param["stride"],
            conv_param["pad"],
        )
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])

        self.last_layer = SoftmaxWithLoss()

    # two_layer_netと同一
    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    # two_layer_netと同一
    def loss(self, x, t):
        """損失関数を求める

        Args:
            x (_type_): 入力データ
            t (_type_): 教師ラベル

        Returns:
            _type_: 損失関数の値
        """
        # softmaxwithlossの手前までを計算
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    # 今までと少し違う。
    # サンプル数がbatch_sizeで割り切れない場合は最後の端数を捨てることになる。
    # また、バッチサイズごとにfor文を回しているが、今まで通り一気にやることもできる。（おそらくtwo_layerと同じ実装でも動く）
    def accuracy(self, x, t, batch_size=100):

        # tがベクトルになっていない（つまり正解ラベルの配列でない場合）は、そのように直す（例えばone-hot行ベクトルが並んでいた場合、下記で正解ラベルのベクトルになる。）
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        # 正解数のカウント変数
        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            # i番目からバッチサイズ分だけ取得
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx)
            # 各サンプルのmaxのindex。つまり、yは予測ラベルを並べたベクトル
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]  # x.shape[0]はデータ数を表す

    # twoとほぼ同じ
    def numerical_gradient(self, x, t):
        def loss_W(_):
            """
            この関数は見た目ではWを使用していないが、勾配を計算できることに注意
            また、外部関数のnumerical_gradientの内部の実装で第一引数の関数に第二引数を渡す処理があるため、loss_Wは引数を一つsetしておかなければならない。
            ただし、あくまで形式的に渡されるだけでloss_Wに渡された引数が直接は使われないことに注意。


            """
            return self.loss(x, t)

        grads = {}

        # multi_layerと、ループ数を明示的に書いているところ以外は全く同じ
        for idx in (1, 2, 3):
            grads["W" + str(idx)] = numerical_gradient(
                loss_W, self.params["W" + str(idx)]
            )
            grads["b" + str(idx)] = numerical_gradient(
                loss_W, self.params["b" + str(idx)]
            )

        return grads

    # BPを使用したgradientの関数
    # two_layerとほぼ同じ

    def gradient(self, x, t):
        """
        各レイヤーを部品としてうまく分離して実装しているため、単純なfor文でforwardとbackwardを行うことができる。
        """

        # forward
        # lossを呼ぶことでその中でpredictが呼ばれ、順伝播が走る。
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        # 入出力のshapeは対応するようになっているので下記のような単純なfor文でok
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        # backwardを走らせた後なので、各レイヤーのインスタンス変数は勾配を保持している。

        grads["W1"], grads["b1"] = self.layers["Conv1"].dW, self.layers["Conv1"].db
        grads["W2"], grads["b2"] = self.layers["Affine1"].dW, self.layers["Affine1"].db
        grads["W3"], grads["b3"] = self.layers["Affine2"].dW, self.layers["Affine2"].db

        return grads

    # 畳み込み層の可視化用
    def save_params(self, file_name="params.pkl"):
        params = {}

        # 単にself.paramsをparamsにコピーしているだけ
        for key, val in self.params.items():
            params[key] = val

        # paramsというdictオブジェクトをpklとして保存
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)

        # 単にpklから読んだパラメータで、self.paramsを上書きしているだけ
        for key, val in params.items():
            self.params[key] = val

        # initでは、乱数でself.paramsを初期化し、それを引数として各レイヤー（クラスのインスタンス）を生成した。
        # ここではインスタンスはすでに生成されているので、各レイヤーのWなどのインスタンス変数を直接self.paramsで上書きする。
        # フィルターを可視化するだけならこれは不要？学習されたフィルターによる畳み込み処理の結果とかも見たいかららlayerの重みも更新してる？
        for i, key in enumerate(["Conv1", "Affine1", "Affine2"]):
            self.layers[key].W = self.params["W" + str(i + 1)]
            self.layers[key].b = self.params["b" + str(i + 1)]

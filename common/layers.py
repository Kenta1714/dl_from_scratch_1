import sys
import os

import numpy as np

sys.path.append(os.pardir)
from ch03.softmax import softmax
from ch04.loss_function import cross_entropy_error
from common.utill import im2col, col2im

"""
基本的にNNで使うレイヤーのbackwardは、逆伝播のfor文がレイヤーの組み合わせによらず単純なfor文になるように、入力も出力も一つにする。

forwardも同様
"""


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        # backwardの計算で必要なので保持しておく
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        # 下記二つは下流には伝えないが、更新のために保持しておく必要があるのでインスタンス変数にする
        self.dW = None
        self.db = None

    def forward(self, x):
        """
        テンソル対応でflattenのような操作を入れる
        """
        # テンソル対応
        # 例えばカラー画像なら4階のテンソルなので(10,3,32,32)のようなshape
        self.original_x_shape = x.shape

        # ここでflatttenのような操作を入れる。各データを一本の横ベクトルにするイメージ
        # x.shape[0]は常にデータ数(バッチ数)
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        """

        Affineの内部計算の都合で2次元に変形するが、外から見た入出力の形は壊してはいけないので元に戻す
        """

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # 元のデータ形状に戻す。これは単に偏微分の並べ方を変えているだけなので問題ない（見かけを変えているだけ）
        # shapeはタプルになっているのでアンパックして引数として渡しているだけ
        dx = dx.reshape(*self.original_x_shape)

        # backwardの処理が単純なfor文でいいようにdWやdbは保持するだけで出力はしない。（下流に伝える必要があるものだけにする）
        # 各レイヤーを一つの演算のように扱っているので、基本的に下流に伝える偏微分は一つだけで済む。
        return dx


class SoftmaxWithLoss:
    """
    x,y,tはバッチを想定していることに注意
    """

    def __init__(self):
        self.loss = None
        # 下記二つはbackwardの計算時に必要
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        # backwardの計算自体にはlossは不要だが、便利なので保持しておく
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        # 教師データがone-hot（つまり出力と同じ形のベクトル）
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size

        else:
            dx = self.y.copy()
            # バッチの各行について、正解のindexのみ1を引く
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx


class Dropout:
    """dropoutのレイヤ

    dropout_ratio:ニューロンを落とす割合
    """

    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        # forwardで使ったmaskをbackwardでも使うので
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            # randnと違いrandは0から1で一様分布に従う乱数を生成させる。入力xと同じshapeの乱数配列を生成している。
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # ブールインデックスは掛け算の時は0or1として扱われる。よってtrueの箇所のみ残る。
            return x * self.mask
        else:
            # TODO 復習する
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        # maskがxの値によるのかランダムかの違いだけで、基本的にはReluと同じ要領。
        # 落としたニューロンがどれだけ動いても、強制的に0になるため最終的な損失が変わらない。よってbackwardでも0をかけることで勾配を流さないようにする
        return dout * self.mask


# TODO 下記は一旦コピペなので自分で実装し直す。
class BatchNormalization:
    """
    http://arxiv.org/abs/1502.03167
    """

    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # Conv層の場合は4次元、全結合層の場合は2次元

        # テスト時に使用する平均と分散
        self.running_mean = running_mean
        self.running_var = running_var

        # backward時に使用する中間データ
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * mu
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * var
            )
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


# TODO 特にbackwardがよくわからないので復習する
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):

        # 重み=フィルターなので4次元テンソルの想定？
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中間データ
        self.x = None
        self.col = None
        self.col_W = None

        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        # 重みテンソルのshapeがそのままフィルターの個数、チャネル数、高さ、幅に対応
        FN, C, FH, FW = self.W.shape
        # 入力データは4次元テンソル
        N, C, H, W = x.shape
        out_h = int((H + 2 * self.pad - FH) / self.stride) + 1
        out_w = int((W + 2 * self.pad - FW) / self.stride) + 1

        # 入力側の一回のフィルター適用要素が一行となっている配列を作成
        col = im2col(x, FH, FW, self.stride, self.pad)

        # 各列が一つのフィルタとなっている配列col_Wを作成
        # フィルターの個数行ある配列にしてから、転置してそれらを列にする。

        # 基本的にshapeの一番左がFN（つまりフィルタ数がFN）なので、メモリに並んでいるものについて、先頭からW/FN件ずつ取ってきたい。
        # reshap(FN,-1)だと一行にW/FN個ずつ並べるのでそれができる
        # reshape(-1,FN)にしてしまうと、例えば2かけ2のチャネル数1のフィルターが2個あったとして合計8個の要素が4行2列の配列に変形される
        # そうすると、先頭から2個ずつ区切って置かれることになるので、8/2=4ずつ取ることができない
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        # affineと同じ
        self.db = np.sum(dout, axis=0)
        # affineと同じ
        self.dW = np.dot(self.col.T, dout)
        # ここは違う
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


# TODO 特にbackwardがよくわからないので復習する
class Pooling:
    def __init__(self, pool_h, pool_w, stride=2, pad=0):

        # プーリング領域の高さと幅
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # im2colの返り値では、各行には同じウィンドウの全チャンネルの要素が横に全部くっついている。（畳み込みでは内積をとった後チャネル方向に足すため）
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        # しかし、poolingではチャネル方向は混ぜ合わせないので、一行は一チャネルにしたい。よって、一行がフィルタのマス数になるようにreshapeする
        # この時点では、左上の位置のチャネル1でスキャンされる要素、チャネル2でスキャンされる要素、、、その一つ右の領域のチャネル1でスキャンされる要素、チャネル2、、のように並んでいる
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 1列に潰して最大値取得したい。
        # 引数を取得
        arg_max = np.argmax(col, axis=1)

        # 値そのものを取得（一次元縦ベクトルのイメージ）
        out = np.max(col, axis=1)

        # ここで、reshapeすると各位置ごとに最大値がチャネル数分並んでいるような配列になる
        # それを、チャネル軸方向に取りたいのでtransposeしている
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    # reluの実装で使用したmaxの逆伝播が参考になるらしい
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx

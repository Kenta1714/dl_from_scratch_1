import numpy as np


def softmax_v1(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def _softmax_1d(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def softmax(x):
    """
    通常softmax関数は各行ごとに正規化したいので、このような実装にする
    """
    # keepdims=Trueにすることで例えば通常二次元配列に対してaxis指定してmaxを求めると一次元ベクトルになってしまうが、それを元の二次元配列のままにしてくれる。
    c = np.max(
        x, axis=-1, keepdims=True
    )  # axis=-1は最後の軸。二次元配列なら1列に潰す。
    exp_x = np.exp(x - c)  # オーバーフロー対策
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    y = exp_x / sum_exp_x
    return y

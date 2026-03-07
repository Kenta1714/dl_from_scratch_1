# %%
import numpy as np


# %%
def sum_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# どの入力でも対応できるやつ
def cross_entropy_error(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)  # argmaxを使いたいのでここではtも必ず二次元配列にする

    # one-hotの場合
    if t.size == y.size:
        t = t.argmax(axis=1)  # ラベルを格納した一次元配列が返る
    else:
        t = t.reshape(-1)  # one-hotでない場合も必ず一次元配列にする

    delta = 1e-7

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size


# %%
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

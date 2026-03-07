# %%
import numpy as np
import matplotlib.pyplot as plt


# ブロードキャストにより配列の要素ごとに計算が行われる
# sigmoidは（softmaxとは異なり）入力の配列の形によらず、各要素に対して独立な演算なので下記で任意の入力に対応できる。
def sigmoid(x):

    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


# %%

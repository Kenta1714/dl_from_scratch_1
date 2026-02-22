# %%
import numpy as np
import matplotlib.pyplot as plt


# ブロードキャストにより配列の要素ごとに計算が行われる
def sigmoid(x):

    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


# %%

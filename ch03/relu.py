# %%
import numpy as np
import matplotlib.pyplot as plt


# ブロードキャストにより配列の要素ごとに計算が行われる
def relu(x):

    return np.maximum(0, x)


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1, 5.5)
    plt.show()

    # maximumもブロードキャスト演算が行えることに注意
    x = np.array([-1, 2, -3])
    print(np.maximum(0, x))


# %%

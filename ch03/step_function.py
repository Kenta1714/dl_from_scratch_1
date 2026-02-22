# %%
import numpy as np
import matplotlib.pyplot as plt


# これだと実数しか扱えない
def step_function_v1(x):
    if x > 0:
        return 1
    else:
        return 0


# 配列を扱える
def step_function(x):
    y = x > 0
    return y.astype(int)


if __name__ == "__main__":
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


# %%

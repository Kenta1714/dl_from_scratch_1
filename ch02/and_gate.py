# %%
import numpy as np


# %%
def AND_v1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif (
        tmp > theta
    ):  # 両方1の場合のみtmpが1になり閾値0.7を超える（他は0.5か0になるので超えない）
        return 1


# %%
def AND(x1, x2):
    x = np.array([x1, x2])  # 入力を配列にする
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    elif tmp > 0:  # 両方1の場合のみbias0.7を引いても0.3残るので0を超える
        return 1


# この中に入れないとimportしたときに下記のprint文などが全て実行されてしまう
if __name__ == "__main__":
    print(AND_v1(0, 0))
    print(AND_v1(1, 0))
    print(AND_v1(0, 1))
    print(AND_v1(1, 1))

    x = np.array([0, 1])
    w = np.array([0.5, 0.5])
    b = -0.7

    print(w * x)  # 要素ごと(element-wise)の積
    print(np.sum(w * x))  # 総和を計算することで内積になる
    print(np.sum(w * x) + b)

    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet

sys.path.append(os.pardir)
from common.utill import output_plot


def filter_show(filters, filename=None, nx=8, margin=3, scale=10):
    """_summary_

    Args:
        filters (_type_): フィルター群を格納する4次元テンソル
        nx (int, optional): 一行に何フィルター並べるか. Defaults to 8.
        margin (int, optional): _description_. Defaults to 3.
        scale (int, optional): _description_. Defaults to 10.
    """
    FN, C, FH, FW = filters.shape
    # 何行必要かの計算。ceilは切り上げ
    ny = int(np.ceil(FN / nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        ax = fig.add_subplot(
            ny, nx, i + 1, xticks=[], yticks=[]
        )  # 空配列を渡して目盛りを消す)
        # i番目のフィルターの第0チャネル（ただし白黒なので唯一のフィルター）
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation="nearest")

    output_plot(filename=filename)


network = SimpleConvNet()
# デフォルト引数でのsimpleconnetのランダム初期化第一層重み
filter_show(network.params["W1"], "ランダム初期化重み")

# 学習後の重みとして保存したparams.pklの重みでnetの重みを上書き
network.load_params("params.pkl")
filter_show(network.params["W1"], "全学習データ学習後の重み")

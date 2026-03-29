import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# ch05のtwolayernetを使用する
from ch05.two_layer_net import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# ハイパラ
iters_num = 10000
train_size = x_train.shape[0]  # データ数
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []
# ミニバッチを何回繰り返したら1エポックになるか
iter_per_epoch = max(train_size / batch_size, 1)  # 今回ならtrain6万件より600になる

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    batch_mask = np.random.choice(
        train_size, batch_size
    )  # trainのindexの中からbatch_size個の数字を選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    # ここのみ変更
    grad = network.gradient(x_batch, t_batch)

    # gradはパラ名をキー、その勾配をvalueとした辞書なので、for文で各パラの勾配にアクセスし更新
    for key in ("W1", "W2", "b1", "b2"):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train acc, test acc | {train_acc}, {test_acc}")

# グラフの描画
markers = {"train": "o", "test": "s"}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label="train acc")
plt.plot(x, test_acc_list, label="test acc", linestyle="--")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc="lower right")
plt.show()

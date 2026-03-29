import sys
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from common.utill import smooth_curve

# ch05のtwolayernetを使用する
from ch06.optimizer import SGD,Momentum,AdaGrad,Adam 
from ch06.multi_layer_net import MultiLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)


train_size = x_train.shape[0]  # データ数
batch_size = 128
max_iterations=2000


# 実験の設定
# lrを引数として渡していないので、各optimizerのデフォルトの値が使用される
optimizers={}
optimizers["SGD"] = SGD()
optimizers["Momentum"] = Momentum()
optimizers["AdaGrad"] = AdaGrad()
optimizers["Adam"] = Adam()

# networkもloss_listもoptimezer名をkeyとして、実験ごとにvalueを持てるようにする。
networks={}
train_loss={}
for optimizer_name in optimizers.keys():
    # 下記三つの引数以外に指定していないので、reluと正則化なしが適用される
    networks[optimizer_name] = MultiLayerNet(
        input_size=784, hidden_size_list=[100,100,100,100], 
        output_size=10)
    # 今回はtrainの損失のみを見る
    train_loss[optimizer_name] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(
        train_size, batch_size
    )  # trainのindexの中からbatch_size個の数字を選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for optimizer_name in optimizers.keys():
        # gradientを呼ぶことで、順伝播と逆伝播が一回ずつ行われ、BPにより求めた勾配が返される
        grads = networks[optimizer_name].gradient(x_batch, t_batch)
        
        # 今まではここで下記のように各パラ行列に対して直接lr分更新する操作をforループで記述していたが、update側に押し付けている
        """
        for key in ("W1", "W2", "b1", "b2"):
            networks.params[key] -= learning_rate * grad[key]
        """
        optimizers[optimizer_name].update(networks[optimizer_name].params,grads)
        
        loss = networks[optimizer_name].loss(x_batch, t_batch)
        train_loss[optimizer_name].append(loss)    

    if i % 100 == 0:
        print(f"========iteration{i}========")
        for optimizer_name in optimizers.keys():
            # ここでなぜかもう一度lossを計算しているが、loss[-1]で取り出しても問題なさそう？
            loss = networks[optimizer_name].loss(x_batch, t_batch)
            print(f"{optimizer_name}:{loss}")
            
# 以下コピペ
# グラフの描画
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

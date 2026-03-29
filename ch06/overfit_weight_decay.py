import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# ch05のtwolayernetを使用する
from ch06.optimizer import SGD,Adam 
from ch06.multi_layer_net import MultiLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 過学習が起こるようにわざと学習データを削減
x_train=x_train[:300]
t_train=t_train[:300]

# weight_decayの設定=====================================
# weight_decay_lambda=0
weight_decay_lambda=1
# =======================================================

network=MultiLayerNet(input_size=784, hidden_size_list=[100,100,100,100,100,100], # 過学習しやすいように層数を増やす 
                      output_size=10,weight_decay_lambda=weight_decay_lambda) # ここで設定したlambdaを渡す

optimizer = SGD(lr=0.01)


max_epochs=201
train_size = x_train.shape[0]  # データ数
batch_size = 100

# 今回は過学習を見たいのでtrainとtestでそれぞれ用意する
train_loss_list = []
train_acc_list = []
test_acc_list = []
# ミニバッチを何回繰り返したら1エポックになるか
iter_per_epoch = max(train_size / batch_size, 1) 
epoch_cnt=0


for i in range(1000000000):
    batch_mask = np.random.choice(
        train_size, batch_size
    )  # trainのindexの中からbatch_size個の数字を選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # gradientを呼ぶことで、順伝播と逆伝播が一回ずつ行われ、BPにより求めた勾配が返される
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params,grads)


    if i % iter_per_epoch == 0:
        
        # 過学習の様子を確認するためにtrainとtestそれぞれでグラフをプロットする
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        
        print(f"epoch:{epoch_cnt} | train_acc:{train_acc}, test_acc:{test_acc}")
        
        epoch_cnt+=1
        if epoch_cnt >= max_epochs:
            break
        
        
# 以下コピペ

# 3.グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
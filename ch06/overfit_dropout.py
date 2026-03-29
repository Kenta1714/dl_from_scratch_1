import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# ch05のtwolayernetを使用する
from ch06.optimizer import SGD,Adam 
from ch06.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 過学習が起こるようにわざと学習データを削減
x_train=x_train[:300]
t_train=t_train[:300]

# dropoutの設定=====================================
use_dropout=True
dropout_ratio=0.2
# =======================================================

network=MultiLayerNetExtend(input_size=784, hidden_size_list=[100,100,100,100,100,100], # 過学習しやすいように層数を増やす 
                      output_size=10,use_dropout=use_dropout,dropout_ration=dropout_ratio) # ここでdropoutについて設定した内容を渡す

# 実験の設定や途中経過の表示などは全てtrainer側に任せている。
trainer=Trainer(network,x_train,t_train,x_test,t_test,
                epochs=1001,mini_batch_size=100,
                optimizer="sgd",optimizer_param={"lr":0.01},verbose=True)

trainer.train()

# 途中経過のlogはtrainerクラス側で勝手に吐き出されるが、下記のリストはこちら側からインスタンス変数にアクセスしてこちら側で表示させる。
train_acc_list,test_acc_list=trainer.train_acc_list,trainer.test_acc_list

# 以下コピペ
# グラフの描画==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='o', label='train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='test', markevery=10)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
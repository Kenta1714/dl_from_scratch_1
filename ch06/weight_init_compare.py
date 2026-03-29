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
# key名はprint文などでの表示用、valueはmulti_layer_netの初期化の際にweight_init_std引数に渡す用
weight_init_types={"std=0.01":0.01,"Xavier":"sigmoid","He":"relu"}
optimizer = SGD(lr=0.01)


# networkもloss_listもoptimezer名をkeyとして、実験ごとにvalueを持てるようにする。
networks={}
train_loss={}
for weight_type_name,weight_type in weight_init_types.items():
    networks[weight_type_name] = MultiLayerNet(
        input_size=784, hidden_size_list=[100,100,100,100], 
        output_size=10,weight_init_std=weight_type) # ここでweight_typeを渡す
    # 今回はtrainの損失のみを見る
    train_loss[weight_type_name] = []

for i in range(max_iterations):
    batch_mask = np.random.choice(
        train_size, batch_size
    )  # trainのindexの中からbatch_size個の数字を選ぶ
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for weight_type_name in weight_init_types.keys():
        # gradientを呼ぶことで、順伝播と逆伝播が一回ずつ行われ、BPにより求めた勾配が返される
        grads = networks[weight_type_name].gradient(x_batch, t_batch)
        
        # 今まではここで下記のように各パラ行列に対して直接lr分更新する操作をforループで記述していたが、update側に押し付けている
        """
        for key in ("W1", "W2", "b1", "b2"):
            networks.params[key] -= learning_rate * grad[key]
        """
        optimizer.update(networks[weight_type_name].params,grads)
        
        loss = networks[weight_type_name].loss(x_batch, t_batch)
        train_loss[weight_type_name].append(loss)    

    if i % 100 == 0:
        print(f"========iteration{i}========")
        for weight_type_name in weight_init_types.keys():
            loss = networks[weight_type_name].loss(x_batch, t_batch)
            print(f"{weight_type_name}:{loss}")
            
# 以下コピペ
# グラフの描画
markers = {'std=0.01': 'o', 'Xavier': 's', 'He': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()

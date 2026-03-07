import sys
import os

sys.path.append(os.pardir)
import numpy as np
from ch03.softmax import softmax
from ch04.loss_function import cross_entropy_error
from ch04.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.rand(2, 3)

    def predict(self, x):
        # 横ベクトルと行列の積より結果は要素数が3の横ベクトルになる。
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print("重み行列")
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print("予測結果")
print(p)
print("最大値のラベル")
print(np.argmax(p))

t = np.array([0, 0, 1])
print("正解ラベルが3番目だった場合の損失")
print(net.loss(x, t))

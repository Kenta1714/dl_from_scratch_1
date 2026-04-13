import sys
import os
import numpy as np
from tqdm import tqdm

sys.path.append(os.pardir)

from ch06.optimizer import SGD, Momentum, AdaGrad, Adam


class Trainer:
    """NNの訓練を行うクラス"""

    def __init__(
        self,
        network,
        x_train,
        t_train,
        x_test,
        t_test,
        epochs=20,
        mini_batch_size=100,
        optimizer="SGD",
        optimizer_param={"lr": 0.01},
        # evaluate..引数はaccを測るときに先頭何件を使用するか。（全件だと重すぎる場合があるため）
        # verbose引数はaccなどをprintするかどうか
        evaluate_sample_num_per_epoch=None,
        verbose=True,
    ):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimizer
        # この辞書作成時点ではインスタンス化をしていないことに注意。
        # 辞書で全小文字と正しいクラス名の対応を作成しておくことで、表記揺れに対処できる。
        optimizer_class_dict = {
            "sgd": SGD,
            "momentum": Momentum,
            "adagrad": AdaGrad,
            "adam": Adam,
        }

        # ユーザーの入力を全部小文字にした後、上記のdictで正しいクラス名を引っ張ってきてインスタンス化を行う
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)

        # 学習のための設定など
        # 下記は従来までは直接の変数としていたが、今回はインスタンス変数として保持することに注意
        self.train_size = x_train.shape[0]  # データ数
        # iter_per_epochは、ミニバッチを何回繰り返したら1エポックになるか
        # ただし相変わらず復元抽出なので、厳密な意味でのエポックではないことに注意

        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        # 全エポックにわたり、合計で何回ミニバッチを取り出すかがmax_iter
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0  # 今まではfor文のiに保持していた。今回はfor文の中身からtrain_step関数を切り出しているので、そっち側でもiter数を持つ必要があるので、インスタンス変数としている
        # また、設計思想的に学習がどこまで進んでいるかを内部に持つような設計にしていると言える。
        # （スクリプト側で、インスタンス変数にアクセスできる。）
        self.current_epoch = 0  # epoch数のカウント

        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        # trainのindexの中からbatch_size個の数字を選ぶ
        batch_mask = np.random.choice(self.train_size, self.batch_size)

        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        # gradientを呼ぶことで、順伝播と逆伝播が一回ずつ行われ、BPにより求めた勾配が返される
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)

        if self.verbose:
            print(f"train loss:{loss}")

        # 今までfor文だったのをインスタンス変数に変えているので、明示的に+1する
        # 公式の実装では1エポック判定の後に書かれているが、そうすると1回目はcurrent_iter=0でif文の判定が行われ表示されてしまう。
        self.current_iter += 1

        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1

            # 評価用のデータセットを作成する。基本的には全データとする。
            x_train_sample, t_train_sample = self.x_train, self.t_train
            x_test_sample, t_test_sample = self.x_test, self.t_test

            # もし評価用のデータ件数について指定があるなら先頭t件を使用したデータセットにする。（上書きする）
            # ただし本来はランダムの1000件サンプリングが適切
            if self.evaluate_sample_num_per_epoch is not None:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample, t_train_sample = self.x_train[:t], self.t_train[:t]
                x_test_sample, t_test_sample = self.x_test[:t], self.t_test[:t]

            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print(
                    f"epoch:{self.current_epoch} | train_acc:{train_acc}, test_acc:{test_acc}"
                )

    def train(self):
        for i in tqdm(range(self.max_iter)):
            self.train_step()

        # ここはsampleではなく必ず全体を使う。
        test_acc = self.network.accuracy(self.x_test, self.t_test)
        if self.verbose:
            print("======================= Final Test Accuracy=======================")
            print(f"test_acc:{test_acc}")

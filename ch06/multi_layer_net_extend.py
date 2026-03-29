import sys
import os
import numpy as np
from collections import OrderedDict

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

from ch04.gradient import numerical_gradient
from ch05.layers import Sigmoid,Relu,Affine,SoftmaxWithLoss,BatchNormalization,Dropout

# 主にmulti_layer_net.pyからの変更点をメモする
class MultiLayerNetExtend:
    """全結合層による多層NN
    
    Parameters
    input_size:入力サイズ
    hidden_size_list:隠れ層のニューロンのリスト
    output_size:出力サイズ
    activation="relu" or "sigmoid"
    weight_init_std=重みの標準偏差を指定
    "relu"または"he"を指定した場合は「Heの初期値」を設定
    "sigmoid"または"xavier"を指定した場合は「Xavierの初期値」を設定
    weight_decay_lambda:L2ノルムの強さ（L2ノルムの係数）
    use_dropout:Dropoutを使用するかどうか
    dropout_ration:Dropoutの割合
    use_bathnorm:BNを使用するかどうか
    """
    def __init__(self, input_size, hidden_size_list, output_size, 
                 activation="relu",weight_init_std="relu",weight_decay_lambda=0,
                 use_dropout=False,dropout_ration=0.5,use_batchnorm=False):
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_size_list=hidden_size_list
        self.hidden_layer_num=len(hidden_size_list)
        self.use_dropout=use_dropout
        self.weight_decay_lambda =weight_decay_lambda
        self.use_batchnorm=use_batchnorm
        self.params={}
        
        # 重みの初期化
        self.__init_weight(weight_init_std)
        
        # レイヤの生成
        # valueはlayers.pyのクラス名。()をつけてないのでインスタンス化はまだされていない
        activation_layer={"sigmoid":Sigmoid, "relu":Relu}
        self.layers=OrderedDict()
        # hidden_layer_numが3ならidx=3まで回す。最終層はそのあと。
        # twoでは"Affine1"のように層名を手書きしていたのを、ループのidxを使用して半自動的に命名している。
        for idx in range(1,self.hidden_layer_num+1):
            self.layers["Affine"+str(idx)]=Affine(self.params["W"+str(idx)],
                                                  self.params["b"+str(idx)]) # この時点ですでにself.paramsの辞書のvalueは初期化済みのパラであることに注意
            
            # ここのAffineとactivationの間のタイミングでBN層を追加
            if self.use_batchnorm:
                # BNのパラも学習するのでselfのparamsに追加する
                # 例えば2層目のデータはhidden_size_listの2番目の値よりindex=1のように、indexと層の数字がずれていることに注意。あくまでBNは自分の層の特徴量ごとにgammaとbetaを持つ。
                self.params["gamma"+str(idx)]=np.ones(hidden_size_list[idx-1])
                self.params["beta"+str(idx)]=np.zeros(hidden_size_list[idx-1])
                self.layers["BatchNorm"+str(idx)]=BatchNormalization(self.params["gamma"+str(idx)],self.params["beta"+str(idx)])
            
            self.layers["Activation_function"+str(idx)]=activation_layer[activation]()
            
            # 活性化関数を通した後で、dropoutでランダムにニューロンの値を0にする
            if self.use_dropout:
                self.layers["Dropout"+str(idx)]=Dropout(dropout_ration)
            
        # 最後の層は線形変換のみ（softmax等の活性化関数はlast_layerに入れる）
        idx=self.hidden_layer_num+1
        self.layers["Affine"+str(idx)]=Affine(self.params["W"+str(idx)],
                                                  self.params["b"+str(idx)])
        
        self.last_layer=SoftmaxWithLoss()        
        
    def __init_weight(self,weight_init_std):
        """重みの初期値設定

        Args:
            weight_init_std (_type_): 重みの標準偏差を指定
            "relu"または"he"を指定した場合はheの初期値を設定
             "sigmoid"または"xavier"を指定した場合は「Xavierの初期値」を設定

        Returns:
            _type_: _description_
        """
    
        # 下記のように一つのリストにすることで、(前の層, 自分の層)のように重み行列のshapeを記述しやすくなる
        all_size_list=[self.input_size]+self.hidden_size_list+[self.output_size]
        
        # 1層目に向かう重みから設定するので、1から始める
        for idx in range(1, len(all_size_list)):
            scale=weight_init_std
            if str(weight_init_std).lower() in ("relu","he"): # 引数（を小文字にした文字列）がreluまたはheの時
                # σ = sqrt(2 / 一つ前の層のニューロン数)に設定する
                scale=np.sqrt(2.0/all_size_list[idx-1])
            elif str(weight_init_std).lower() in ("sigmoid","xavier"): # 引数（を小文字にした文字列）がreluまたはheの時
                # σ = sqrt(2 / 一つ前の層のニューロン数)に設定する
                scale=np.sqrt(1.0/all_size_list[idx-1])
                
            # データは横ベクトル想定なので、行数が前の層のニューロン数、列数が現在の層のニューロン数となる重み行列をscaleを使用して初期化する
            self.params["W"+str(idx)]=scale*np.random.randn(all_size_list[idx-1], all_size_list[idx])
            # 自層のニューロン数の分だけバイアスがある。バイアスは0で初期化している
            self.params["b"+str(idx)]=np.zeros(all_size_list[idx])
        
    # train_flgが追加されている
    def predict(self, x,train_flg=False):

        for key,layer in self.layers.items():
            # TODO ここは特にBN側は実装を確認する
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x,train_flg)
            else:
                x = layer.forward(x)

        return x
    
    # train_flgが追加されている
    def loss(self, x, t,train_flg=False):
        """損失関数を求める

        Args:
            x (_type_): 入力データ
            t (_type_): 教師ラベル

        Returns:
            _type_: 損失関数の値
        """
        # softmaxwithlossの手前までを計算
        y = self.predict(x,train_flg)
        
        weight_decay=0
        # ここでは最終層まで（隠れ層+1まで）ループを回す
        for idx in range(1, self.hidden_layer_num+2):
            W=self.params["W"+str(idx)]
            # 層ごとに0.5*lambdaをかけているが、ループ後にまとめてかけても同じ。ただし、層ごとにlambdaを変えるなどの拡張がしやすい形。
            weight_decay+=0.5*self.weight_decay_lambda*np.sum(W**2) # 単純に0.5*lambda*L2ノルムを計算しているだけ。
        # softmaxwithlossまで計算
        return self.last_layer.forward(y, t)+weight_decay
    
    # twolayernetと同じ。
    def accuracy(self, x, t):
        y = self.predict(x,train_flg=False)
        # 各サンプルのmaxのindex。つまり、yは予測ラベルを並べたベクトル
        y = np.argmax(y, axis=1)
        # tがベクトルになっていない（つまり正解ラベルの配列でない場合）は、そのように直す（例えばone-hot行ベクトルが並んでいた場合、下記で正解ラベルのベクトルになる。）
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])  # x.shape[0]はデータ数を表す

        return accuracy
    
    # twoとほぼ同じ
    def numerical_gradient(self, x, t):
        def loss_W(_):
            """
            この関数は見た目ではWを使用していないが、勾配を計算できることに注意
            また、外部関数のnumerical_gradientの内部の実装で第一引数の関数に第二引数を渡す処理があるため、loss_Wは引数を一つsetしておかなければならない。
            ただし、あくまで形式的に渡されるだけでloss_Wに渡された引数が直接は使われないことに注意。


            """
            return self.loss(x, t)

        grads = {}
        # self.numerical_gradientではないので、再帰ではなく単に外部関数のnumerical_gradientを呼び出しているだけ。

        # 例えば下記の時、self.params["W1"]は配列そのものを指していることに注意。
        # つまり外部関数のx[idx] = tmp_val + hはself.params['W1'][idx] = tmp_val + hと同じ。
        # よってその直後のf(x)は、すでにW1のある要素が微小変化した状態のfがつかわれる。

        # lossの中のpredictはself.params["W1"]などを使用して出力を計算するが、このself.params["W1"]は外部関数の引数としても渡されており微小変化させられている。
        
        # twoからの変更は下記を直書きからforループに変えただけ
        for idx in range(1,self.hidden_layer_num+2):
            grads["W"+str(idx)] = numerical_gradient(loss_W, self.params["W"+str(idx)])
            grads["b"+str(idx)] = numerical_gradient(loss_W, self.params["b"+str(idx)])
            
            # ここだけ変更点
            # BNありで最終層でない時
            if self.use_batchnorm and idx !=self.hidden_layer_num+1:
                grads["gamma"+str(idx)] = numerical_gradient(loss_W, self.params["gamma"+str(idx)])
                grads["beta"+str(idx)] = numerical_gradient(loss_W, self.params["beta"+str(idx)])
            

        return grads
    
    # BPを使用したgradientの関数
    def gradient(self, x, t):
        """
        各レイヤーを部品としてうまく分離して実装しているため、単純なfor文でforwardとbackwardを行うことができる。
        """

        # forward
        # lossを呼ぶことでその中でpredictが呼ばれ、順伝播が走る。
        self.loss(x, t,train_flg=True)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()

        # 入出力のshapeは対応するようになっているので下記のような単純なfor文でok
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        # backwardを走らせた後なので、各レイヤーのインスタンス変数は勾配を保持している。
        
        # twoからの変更は下記を直書きからforにしたことと、正則化への対応
        # あくまで各勾配は（正則化項を含む）最終的な損失関数を各パラで微分したものとしたい。ただし、BPによるdWは正則化項なしの誤差関数の微分になっている。
        # 最終的な損失関数のWによる微分は、正則化項目を除く部分の微分+λWなので、λW部分を加える必要がある。
        for idx in range(1,self.hidden_layer_num+2):
            grads["W"+str(idx)] =self.layers["Affine"+str(idx)].dW+self.weight_decay_lambda*self.layers["Affine"+str(idx)].W
            grads["b"+str(idx)] = self.layers["Affine"+str(idx)].db
            
            # ここだけ変更点
            # BNありで最終層でない時
            if self.use_batchnorm and idx !=self.hidden_layer_num+1:
                grads["gamma"+str(idx)] = self.layers["BatchNorm"+str(idx)].dgamma
                grads["beta"+str(idx)] = self.layers["BatchNorm"+str(idx)].dbeta

        return grads

    



    

    

    

    
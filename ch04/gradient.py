import numpy as np


def _numerical_gradient_1d(f, x):
    h = 1e-4
    # 関数の変数の個数分の次元を持つ（勾配用の）空配列を用意する
    grad = np.zeros_like(x)

    # 変数の個数だけ繰り返す
    # 毎回ある特定の要素だけを変化させた違う配列を作る必要があるため、ベクトル化演算のようなことはできず、ループを回す必要がある。
    for idx in range(x.size):

        tmp_val = x[idx]  # 注目している変数の元の値を保持しておく
        # 注目しているindexの要素のみhを加えたものに置き換える
        x[idx] = tmp_val + h
        # print(f"{idx}番目の変数についてhを加えたもの{x}")
        # 注目indexのみ上書きした配列xを代入することでf(x+h)の計算
        fxh1 = f(x)

        x[idx] = tmp_val - h
        # print((f"{idx}番目の変数についてhを引いたもの{x}"))
        # f(x-h)の計算
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        # 値を元に戻す、これをしないと次の（ループの）要素の偏微分で、一つ前の変数がすでに変化した状態になってしまう。
        # つまり、あくまで注目している変数以外は現在の値という状態を保つために必要
        x[idx] = tmp_val

    return grad


def numerical_gradient_2d(f, X):
    """
    行列を入力として、各行に対して_numerical_gradient_1dを適用する関数。
    特に実用性はなさそうな気がする

    """
    if X.ndim == 1:
        return _numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_1d(
                f, x
            )  # gradの各要素を、numerical_gradient_1dの結果（つまりベクトル）にする。→結果的にgradは二次元配列になる

        return grad


def numerical_gradient(f, x):
    """
    一般の多次元配列に対する数値勾配を求める。
    入力が行列でもテンソルでもそれを多数の独立変数の集まりとして見ている。入力の形式は関数の引数の表記方法なだけで本質的ではないことに注意

    今までは関数の入力がベクトルで、勾配もベクトルの各要素を関数をその要素の変数で微分したものになってたが
    今回は関数の入力が行列でも対応できる。
    勾配も行列orベクトルの各要素を、偏微分に置き換えたものになっている。
    """
    h = 1e-4
    # 関数の変数の個数分の次元を持つ（勾配用の）空配列を用意する
    grad = np.zeros_like(x)

    # np.nditer は、NumPy配列を要素単位で順にたどるためのイテレータ。
    # これを使用しないと入力配列の次元数に応じてその分だけのfor文を回さないといけない。

    it = np.nditer(
        x, flags=["multi_index"], op_flags=["readwrite"]
    )  # op_flagsはxの要素を書き換えるため。

    while not it.finished:
        idx = (
            it.multi_index
        )  # 今見ている要素の添え字をタプルとして取得。例えば x が2次元なら idx は (0, 1) みたいなタプルになる。すると x[idx] は x[0, 1] と同じ意味になる。
        tmp_val = x[idx]

        x[idx] = tmp_val + h
        fxh1 = f(x)  # 配列全体を関数に渡している（つまりちゃんと全変数を渡している）

        x[idx] = tmp_val - h
        fxh2 = f(x)  # 配列全体を関数に渡している（つまりちゃんと全変数を渡している）

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val

        it.iternext()

    return grad


"""
上記は下記のような実装にすることもできる。（np.ndindexは多次元添え字を順番に返してくれる）
for idx in np.ndindex(x.shape):
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x)

    x[idx] = tmp_val - h
    fxh2 = f(x)

    grad[idx] = (fxh1 - fxh2) / (2*h)
    x[idx] = tmp_val"""


def main():
    def function_2(x):
        return x[0] ** 2 + x[1] ** 2 + x[0] * x[1]

    x = np.array([3.0, 4.0])
    print(_numerical_gradient_1d(function_2, x))


if __name__ == "__main__":
    main()

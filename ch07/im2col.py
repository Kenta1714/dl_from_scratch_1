import numpy as np

def im2col(input_data,filter_h,filter_w,stride=1,pad=0):
    
    N,C,H,W=input_data.shape
    out_h=(H+2*pad-filter_h) // stride +1
    out_w=(W+2*pad-filter_w) // stride +1
    
    # np.padで配列の端に値を足すことができる。constantの場合は0埋め。
    # 引数については、N軸に前後0、C軸に前後0、H軸に前後padずつ（上下のpadに対応）、W軸に前後padずつ（左右のpadに対応）
    img=np.pad(input_data, [(0,0), (0,0), (pad,pad),(pad,pad)], "constant")
    
    # 一旦6次元配列を作る？
    col=np.zeros((N,C,filter_h,filter_w,out_h,out_w))
    
    # フィルターの位置についてループする
    # 例えば2×2なら左上、右上、左下、右下
    for y in range(filter_h):
        y_max=y+stride*out_h
        for x in range(filter_w):
            x_max=x+stride*out_w
            col[:,:,y,x,:,:]=img[:,:,y:y_max:stride,x:x_max:stride]
            
    col=col.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w,-1)
    
    return col
        

import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from alifebook_lib.visualizers import MatrixVisualizer

import time
delay = 10 #ms

#左から：定常的なパターン、縞模様、二相の振動、移動するパターン
T_L =    1.   # 1. # 2.5  # 1.25            
T_H =    1.2  # 5. # 5.   # 1.5
F =      2    # 2  # 2.   # 2.




def dec(x):
    return x/F
def inc(x):
    return x/F + (F-1)/F

# visualizerの初期化 (Appendix参照)
visualizer = MatrixVisualizer()

WIDTH = 100
HEIGHT = 100

state = np.zeros((HEIGHT,WIDTH), dtype=np.float)
next_state = np.empty((HEIGHT,WIDTH), dtype=np.float)

# 初期化
### ランダム ###
state = np.random.random_sample((HEIGHT,WIDTH))
### game_of_life_patterns.pyの中の各パターンを利用. 左上(2,2)の位置にセットする. ###
# pattern = game_of_life_patterns.OSCILLATOR
# state[2:2+pattern.shape[0], 2:2+pattern.shape[1]] = pattern
while visualizer:  # visualizerはウィンドウが閉じられるとFalseを返す
    for i in range(HEIGHT):
        for j in range(WIDTH):
            # 自分と近傍のセルの状態を取得
            # c: center (自分自身)
            # nw: north west, ne: north east, c: center ...
            nw = state[i-1,j-1]
            n  = state[i-1,j]
            ne = state[i-1,(j+1)%WIDTH]
            w  = state[i,j-1]
            c  = state[i,j]
            e  = state[i,(j+1)%WIDTH]
            sw = state[(i+1)%HEIGHT,j-1]
            s  = state[(i+1)%HEIGHT,j]
            se = state[(i+1)%HEIGHT,(j+1)%WIDTH]
            neighbor_cell_sum = nw + n + ne + w + e + sw + s + se

            if neighbor_cell_sum >= T_L and neighbor_cell_sum <= T_H:
                next_state[i,j] = inc(state[i,j])
            else:
                next_state[i,j] = dec(state[i,j])
    state, next_state = next_state, state
    # 表示をアップデート

    time.sleep(delay/1000)
    visualizer.update(1-state) # 1を黒, 0を白で表示する


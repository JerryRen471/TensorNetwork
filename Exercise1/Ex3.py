'''
编程练习：随机生成一个(2 × 3 × 3)维的 3 阶张量𝑇𝑎𝑏𝑐，并完成如下计算：
（a）计算收缩∑ 𝑇𝑎𝑏𝑐𝑇𝑎𝑐𝑑 𝑇𝑎𝑑𝑏 𝑎𝑏𝑐𝑑 ；
（b）引入高阶单位张量，画出上述收缩对应的图形表示。
'''

import numpy as np
import torch

T_abc = np.random.randint(0, 10, (2, 3, 3))
T_abc = torch.from_numpy(T_abc)

A_abed = torch.einsum('abc, ecd -> abed', T_abc, T_abc)
B_aef = torch.einsum('abed, fdb -> aef', A_abed, T_abc)
delta = torch.zeros(B_aef.shape)
for i in range(min(delta.shape)):
    delta[i,i,i] = 1

print(delta)
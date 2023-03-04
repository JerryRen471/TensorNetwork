'''
编程练习：建立所有张量元等于 2 的(2 × 2 × 3)维的 3 阶张量𝑇𝑎𝑏𝑐，进行如下计算：
（a）获得其矩阵化𝑻[𝑎]与𝑻[𝑏]，并计算两个矩阵的相乘𝑨 = 𝑻[𝑎] T 𝑻[𝑏]；
（b）计算𝐵𝑏𝑐𝑑𝑒 = ∑ 𝑇𝑎𝑏𝑐𝑇𝑑𝑎𝑒 𝑎；
（c）对（b）中所得的张量𝑩进行变形与指标交换，使得其与（a）中所得的𝑨同阶同维，验证𝑨是否与 𝑩完全相等。
'''
import numpy as np

def ten2mat(tensor, first_label=0):
    try:
        mat = np.reshape(np.moveaxis(tensor, first_label, 0), (tensor.shape[first_label], -1))
    except ValueError:
        dim1 = 1
        for i in first_label:  # type: ignore
            dim1 = dim1 * tensor.shape[i]
        mat = np.reshape(np.moveaxis(tensor, first_label, list(range(len(first_label)))), (dim1, -1))  # type: ignore
    return mat

T_abc = np.array(list(range(2*2*3))).reshape((2, 2, 3))
print(T_abc)
T_a = ten2mat(T_abc, 0)
print(T_a)
T_b = ten2mat(T_abc, 1)
print(T_b)
A = T_a.T @ T_b
print(A)

B = np.tensordot(T_abc, T_abc, axes=(0, 1))
print(B)

C = np.reshape(B, A.shape)
print(A==C)
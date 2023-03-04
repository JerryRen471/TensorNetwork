# 随机生成一个(2 × 6 × 4 × 3)维的 4 阶张量，依次将其变形成(12 × 4 × 3)维，(2 × 24 × 3) 维，(12 × 12)维，最后变形回(2 × 6 × 4 × 3)维，验证最后所得的张量与原张量完全相等。
import numpy as np

np.random.seed(10)
a = np.array(list(range(144)))
a0 = a.reshape((2, 6, -1, 3))

a1 = a.reshape((12, 4, 3))

a2 = a.reshape((2, 24, 3))

a3 = a.reshape((12, 12))

a4 = a.reshape((2, 6, 4, 3))
print('(2 × 6 × 4 × 3)')
print(a0)
print('(12 × 4 × 3)')
print(a1)
print('(2 × 24 × 3)')
print(a2)
print('(12 × 12)')
print(a3)
print('(2 × 6 × 4 × 3)')
print(a4)
print(a4==a0)

# reshape函数将张量依序折叠或展开
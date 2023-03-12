import torch as tc

def ten_perm(x, pos, pos_first=False):
    perm = list(_ for _ in range(len(x.shape)))
    pos_dim = 1
    shape = list(x.shape)
    for _ in pos:
        perm.remove(_)
        d_ = x.shape[_]
        pos_dim *= d_
        shape.remove(d_)
    if pos_first:
        perm = pos + perm
    else:
        perm = perm + pos
    return x.permute(perm), pos_dim, shape

def tmul(x, y, pos_x=[], pos_y=[]):
    x_new, mul_dim_x, shape_x = ten_perm(x, pos_x)
    y_new, mul_dim_y, shape_y = ten_perm(y, pos_y, pos_first=True)

    shape = shape_x + shape_y
    result = x_new.reshape(-1, mul_dim_x).mm(y_new.reshape(mul_dim_y, -1))
    result = result.reshape(shape)
    return result

if __name__ == '__main__':
    y = tc.randn(5, 2, 4, 7, 8)
    x = tc.randn(6, 2, 3, 4)
    z = tmul(x, y, pos_x=[1, 3], pos_y=[1, 2])
    n = len(y)
    i = 1
    perm = [_ for _ in range(2, i+2)] + [0, 1] + [_ for _ in range(i+2, n)]
    z = z.permute(perm)
    print(z.shape)
    pass
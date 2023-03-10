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

def tmul(x, y, pos_x, pos_y):
    x_new, mul_dim_x, shape_x = ten_perm(x, pos_x)
    y_new, mul_dim_y, shape_y = ten_perm(y, pos_y, pos_first=True)

    shape = shape_x + shape_y
    result = x_new.reshape(-1, mul_dim_x).mm(y_new.reshape(mul_dim_y, -1))
    result = result.reshape(shape)
    return result

class HeisenbergModel:
    def __init__(self, num_sys, J=[0, 0, 1], h=[0, 0, 1], device = 'cpu'):
        self.num_sys = num_sys
        self.H = []
        Sx = tc.tensor([[0., 1.], [1., 0.]], dtype=tc.complex64, device=device)
        Sy = tc.tensor([[0., 0.-1.j], [0.-1.j, 0.]], dtype=tc.complex64, device=device)
        Sz = tc.tensor([[1., 0.], [0., -1.]], dtype=tc.complex64, device=device)
        S = [Sx, Sy, Sz]
        I2 = tc.tensor([[1., 0.], [0., 1.]], dtype=tc.complex64, device=device)
        Hij = 0
        for i in range(3):
            Hij += J[i]*tc.kron(S[i], S[i]) + h[i]*tc.kron(S[i], I2) + h[i]*tc.kron(I2, S[i])
        self.Hij = Hij
        pass

if __name__ == '__main__':
    num_sys = 6
    J = [0, 0, 2]
    h = [0, 0, 1]
    model = HeisenbergModel(num_sys, J, h)
    print(model.Hij)
    x = tc.ones([2, 3, 4, 5, 2])
    y = tc.ones([2, 5, 4, 3])
    print(x.shape, '\n', y.shape)
    z = tmul(x, y, pos_x=[0, 2, 3], pos_y=[0, 2, 1])
    print(z.shape)
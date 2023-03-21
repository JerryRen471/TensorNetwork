import torch as tc
import numpy as np
import scipy as sp
from scipy.sparse.linalg import LinearOperator

from Tools import *

class HeisenbergModel:
    def __init__(self, num_sys, J=[0, 0, 1], h=[0, 0, 1], device = 'cpu'):
        self.device = device
        self.num_sys = num_sys
        self.H = []
        Sx = tc.tensor([[0., 1.], [1., 0.]], dtype=tc.complex64, device=device)
        Sy = tc.tensor([[0., 0.-1.j], [0.-1.j, 0.]], dtype=tc.complex64, device=device)
        Sz = tc.tensor([[1., 0.], [0., -1.]], dtype=tc.complex64, device=device)
        S = [Sx, Sy, Sz]
        I2 = tc.tensor([[1., 0.], [0., 1.]], dtype=tc.complex64, device=device)
        Hij = 0
        Hi = 0
        for i in range(3):
            Hij += J[i]*tmul(S[i], S[i]) + h[i]*tmul(I2, S[i])
            Hi += h[i]*S[i]
        self.Hij = Hij
        self.Hi = Hi
        pass

    def H_function(self, phi):
        phi_1 = phi
        n = int(np.log2(len(phi_1)))
        shape = [2 for _ in range(n)]
        phi_1 = phi_1.reshape(shape)
        phi_2 = 0
        for i in range(n-1):
            phi_3 = tmul(self.Hij, phi_1, pos_x=[1, 3], pos_y=[i, i+1])
            perm = [_ for _ in range(2, i+2)] + [0, 1] + [_ for _ in range(i+2, n)]
            phi_3 = phi_3.permute(perm)
            phi_2 += phi_3
        phi_2 += tmul(self.Hi, phi_1, pos_x=[1], pos_y=[0])
        return phi_2.reshape(2**n, -1)
    
    def H_matmat(self, A):
        B = tc.zeros(A.shape, dtype=tc.complex64)
        for i in range(A.shape[1]):
            B[:, [i]] = self.H_function(A[:, i])
        return B

    def ground_state_grad(self, phi0, lr=0.01):
        phi0.requires_grad_()
        optimizer = tc.optim.Adam([phi0], lr=lr)
        for _ in range(10000):
            loss = tc.dot(phi0.conj(), self.H_function(phi0).flatten()) / (tc.dot(phi0.conj(), phi0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if _%1000 == 0:
                print('step={}: phi={}, E={}'.format(_, phi0, loss))
        return phi0, loss

if __name__ == '__main__':
    num_sys = 3
    J = [0, 0, 2]
    h = [0, 0, 1]
    model = HeisenbergModel(num_sys, J, h)
    print(model.Hij)
    print(model.Hij.shape)
    phi = tc.randn(2**num_sys, dtype=tc.complex64)
    print(phi.shape)
    e_vals, e_vecs = tc.eig(model.H_matmat(tc.eye(2**num_sys, dtype=tc.complex64)), eigenvectors=True)
    print(e_vals)
    print(e_vecs[0])
    print(model.H_function(e_vecs[0]))
    phi0, loss = model.ground_state_grad(phi0=tc.randn(2**num_sys, dtype=tc.complex64), lr=1e-3)
    # x = tc.ones([2, 3])
    # y = tc.ones([2])
    # print(x.shape, '\n', y.shape)
    # z = tmul(x, y, pos_x=[], pos_y=[])
    # print(z)
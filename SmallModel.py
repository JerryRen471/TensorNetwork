import torch as tc

class HeisenbergModel:
    def __init__(self, num_sys, J=[0, 0, 1], h=[0, 0, 1], device = 'cpu'):
        self.num_sys = num_sys
        self.H = []
        Sx = tc.tensor([[0., 1.], [1., 0.]], dtype=complex64, device=device)
        Sy = tc.tensor([[0., 0.-1.j], [0.-1.j, 0.]], dtype=complex64, device=device)
        Sz = tc.tensor([[1., 0.], [0., -1.]], dtype=complex64, device=device)
        S = [Sx, Sy, Sz]
        I2 = tc.tensor([1., 0.], [0., 1.])
        Hij = 0
        for i in range(3):
            Hij += J[i]*tc.kron(S[i], S[i]) + h[i]*tc.kron(S[i], I2) + h[i]*tc.kron(I2, S[i])
        pass
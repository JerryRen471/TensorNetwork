import numpy as np
import torch

from PermList import *

def normalize(x):
    pass

class PureState():
    def __init__(self, tensor=None, nq=None):
        if tensor == None:
            tensor = torch.randn([2]*nq)
        self.tensor = normalize(tensor)
        if nq == None:
            nq = len(tensor.shape)
        self.nq = nq
    
    def act_gate(self, gate, pos_act, pos_crl=None):
        if pos_crl == None:
            pos_crl = []
        perm = list(i for i in range(self.nq))
        for i in (pos_act + pos_crl):
            perm.pop(i)
        perm = pos_act + perm + pos_crl
        n_act = len(pos_act)
        n_crl = len(pos_crl)
        shape = self.tensor.permute(perm).shape
        state1 = self.tensor.permute(perm).reshape(2**n_act, -1, 2**n_crl)
        state1_ = state1[:, :, :-1] # the part that is not operated by gate
        state2_ = gate.reshape(-1, 2**n_act).mm(state1[:, :, -1])
        state1 = torch.cat([state1_, state2_.reshape(state2_.shape + (1,))], dim = -1)
        state1 = state1.reshape(shape)
        perm_ = PermList(perm)
        self.tensor = state1.permute(perm_.inverse())
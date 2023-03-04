def smallest_first(x):
    first = x.index(min(x))
    full_size = list(i for i in range(max(x)))
    x_ = x[:first]
    x = x[first:] + x_
    return x

class SwitchGroup(list):
    def __init__(self, a):
        super(SwitchGroup, self).__init__(a)
        # self.list = smallest_first(a)
        self.order = len(a)
        pass

    def init(self):
        first = self.index(min(self))
        self_ = self[:first]
        self[:] = self[first:] + self_
        self.order = len(self[:])

    def power(self, n):
        n = n % self.order
        a2n = []
        for i in range(self.order):
            a2n.append(self[(i * n) % self.order])
        return SwitchGroup(a2n)

    def operate(self, l):
        self.append(self[0])
        l_new = l[:]
        for i in range(len(self)-1):
            l_new[self[i]] = l[self[i+1]]
        self.pop()
        return l_new

class PermList:
    def __init__(self, x):
        self.list = x
        self.op_form = self.breakdown()
    
    def breakdown(self):
        operations = []
        for i in self.list:
            op = SwitchGroup([])
            a = i
            while (a in op) != True:
                op.append(a)
                a = self.list[a]
            op.init()
            if (op in operations) or (len(op) <= 1):
                continue
            operations.append(op)
        return operations

    def build_up(self):
        l = list(i for i in range(len(self.list)))
        for op in self.op_form:
            l = op.operate(l)
        return l

    def inverse(self):
        l = list(i for i in range(len(self.list)))
        for op in self.op_form:
            l = op.power(op.order-1).operate(l)
        return l

if __name__ == '__main__':
    x = PermList([2, 3, 4, 0, 1])
    print(x.list)
    l = list(i for i in range(len(x.list)))
    y = x.inverse()
    print(x.list, y)
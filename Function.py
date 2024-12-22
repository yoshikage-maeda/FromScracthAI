import numpy as np
import heapq

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class CustomTapple:
    def __init__(self, priority, data):
        self.priority = priority # 比較に使う値
        self.data = data # 比較に使わない
    
    def __lt__(self, other):
        return self.priority < other.priority # priorityだけで比較

class Variable:
    # 変数
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad = None # 微分が計算されたときにその値を設定する。
        self.creator = None
        self.generation = 0

    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()
        def add_func(f):
            if f not in seen_set:
                heapq.heappush(funcs, CustomTapple(-f.generation, f))
                seen_set.add(f)
        add_func(self.creator)
        while funcs:
            f_info = heapq.heappop(funcs)
            f = f_info.data  # 関数の情報を取得
            
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)  # 1要素のタプルに変換する。
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)
    def cleargrad(self):
        # 微分を初期化
        self.grad = None


class Function:
    # 関数
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)  # 1要素のタプルに変換する。
        outputs = [Variable(as_array(y)) for y in ys]

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self) # 出力変数に生みの親を覚えさせる。
        self.inputs = inputs
        self.outputs = outputs # 出力も覚える。
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        # Functionクラスのforwardクラスを使った人に、継承先で実装すべきなことを教えてあげる
        raise NotImplementedError()
    
    def backward(self, gy):
        # 逆伝播
        raise NotImplementedError()

class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(np.asarray(x.data - eps))
    x1 = Variable(np.asarray(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


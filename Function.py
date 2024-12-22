import numpy as np

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Variable:
    # 変数
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad = None # 微分が計算されたときにその値を設定する。
        self.creator = None

    
    def set_creator(self, func):
        self.creator = func
    
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 関数を取得
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function:
    # 関数
    def __call__(self, inputs: Variable):
        xs = [x.data for x in inputs.data]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self) # 出力変数に生みの親を覚えさせる。
        self.inputs = inputs
        self.outputs = outputs # 出力も覚える。
        return outputs
    
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
        x = self.input.data
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

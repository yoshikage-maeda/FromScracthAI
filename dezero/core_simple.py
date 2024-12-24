import numpy as np
import heapq
import weakref
import contextlib

class Function:
    # 関数
    def __call__(self, *inputs):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)  # 1要素のタプルに変換する。
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self) # 出力変数に生みの親を覚えさせる。
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs] # 出力も覚える。
        return outputs if len(outputs) > 1 else outputs[0]
    
    def forward(self, x):
        # Functionクラスのforwardクラスを使った人に、継承先で実装すべきなことを教えてあげる
        raise NotImplementedError()
    
    def backward(self, gy):
        # 逆伝播
        raise NotImplementedError()

class Variable:
    # 変数
    __array_priority__ = 200
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported")
        self.data = data
        self.grad = None # 微分が計算されたときにその値を設定する。
        self.creator = None
        self.generation = 0
        self.name = name
    

    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' '* 9 )
        return 'variable(' + p + ')'
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1
    
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            self.grad = Variable(np.ones_like(self.data))
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
            gys = [output().grad for output in f.outputs] #[output().grad for output in f.outputs]

            with using_config('enable_backprop', create_graph):
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
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # yはweakref
    def cleargrad(self):
        # 微分を初期化
        self.grad = None

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def dtype(self):
        return self.data.dtype
    
    @property
    def size(self):
        return self.data.size

class CustomTapple:
    def __init__(self, priority, data):
        self.priority = priority # 比較に使う値
        self.data = data # 比較に使わない
    
    def __lt__(self, other):
        return self.priority < other.priority # priorityだけで比較

# 演算子
class Add(Function):
    def forward(self, x0, x1):
        y = x0 + x1
        return y
    
    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

class Sub(Function):
    def forward(self, x0, x1):
        y = x0 - x1
        return y
    
    def backward(self, gy):
        return gy, -gy

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs
        return gy / x1, -gy * (x0 / x1 ** 2)

class Pow(Function):
    def __init__(self, c):
        self.c = c
    
    def forward(self, x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        x,  = self.inputs
        c = self.c
        gy = gy * c * (x ** (c - 1))
        return gy

def pow(x, c):
    return Pow(c)(x)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

# Config 微分するかを管理
class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config('enable_backprop', False)

# 補助関数
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

#　演算子のオーバーロード
def setup_variable():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow

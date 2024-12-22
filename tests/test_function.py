import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# 親ディレクトリのパスを取得してsys.pathに追加

import numpy as np
from Function import Function, Variable, square

def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(np.asarray(x.data - eps))
    x1 = Variable(np.asarray(x.data + eps))
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


class TestSquare:
    def test_forward(self):
        x = Variable(np.array(2.0))
        expected = np.array(4.0)
        assert expected == square(x).data
    
    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected_grad = np.array(6.0)
        assert expected_grad == x.grad.data
    
    def test_grad_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        numerical_grad = numerical_diff(square, x)
        assert np.allclose(numerical_grad, x.grad.data)

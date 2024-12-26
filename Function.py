import numpy as np
from dezero import Variable, Function
from dezero.utils import plot_dot_graph
import dezero.functions  as F

if '__file__' in globals():
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
def sphere(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * (30 + (2*x - 3*y)**2  * (18 -32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

def rosenrock(x0, x1):
    y = 100 * (x1 - x0**2) ** 2 + (x0 - 1) ** 2
    return y


# x = Variable(np.array(2.0))
# y = x ** 2
# y.backward(create_graph=True)
# gx = x.grad
# print(gx)
# gx.name = 'gx'
# x.cleargrad()

# z = gx ** 3 + y
# z.backward()
# x.name = 'x'
# y.name = 'y'
# z.name = 'z'
# print(x.grad)
# plot_dot_graph(z, verbose=False, to_file=f'samle.png')

# x = Variable(np.array(3.0))
# y = F.sin(x)
# y.backward(create_graph=True)
# gx = x.grad

# x.cleargrad()
# gx.backward(create_graph=True)
# gx2 = x.grad
# gx.name = 'gx'
# x.name = 'x'
# y.name ='y'
# plot_dot_graph(gx2, verbose=False, to_file=f'sin.png')

x = Variable(np.array(3.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)
gx_list = []
for iters in range(3):
    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        gx.name = 'gx' + str(i)
        gx_list.append(gx)

    gxn = x.grad
    gxn.name = 'gx' + str(iters+1)
    plot_dot_graph(gxn, verbose=False, to_file=f'tanh_{iters}.png')













# def numerical_diff(f, x, eps=1e-4):
#     x0 = Variable(np.asarray(x.data - eps))
#     x1 = Variable(np.asarray(x.data + eps))
#     y0 = f(x0)
#     y1 = f(x1)
#     return (y1.data - y0.data) / (2 * eps)


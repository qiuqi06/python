import scipy
import numpy as np
import matplotlib.pyplot as plt

a = np.array(range(0, 10))
b = 2 * a + np.random.randn(10)
# print(a)
# print(b)

plt.scatter(a, b, marker="x")


def hello():
    """
    >>> hello()
    >>> print("hello")
    """
    print("ss")


def train_lr(Xb, y):
    pass


def get_derivative(matrix, thi, y):
    """
    :param matrix: m,n
    :param thi: n,1
    :param y:
    :return:
    """
    return matrix.T * (matrix * thi - y)


def ss() -> str:
    return 's'


def sigmoid(x):
    """
    :param x:
    :return:
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def vo(x):
    s:str=f'ss{x}'
import numpy as np
import matplotlib.pyplot as plt
import tensor as tf
weight_l=tf.Variable(tf.random_normal([2,10]))
def get_bias(shape):
    b=tf.Variable(tf.constant(0.001),shape=shape)
    return b
w=get_bias([11])
print(w)

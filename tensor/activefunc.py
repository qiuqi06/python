import numpy as np
import tensorflow as tf
def sigmoid(input):
    y=[1/float(1+np.exp(-x)) for x in input]
    return y
def relu(input):
    y=[x*[x>0] for x in input]
    return y
def tanh(input):
    y=[(np.exp(x)-np.exp(-x)/float(np.exp(x)+np.exp(-x))) for x in input]
    return y
def softplus(input):
    y=[np.log(1+np.exp(x)) for x in input]
    return y
x=[1,3,4]
y_sigmoid=tf.nn.sigmoid(x)
with tf.Session as sess:
    y_sigmoid=sess.run([y_sigmoid])
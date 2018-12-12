import numpy as np
import tensorflow as tf

x_data =np.random.rand(100).astype(np.float)
y_data=x_data*0.1+0.3
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
print(type(Weights))
biases=tf.Variable(tf.zeros([1]))
y=Weights*x_data+biases


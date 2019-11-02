import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data =np.random.rand(100).astype(np.float)

x=np.sort(x_data,axis=0)
print(x)
z=np.linspace(0,1,100)

plt.plot(x,z)
plt.show()

y_data=x_data*0.1+0.3
Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
print(type(Weights))
biases=tf.Variable(tf.zeros([1]))
y=Weights*x_data+biases



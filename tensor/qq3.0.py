import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# produce 200 random dot
x_data=np.linspace(-0.5,0.5,200)[:,np.newaxis]
print(x_data.shape)
noise=np.random.normal(0,0.02,x_data.shape)
y_date=np.square(x_data)+noise
# define two palceholder
x=tf.placeholder(tf.float32,[None,1])
y=tf.placeholder(tf.float32,[None,1])
# define neural middle layer
weight_l=tf.Variable(tf.random_normal([1,10]))
biase_l=tf.Variable(tf.zeros([1,10]))
plus_1= tf.matmul(x, weight_l) + biase_l
l1=tf.nn.tanh(plus_1)
# define output layer
weight_2=tf.Variable(tf.random_normal([10,1]))
biase_2=tf.Variable(tf.zeros([1,10]))
plus_2=tf.matmul(l1,weight_2)+biase_2
prediction=tf.nn.tanh(plus_2)
loss=tf.reduce_mean(tf.square(y-prediction))
# use descent train data
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
print(train_step)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict={x:x_data,y:y_date})
    prediction_value=sess.run(prediction,feed_dict={x:x_data})
    plt.figure()
    plt.scatter(x_data,y_date)
    plt.plot(x_data,prediction_value,'r-',lw=5)
    plt.show()


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
BATCH_SIZE=30
REGULARIZER=0.01
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DECAY=0.999
seed=2
STEP=40000
##########正则化################
# plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
# plt.show()
def generate():
    rdm = np.random.RandomState(seed)
    X = rdm.randn(300, 2)
    Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
    Y_c = [['red' if y else 'blue'] for y in Y_]
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)
    return X,Y_,Y_c

def get_weight(shape,regularizer):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w
def get_bias(shape):
    b=tf.Variable(tf.constant(0.001,shape=shape))
    return b
def forward(x,regularizer):
    w1 = get_weight([2, 11], regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    w2 = get_weight([11, 1], regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1, w2) + b2
    return y

def backward():
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    X, Y_, Y_c=generate()
    y=forward(x,REGULARIZER)
    global_step=tf.Variable(0,REGULARIZER)
    # 学习率指数衰减
    learning_rate=tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,300/BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )
    loss_mes = tf.reduce_mean(tf.square(y - y_))
    loss_total = loss_mes + tf.add_n(tf.get_collection("losses"))
    train_step=tf.train.AdamOptimizer(0.0001).minimize(loss_mes)
    init=tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        ckpt=tf.train.get_checkpoint_state("/tmp")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_pat)
        for i in range(STEP):
            start=(i*BATCH_SIZE)%300
            end=start+BATCH_SIZE
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
            if i%2000==0:
                loss_mes_v=sess.run(loss_mes,feed_dict={x:X,y_:Y_})
                print("after %d steps loss is %f"%(i,loss_mes_v))
                # saver.save(sess,os.path.join("path"),global_step=global_step)
        xx,yy=np.mgrid[-3:3:.01,-3:-3:.01]
        grid=np.c_[xx.ravel(),yy.ravel()]
        probs=sess.run(y,feed_dict={x:grid})
        probs=probs.reshape(xx.shape)

    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show()
backward()


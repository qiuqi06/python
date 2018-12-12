import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
print(type(mnist))
print(mnist)
batch_size=100
n_batch=mnist.train.num_examples//batch_size
# define placeholder
with tf.name_scope('input'):
    x=tf.placeholder(tf.float32,[None,784],name='x_input')
    y=tf.placeholder(tf.float32,[None,10],name='y_input')
keep_prob=tf.placeholder(tf.float32)
lr=tf.Variable(0.001,dtype=tf.float32)

# neural
w1=tf.Variable(tf.truncated_normal([784,500],stddev=0.1))
b1=tf.Variable(tf.zeros([500])+0.1)
L1=tf.nn.tanh(tf.matmul(x,w1)+b1)
L1_drop=tf.nn.dropout(L1,keep_prob)

w2=tf.Variable(tf.truncated_normal([500,300],stddev=0.1))
b2=tf.Variable(tf.zeros([300])+0.1)
L2=tf.nn.tanh(tf.matmul(L1_drop,w2)+b2)
L2_drop=tf.nn.dropout(L2,keep_prob)

w3=tf.Variable(tf.truncated_normal([300,10],stddev=0.1))
b3=tf.Variable(tf.zeros([10])+0.1)
prediction=tf.nn.softmax(tf.matmul(L2_drop,w3)+b3)

# loss
# loss=tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# grad descent
train_step=tf.train.AdamOptimizer(lr).minimize(loss)

init=tf.global_variables_initializer()
# bool list
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #返回一维张量最大值的位置
# correct rate
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    writer=tf.summary.FileWriter('c:\\zoodata',tf.get_default_graph())
    for epoch in range(51):
        sess.run(tf.assign(lr,0.001*(0.95**epoch)))
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch((batch_size))
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})
        learning_rate=sess.run(lr)
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        # train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels})
        print("ITer"+str(epoch)+",Test Accuracy"+str(test_acc)+"，train Accuracy"+str(learning_rate))



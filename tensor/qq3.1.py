import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples//batch_size
# define placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

# neural
w=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
prediction=tf.nn.softmax(tf.matmul(x,w)+b)

# loss
# loss=tf.reduce_mean(tf.square(y-prediction))
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
# grad descent
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init=tf.global_variables_initializer()
# bool list
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #返回一维张量最大值的位置
# correct rate
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch((batch_size))
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        train_acc=sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels})
        print("ITer"+str(epoch)+",Test Accuracy"+str(test_acc)+"，train Accuracy"+str(train_acc))


